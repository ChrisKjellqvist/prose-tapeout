package prose.nn_util.norm

import beethoven.Generation.CppGeneration
import beethoven._
import beethoven.common._
import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper.FPFloatFormat


class LayerNormCore(maxSetSize: Int,
                    maxIntermediateSets: Int,
                    maxNRepeats: Int,
                    dataType: FPFloatFormat.Type,
                    subLatency: Int = 2,
                    fmaLatency: Int = 3,
                    sqrtLUTLatency: Int = 1,
                   )(implicit p: Parameters) extends AcceleratorCore {
  // make sure all of the necessary LUTs are generated
  locally {
    os.makeDir.all(os.pwd / "luts")
    try {
      os.proc("cmake", os.pwd / "src" / "main" / "c" / "generate_verilog").call(cwd = os.pwd / "luts")
    } catch {
      case _: os.SubprocessException => // try to delete the lut directory and try again
        os.remove.all(os.pwd / "luts")
        os.makeDir.all(os.pwd / "luts")
        os.proc("cmake", os.pwd / "src" / "main" / "c" / "generate_verilog").call(cwd = os.pwd / "luts")
    }
    os.proc("make", "generate_invsqrt").call(cwd = os.pwd / "luts")
    os.proc("./generate_invsqrt", os.pwd / "luts").call(cwd = os.pwd / "luts")
  }

  implicit val dataWidthBytes = dataType match {
    case fpwrapper.FPFloatFormat.Fp16Alt => 2
    case fpwrapper.FPFloatFormat.Fp32 => 4
    case fpwrapper.FPFloatFormat.Fp64 => 8
    case _ => throw new Exception("Unsupported data type")
  }
  val dataWidthBits = dataWidthBytes * 8
  implicit val fpuDataType = dataType match {
    case fpwrapper.FPFloatFormat.Fp16Alt => fpwrapper.FPFloatFormat.Fp32
    case _ => dataType
  }
  implicit val fpuLoadData: UInt => UInt = dataType match {
    case fpwrapper.FPFloatFormat.Fp16Alt => (x: UInt) => Cat(x, 0.U(16.W))
    case _ => (x: UInt) => x
  }
  implicit val fpuUnpackResult: UInt => UInt = dataType match {
    case fpwrapper.FPFloatFormat.Fp16Alt => (x: UInt) => x(31, 16)
    case _ => (x: UInt) => x
  }

  // how many variances do we need to store at once to support this stream?
  val io = BeethovenIO(new AccelCommand("norm") {
    // batch size of the input matrix (batch is the last dimension in input matrix)!
    val nIntermediateSets = UInt(log2Up(maxIntermediateSets + 1).W)
    // address of gamma_beta segment in memory
    val gamma_beta = Address()
    // signal to fetch new gamma_beta parameters
    val refresh_gb = Bool()
    // input matrix address (will be re-used multiple times)
    val input = Address()
    // output matrix address
    val output = Address()
    // When computing E[x] and Var[x], we need to divide by N. We do this by multiplying by 1/N which is provided here
    // as a BF16 value. This means we don't need a reciprocal unit to compute the division.
    val norm = UInt(dataWidthBits.W)
    // max number of elements in a set. e.g., |x| for which you might compute E[x] or Var[x]
    val setSize = UInt(log2Up(maxSetSize + 1).W)
    // how many times to consecutively run layernorm on contiguous sets of data
    val nRepeats = UInt(log2Up(maxNRepeats + 1).W)
    // normType
    val norm_ty = UInt(NormType.width.W)
  }, EmptyAccelResponse())
  val normTyReg = Reg(UInt(NormType.width.W))

  CppGeneration.addUserCppDefinition("uint8_t", "flagLayerNorm", NormType.LayerNorm, None)
  CppGeneration.addUserCppDefinition("uint8_t", "flagRMSNorm", NormType.RMSNorm, None)

  // stream in input set multiple times so we don't have to store it
  val meanRead = getReaderModule("mean")
  val varRead = getReaderModule("variance")
  val layerNormRead = getReaderModule("lnorm")

  // gamma,beta parameters for each row should be stored next to each other: g1, b1, g2, b2, etc...
  val gammaBeta = getScratchpad("gamma_beta")

  // output stream for normalized operands
  val outputWrite = getWriterModule("output")

  Seq(
    (meanRead.requestChannel, io.req.bits.input, false),
    (varRead.requestChannel, io.req.bits.input, true),
    (layerNormRead.requestChannel, io.req.bits.input, false),
    (outputWrite.requestChannel, io.req.bits.output, false)).foreach {
    case (reqChan, addr, maskOnRMS) =>
      reqChan.valid := (if (maskOnRMS) io.req.bits.norm_ty === NormType.LayerNorm.U && io.req.fire else io.req.fire)
      reqChan.bits.addr := addr
      // shift by 1 to account for width of BF16
      reqChan.bits.len := Cat(io.req.bits.setSize * io.req.bits.nIntermediateSets * io.req.bits.nRepeats, 0.U(1.W))
      when(io.req.fire) {
        if (maskOnRMS) {
          assert(reqChan.ready || io.req.bits.norm_ty =/= NormType.LayerNorm.U)
        } else {
          assert(reqChan.ready)
        }
      }
  }

  gammaBeta.dataChannels(0).req.valid := false.B

  val tagCounterMean, tagCounterVariance, tagCounterLN, tagMax, tileRowCntr, tileMax = Reg(UInt(log2Up(maxIntermediateSets + 1).W))
  val meanRoundCounter, varRoundCounter, LNRoundCounter, setSize = Reg(UInt(log2Up(maxSetSize + 1).W))
  val LNTileCounter = Reg(UInt(log2Up(maxNRepeats + 1).W))
  val norm = Reg(UInt(dataWidthBits.W))

  val means = Module(new Mean(maxIntermediateSets, dataWidthBytes, fmaLatency, fpuDataType, fpuLoadData, fpuUnpackResult))
  means.io.norm_ty := normTyReg
  val mean_s_idle :: mean_core_start :: mean_stream :: mean_transfer :: mean_finish :: Nil = Enum(5)
  val meanState = RegInit(mean_s_idle)
  val trigger_mean_output = Reg(Bool())
  val gamma_beta_valid = RegInit(false.B)

  val refresh_is_idle = { // ---------------------- gamma beta refresh ----------------------
    val refresh_idle :: refresh_active :: Nil = Enum(2)
    val s_refresh = RegInit(refresh_idle)

    gammaBeta.requestChannel.init.valid := false.B
    // multiple by 4 (cat with 00) because it's two 2-byte operands (gamma and beta)
    gammaBeta.requestChannel.init.bits.len := Cat(io.req.bits.setSize, 0.U(2.W))
    gammaBeta.requestChannel.init.bits.memAddr := io.req.bits.gamma_beta
    // TODO THIS USED TO BE 1.U, MAKE SURE I DIDN'T MAKE A MISTAKE TURNING IT TO 0
    gammaBeta.requestChannel.init.bits.scAddr := 0.U

    when(io.req.fire && io.req.bits.refresh_gb) {
      s_refresh := refresh_active
      gamma_beta_valid := false.B
      gammaBeta.requestChannel.init.valid := true.B
      assert(gammaBeta.requestChannel.init.ready)
    }

    when(s_refresh === refresh_active) {
      when(gammaBeta.requestChannel.init.ready) {
        s_refresh := refresh_idle
        gamma_beta_valid := true.B
      }
    }

    s_refresh === refresh_idle
  }

  io.req.ready := meanState === mean_s_idle && refresh_is_idle
  means.io.begin.valid := meanState === mean_core_start
  means.io.begin.bits := norm

  val mean_can_stream = meanState === mean_stream
  means.io.input.valid := meanRead.dataChannel.data.valid && mean_can_stream
  means.io.input.bits.tag := tagCounterMean
  means.io.input.bits.data := meanRead.dataChannel.data.bits
  means.io.done := false.B
  means.io.output.ready := false.B
  meanRead.dataChannel.data.ready := means.io.input.ready && mean_can_stream
  when(means.io.input.fire) {
    tagCounterMean := tagCounterMean + 1.U
    when(tagCounterMean === tagMax) {
      tagCounterMean := 0.U
      meanRoundCounter := meanRoundCounter + 1.U
      when(meanRoundCounter === setSize) {
        meanState := mean_transfer
        meanRoundCounter := 1.U
        trigger_mean_output := true.B
      }
    }
  }

  val variance = Module(new Variance(maxIntermediateSets,
    subLatency,
    fmaLatency,
    sqrtLUTLatency,
    dataWidthBytes,
    fpuDataType,
    fpuLoadData,
    fpuUnpackResult))
  variance.io.begin.bits.means := means.io.output.bits
  variance.io.begin.bits.norm := norm
  variance.io.begin.valid := false.B
  variance.io.norm_ty := normTyReg


  when(io.req.fire) {
    tagCounterMean := 0.U
    norm := io.req.bits.norm
    tagMax := io.req.bits.nIntermediateSets - 1.U
    tileMax := io.req.bits.nRepeats
    tileRowCntr := 0.U
    meanRoundCounter := 1.U
    setSize := io.req.bits.setSize
    meanState := mean_core_start
    gamma_beta_valid := false.B
    normTyReg := io.req.bits.norm_ty
  }
  when(meanState === mean_core_start) {
    trigger_mean_output := false.B
    when(means.io.begin.fire) {
      tileRowCntr := tileRowCntr + 1.U
      meanState := mean_stream
    }
  }.elsewhen(meanState === mean_finish) {
    when(outputWrite.requestChannel.ready) {
      meanState := mean_s_idle
    }
  }.elsewhen(meanState === mean_transfer) {
    means.io.done := trigger_mean_output
    trigger_mean_output := false.B
    variance.io.begin.valid := means.io.output.valid
    means.io.output.ready := variance.io.begin.ready
    when(means.io.output.fire) {
      tagCounterVariance := 0.U
      varRoundCounter := 1.U
      when(tileRowCntr === tileMax) {
        meanState := mean_finish
      }.otherwise {
        meanState := mean_core_start
      }
    }
  }

  val variance_s_idle :: variance_s_acc :: variance_s_transfer :: Nil = Enum(3)
  val variance_state = RegInit(variance_s_idle)
  val trigger_variance_output = RegInit(false.B)

  variance.io.input.bits.tag := tagCounterVariance
  variance.io.input.bits.data := varRead.dataChannel.data.bits
  variance.io.input.valid := varRead.dataChannel.data.valid && variance_state === variance_s_acc
  varRead.dataChannel.data.ready := variance.io.input.ready && variance_state === variance_s_acc

  when(variance.io.input.fire) {
    tagCounterVariance := tagCounterVariance + 1.U
    when(tagCounterVariance === tagMax) {
      tagCounterVariance := 0.U
      varRoundCounter := varRoundCounter + 1.U
      when(varRoundCounter === setSize) {
        variance_state := variance_s_transfer
        varRoundCounter := 1.U
        trigger_variance_output := true.B
      }
    }
  }

  variance.io.computeVariances_valid := trigger_variance_output

  val layernorm = Module(new LayerNormStream(maxIntermediateSets, fmaLatency, dataWidthBytes, fpuDataType, fpuLoadData, fpuUnpackResult))
  layernorm.io.norm_ty := normTyReg

  layernorm.io.start_handshake.valid := variance.io.output.valid
  variance.io.output.ready := layernorm.io.start_handshake.ready
  layernorm.io.start_handshake.bits.means := variance.io.output.bits.means
  layernorm.io.start_handshake.bits.norms := variance.io.output.bits.variances

  when(variance_state === variance_s_idle) {
    when(variance.io.begin.fire) {
      variance_state := variance_s_acc
      trigger_variance_output := false.B
    }
  }.elsewhen(variance_state === variance_s_transfer) {
    when(variance.io.computeVariances_valid && variance.io.computeVariances_ready) {
      trigger_variance_output := false.B
    }
    when(!trigger_variance_output && variance.io.output.fire) {
      variance_state := variance_s_idle
    }
  }
  val gammabeta_queue = Module(new Queue(new Bundle {
    val gamma = UInt(16.W)
    val beta = UInt(16.W)
  }, 2, hasFlush = true))
  gammabeta_queue.io.flush.get := false.B
  val gamma_beta_idx = Reg(UInt(log2Up(maxSetSize + 1).W))
  val read_in_flight = Reg(Bool())

  when(io.req.fire) {
    gammabeta_queue.io.flush.get := true.B
    gamma_beta_idx := 0.U
    read_in_flight := false.B
  }
  gammaBeta.dataChannels(0).req.bits.write_enable := false.B
  gammaBeta.dataChannels(0).req.bits.data := DontCare
  gammaBeta.dataChannels(0).req.bits.addr := gamma_beta_idx
  when(gamma_beta_valid && gammabeta_queue.io.enq.ready && !read_in_flight) {
    gammaBeta.dataChannels(0).req.valid := true.B
    assert(gammaBeta.dataChannels(0).req.ready)
    read_in_flight := true.B
  }
  gammabeta_queue.io.enq.valid := gammaBeta.dataChannels(0).res.valid
  gammabeta_queue.io.enq.bits.gamma := gammaBeta.dataChannels(0).res.bits(31, 16)
  gammabeta_queue.io.enq.bits.beta := gammaBeta.dataChannels(0).res.bits(15, 0)
  when(gammaBeta.dataChannels(0).res.valid) {
    assert(gammabeta_queue.io.enq.ready)
    read_in_flight := false.B
    val next = gamma_beta_idx + 1.U
    gamma_beta_idx := next
    when(next === setSize) {
      gamma_beta_idx := 0.U
    }
  }
  layernorm.io.input_stream.bits.beta := gammabeta_queue.io.deq.bits.beta
  layernorm.io.input_stream.bits.gamma := gammabeta_queue.io.deq.bits.gamma
  layernorm.io.input_stream.bits.tag := tagCounterLN
  layernorm.io.input_stream.bits.x := layerNormRead.dataChannel.data.bits
  layernorm.io.resetStream := false.B

  val ln_s_idle :: ln_s_stream :: ln_s_transfer :: Nil = Enum(3)
  val ln_state = RegInit(ln_s_idle)

  layernorm.io.input_stream.valid := gammabeta_queue.io.deq.valid && layerNormRead.dataChannel.data.valid && ln_state === ln_s_stream
  gammabeta_queue.io.deq.ready := layernorm.io.input_stream.ready && layerNormRead.dataChannel.data.valid && tagCounterLN === tagMax && ln_state === ln_s_stream
  layerNormRead.dataChannel.data.ready := layernorm.io.input_stream.ready && gammabeta_queue.io.deq.valid && ln_state === ln_s_stream

  when(io.req.fire) {
    LNRoundCounter := 1.U
    LNTileCounter := 0.U
    tagCounterLN := 0.U
  }
  when(ln_state === ln_s_idle) {
    when(layernorm.io.start_handshake.fire) {
      ln_state := ln_s_stream
    }
  }.elsewhen(ln_state === ln_s_stream) {
    when(layernorm.io.input_stream.fire) {
      tagCounterLN := tagCounterLN + 1.U
      when(tagCounterLN === tagMax) {
        tagCounterLN := 0.U
        LNRoundCounter := LNRoundCounter + 1.U
        when(LNRoundCounter === setSize) {
          LNRoundCounter := 1.U
          ln_state := ln_s_transfer
          LNTileCounter := LNTileCounter + 1.U
        }
      }
    }
  }.elsewhen(ln_state === ln_s_transfer) {
    when(LNTileCounter === tileMax) {
      when (outputWrite.requestChannel.ready) {
        io.resp.valid := true.B
        when (io.resp.fire) {
          ln_state := ln_s_idle
          layernorm.io.resetStream := true.B
        }
      }
    }.otherwise {
      ln_state := ln_s_idle
      layernorm.io.resetStream := true.B
    }
  }

  outputWrite.dataChannel.data <> layernorm.io.output
}
