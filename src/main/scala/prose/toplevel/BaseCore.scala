package prose.toplevel

import beethoven.Generation.CppGeneration
import beethoven.MemoryStreams.{Memory, WriterDataChannelIO}
import chipsalliance.rocketchip.config._
import chisel3._
import chisel3.util._
import beethoven._
import beethoven.common.Address.addrBits
import beethoven.common._
import prose.activations._
import prose.err_handling.{ErrorHandle, ProseErr}
import prose.nn_util.SpecialFunction
import prose.nn_util.SpecialFunction.SpecialFunction
import prose.random.LinearFeedbackShiftRegister
import prose.splitToChunks
import prose.systolic_array.SystolicArray

import scala.util.Random

/**
 * @param kMax        maximum length row/column stream to stream in
 * @param Nmin        min dimension of systolic array on the accelerator. This determines alignment
 * @param N           size of systolic array in this core
 * @param SRAMLatency SRAM latency in cycles
 * @param fpuLatency  FPU FMA latency in cycles
 * @param maxBatch    maximum batch size. Minimum batch is determined by Nmax / N
 * @param specialFunc special operation with latency
 * @param p           constructor argument for Beethoven
 */
abstract class BaseCore(kMax: Int,
                        Nmin: Int,
                        N: Int,
                        SRAMLatency: Int,
                        fpuLatency: Int,
                        maxBatch: Int,
                        supportWideBias: Boolean,
                        maxColTilesSimul: Int,
                        specialFunc: Option[(SpecialFunction, Int)] = None,
                       )(implicit p: Parameters) extends AcceleratorCore {
  val useFakeLUTs: Boolean = false

  CppGeneration.addPreprocessorDefinition("PROSE")
  CppGeneration.addPreprocessorDefinition(
    Seq(
      (f"PROSE_${outer.systemParams.name}_N", N),
      (f"PROSE_Nmin", Nmin),
      (f"PROSE_maxBatch", maxBatch),
      (f"PROSE_kMax", kMax),
      (f"PROSE_${N}Core_maxColTiles", maxColTilesSimul)
    )
  )

  val useUpShift = specialFunc match {
    case Some((q, _)) => q != SpecialFunction.EXP
    case None => true
  }
  // TODO why do we need this...
  //  assert(maxBatch >= N / Nmin)

  val useFakeIO = RegInit(false.B)
  val useNorms = RegInit(false.B)

  val fio_cmd = BeethovenIO(new AccelCommand("setFakeIO") {
    val enable = Bool()
  })

  fio_cmd.req.ready := true.B
  when(fio_cmd.req.fire) {
    useFakeIO := fio_cmd.req.bits.enable
  }

  val rgen = new Random(1284884)

  def getReaderWrapper(name: String): (List[DecoupledIO[ChannelTransactionBundle]], List[DataChannelIO]) = if (useFakeLUTs) {
    val (reqs, dats) = getReaderModules(name)
    reqs.zipWithIndex.foreach(a => a._1.suggestName(f"requestChannel_${name}_${a._2}"))
    dats.zipWithIndex.foreach(a => a._1.suggestName(f"dataChannel_${name}_${a._2}"))
    val maxTxLen = kMax * N * maxBatch * 2
    val dwidth = dats(0).data.bits.getWidth
    val maxBeats = 1 + maxTxLen / (dwidth / 8)
    val beatCount = RegInit(0.U(log2Up(maxBeats + 1).W))
    beatCount.suggestName(f"beatCount_${name}")

    val _ = {
      // build the lut
      if (!os.exists(os.pwd / "luts" / "Normal.v")) {
        os.proc("cmake", os.pwd /  "src" / "main" / "c" / "generate_verilog").call(cwd = os.pwd / "luts")
        os.proc("make").call(cwd = os.pwd / "luts")
        os.proc("./generate_normal").call(cwd = os.pwd / "luts")
      }
    }

    val req_wrapper = reqs.zip(dats).zipWithIndex.map { case ((r, d), idx) =>
      r.suggestName(f"requestChannel_${name}_$idx")
      d.suggestName(f"dataChannel_${name}_$idx")
      val wrapperReq = Wire(r.cloneType)
      wrapperReq.suggestName(f"requestWrapper_${name}_$idx")
      val wrapperDat = Wire(d.cloneType)
      wrapperDat.suggestName(s"dataWrapper_${name}_$idx")

      wrapperReq.ready := r.ready
      r.valid := wrapperReq.valid
      r.bits := wrapperReq.bits

      wrapperDat.data.valid := d.data.valid
      wrapperDat.in_progress := d.in_progress
      wrapperDat.data.bits := d.data.bits
      d.data.ready := wrapperDat.data.ready


      val randomFloatGen = (0 until dwidth / 16) map { _ =>
        val perSlotUInt = VecInit((0 until 16) map { _ =>
          val lfsr = Module(new LinearFeedbackShiftRegister(13, List(0, 1, 2, 7)))
          lfsr.io.clock := clock.asBool
          lfsr.io.increment := useFakeIO && wrapperDat.data.ready
          lfsr.io.set_valid := reset.asBool
          val rinit = rgen.nextInt(1 << 13)
          lfsr.io.set_data := rinit.U
          lfsr.io.out
        }).asUInt

        val randomGen = Module(new LookupTableWithLatency("Normal", 1))
        randomGen.io.in := perSlotUInt
        randomGen.io.out
      }


      when(useFakeIO) {
        r.valid := false.B
        r.bits := DontCare
        wrapperReq.ready := beatCount === 0.U
        when(wrapperReq.fire) {
          beatCount := (wrapperReq.bits.len >> CLog2Up(dwidth / 8)).asUInt
        }

        d.data.ready := false.B
        wrapperDat.data.valid := beatCount =/= 0.U
        wrapperDat.data.bits := Cat(randomFloatGen.reverse)
        when(wrapperDat.data.fire) {
          beatCount := beatCount - 1.U
        }
      }

      (wrapperReq, wrapperDat)
    }
    (req_wrapper.map(_._1), req_wrapper.map(_._2))
  } else {
    getReaderModules(name)
  }

  def getWriterWrapper(name: String): (List[DecoupledIO[ChannelTransactionBundle]], List[WriterDataChannelIO]) = if (useFakeLUTs) {
    val (reqs, dats) = getWriterModules(name)
    reqs.zipWithIndex.foreach(a => a._1.suggestName(f"requestChannel_${name}_${a._2}"))
    dats.zipWithIndex.foreach(a => a._1.suggestName(f"dataChannel_${name}_${a._2}"))
    val maxTxLen = kMax * Nmin * maxBatch * 2
    val dwidth = dats(0).data.bits.getWidth
    val maxBeats = 1 + maxTxLen / dwidth
    val beatCount = RegInit(0.U(log2Up(maxBeats + 1).W))
    beatCount.suggestName(f"beatCount_${name}")
    //    println(s"beatCount $name: " + beatCount.getWidth)
    val req_wrapper = reqs.zip(dats).zipWithIndex.map { case ((r, d), idx) =>
      r.suggestName(f"requestChannel_${name}_$idx")
      d.suggestName(f"dataChannel_${name}_$idx")
      val wrapperReq = Wire(r.cloneType)
      wrapperReq.suggestName(f"requestWrapper_${name}_$idx")
      val wrapperDat = Wire(d.cloneType)
      wrapperDat.suggestName(s"dataWrapper_${name}_$idx")

      wrapperReq.ready := r.ready
      r.valid := wrapperReq.valid
      r.bits := wrapperReq.bits

      d.data.valid := wrapperDat.data.valid
      d.data.bits := wrapperDat.data.bits
      wrapperDat.isFlushed := d.isFlushed
      wrapperDat.data.ready := d.data.ready


      when(useFakeIO) {
        r.valid := false.B
        r.bits := DontCare
        wrapperReq.ready := beatCount === 0.U
        when(wrapperReq.fire) {
          beatCount := (wrapperReq.bits.len >> CLog2Up(dwidth / 8)).asUInt
          //          println("shifting by " + CLog2Up(dwidth/8))
        }

        when(wrapperDat.data.fire) {
          beatCount := beatCount - 1.U
        }

        d.data.valid := false.B
        wrapperDat.data.ready := beatCount =/= 0.U
      }

      (wrapperReq, wrapperDat)
    }
    (req_wrapper.map(_._1), req_wrapper.map(_._2))

  } else getWriterModules(name)

  // store activations as transposition (A matrix)
  val (weights_req, weights) = getReaderWrapper("weight_stream")
  val (activations_req, activations) = getReaderWrapper("activation_stream")
  val (activations_out_req_left, activations_out_left) = getWriterWrapper("activation_out")
  (weights_req ++ activations_req ++ activations_out_req_left).foreach { q =>
    q.valid := false.B
    q.bits := DontCare
  }

  val kBits = log2Up(kMax)
  val batchBits = log2Up(maxBatch + 1)

  val bias_tx_bytes_max_len = maxBatch * N * kMax


  val matrixOpCmd = BeethovenIO(new AccelCommand("matrixOp") {
    // ---------- WEIGHT ----------- //
    val addrIn = Address() // weight stream
    // ---------- OUTPUT ----------- //
    val addrOut = Address() // output address for MatAdd and MatMul
    val outputRowOffset = UInt(addrBits.W) // byte offset for a stripe in the output matrix
    // ---------- CONTROL ----------- //
    val outputTranspose = if (useUpShift) Some(Bool()) else None
    val nColTilesToDo_MO = UInt(log2Up(maxColTilesSimul).W)
    val weightsAreBatched = Bool()
    // ---------- SOFTMAX ----------- //
    val softmax_out = if (specialFunc.exists(_._1 == SpecialFunction.EXP)) Some(Address()) else None
    // ---------- ACTIVATIONS ----------- //
    val activationAddr = Address() // activation stream
    val batchSize = UInt(batchBits.W)
    val matrixLen = UInt((kBits + 1).W) // k (shared matrix dimension)
    // ---------- BIAS ----------- //
    val bias_addr = Address()
    val bias_tx_bytes = UInt(log2Up(bias_tx_bytes_max_len * 2 + 1).W)
    val bias_mode = UInt(2.W)
    // ---------- NORM ----------- //
    val norm_enable = Bool()
    val norm_addr = Address()
    val norm_per_batch = Bool()
  }, new AccelResponse("opErr") {
    val err = UInt(3.W)
  })

  //  println("ncw is " + matrixOpCmd.req.bits.nColTilesToDo_MO.getWidth)

  val biasNONE = 0
  val biasCOLS = 1
  val biasMATRIX = 2
  val biasBATCHEDMATRIX = 3
  val biasMode = Reg(UInt(2.W))

  CppGeneration.addUserCppDefinition(Seq(
    ("uint8_t", f"PROSE_biasNONE", biasNONE),
    ("uint8_t", f"PROSE_biasCOLS", biasCOLS),
    ("uint8_t", f"PROSE_biasMATRIX", biasMATRIX),
    ("uint8_t", f"PROSE_biasBATCHEDMATRIX", biasBATCHEDMATRIX)
  ))

  implicit val errReg = RegInit(ProseErr.NoError)
  matrixOpCmd.resp.bits.err := errReg.asUInt

  /** *************************** ROW NORMALIZATION CONSTANT READ IN **************************** */
  val norm_idle :: norm_read :: Nil = Enum(2)
  val norm_state = RegInit(norm_idle)
  val norm_counter_row = Reg(UInt(log2Up(N).W))
  val norm_counter_batch = Reg(UInt(log2Up(maxBatch).W))
  val norm_batch_max = Reg(UInt(log2Up(maxBatch).W))
  val norms = Reg(Vec(N, Vec(maxBatch, UInt(16.W))))

  val norm_is_ready = {
    val (List(norm_req), List(norm_dat)) = getReaderWrapper("norm_stream")
    norm_req.valid := false.B
    norm_req.bits := DontCare
    norm_dat.data.ready := false.B

    val w_norm_addr = matrixOpCmd.req.bits.norm_addr
    val w_norm_batched = matrixOpCmd.req.bits.norm_per_batch

    val use_per_batch = Reg(Bool())
    when(norm_state === norm_idle) {
      when(matrixOpCmd.req.fire) {
        when (matrixOpCmd.req.bits.norm_enable) {
          norm_state := norm_read
        }
        useNorms := matrixOpCmd.req.bits.norm_enable
        norm_req.valid := matrixOpCmd.req.bits.norm_enable
        norm_req.bits.addr := w_norm_addr
        norm_req.bits.len := (N * 2).U * Mux(w_norm_batched, matrixOpCmd.req.bits.batchSize, 1.U)
        norm_batch_max := Mux(w_norm_batched, matrixOpCmd.req.bits.batchSize - 1.U, 0.U)
        norm_counter_row := 0.U
        norm_counter_batch := 0.U
      }
    }.elsewhen(norm_state === norm_read) {
      norm_dat.data.ready := true.B
      when(norm_dat.data.fire) {
        when(use_per_batch) {
          norms(norm_counter_row)(norm_counter_batch) := norm_dat.data.bits
        }.otherwise {
          norms(norm_counter_row).foreach(_ := norm_dat.data.bits)
        }
        norm_counter_batch := norm_counter_batch + 1.U
        when(norm_counter_batch === norm_batch_max) {
          norm_counter_batch := 0.U
          norm_counter_row := norm_counter_row + 1.U
          when(norm_counter_row === (N - 1).U) {
            norm_counter_row := 0.U
            norm_state := norm_idle
          }
        }
      }
    }

    norm_state === norm_idle
  }

  val biasReadCtr = RegInit(0.U(log2Up(N + 1).W))
  val biasBatchCtr = Reg(UInt(log2Up(maxBatch).W))
  val bias_streams = {
    val (bias_reqs, bias_dats) = getReaderWrapper("bias_stream")
    bias_reqs.foreach {
      bias_req =>
        bias_req.valid := false.B
        bias_req.bits := DontCare
    }
    bias_dats.foreach { dat =>
      dat.data.ready := false.B
    }
    val wire_b_addr = matrixOpCmd.req.bits.bias_addr
    val wire_b_mode = matrixOpCmd.req.bits.bias_mode
    val wire_b_len = matrixOpCmd.req.bits.bias_tx_bytes

    when(matrixOpCmd.req.fire) {
      if (!supportWideBias) {
        assert(wire_b_mode =/= biasMATRIX.U && wire_b_mode =/= biasBATCHEDMATRIX.U,
          "This core was built with `supportWideBias` = false")
      }
      bias_reqs(0).valid := wire_b_mode =/= biasNONE.U
      bias_reqs(0).bits.len := wire_b_len
      bias_reqs(0).bits.addr := wire_b_addr

      bias_reqs.zipWithIndex.tail.foreach { case (req, idx) =>
        req.bits.addr := wire_b_addr + wire_b_len * idx.U
        req.bits.len := wire_b_len
        req.valid := (wire_b_mode === biasMATRIX.U) || (wire_b_mode === biasBATCHEDMATRIX.U)
      }
      biasBatchCtr := 0.U
      biasReadCtr := 0.U
      biasMode := wire_b_mode
    }
    bias_dats
  }

  val CTRL_outputTranspose = if (useUpShift) Reg(Bool()) else WireInit(true.B)
  val CTRL_weightsAreBatched = Reg(Bool())
  val CTRL_outputRowOffset = Reg(UInt(addrBits().W))
  val CTRL_nColTiles_MO = Reg(UInt(log2Up(maxColTilesSimul).W))
  val CTRL_weightsBaseAddress = Reg(Address())
  val CTRL_outBaseAddress = Reg(Address())
  Seq(matrixOpCmd) foreach { q =>
    q.req.ready := false.B
  }

  val s_idle :: s_emit_weight_request :: s_launch_mac :: s_mac :: s_flush :: s_finish :: s_aux :: Nil = Enum(7)
  val state = RegInit(s_idle)

  val array = Module(new SystolicArray(N, kMax, maxBatch, useUpShift, fpuLatency))
  array.io.weightsAreBatched := CTRL_weightsAreBatched
  val weight_consume = array.io.consume_weights
  val activation_consume = array.io.consume_activations

  weights foreach { w => w.data.ready := weight_consume }
  activations foreach { a => a.data.ready := activation_consume }

  array.io.cmd.valid := false.B
  array.io.cmd.bits := DontCare

  val activation_buffer = Memory(
    latency = SRAMLatency,
    dataWidth = N * 16,
    nRows = kMax * maxBatch,
    nReadPorts = 0,
    nWritePorts = 0,
    nReadWritePorts = 1,
    allowFallbackToRegister = false)
  activation_buffer.initLow(clock)
  val activeWriteToActivationBuffer = RegInit(false.B)
  val activationBuffer_valid = RegInit(false.B)
  val activationK = Reg(UInt(kBits.W))
  val activationBatch = Reg(UInt(batchBits.W))


  def parse(a: Seq[UInt]): Vec[UInt] = VecInit(a.reverse.flatMap(q => splitToChunks(q, 16)))

  val activationFIFO = Module(new Queue(Vec(N, UInt(16.W)), SRAMLatency + 1, hasFlush = true))
  activationFIFO.io.flush.get := false.B

  val actReadCnt = RegInit(0.U(log2Up(kMax * N * maxBatch).W))
  val activationSpace = Reg(UInt(log2Up(SRAMLatency + 2).W))
  activationFIFO.io.enq.valid := false.B
  activationFIFO.io.enq.bits := DontCare
  activationFIFO.io.deq.ready := false.B
  val weights_valid = weights.map(_.data.valid).fold(true.B)(_ && _)
  val activations_valid = Wire(Bool())
  val read_active = activations_valid && array.io.consume_activations

  val actValue = parse(activations.map(_.data.bits)).reverse
  val actChosen = Wire(Vec(N, UInt(16.W)))
  activation_buffer.write_enable(0) := activeWriteToActivationBuffer
  activation_buffer.read_enable(0) := !activeWriteToActivationBuffer
  when(activeWriteToActivationBuffer) {
    actChosen := actValue
    activations_valid := activations.map(_.data.valid).fold(true.B)(_ && _)
    activation_buffer.chip_select(0) := read_active
    activation_buffer.addr(0) := actReadCnt
    activation_buffer.data_in(0) := Cat(actValue.reverse)
    when(read_active) {
      actReadCnt := actReadCnt + 1.U
      when(activations_req(0).ready) {
        actReadCnt := 0.U
      }
    }
  }.otherwise {
    activation_buffer.addr(0) := actReadCnt
    val read_buffer_valid = state === s_mac && activationSpace > 0.U && !activeWriteToActivationBuffer
    activation_buffer.chip_select(0) := read_buffer_valid
    when(read_buffer_valid) {
      when(!activationFIFO.io.deq.fire) {
        activationSpace := activationSpace - 1.U
      }
    }.elsewhen(activationFIFO.io.deq.fire) {
      activationSpace := activationSpace + 1.U
    }
    when(read_buffer_valid) {
      actReadCnt := actReadCnt + 1.U
    }
    activationFIFO.io.enq.valid := ShiftReg(read_buffer_valid, SRAMLatency, clock)
    activationFIFO.io.enq.bits := splitToChunks(activation_buffer.data_out(0), 16).reverse

    actChosen := activationFIFO.io.deq.bits
    activations_valid := activationFIFO.io.deq.valid
    activationFIFO.io.deq.ready := array.io.consume_activations
  }

  val weightsSplit = VecInit(weights.reverse.flatMap(q => splitToChunks(q.data.bits, 16)).reverse)
  array.io.a_in := actChosen
  array.io.b_in := weightsSplit
  array.io.activations_valid := activations_valid
  array.io.weights_valid := weights_valid

  val simd_engine = Module(new ProseSIMD(N, fpuLatency, true, 16, specialFunc))
  val bias_data_valid = Wire(Bool())

  val can_consume_array_output = simd_engine.io.readyIn && simd_engine.io.validIn
  array.io.back_pressure := !can_consume_array_output
  array.io.cmd.bits.batchSizeMO := activationBatch - 1.U

  when(biasMode === biasCOLS.U) {
    bias_data_valid := bias_streams(0).data.valid
  }.elsewhen(biasMode === biasMATRIX.U || biasMode === biasBATCHEDMATRIX.U) {
    bias_data_valid := bias_streams.map(_.data.valid).fold(true.B)(_ && _)
  }.otherwise {
    bias_data_valid := true.B
  }
  BeethovenBuild.requestSeparateCompileCell(simd_engine.desiredName)

  simd_engine.io.validIn := array.io.c_out.valid && bias_data_valid
  val biasColChoice = splitIntoChunks(bias_streams(0).data.bits, 16)(biasReadCtr)
  simd_engine.io.accumulator.zipWithIndex.foreach { case (acc, idx) =>
    val groupIdx = idx / Nmin
    val biasIdx = idx % Nmin
    //    println(s"biasIdx: $biasIdx, groupIdx: $groupIdx")
    when(biasMode === biasCOLS.U) {
      acc := biasColChoice
    }.elsewhen(biasMode === biasMATRIX.U || biasMode === biasBATCHEDMATRIX.U) {
      if (supportWideBias) {
        acc := splitIntoChunks(bias_streams(groupIdx).data.bits, 16)(biasIdx)
      } else {
        acc := 0.U
      }
    }.otherwise {
      acc := 0.U
    }
  }

  val SIMD_inFire = simd_engine.io.validIn && simd_engine.io.readyIn
  when(SIMD_inFire) {
    norm_counter_batch := norm_counter_batch + 1.U
    when(norm_counter_batch === norm_batch_max) {
      norm_counter_batch := 0.U
    }

    val nxt = biasBatchCtr +& 1.U
    biasBatchCtr := nxt
    when(nxt === activationBatch || biasMode === biasBATCHEDMATRIX.U) { // if the batch shares the bias then we need to count
      biasBatchCtr := 0.U
      biasReadCtr := biasReadCtr + 1.U
      when(biasReadCtr === (Nmin - 1).U || biasMode === biasBATCHEDMATRIX.U || biasMode === biasMATRIX.U) {
        biasReadCtr := 0.U
        bias_streams.foreach(_.data.ready := true.B)
      }
    }
  }

  simd_engine.io.operand := array.io.c_out.bits
  simd_engine.io.operandMult.get.zip(norms.map(_.apply(norm_counter_batch))) foreach {
    case (sink, src) =>
      sink := src
      when(!useNorms) {
        sink := 0x3f80.U
      }
  }

  simd_engine.io.readyOut := activations_out_left.map(_.data.ready).fold(true.B)(_ && _)
  activations_out_left.zip(simd_engine.io.out.grouped(Nmin)) foreach { case (o_stream, dat) =>
    o_stream.data.bits := Cat(dat.reverse)
    o_stream.data.valid := simd_engine.io.validOut && state =/= s_idle
  }

  implicit val returnWithError = Wire(Bool())
  returnWithError := DontCare

  val can_move_to_finish = WireInit(true.B)

  when(state === s_idle) {
    matrixOpCmd.req.ready := true.B
    returnWithError := false.B
    errReg := ProseErr.NoError
    when(matrixOpCmd.req.fire) {
      /** ******************* Initialize Activation Stream ****************** */
      activeWriteToActivationBuffer := true.B
      activationK := matrixOpCmd.req.bits.matrixLen - 1.U
      activationBatch := matrixOpCmd.req.bits.batchSize
      activationFIFO.io.flush.get := true.B
      val readLenBatched = ((matrixOpCmd.req.bits.matrixLen * matrixOpCmd.req.bits.batchSize) << log2Up(Nmin * 2)).asUInt
      activations_req.zipWithIndex.foreach { case (req_port, idx) =>
        req_port.valid := matrixOpCmd.req.bits.batchSize > 0.U
        req_port.bits.len := readLenBatched
        req_port.bits.addr := matrixOpCmd.req.bits.activationAddr + (readLenBatched * idx.U)

        // error handling. In deployment we have return codes, in simulation we have asserts
        ErrorHandle(matrixOpCmd.req.bits.batchSize === 0.U,
          "Cannot have batch size == 0",
          ProseErr.InvalidBatch)
        ErrorHandle(
          !req_port.ready,
          "Check command stream. We weren't ready for a stream initialization",
          ProseErr.OverlappingActivationInit)
      }

      /** ****************************** MAT MUL *************************** */
      //      ErrorHandle(!(activationBuffer_valid || activeWriteToActivationBuffer),
      //        "Command must have a valid activation stream, though the current activation buffer is marked invalid and" +
      //          "the activation stream is unprimed. Please prime the activation stream.",
      //        ProseErr.InvalidActivation)

      if (useUpShift) CTRL_outputTranspose := matrixOpCmd.req.bits.outputTranspose.get
      CTRL_weightsAreBatched := matrixOpCmd.req.bits.weightsAreBatched
      CTRL_nColTiles_MO := matrixOpCmd.req.bits.nColTilesToDo_MO
      CTRL_weightsBaseAddress := matrixOpCmd.req.bits.addrIn
      CTRL_outBaseAddress := matrixOpCmd.req.bits.addrOut
      CTRL_outputRowOffset := matrixOpCmd.req.bits.outputRowOffset
      activations_out_req_left.zipWithIndex.foreach { case (w_o, idx) =>
        w_o.valid := matrixOpCmd.req.bits.outputTranspose.getOrElse(true.B)
        val wLen = (Nmin * N * 2).U * matrixOpCmd.req.bits.batchSize * (1.U +& matrixOpCmd.req.bits.nColTilesToDo_MO)
        val wAddrStride = matrixOpCmd.req.bits.outputRowOffset
        w_o.bits.addr := matrixOpCmd.req.bits.addrOut + wAddrStride * idx.U
        w_o.bits.len := wLen
        ErrorHandle(!w_o.ready,
          "Tried to launch an output stream but channel was not ready. This is unexpected.",
          ProseErr.DesignError)
      }
      state := s_emit_weight_request
    }
    when(matrixOpCmd.req.fire && returnWithError) {
      state := s_finish
    }
  }.elsewhen(state === s_emit_weight_request) {
    val can_emit = Wire(Bool())

    // consts
    val wLen = Misc.multByIntPow2(activationBatch, Nmin * N * 2)
    val wAddrStride = CTRL_outputRowOffset
    val sharedDim = activationK +& 1.U
    val readLenNonBatched = Misc.multByIntPow2(sharedDim, Nmin * 2)
    val readLenBatched = Misc.multByIntPow2(sharedDim * activationBatch, Nmin * 2)
    val readLen = Mux(CTRL_weightsAreBatched, readLenBatched, readLenNonBatched)

    val outs_ready = if (useUpShift) {
      when(!CTRL_outputTranspose) {
        activations_out_req_left.zipWithIndex.foreach { case (w_o, idx) =>
          w_o.valid := can_emit
          w_o.bits.addr := CTRL_outBaseAddress + wAddrStride * idx.U
          w_o.bits.len := wLen
        }
      }
      activations_out_req_left.map(_.ready).fold(true.B)(_ && _) || CTRL_outputTranspose
    } else true.B

    weights_req.zipWithIndex.foreach { case (w_r, idx) =>
      w_r.valid := can_emit
      w_r.bits.addr := CTRL_weightsBaseAddress + readLen * idx.U
      w_r.bits.len := readLen
      ErrorHandle(!w_r.ready,
        "Tried to launch a weight stream but channel was not ready. This is unexpected.",
        ProseErr.DesignError)
    }
    val weights_ready = weights_req.map(_.ready).fold(true.B)(_ && _)
    can_emit := weights_ready && outs_ready && norm_is_ready

    when(can_emit) {
      state := s_launch_mac
      CTRL_weightsBaseAddress := CTRL_weightsBaseAddress + Misc.multByIntPow2(readLen, N / Nmin)
      CTRL_outBaseAddress := CTRL_outBaseAddress +
        Misc.multByIntPow2(Mux(CTRL_outputTranspose, wLen, CTRL_outputRowOffset), N / Nmin)
    }
  }.elsewhen(state === s_launch_mac) {
    array.io.cmd.valid := true.B
    array.io.cmd.bits.k := activationK
    if (useUpShift)
      array.io.cmd.bits.outputTranspose.get := CTRL_outputTranspose
    state := s_mac
  }.elsewhen(state === s_mac) {
    when(CTRL_nColTiles_MO =/= 0.U) {
      when(array.io.array_idle) {
        state := s_emit_weight_request
        CTRL_nColTiles_MO := CTRL_nColTiles_MO - 1.U
        activationFIFO.io.flush.get := true.B
        when(activeWriteToActivationBuffer) {
          activeWriteToActivationBuffer := false.B
          activationBuffer_valid := true.B
        }
        actReadCnt := 0.U
        activationSpace := (SRAMLatency + 1).U
      }
    }.otherwise {
      val flushed = activations_out_left.map(_.isFlushed).fold(true.B)(_ && _)
      val req_complete = activations_out_req_left.map(_.ready).fold(true.B)(_ && _)
      when(flushed && req_complete) {
        // we move to aux if there's some other hardware that needs to run some additional stuff before we return
        state := Mux(can_move_to_finish, s_finish, s_aux)
        when(activeWriteToActivationBuffer) {
          activeWriteToActivationBuffer := false.B
          activationBuffer_valid := true.B
        }
      }
    }
  }.elsewhen(state === s_finish) {
    matrixOpCmd.resp.valid := true.B
    activationFIFO.io.flush.get := true.B
    actReadCnt := 0.U
    activationSpace := (SRAMLatency + 1).U
    when(matrixOpCmd.resp.fire) {
      state := s_idle
    }
  }.elsewhen(state === s_aux) {
    when(can_move_to_finish) {
      state := s_finish
    }
  }
}