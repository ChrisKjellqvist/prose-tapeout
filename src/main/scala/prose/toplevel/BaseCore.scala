package prose.toplevel

import beethoven.Generation.CppGeneration
import beethoven.MemoryStreams.Memory
import beethoven.MemoryStreams._
import chipsalliance.rocketchip.config._
import chisel3._
import chisel3.util._
import beethoven._
import beethoven.common.Address.addrBits
import beethoven.common._
import prose.SpecialFunction.SpecialFunction
import prose.activations._
import prose.err_handling.{ErrorHandle, ProseErr}
import prose.random.LinearFeedbackShiftRegister
import prose.{SpecialFunction, splitToChunks}
import prose.systolic_array.SystolicArray

import scala.util.Random

/**
 * Base Systolic array core without any additional functionalities
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
                        simdLatency: Int,
                        maxBatch: Int,
                        supportWideBias: Boolean,
                        maxColTilesSimul: Int,
                        n_arrays: Int,
                        specialFunc: Option[(SpecialFunction, Int)] = None,
                       )(implicit p: Parameters) extends AcceleratorCore {
  val useFakeLUTs: Boolean = true
  class MatrixOpCmd extends AccelCommand("matrixOp") {
    // ---------- WEIGHT ----------- //
    val addrIn = Address() // weight stream
    // ---------- OUTPUT ----------- //
    val addrOut = Address() // output address for MatAdd and MatMul
    val outputRowOffset = UInt(addrBits.W) // byte offset for a stripe in the output matrix
    // ---------- CONTROL ----------- //
    val outputTranspose = if (useUpShift) Some(Bool()) else None
    val nColTilesToDo_MO = UInt(log2Up(maxColTilesSimul).W)
    val nRowTilesToDo_MO = UInt(log2Up(maxColTilesSimul).W)
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
  }
  CppGeneration.addPreprocessorDefinition("PROSE")
  CppGeneration.addPreprocessorDefinition(
    Seq(
      (f"PROSE_${outer.systemParams.name}_N", N),
      (f"PROSE_Nmin", Nmin),
      (f"PROSE_maxBatch", maxBatch),
      (f"PROSE_${outer.systemParams.name}Core_maxColTiles", maxColTilesSimul),
      (s"PROSE_${outer.systemParams.name}_kMax", kMax)
    )
  )

  val useUpShift = specialFunc match {
    case Some((q, _)) => q != SpecialFunction.EXP
    case None => true
  }

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

  /**
   * For test-chip purposes, we want to be able to test the power-consumption of the chip under load
   * which we can't actually due with the HyperRAM bandwidth (300MBs). To get around this, we generate
   * random normal numbers. Box-Mueller too expensive so we do 1cy LFSR + 2cy LUT but we can do this
   * at a throughput of 1/cy
   */
  def getReaderWrapper(name: String): (List[DecoupledIO[ChannelTransactionBundle]], List[DataChannelIO]) = if (useFakeLUTs) {
    val (reqs, dats) = getReaderModules(name)
    reqs.zipWithIndex.foreach(a => a._1.suggestName(f"requestChannel_${name}_${a._2}"))
    dats.zipWithIndex.foreach(a => a._1.suggestName(f"dataChannel_${name}_${a._2}"))
    val maxTxLen = kMax * N * n_arrays * maxBatch * 2
    val dwidth = dats(0).data.bits.getWidth
    val maxBeats = 1 + maxTxLen / (dwidth / 8)
    val beatCount = RegInit(0.U(log2Up(maxBeats + 1).W))
    beatCount.suggestName(f"beatCount_${name}")
    // build the lut
    val _ = {
      os.makeDir.all(os.pwd / "luts")
      os.proc("cmake", os.pwd / "src" / "main" / "c" / "generate_verilog").call(cwd = os.pwd / "luts")
      os.proc("make").call(cwd = os.pwd / "luts")
      os.proc("./generate_normal", os.pwd/"luts").call(cwd = os.pwd / "luts")
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


  val matrixOpCmd = BeethovenIO(new MatrixOpCmd, new AccelResponse("opErr") {
      val err = UInt(3.W)
    })

  // state machine for the systolic array manager
  val s_idle :: s_start_row :: s_emit_weight_request :: s_launch_mac :: s_mac :: s_flush :: s_finish :: s_aux :: Nil = Enum(8)
  val state = RegInit(s_idle)

  // stage the command
  val CTRL_cmdCopy = Reg(new MatrixOpCmd)
  val active_cores_MO = Wire(UInt(log2Up(n_arrays).W))

  // we hold n_arrays systolic arrays inside the module. They will share the weight stream
  val n_arrays_mo = n_arrays - 1
  active_cores_MO := Mux(CTRL_cmdCopy.nRowTilesToDo_MO >= n_arrays_mo.U, n_arrays_mo.U, CTRL_cmdCopy.nRowTilesToDo_MO)
  val CTRL_weightsBaseAddressAccumulator = Reg(Address())
  val CTRL_outputsBaseAddressAccumulator = Reg(Address())
  val CTRL_nColTiles_MO_accumulator = Reg(UInt(log2Up(maxColTilesSimul).W))

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
  val norm_counter_row = Reg(UInt(log2Up(N * n_arrays).W))
  val norm_counter_batch = Reg(UInt(log2Up(maxBatch).W))
  val norm_batch_max = Reg(UInt(log2Up(maxBatch).W))
  val norms = Reg(Vec(N * n_arrays, Vec(maxBatch, UInt(16.W))))

  val norm_is_ready = {
    val (List(norm_req), List(norm_dat)) = getReaderWrapper("norm_stream")
    norm_req.valid := false.B
    norm_req.bits := DontCare
    norm_dat.data.ready := false.B

    val w_norm_addr = CTRL_cmdCopy.norm_addr
    val w_norm_batched = CTRL_cmdCopy.norm_per_batch

    val use_per_batch = Reg(Bool())
    when(norm_state === norm_idle) {
      when(state === s_start_row) {
        when(CTRL_cmdCopy.norm_enable) {
          norm_state := norm_read
        }
        useNorms := CTRL_cmdCopy.norm_enable
        norm_req.valid := CTRL_cmdCopy.norm_enable
        norm_req.bits.addr := w_norm_addr
        val norm_req_len = (N * 2 * (n_arrays)).U * Mux(w_norm_batched, CTRL_cmdCopy.batchSize, 1.U)
        norm_req.bits.len := norm_req_len
        CTRL_cmdCopy.norm_addr := CTRL_cmdCopy.norm_addr + norm_req_len
        norm_batch_max := Mux(w_norm_batched, CTRL_cmdCopy.batchSize - 1.U, 0.U)
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
          when(norm_counter_row === (N * n_arrays - 1).U) {
            norm_counter_row := 0.U
            norm_state := norm_idle
          }
        }
      }
    }

    norm_state === norm_idle
  }

  // biases are per-col
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
    val wire_b_addr = CTRL_cmdCopy.bias_addr
    val wire_b_mode = CTRL_cmdCopy.bias_mode
    val wire_b_len = CTRL_cmdCopy.bias_tx_bytes

    when(state === s_start_row) {
      if (!supportWideBias) {
        assert(wire_b_mode =/= biasMATRIX.U && wire_b_mode =/= biasBATCHEDMATRIX.U,
          "This core was built with `supportWideBias` = false")
      }
      bias_reqs(0).valid := wire_b_mode =/= biasNONE.U
      bias_reqs(0).bits.len := wire_b_len
      bias_reqs(0).bits.addr := wire_b_addr

      bias_reqs.zipWithIndex.tail.foreach { case (req, idx) =>
        val array_idx = idx / (N / Nmin)
        req.bits.addr := wire_b_addr + wire_b_len * idx.U
        req.bits.len := wire_b_len
        req.valid := ((wire_b_mode === biasMATRIX.U) || (wire_b_mode === biasBATCHEDMATRIX.U)) && (active_cores_MO >= array_idx.U)
      }
      biasBatchCtr := 0.U
      biasReadCtr := 0.U
      biasMode := wire_b_mode
    }
    bias_dats
  }

  Seq(matrixOpCmd) foreach { q =>
    q.req.ready := false.B
  }

  println(f"${outer.systemParams.name} has memory size ${kMax * maxBatch} x ${N * 16 * n_arrays}")
  val activation_buffer = Memory(
    latency = SRAMLatency,
    dataWidth = N * 16 * n_arrays,
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

  val activationFIFO = Module(new Queue(Vec(N * n_arrays, UInt(16.W)), SRAMLatency + 1, hasFlush = true))
  activationFIFO.io.flush.get := false.B

  val actReadCnt = RegInit(0.U(log2Up(kMax * N * maxBatch).W))
  val activationSpace = Reg(UInt(log2Up(SRAMLatency + 2).W))
  activationFIFO.io.enq.valid := false.B
  activationFIFO.io.enq.bits := DontCare
  activationFIFO.io.deq.ready := false.B
  val weights_valid = weights.map(_.data.valid).fold(true.B)(_ && _)
  val activations_valid = Wire(Bool())

  val actChosen = Wire(Vec(N * n_arrays, UInt(16.W)))
  val arrays = Seq.tabulate(n_arrays) { array_idx =>
    val array = Module(new SystolicArray(N, kMax, maxBatch, useUpShift, fpuLatency))
    array.suggestName(f"SystolicArray_$array_idx")
    array.io.weightsAreBatched := CTRL_cmdCopy.weightsAreBatched
    array.io.cmd.valid := false.B
    array.io.cmd.bits := DontCare

    val weightsSplit = VecInit(weights.reverse.flatMap(q => splitToChunks(q.data.bits, 16)).reverse)
    (0 until N) foreach { i =>
      array.io.a_in(i) := actChosen(array_idx * N + i)
    }
    //    array.io.a_in := actChosen
    array.io.b_in := weightsSplit
    array.io.activations_valid := activations_valid
    array.io.weights_valid := weights_valid

    array
  }

  val weight_consume = arrays(0).io.consume_weights
  val activation_consume = arrays(0).io.consume_activations

  weights foreach { w => w.data.ready := weight_consume }
  activations foreach { a => a.data.ready := activation_consume }
  val read_active = activations_valid && arrays(0).io.consume_activations

  val actValue = parse(activations.map(_.data.bits)).reverse
  activation_buffer.write_enable(0) := activeWriteToActivationBuffer
  activation_buffer.read_enable(0) := !activeWriteToActivationBuffer
  when(activeWriteToActivationBuffer) {
    actChosen := actValue
    activations_valid := activations.zipWithIndex.map { case (act, act_idx) =>
      val array_idx = act_idx / (N / Nmin)
      act.data.valid || active_cores_MO < array_idx.U
    }.fold(true.B)(_ && _)
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
    activationFIFO.io.deq.ready := arrays(0).io.consume_activations
  }

  val simd_engine = Module(new ProseSIMD(N * n_arrays, simdLatency, true, 16, specialFunc))
  val bias_data_valid = Wire(Bool())
  val can_consume_array_output = simd_engine.io.readyIn && simd_engine.io.validIn
  arrays.foreach(_.io.back_pressure := !can_consume_array_output)
  arrays.foreach(_.io.cmd.bits.batchSizeMO := activationBatch - 1.U)

  when(biasMode === biasCOLS.U) {
    bias_data_valid := bias_streams(0).data.valid
  }.elsewhen(biasMode === biasMATRIX.U || biasMode === biasBATCHEDMATRIX.U) {
    bias_data_valid := bias_streams.zipWithIndex.map { case (bstream, stream_idx) =>
      val array_idx = stream_idx / (N / Nmin)
      bstream.data.valid || array_idx.U > CTRL_cmdCopy.nColTilesToDo_MO
    }.fold(true.B)(_ && _)
  }.otherwise {
    bias_data_valid := true.B
  }
  simd_engine.io.validIn := arrays(0).io.c_out.valid && bias_data_valid
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

  (0 until n_arrays) foreach { array_i =>
    (0 until N) foreach { i =>
      val idx = array_i * N + i
      simd_engine.io.operand(idx) := arrays(array_i).io.c_out.bits(i)
    }
  }
  simd_engine.io.operandMult.get.zip(norms.map(_.apply(norm_counter_batch))) foreach {
    case (sink, src) =>
      sink := src
      when(!useNorms) {
        sink := 0x3f80.U
      }
  }

  // when there are many writers, the complexity of the reader/writer can mix with the SIMD unit and make
  // it hard to pass hold time
  val simd_extra_slack_queue = Module(new Queue[Vec[UInt]](simd_engine.io.out.cloneType, 2))

  simd_engine.io.readyOut := simd_extra_slack_queue.io.enq.ready
  simd_extra_slack_queue.io.enq.valid := simd_engine.io.validOut && state =/= s_idle
  simd_extra_slack_queue.io.enq.bits := simd_engine.io.out

  simd_extra_slack_queue.io.deq.ready := activations_out_left.zipWithIndex.map { case (out_stream, stream_idx) =>
    val array_idx = stream_idx / (N / Nmin)
    out_stream.data.ready || array_idx.U > active_cores_MO
  }.fold(true.B)(_ && _)

  activations_out_left.zip(simd_extra_slack_queue.io.deq.bits.grouped(Nmin)).zipWithIndex foreach { case ((o_stream, dat), stream_idx) =>
    val array_idx = stream_idx / (N / Nmin)
    o_stream.data.bits := Cat(dat.reverse)
    o_stream.data.valid := simd_extra_slack_queue.io.deq.valid && array_idx.U <= active_cores_MO
  }

  implicit val returnWithError = Wire(Bool())
  returnWithError := DontCare

  val can_move_to_finish = WireInit(true.B)

  def increment_on_row_incr(): Unit = {
    // activation buffer is no longer valid
    activeWriteToActivationBuffer := true.B
    actReadCnt := 0.U
    // ------ reset addresses/accumulate regs for the next row ------
    // number of rows left to do
    CTRL_cmdCopy.nRowTilesToDo_MO := CTRL_cmdCopy.nRowTilesToDo_MO - n_arrays.U
    // from c++: out_acc += PROSE_GCore_N * N * 2 * chosen_batch_size
    // N in this context is a M x K x N matmul
    //   auto output_row_increment = 2 * chosen_batch_size * PROSE_MCore_N * (output_transpose ? N : PROSE_Nmin);
    require(isPow2(n_arrays), "need n_arrays to be power of 2 to simplify address computation. Can change implementation to not need this")
    when (CTRL_cmdCopy.outputTranspose.getOrElse(true.B)) {
      CTRL_cmdCopy.addrOut := CTRL_cmdCopy.addrOut +
        Misc.multByIntPow2(CTRL_cmdCopy.batchSize * Misc.multByIntPow2(CTRL_cmdCopy.nColTilesToDo_MO +& 1.U, N), N * 2 * n_arrays)
    }.otherwise {
      CTRL_cmdCopy.addrOut := CTRL_cmdCopy.addrOut + Misc.multByIntPow2(CTRL_cmdCopy.batchSize, 2 * N * Nmin * n_arrays)
    }

    when (CTRL_cmdCopy.bias_mode === biasMATRIX.U || CTRL_cmdCopy.bias_mode === biasBATCHEDMATRIX.U) {
      CTRL_cmdCopy.bias_addr := CTRL_cmdCopy.bias_addr +
        Misc.multByIntPow2(CTRL_cmdCopy.bias_tx_bytes, N / Nmin * n_arrays)
    }

    CTRL_cmdCopy.activationAddr := CTRL_cmdCopy.activationAddr +
      Misc.multByIntPow2(CTRL_cmdCopy.matrixLen * CTRL_cmdCopy.batchSize, N * n_arrays * 2)

  }

  when(state === s_idle) {
    matrixOpCmd.req.ready := true.B
    returnWithError := false.B
    errReg := ProseErr.NoError
    when(matrixOpCmd.req.fire) {
      /** ******************* Initialize Activation Stream ****************** */
      activeWriteToActivationBuffer := true.B
      activationK := matrixOpCmd.req.bits.matrixLen - 1.U
      activationBatch := matrixOpCmd.req.bits.batchSize
      CTRL_cmdCopy := matrixOpCmd.req.bits
      state := s_start_row
    }
    when(matrixOpCmd.req.fire && returnWithError) {
      state := s_finish
    }
  }.elsewhen(state === s_start_row) {
    activationFIFO.io.flush.get := true.B
    CTRL_nColTiles_MO_accumulator := CTRL_cmdCopy.nColTilesToDo_MO
    CTRL_weightsBaseAddressAccumulator := CTRL_cmdCopy.addrIn
    CTRL_outputsBaseAddressAccumulator := CTRL_cmdCopy.addrOut
    val readLenBatched = Misc.multByIntPow2(CTRL_cmdCopy.matrixLen * CTRL_cmdCopy.batchSize, Nmin * 2)
    activations_req.zipWithIndex.foreach { case (req_port, idx) =>
      val array_idx = idx / (N / Nmin)
      req_port.valid := CTRL_cmdCopy.batchSize > 0.U && array_idx.U <= active_cores_MO
      req_port.bits.len := readLenBatched
      req_port.bits.addr := CTRL_cmdCopy.activationAddr + (readLenBatched * idx.U)

      ErrorHandle(CTRL_cmdCopy.batchSize === 0.U,
        "Cannot have batch size == 0",
        ProseErr.InvalidBatch)
      ErrorHandle(
        !req_port.ready,
        "Check command stream. We weren't ready for a stream initialization",
        ProseErr.OverlappingActivationInit)
    }
    activations_out_req_left.zipWithIndex.foreach { case (w_o, idx) =>
      val array_idx = idx / (N / Nmin)
      w_o.valid := CTRL_cmdCopy.outputTranspose.getOrElse(true.B) && (active_cores_MO >= array_idx.U)
      val wLen = (Nmin * N * 2).U * CTRL_cmdCopy.batchSize * (1.U +& CTRL_cmdCopy.nColTilesToDo_MO)
      val wAddrStride = CTRL_cmdCopy.outputRowOffset
      w_o.bits.addr := CTRL_cmdCopy.addrOut + wAddrStride * idx.U
      w_o.bits.len := wLen
      ErrorHandle(!w_o.ready,
        "Tried to launch an output stream but channel was not ready. This is unexpected.",
        ProseErr.DesignError)
    }
    state := s_emit_weight_request
  }.elsewhen(state === s_emit_weight_request) {
    val can_emit = Wire(Bool())

    val wLen = Misc.multByIntPow2(activationBatch, Nmin * N * 2)
    val wAddrStride = CTRL_cmdCopy.outputRowOffset
    val sharedDim = activationK +& 1.U
    val readLenNonBatched = Misc.multByIntPow2(sharedDim, Nmin * 2)
    val readLenBatched = Misc.multByIntPow2(sharedDim * activationBatch, Nmin * 2)
    val readLen = Mux(CTRL_cmdCopy.weightsAreBatched, readLenBatched, readLenNonBatched)

    val outs_ready = if (useUpShift) {
      when(!CTRL_cmdCopy.outputTranspose.get) {
        activations_out_req_left.zipWithIndex.foreach { case (w_o, idx) =>
          val array_idx = idx / (N / Nmin)
          val loc_idx = idx % (N / Nmin)
          w_o.valid := can_emit && array_idx.U <= active_cores_MO
          w_o.bits.addr := CTRL_outputsBaseAddressAccumulator + wAddrStride * loc_idx.U + wLen * array_idx.U
          w_o.bits.len := wLen
        }
      }
      activations_out_req_left.map(_.ready).fold(true.B)(_ && _) || CTRL_cmdCopy.outputTranspose.get
    } else true.B

    weights_req.zipWithIndex.foreach { case (w_r, idx) =>
      w_r.valid := can_emit
      w_r.bits.addr := CTRL_weightsBaseAddressAccumulator + readLen * idx.U
      w_r.bits.len := readLen
      ErrorHandle(!w_r.ready,
        "Tried to launch a weight stream but channel was not ready. This is unexpected.",
        ProseErr.DesignError)
    }
    val weights_ready = weights_req.map(_.ready).fold(true.B)(_ && _)
    can_emit := weights_ready && outs_ready && norm_is_ready

    when(can_emit) {
      state := s_launch_mac
      CTRL_weightsBaseAddressAccumulator := CTRL_weightsBaseAddressAccumulator + Misc.multByIntPow2(readLen, N / Nmin)
      // don't need to multiply by n_arrays because the arrays are just multiplex downwards and this offset strides sideways
      CTRL_outputsBaseAddressAccumulator := CTRL_outputsBaseAddressAccumulator +
        Misc.multByIntPow2(Mux(CTRL_cmdCopy.outputTranspose.getOrElse(true.B), wLen, CTRL_cmdCopy.outputRowOffset), N / Nmin)
    }
  }.elsewhen(state === s_launch_mac) {
    arrays.zipWithIndex.foreach { case (array, idx) =>
      array.io.cmd.valid := idx.U <= active_cores_MO
      array.io.cmd.bits.k := activationK
      if (useUpShift)
        array.io.cmd.bits.outputTranspose.get := CTRL_cmdCopy.outputTranspose.get
    }
    state := s_mac
  }.elsewhen(state === s_mac) {
    when(CTRL_nColTiles_MO_accumulator =/= 0.U) {
      when(arrays(0).io.array_idle) {
        state := s_emit_weight_request
        CTRL_nColTiles_MO_accumulator := CTRL_nColTiles_MO_accumulator - 1.U
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
        when(activeWriteToActivationBuffer) {
          activeWriteToActivationBuffer := false.B
          activationBuffer_valid := true.B
        }
        // we move to aux if there's some other hardware that needs to run some additional stuff before we return
        when (can_move_to_finish) {
          when(CTRL_cmdCopy.nRowTilesToDo_MO <= n_arrays_mo.U) {
            state := s_finish
          }.otherwise {
            state := s_start_row
            increment_on_row_incr()
          }
        }.otherwise {
          state := s_aux
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
      when (CTRL_cmdCopy.nRowTilesToDo_MO <= n_arrays_mo.U) {
        println("n_arrays_mo (" + outer.systemParams.name + "): " + n_arrays_mo)
        state := s_finish
      }.otherwise {
        state := s_start_row
        increment_on_row_incr()
      }
    }
  }
}