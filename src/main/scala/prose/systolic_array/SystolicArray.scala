package prose.systolic_array

import beethoven._
import org.chipsalliance.cde.config._
import chisel3._
import chisel3.util._
import beethoven.common._
import prose.systolic_array.util.{ShiftArray, ShiftScan}

class SystolicArray(N: Int,
                    kMax: Int,
                    maxBatch: Int,
                    withUpShift: Boolean,
                    fpuLatency: Int = 1)(implicit p: Parameters) extends Module {
  override val desiredName = s"SystolicArray_${N}x${kMax}x${maxBatch}x${fpuLatency}_shiftUp$withUpShift"
  println(f"CREATING $desiredName")
  require(N > 0 && kMax > 0 && kMax >= N)
  val io = IO(new Bundle {

    val cmd = Flipped(Decoupled(new Bundle {
      val k = UInt(log2Up(kMax).W)
      val outputTranspose = if (withUpShift) Some(Bool()) else None
      val batchSizeMO = Input(UInt(log2Up(maxBatch).W))
    }))

    val weightsAreBatched = Input(Bool())

    val activations_valid = Input(Bool())
    val weights_valid = Input(Bool())
    val consume_activations = Output(Bool())
    val consume_weights = Output(Bool())

    val a_in, b_in = Flipped(Vec(N, UInt(16.W)))

    val c_out = Valid(Vec(N, UInt(16.W)))

    val back_pressure = Input(Bool())

    val array_idle = Output(Bool())
  })

  /**
   * s_idle9
   * s_shift_in - shifting values in on (hopefully) every cycle for alternating batches in set
   * s_flush - let shifted values propagate before shifting left out
   * s_shift_out - shift left out
   */
  val s_idle :: s_d1 :: s_shift_in :: s_flush :: s_shift_out :: Nil = Enum(5)
  val state = RegInit(s_idle)
  io.array_idle := state === s_idle
  val outputTranpose = Reg(Bool())
  val maxloopSize = Math.max(fpuLatency, maxBatch)
  val loopCounter = Reg(UInt(log2Up(maxloopSize + 1).W))
  val currentBatchMO = Reg(UInt(log2Up(maxBatch).W))

  val kCount, cmdK = RegInit(0.U(log2Up(Math.max(kMax, N * fpuLatency)).W))

  val must_consume_activation = loopCounter <= currentBatchMO
  val am_consuming_activation = WireInit(false.B)
  dontTouch(am_consuming_activation)
  val must_consume_weight = Mux(io.weightsAreBatched, must_consume_activation, loopCounter === maxloopSize.U)

  /** ----------------IO init------------------ */
  io.c_out.valid := state === s_shift_out
  val currentPEMode = WireInit(PEMode.idle)
  val routingDelay = 2
  val shiftOutRoutingDelay = Reg(UInt(2.W))

  io.consume_weights := false.B
  io.consume_activations := false.B

  io.cmd.ready := state === s_idle

  val shift_anyway = WireInit(false.B)
  val routeShiftSignal = shift_anyway || !io.back_pressure

  val kCpO = kCount + 1.U
  val kc_en = WireInit(false.B)
  val kc_sel = WireInit(false.B)
  when (kc_en) {
    kCount := Mux(kc_sel, kCpO, 0.U)
  }

  when(state === s_idle) {
    when(io.cmd.valid) {
      cmdK := io.cmd.bits.k
      outputTranpose := io.cmd.bits.outputTranspose.getOrElse(true.B)
      state := s_d1
      currentBatchMO := io.cmd.bits.batchSizeMO
      loopCounter := 0.U
      kc_en := true.B
    }
  }.elsewhen(state === s_d1) {
    state := s_shift_in
  }.elsewhen(state === s_shift_in) {
    currentPEMode := PEMode.MAC
    // logical operation "a -> b"
    def implies(a: Bool, b: Bool): Bool = (a && b) || !a
    when(
      implies(must_consume_activation, io.activations_valid && io.weights_valid) &&
        implies(must_consume_weight, io.weights_valid && Mux(io.weightsAreBatched, io.activations_valid, true.B))) {
      am_consuming_activation := must_consume_activation
      loopCounter := loopCounter + 1.U
      io.consume_weights := must_consume_weight
      io.consume_activations := must_consume_activation

      when(loopCounter === maxloopSize.U) {
        loopCounter := 0.U
        kc_en := true.B
        kc_sel := true.B
        when(kCount === cmdK) {
          kc_sel := false.B
          state := s_flush
        }
      }
    }
  }.elsewhen(state === s_flush) {
    currentPEMode := PEMode.MAC
    kc_en := true.B
    kc_sel := true.B
    // now that we've streamed everything in, just make sure it's all propagated through
    when(kCount === (N * fpuLatency).U) {
      kc_sel := false.B
      state := s_shift_out
      shiftOutRoutingDelay := Mux(outputTranpose, 0.U, 2.U)
    }
  }.elsewhen(state === s_shift_out) {
    currentPEMode := PEMode.left_shift
    // shift out
    io.c_out.valid := shiftOutRoutingDelay === 0.U
    when(shiftOutRoutingDelay =/= 0.U) {
      shiftOutRoutingDelay := shiftOutRoutingDelay - 1.U
      shift_anyway := true.B
    }
    when(!io.back_pressure) {
      when(shiftOutRoutingDelay === 0.U) {
        loopCounter := loopCounter + 1.U
        when(loopCounter === currentBatchMO) {
          loopCounter := 0.U
          kc_en := true.B
          kc_sel := true.B
          when(kCount === (N - 1).U) {
            kc_sel := false.B
            state := s_idle
          }
        }
      }
    }
  }

  /** --------------end IO init --------------- */

  val leftShiftedEles = Wire(Vec(N, UInt(16.W)))
  val altShiftedEles = Wire(Vec(N, UInt(16.W)))
  leftShiftedEles := DontCare
  altShiftedEles := DontCare
  if (withUpShift) {
    io.c_out.bits := ShiftRegEnable(altShiftedEles, routingDelay, routeShiftSignal, clock)
  } else {
    io.c_out.bits := DontCare
  }
  when(outputTranpose) {
    io.c_out.bits := leftShiftedEles
  }

  val pe_array = Seq.tabulate(N)(row_idx =>
    Seq.tabulate(N) { col_idx =>
      val mod = Module(new OutputStationaryPE(fpuLatency, maxBatch, withUpShift))
      BeethovenBuild.requestModulePartition(mod.desiredName)
      mod.suggestName(f"SysArray_r${row_idx}d_c$col_idx")
    })
  pe_array.flatten.foreach {
    ospe: OutputStationaryPE =>
      ospe.io.shiftOutLeft := currentPEMode === PEMode.left_shift && outputTranpose && routeShiftSignal
      if (withUpShift)
        ospe.io.shiftOutAlt.get := currentPEMode === PEMode.left_shift && !outputTranpose && routeShiftSignal
      ospe.io.syncResetIn := io.cmd.fire
  }
  if (withUpShift)
    altShiftedEles.zip(pe_array.head).foreach { case (outWire, pe) =>
      outWire := pe.io.north_out.get
    }


  val valid_in_scan = Module(new ShiftScan(N, 1))
  valid_in_scan.io.in := am_consuming_activation

  //   connect PEs to each other
  pe_array.zipWithIndex.foreach { case (peRow, rowIdx) =>
    leftShiftedEles(rowIdx) := peRow(0).io.shift_out

    peRow(0).io.validIn := valid_in_scan.io.out(rowIdx)
    peRow(0).io.batchSizeMOInLeft := currentBatchMO
    if (rowIdx != N - 1) {
      val nextRow = pe_array(rowIdx + 1)
      peRow.zip(nextRow).foreach { case (col, nCol) =>
        nCol.io.north_in := col.io.south_out
        if (withUpShift)
          col.io.south_in.get := nCol.io.north_out.get
      }
    } else {
      if (withUpShift)
        peRow.foreach(_.io.south_in.get := DontCare)
    }
    (0 until N - 1) foreach { idx =>
      val col = peRow(idx)
      val nCol = peRow(idx + 1)
      nCol.io.west_in := col.io.east_out
      nCol.io.validIn := col.io.validOut
      col.io.shift_in := nCol.io.shift_out
      nCol.io.batchSizeMOInLeft := col.io.batchSizeMOOutRight
    }
    peRow.last.io.shift_in := DontCare
  }

  // connect top row to inputs (account for padding first)
  // for sanity, ensure that valid bits for both matrices are always matching

  val Seq(a_padShift, b_padShift) = Seq((io.a_in, "a_in"), (io.b_in, "b_in")).map { case (vec, name) =>
    val shifter = Module(new ShiftArray(N, 16))
    shifter.suggestName(f"Shifter_${name}")
    shifter.io.in := vec
    shifter.io.out
  }

  pe_array.head.zip(b_padShift).foreach { case (topPE, b_ele) =>
    topPE.io.north_in := b_ele
  }
  pe_array.map(_.head).zip(a_padShift).foreach { case (leftPe, a_ele) =>
    leftPe.io.west_in := a_ele
  }

  dontTouch(currentPEMode)
}
