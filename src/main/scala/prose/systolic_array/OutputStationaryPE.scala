package prose.systolic_array

import beethoven.Platforms.{BuildModeKey, PlatformType}
import beethoven.{BuildMode, platform}
import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper._
import fpwrapper.impl.fpnew._
import fpwrapper.impl.xilinx.{XilinxFPUImplementation, XilinxOperator}
import prose.FPUBuildMode

object PEMode extends chisel3.ChiselEnum {
  val idle, MAC, left_shift = Value
}

class OutputStationaryPE(fpuLatency: Int, maxBatch: Int, withUpShift: Boolean)(implicit p: Parameters) extends Module {
  override val desiredName = s"OutputStationaryPE_b${maxBatch}_l${fpuLatency}_up$withUpShift"
  val io = IO(new Bundle {
    val syncResetIn = Input(Bool())
    val shiftOutLeft = Input(Bool())
    val shiftOutAlt = if (withUpShift) Some(Input(Bool())) else None
    val validIn = Input(Bool())

    // The rest of the wires are local-only and shouldn't cause much of a problem
    val south_in = if (withUpShift) Some(Input(UInt(16.W))) else None
    val north_in = Input(UInt(16.W))
    val west_in = Input(UInt(16.W))

    val south_out = Output(UInt(16.W))
    val east_out = Output(UInt(16.W))
    val north_out = if (withUpShift) Some(Output(UInt(16.W))) else None

    val validOut = Output(Bool())

    val shift_out = Output(UInt(16.W))
    val shift_in = Input(UInt(16.W))

    // we can make these values local because they're set whenever the accumulator is initialized which is
    // ~100s-1000s of cycles before the matrix will likely make an appearance
    //
    val batchSizeMOInLeft = Input(UInt(log2Up(maxBatch).W))
    val batchSizeMOOutRight = Output(UInt(log2Up(maxBatch).W))
  })
  /**
   * NOTE: this implementation DOES NOT consider the FPU latency, the master NEEDS to account for this in the
   * control of the `validIn` signal.
   */
  val batchSizeMO = Reg(UInt(log2Up(maxBatch).W))
  val batchCountRead, batchCountWrite = RegInit(0.U(log2Up(maxBatch).W))
  batchSizeMO := io.batchSizeMOInLeft
  io.batchSizeMOOutRight := batchSizeMO


  val accumulatorVec = Reg(Vec(maxBatch, UInt(32.W)))
//  val acc_debug_asBF16 = accumulatorVec.map(acc => Cat(acc(31, 16), 0.U(16.W)))
//  if (p(BuildModeKey) != BuildMode.Synthesis)
//    dontTouch(acc_debug_asBF16)

  val fma = Module(
    new FPU(
      if (platform.platformType == PlatformType.FPGA && p(BuildModeKey)==BuildMode.Synthesis) {
        XilinxFPUImplementation(XilinxOperator.FMA, fpuLatency)
      } else {
        FPUNewImplementation(ADDMUL = Some(fpuLatency))
      },
      FPFloatFormat.Fp32,
      lanes = 1,
      sourceTy = p(FPUBuildMode)))
  // BPF16 has same # exp bits as FP32
  fma.io.req.bits.operands(0)(0) := Cat(io.north_in, 0.U(16.W))
  fma.io.req.bits.operands(1)(0) := Cat(io.west_in, 0.U(16.W))
  fma.io.req.bits.operands(2)(0) := accumulatorVec(batchCountRead)

  def incrementBatch(a: UInt): Unit = {
    when(a === io.batchSizeMOOutRight) {
      a := 0.U
    }.otherwise {
      a := a + 1.U
    }
  }

  when(fma.io.req.fire) {
    incrementBatch(batchCountRead)
  }
  io.validOut := RegNext(io.validIn)
  when(fma.io.resp.valid) {
    incrementBatch(batchCountWrite)
    accumulatorVec(batchCountWrite) := fma.io.resp.bits.result(0)
  }
  when(io.syncResetIn) {
    accumulatorVec foreach (_ := 0.U)
    batchCountRead := 0.U
    batchCountWrite := 0.U
  }

  fma.io.req.bits.op := FPOperation.FMADD
  fma.io.req.bits.opModifier := false.B
  fma.io.req.bits.srcFormat := FPFloatFormat.Fp32
  fma.io.req.bits.dstFormat := FPFloatFormat.Fp32
  fma.io.req.bits.intFormat := FPIntFormat.Int32
  fma.io.req.bits.roundingMode := FPRoundingMode.RNE
  fma.io.req.valid := io.validIn
  fma.io.resp.ready := true.B

  io.east_out := RegNext(io.west_in)
  io.south_out := RegNext(io.north_in)
  io.shift_out := DontCare
  if (withUpShift)
    io.north_out.get := io.shift_out

  when(io.shiftOutLeft || io.shiftOutAlt.getOrElse(false.B)) {
    val currentRead = accumulatorVec(batchCountRead)
    val roundBF16 = currentRead(31, 16) + currentRead(15)
    val cut = currentRead(31, 16)
    val isVerySmall = cut(14, 0) === 0.U
    val isNonZero = currentRead(15, 0) =/= 0.U
    val sign = currentRead(31)

    io.shift_out := Mux(isVerySmall && isNonZero, Cat(sign, 1.U(15.W)), roundBF16) // this is RNE rounding, maybe dont use?
    when (io.shiftOutLeft) {
      accumulatorVec(batchCountRead) := Cat(io.shift_in, 0.U(16.W))
    }
      when(io.shiftOutAlt.getOrElse(false.B)) {
        accumulatorVec(batchCountRead) := Cat(io.south_in.getOrElse(0.U), 0.U(16.W))
      }
    batchCountRead := batchCountRead + 1.U
    when(batchCountRead === batchSizeMO) {
      batchCountRead := 0.U
    }
  }

   // Uncomment this to make verilator log a version that is in FP32 format
  val north_as_float, west_as_float = Wire(UInt(32.W))
  north_as_float := Cat(io.north_in, 0.U(16.W))
  west_as_float := Cat(io.west_in, 0.U(16.W))
  dontTouch(north_as_float)
  dontTouch(west_as_float)
}
