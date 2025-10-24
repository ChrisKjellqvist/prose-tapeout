package prose.activations

import beethoven._
import chisel3._
import chisel3.util.Cat
import beethoven.Platforms._
import beethoven.common._
import fpwrapper._
import fpwrapper.impl.fpnew._
import fpwrapper.impl.xilinx.{XilinxFPUImplementation, XilinxOperator}
import prose.{FPUBuildMode, SpecialFunction}
import prose.SpecialFunction.SpecialFunction
import prose.random.ShiftRegEnableReset
import org.chipsalliance.cde.config._

class ProseSIMD(vectorDepth: Int,
                fpuLatency: Int,
                withScalarMult: Boolean,
                accumulatorWidth: Int,
                specialFn: Option[(SpecialFunction, Int)])(implicit p: Parameters) extends Module {
  /**
   * The point of making a separate module for this is that since we're doing a separate synthesis/route
   * for FPUs (ie, not global), we need the FPU module to include the knowledge that the scalar multiple is
   * broadcast to the entire SIMD, that operand(0) is the same all the way down the column.
   */

  override val desiredName = s"ProSESIMD_v${vectorDepth}_fpu${fpuLatency}${
    specialFn match {
      case None => ""
      case Some((SpecialFunction.EXP, l)) => "_e" + l.toString
      case Some((SpecialFunction.GELU, l)) => "_g" + l.toString
    }
  }"

  val io = IO(new Bundle {
    val validIn = Input(Bool())
    val readyIn = Output(Bool())
    val operand = Input(Vec(vectorDepth, UInt(16.W)))
    val accumulator = Input(Vec(vectorDepth, UInt(accumulatorWidth.W)))
    val operandMult = if (withScalarMult) Some(Input(Vec(vectorDepth, UInt(16.W)))) else None

    val validOut = Output(Bool())
    val readyOut = Input(Bool())
    val out = Output(Vec(vectorDepth, UInt(accumulatorWidth.W)))
  })


  class comboFPU extends Module {
    val io = IO(new Bundle {
      val validIn = Input(Bool())
      val readyIn = Output(Bool())
      val operand = Input(UInt(16.W))
      val accumulator = Input(UInt(accumulatorWidth.W))

      val operandMult = if (withScalarMult) Some(Input(UInt(16.W))) else None
      val validOut = Output(Bool())
      val readyOut = Input(Bool())
      val outS = Output(UInt(accumulatorWidth.W))
    })

    override val desiredName = f"comboSIMDFPU_${vectorDepth}_${fpuLatency}_${withScalarMult}_${accumulatorWidth}_${specialFn.map(_._1.toString).getOrElse("_")}"

    val fpu = Module(new FPU(
      implementation =
        if (platform.platformType == PlatformType.FPGA && p(BuildModeKey)==BuildMode.Synthesis) {
          XilinxFPUImplementation(XilinxOperator.FMA, fpuLatency)
        } else {
          FPUNewImplementation(ADDMUL = Some(fpuLatency))
        },
      floattype = FPFloatFormat.Fp32,
      lanes = 1,
      sourceTy = p(FPUBuildMode)))
    fpu.io.req.valid := io.validIn
    fpu.io.req.bits.operands(0)(0) := (if (withScalarMult) Cat(io.operandMult.get, 0.U(16.W)) else DontCare)
    fpu.io.req.bits.operands(1)(0) := Cat(io.operand, 0.U(16.W))
    fpu.io.req.bits.operands(2)(0) := (if (accumulatorWidth == 16) Cat(io.accumulator, 0.U(16.W)) else io.accumulator)
    fpu.io.req.bits.op := (if (withScalarMult) FPOperation.FMADD else FPOperation.ADD)
    fpu.io.req.bits.dstFormat := FPFloatFormat.Fp32
    fpu.io.req.bits.srcFormat := FPFloatFormat.Fp32
    fpu.io.req.bits.intFormat := DontCare
    fpu.io.req.bits.opModifier := 0.U
    fpu.io.req.bits.roundingMode := FPRoundingMode.RNE

    fpu.io.resp.ready := io.readyOut
    io.readyIn := fpu.io.req.ready
    specialFn match {
      case None =>
        io.validOut := fpu.io.resp.valid
        if (accumulatorWidth == 32)
          io.outS := fpu.io.resp.bits.result(0)
        else
          io.outS := fpu.io.resp.bits.result(0)(31, 16)
      case Some((SpecialFunction.EXP, latency: Int)) =>
        val expModule = Module(new LookupTableWithLatencyWithEnable("Exp", latency))
        expModule.io.in := fpu.io.resp.bits.result(0)(31, 16)
        expModule.enable := io.readyOut
        io.validOut := ShiftRegEnableReset(io.readyOut || reset.asBool, false.B, fpu.io.resp.valid, latency, 0, clock)
        //          io.readyOut ||
        //            ShiftReg(reset.asBool, latency))
        if (accumulatorWidth == 32)
          io.outS := Cat(expModule.io.out, 0.U(16.W))
        else {
          io.outS := expModule.io.out
        }
      case Some((SpecialFunction.GELU, latency: Int)) =>
        val geluModule = Module(new LookupTableWithLatencyWithEnable("GeLU", latency))
        geluModule.io.in := fpu.io.resp.bits.result(0)(31, 16)
        val fpu_gelu = Module(new FPU(
          implementation =
            if (platform.platformType == PlatformType.FPGA && p(BuildModeKey)==BuildMode.Synthesis) {
              XilinxFPUImplementation(XilinxOperator.FMA, latency)
            } else {
              FPUNewImplementation(ADDMUL = Some(latency))
            },
          floattype = FPFloatFormat.Fp16Alt,
          lanes = 1,
          sourceTy = p(FPUBuildMode)))
          val geluModuleValidOut = ShiftRegEnable(fpu.io.resp.valid, latency, io.readyOut, clock)
          fpu_gelu.io.req.valid := geluModuleValidOut && io.readyOut
          geluModule.enable := io.readyOut
          val relu = Mux(fpu.io.resp.bits.result(0)(31), 0.U(16.W), fpu.io.resp.bits.result(0)(31, 16))
          fpu_gelu.io.req.bits.operands(0)(0) := 0.U
          fpu_gelu.io.req.bits.operands(1)(0) := ShiftReg(relu, latency, clock)
          fpu_gelu.io.req.bits.operands(2)(0) := geluModule.io.out
          fpu_gelu.io.req.bits.op := FPOperation.ADD
          fpu_gelu.io.req.bits.dstFormat := FPFloatFormat.Fp16Alt
          fpu_gelu.io.req.bits.srcFormat := FPFloatFormat.Fp16Alt
          fpu_gelu.io.req.bits.intFormat := DontCare
          fpu_gelu.io.req.bits.opModifier := 0.U
          fpu_gelu.io.req.bits.roundingMode := FPRoundingMode.RNE
          fpu_gelu.io.req.valid := ShiftRegEnableReset(io.readyOut || reset.asBool, false.B, fpu.io.resp.valid, latency, 0, clock)

          fpu_gelu.io.resp.ready := io.readyOut

          io.validOut := fpu_gelu.io.resp.valid
          if (accumulatorWidth == 32)
            io.outS := Cat(fpu_gelu.io.resp.bits.result(0), 0.U(16.W))
          else
            io.outS := fpu_gelu.io.resp.bits.result(0)
    }

    if (p(BuildModeKey) == BuildMode.Simulation) {
      val operandCast = WireInit(VecInit((io.operand << 16).asUInt))
      val operandMultCast = WireInit(VecInit((io.operandMult.getOrElse(0.U) << 16).asUInt))
      val accumulatorCast = if (accumulatorWidth == 16) WireInit(VecInit((io.accumulator << 16).asUInt)) else io.accumulator
      val outputCast = WireInit(VecInit((io.outS << 16).asUInt))
      Seq(operandCast, operandMultCast, outputCast, accumulatorCast) foreach dontTouch.apply
    }
  }

  for (i <- 0 until vectorDepth) {
    val fpu = Module(new comboFPU)
    if (i == 0) {
      io.validOut := fpu.io.validOut
      io.readyIn := fpu.io.readyIn
    }

    fpu.io.validIn := io.validIn
    fpu.io.accumulator := io.accumulator(i)
    fpu.io.operand := io.operand(i)
    if (fpu.io.operandMult.isDefined) fpu.io.operandMult.get := io.operandMult.get.apply(i)
    io.out(i) := fpu.io.outS
    fpu.io.readyOut := io.readyOut
  }
}


object BuildSIMDALU extends App {
  emitVerilog(new ProseSIMD(8, 2, true, 16, None)(Parameters.empty.alterPartial {
    case BuildModeKey => BuildMode.Synthesis
  }))
}
