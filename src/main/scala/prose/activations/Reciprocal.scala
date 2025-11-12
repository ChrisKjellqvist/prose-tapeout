package prose.activations

import beethoven.BeethovenBuild
import beethoven.common.ShiftReg
import org.chipsalliance.cde.config._
import chisel3._
import chisel3.util._
import fpwrapper.{
  FPFloatFormat,
  FPOperation,
  FPRoundingMode,
  FPU,
  FPUSourceType
}
import fpwrapper.impl.fpnew.{FPNewOpSupport, FPUNewImplementation}
import prose.FPUBuildMode
import fpwrapper.FPIntFormat

class Reciprocal(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(16.W)))
    val out = Decoupled(UInt(16.W))
  })

  val divider = Module(
    new FPU(
      FPUNewImplementation(DIVSQRT = Some(2)),
      floattype = FPFloatFormat.Fp16Alt,
      lanes = 1,
      sourceTy = FPUSourceType.NonSelfContainedSystemVerilog
    )
  )
  divider.io.req.bits.operands(0)(0) := 0x3f80L.U
  divider.io.req.bits.operands(1)(0) := io.in.bits
  divider.io.req.bits.operands(2) := DontCare
  divider.io.req.bits.dstFormat := FPFloatFormat.Fp16Alt
  divider.io.req.bits.srcFormat := FPFloatFormat.Fp16Alt
  divider.io.req.bits.intFormat := FPIntFormat.Int16
  divider.io.req.bits.roundingMode := FPRoundingMode.RNE
  divider.io.req.bits.opModifier := 0.U
  divider.io.req.bits.op := FPOperation.DIV
  io.in.ready := divider.io.req.ready
  divider.io.req.valid := io.in.valid
  io.out.valid := divider.io.resp.valid
  io.out.bits := divider.io.resp.bits.result(0)
  divider.io.resp.ready := io.out.ready
}
