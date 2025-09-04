package prose.activations

import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper.{FPFloatFormat, FPOperation, FPRoundingMode, FPU, FPUSourceType}
import fpwrapper.impl.fpnew.{FPNewOpSupport, FPUNewImplementation}
import prose.FPUBuildMode

class Reciprocal(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(16.W)))
    val out = ValidIO(UInt(16.W))
  })
  val divider = Module(new FPU(FPUNewImplementation(DIVSQRT = Some(2)),
    FPFloatFormat.Fp32, 1,
    sourceTy = p(FPUBuildMode)))
  divider.io.req.valid := io.in.valid

  divider.io.req.bits.op := FPOperation.DIV
  // 1.0 in FP16 = 0x3f80
  divider.io.req.bits.operands(0)(0) := 0x3f800000L.U
  divider.io.req.bits.operands(1)(0) := Cat(io.in.bits, 0.U(16.W))
  divider.io.req.bits.operands(2) := DontCare
  divider.io.req.bits.dstFormat := FPFloatFormat.Fp32
  divider.io.req.bits.srcFormat := FPFloatFormat.Fp32
  divider.io.req.bits.intFormat := DontCare
  divider.io.req.bits.roundingMode := FPRoundingMode.RNE
  divider.io.req.bits.opModifier := 0.U

  io.in.ready := divider.io.req.ready
  io.out.valid := divider.io.resp.valid
  io.out.bits := divider.io.resp.bits.result(0)(31, 16)
  divider.io.resp.ready := true.B
}
