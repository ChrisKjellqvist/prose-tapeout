package prose.nn_util.norm
import chisel3._
import chisel3.util._
import beethoven._
import chipsalliance.rocketchip.config.Parameters
import fpwrapper.{FPFloatFormat, FPIntFormat, FPOperation, FPRoundingMode, FPU, FPUSourceType}
import fpwrapper.impl.fpnew.FPUNewImplementation
import prose.activations.Reciprocal

class InvSqrt()(implicit p: Parameters) extends Module {
  val latency = 2
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(16.W)))
    val out = Decoupled(UInt(16.W))
  })


  val inv = Module(new Reciprocal())
  val sqr = Module(new FPU(FPUNewImplementation(DIVSQRT = Some(latency)),
    floattype = FPFloatFormat.Fp16Alt,
    lanes = 1,
    sourceTy = FPUSourceType.NonSelfContainedSystemVerilog))

  io.in <> inv.io.in
  sqr.io.req.valid := inv.io.out.valid
  sqr.io.req.bits.op := FPOperation.SQRT
  sqr.io.req.bits.opModifier := 0.U
  sqr.io.req.bits.dstFormat := FPFloatFormat.Fp16Alt
  sqr.io.req.bits.srcFormat := FPFloatFormat.Fp16Alt
  sqr.io.req.bits.intFormat := FPIntFormat.Int16
  sqr.io.req.bits.operands(0)(0) := inv.io.out.bits
  sqr.io.req.bits.operands(1) := DontCare
  sqr.io.req.bits.operands(2) := DontCare
  sqr.io.req.bits.roundingMode := FPRoundingMode.RNE
  io.out <> sqr.io.resp.map(_.result(0))
}
