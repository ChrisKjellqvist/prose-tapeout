package prose.nn_util.norm

import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper.impl.fpnew.FPUNewImplementation
import fpwrapper.{FPFloatFormat, FPOperation, FPRoundingMode, FPU}
import prose.FPUBuildMode

// in contrast to mean and variance, this module produces a long stream of elements that needs to support
// backpressure.
class LayerNormStream(concurrency: Int,
                      fmaLatency: Int = 3,
                      dataWidthBytes: Int,
                      fpuDataType: FPFloatFormat.Type,
                      fpuLoadData: UInt => UInt,
                      fpuUnpackResult: UInt => UInt)(implicit p: Parameters) extends Module {
  val dataWidthBits = dataWidthBytes * 8
  val io = IO(new Bundle {
    val start_handshake = Flipped(Decoupled(new Bundle {
      val norms = Vec(concurrency, UInt(dataWidthBits.W))
      val means = Vec(concurrency, UInt(dataWidthBits.W))
    }))

    val input_stream = Flipped(Decoupled(new Bundle {
      val x = UInt(dataWidthBits.W)
      val gamma = UInt(dataWidthBits.W)
      val beta = UInt(dataWidthBits.W)

      val tag = UInt(log2Up(concurrency).W)
    }))

    val norm_ty = Input(UInt(NormType.width.W))

    val resetStream = Input(Bool())
    val output = Decoupled(UInt(dataWidthBits.W))
  })
  val norms = Reg(Vec(concurrency, UInt(dataWidthBits.W)))
  val means = Reg(Vec(concurrency, UInt(dataWidthBits.W)))
  val has_concluded = RegInit(true.B)
  when(io.resetStream) {
    has_concluded := true.B
  }
  io.start_handshake.ready := false.B
  io.input_stream.ready := false.B

  when(has_concluded) {
    io.start_handshake.ready := true.B
  }.otherwise {
    io.input_stream.ready := true.B
  }

  when(io.start_handshake.fire) {
    norms := io.start_handshake.bits.norms
    means := io.start_handshake.bits.means
    has_concluded := false.B
  }

  // this module computes the final (x-E[x])/(sqrt(var[x]+e) * gamma + beta)

  // 1. x - E[x]
  val subFPU = Module(new FPU(FPUNewImplementation(ADDMUL = Some(fmaLatency)), floattype = fpuDataType, 1,
    sourceTy = p(FPUBuildMode)))
  io.input_stream.ready := subFPU.io.req.ready && !has_concluded
  subFPU.io.req.valid := io.input_stream.valid && !has_concluded
  subFPU.io.req.bits := DontCare
  subFPU.io.req.bits.op := FPOperation.ADD
  subFPU.io.req.bits.opModifier := 1.U
  subFPU.io.req.bits.operands(1)(0) := fpuLoadData(io.input_stream.bits.x)
  subFPU.io.req.bits.operands(2)(0) := Mux(io.norm_ty === NormType.LayerNorm.U, fpuLoadData(means(io.input_stream.bits.tag)), 0.U)

  // gamma * 1/sqrt(var[x]+e)
  //   norm is 1/sqrt(var[x]+e) in this context
  val normMul = Module(new FPU(FPUNewImplementation(ADDMUL = Some(fmaLatency)), floattype = fpuDataType, 1,
    sourceTy = p(FPUBuildMode)))
  normMul.io.req.valid := io.input_stream.valid && !has_concluded
  when(subFPU.io.req.fire) {
    assert(normMul.io.req.fire)
  }
  normMul.io.req.bits := DontCare
  normMul.io.req.bits.op := FPOperation.MUL
  normMul.io.req.bits.opModifier := 0.U
  normMul.io.req.bits.operands(0)(0) := fpuLoadData(norms(io.input_stream.bits.tag))
  normMul.io.req.bits.operands(1)(0) := fpuLoadData(io.input_stream.bits.gamma)

  // because paired gamma and beta are provided simultaneously, but only one is consumed on g * norm mult, save the beta for later
  val betaShift = Module(new Queue(io.input_stream.bits.beta.cloneType, fmaLatency + 1))
  betaShift.io.enq.valid := io.input_stream.fire
  betaShift.io.enq.bits := io.input_stream.bits.beta
  assert(normMul.io.resp.valid === subFPU.io.resp.valid)

  // final operation, combing beta and output from normMul
  val layerNormFMA = Module(new FPU(FPUNewImplementation(ADDMUL = Some(fmaLatency)), fpuDataType, 1,
    sourceTy = p(FPUBuildMode)))
  layerNormFMA.io.req.valid := subFPU.io.resp.valid
  subFPU.io.resp.ready := layerNormFMA.io.req.ready
  normMul.io.resp.ready := layerNormFMA.io.req.ready
  betaShift.io.deq.ready := layerNormFMA.io.req.ready
  when(layerNormFMA.io.req.fire) {
    assert(normMul.io.resp.valid)
    assert(io.norm_ty =/= NormType.LayerNorm.U || betaShift.io.deq.valid)
  }
  layerNormFMA.io.req.bits.op := FPOperation.FMADD
  layerNormFMA.io.req.bits.opModifier := 0.U
  layerNormFMA.io.req.bits.operands(0)(0) := normMul.io.resp.bits.result(0)
  layerNormFMA.io.req.bits.operands(1)(0) := subFPU.io.resp.bits.result(0)
  layerNormFMA.io.req.bits.operands(2)(0) := Mux(io.norm_ty === NormType.LayerNorm.U, fpuLoadData(betaShift.io.deq.bits), 0.U)

  Seq(layerNormFMA, normMul, subFPU).foreach { fpu =>
    fpu.io.req.bits.srcFormat := fpuDataType
    fpu.io.req.bits.dstFormat := fpuDataType
    fpu.io.req.bits.intFormat := DontCare
    fpu.io.req.bits.roundingMode := FPRoundingMode.RNE
  }

  io.output.valid := layerNormFMA.io.resp.valid
  layerNormFMA.io.resp.ready := io.output.ready
  io.output.bits := fpuUnpackResult(layerNormFMA.io.resp.bits.result(0))
}
