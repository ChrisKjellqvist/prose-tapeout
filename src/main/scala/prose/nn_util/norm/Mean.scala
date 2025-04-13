package prose.nn_util.norm

import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper.impl.fpnew.FPUNewImplementation
import fpwrapper.{FPFloatFormat, FPOperation, FPU}
import prose.FPUBuildMode

class Mean(concurrentMeans: Int,
           dataWidthBytes: Int,
           fmaLatency: Int,
           fpuDataType: FPFloatFormat.Type,
           fpuLoadData: UInt => UInt,
           fpuUnpackResult: UInt => UInt)(implicit p: Parameters) extends Module {
  val dataWidthBits = dataWidthBytes * 8
  val accumulatorDataWidth = fpuDataType match {
    case FPFloatFormat.Fp16Alt => 32
    case FPFloatFormat.Fp32 => 32
    case FPFloatFormat.Fp64 => 64
  }
  val io = IO(new Bundle {
    val input = Flipped(Decoupled(new Bundle() {
      val data = UInt(dataWidthBits.W)
      val tag = Input(UInt(log2Up(concurrentMeans).W))
    }))
    val norm_ty = Input(UInt(NormType.width.W))

    val begin = Flipped(Decoupled(UInt(dataWidthBits.W)))

    val done = Input(Bool())
    val output = Decoupled(Vec(concurrentMeans, UInt(dataWidthBits.W)))
  })
  val accIdle = Reg(Vec(concurrentMeans, Bool()))
  val accReg = Reg(Vec(concurrentMeans, UInt(accumulatorDataWidth.W)))
  val in_progress = RegInit(false.B)
  val output_latch = RegInit(false.B)
  val norm = Reg(UInt(dataWidthBits.W))
  io.begin.ready := !in_progress && !output_latch
  io.input.ready := in_progress && !output_latch
  io.output.valid := output_latch && accIdle.fold(true.B)(_ && _)
  io.output.bits.zip(accReg).foreach { case (sink, src) =>
    sink := fpuUnpackResult(src)
  }

  when(io.begin.fire) {
    in_progress := true.B
    norm := io.begin.bits
  }
  when(in_progress && io.done) {
    in_progress := false.B
    output_latch := true.B
  }
  when(io.output.fire) {
    output_latch := false.B
  }

  val fpu = Module(new FPU(
    FPUNewImplementation(ADDMUL = Some(fmaLatency)),
    floattype = fpuDataType, 1,
    sourceTy = p(FPUBuildMode)
  ))
  val tagQ = Module(new Queue(io.input.bits.tag.cloneType, concurrentMeans + 1))
  fpu.io.req.valid := io.input.fire
  assert(tagQ.io.enq.ready)

  tagQ.io.enq.valid := io.input.fire
  tagQ.io.enq.bits := io.input.bits.tag
  tagQ.io.deq.ready := false.B
  fpu.io.req.bits := DontCare
  fpu.io.req.bits.operands(0)(0) := Mux(io.norm_ty === NormType.LayerNorm.U, fpuLoadData(norm), fpuLoadData(io.input.bits.data))
  fpu.io.req.bits.operands(1)(0) := fpuLoadData(io.input.bits.data)
  fpu.io.req.bits.operands(2)(0) := accReg(io.input.bits.tag)
  fpu.io.req.bits.op := FPOperation.FMADD
  fpu.io.req.bits.opModifier := 0.U
  fpu.io.req.bits.srcFormat := fpuDataType
  fpu.io.req.bits.dstFormat := fpuDataType

  io.input.ready := accIdle(io.input.bits.tag)
  when(io.input.fire) {
    assert(fpu.io.req.ready)
    accIdle(io.input.bits.tag) := false.B
  }
  fpu.io.resp.ready := true.B
  when(fpu.io.resp.fire) {
    accReg(tagQ.io.deq.bits) := fpu.io.resp.bits.result(0)
    accIdle(tagQ.io.deq.bits) := true.B
    tagQ.io.deq.ready := true.B
    assert(tagQ.io.deq.valid)
  }

  when(io.begin.fire) {
    accReg.foreach(_ := 0.U)
    accIdle.foreach(_ := true.B)
  }
}
