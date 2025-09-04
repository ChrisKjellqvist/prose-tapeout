package prose.nn_util.residual

import beethoven._
import beethoven.common._
import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper.{FPFloatFormat, FPOperation, FPRoundingMode, FPU, FPUSourceType}
import fpwrapper.impl.fpnew.FPUNewImplementation
import prose.FPUBuildMode

class MatAdd(N: Int, maxLength: Int, fpuLatency: Int)(implicit p: Parameters) extends AcceleratorCore{
  val io = BeethovenIO(new AccelCommand("MatAdd") {
    val a = Address()
    val b = Address()
    val c = Address()

    val length = UInt(log2Up(maxLength+1).W)
  }, EmptyAccelResponse())
  val ReaderModuleChannel(reqA, datA) = getReaderModule("A")
  val ReaderModuleChannel(reqB, datB) = getReaderModule("B")
  val WriterModuleChannel(reqC, datC) = getWriterModule("C")

  val s_idle :: s_process :: s_final :: Nil = Enum(3)
  val state = RegInit(s_idle)

  val datAspl = splitIntoChunks(datA.data.bits, 16)
  val datBspl = splitIntoChunks(datB.data.bits, 16)

  val datOut = Wire(Vec(N, UInt(16.W)))
  datC.data.bits := Cat(datOut.reverse)

  datAspl.zip(datBspl).zip(datOut).zipWithIndex.foreach { case (((a, b), out), idx) =>
    val fpu = Module(new FPU(FPUNewImplementation(ADDMUL = Some(fpuLatency)), FPFloatFormat.Fp16Alt, 1,
      sourceTy = p(FPUBuildMode)))
    val bothStreamsValid = datA.data.valid && datB.data.valid
    fpu.io.req.valid := bothStreamsValid
    datA.data.ready := bothStreamsValid && fpu.io.req.ready
    datB.data.ready := bothStreamsValid && fpu.io.req.ready
    fpu.io.req.bits.dstFormat := FPFloatFormat.Fp16Alt
    fpu.io.req.bits.srcFormat := FPFloatFormat.Fp16Alt
    fpu.io.req.bits.intFormat := DontCare
    fpu.io.req.bits.op := FPOperation.ADD
    fpu.io.req.bits.opModifier := 0.U
    fpu.io.req.bits.roundingMode := FPRoundingMode.RNE
    fpu.io.req.bits.operands(0) := DontCare
    fpu.io.req.bits.operands(1)(0) := a
    fpu.io.req.bits.operands(2)(0) := b

    out := fpu.io.resp.bits.result(0)
    fpu.io.resp.ready := datC.data.ready
    if (idx == 0) {
      datC.data.valid := fpu.io.resp.valid
    }
  }

  Seq(io.req.bits.a, io.req.bits.b, io.req.bits.c).zip(Seq(reqA, reqB, reqC)) foreach {
    case (addr, req) =>
      req.bits.addr := addr
      req.bits.len := Cat(io.req.bits.length, 0.U(1.W))
      req.valid := io.req.fire
  }

  io.req.ready := false.B
  when (state === s_idle) {
    io.req.ready := reqA.ready && reqB.ready && reqC.ready
    when (io.req.fire) {
      state := s_process
    }
  }.elsewhen(state === s_process) {
    when (reqA.ready && reqB.ready && reqC.ready) {
      state := s_final
    }
  }.elsewhen(state === s_final) {
    io.resp.valid := true.B
    when (io.resp.fire) {
      state := s_idle
    }
  }
}
