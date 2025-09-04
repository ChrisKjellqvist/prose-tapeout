package prose.nn_util.norm

import beethoven.common.ShiftReg
import chipsalliance.rocketchip.config.Parameters
import chisel3._
import chisel3.util._
import fpwrapper._
import fpwrapper.impl.fpnew.FPUNewImplementation
import prose.FPUBuildMode
import prose.activations._

// compute 1/sqrt(var[x]+eps) in accordance with layernorm
class Variance(concurrency: Int,
               subLatency: Int = 2,
               fmaLatency: Int = 3,
               sqrtLUTLatency: Int = 1,
               dataWidthBytes: Int,
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
    val input = Flipped(Decoupled(new Bundle {
      val tag = UInt(log2Up(concurrency).W)
      val data = UInt(dataWidthBits.W)
    }))

    val norm_ty = Input(UInt(NormType.width.W))

    val begin = Flipped(Decoupled(new Bundle {
      val means = Vec(concurrency, UInt(dataWidthBits.W))
      val norm = UInt(dataWidthBits.W)
    }))

    // layernorm(inputs) = (x-E[x])/(sqrt(var[x]+eps) * gamma + beta
    // gamma and beta are learnable
    // we want to compute the normalization factor 1/sqrt(var[x]+eps)
    // 1/sqrt(var[x]+eps) = 1/sqrt(SE/N+eps)
    val computeVariances_ready = Output(Bool())
    val computeVariances_valid = Input(Bool())
    val output = Decoupled(new Bundle {
      val variances = Vec(concurrency, UInt(dataWidthBits.W))
      val means = Vec(concurrency, UInt(dataWidthBits.W))
    })
  })

  io.output.valid := false.B

  val s_idle :: s_acc :: s_emitNorm :: s_fin :: Nil = Enum(4)
  val accMode = RegInit(s_idle)
  val accs = Reg(Vec(concurrency, UInt(accumulatorDataWidth.W)))
  val accIdle = Reg(Vec(concurrency, Bool()))
  val means = Reg(Vec(concurrency, UInt(dataWidthBits.W)))
  val norm = Reg(UInt(dataWidthBits.W))
  val normCntr = Reg(UInt(log2Up(concurrency).W))
  val inflight = Reg(Bool())

  when(io.begin.fire) {
    accIdle foreach (_ := true.B)
    accs.foreach(_ := 0.U)
    accMode := Mux(io.norm_ty === NormType.LayerNorm.U, s_acc, s_emitNorm)
    normCntr := 0.U
    norm := io.begin.bits.norm
    means := io.begin.bits.means
    inflight := false.B
  }

  val subFPU = Module(new FPU(FPUNewImplementation(ADDMUL = Some(subLatency)), floattype = fpuDataType, 1,
    sourceTy = p(FPUBuildMode)))
  val mulFPU = Module(new FPU(FPUNewImplementation(ADDMUL = Some(fmaLatency)), floattype = fpuDataType, 1,
    sourceTy = p(FPUBuildMode)))

  Seq(mulFPU, subFPU) foreach { fpu =>
    fpu.io.req.bits.dstFormat := fpuDataType
    fpu.io.req.bits.srcFormat := fpuDataType
  }
  val fpuFlushCntdown = RegInit(0.U(log2Up(subLatency + fmaLatency + 1).W))
  when(fpuFlushCntdown =/= 0.U) {
    fpuFlushCntdown := fpuFlushCntdown - 1.U
  }
  when(subFPU.io.req.fire) {
    fpuFlushCntdown := (fmaLatency + subLatency).U
  }

  io.begin.ready := fpuFlushCntdown === 0.U && accMode === s_idle
  io.computeVariances_ready := !io.input.valid && accMode === s_acc && fpuFlushCntdown === 0.U

  // q = (x - E[x])
  subFPU.io.req.valid := io.input.fire
  when(io.input.fire) {
    assert(subFPU.io.req.ready)
  }
  subFPU.io.req.bits := DontCare
  subFPU.io.req.bits.op := FPOperation.ADD
  subFPU.io.req.bits.opModifier := 1.U // ADD with opModifier is SUB
  subFPU.io.req.bits.operands(0) := DontCare
  subFPU.io.req.bits.operands(1)(0) := fpuLoadData(io.input.bits.data)
  subFPU.io.req.bits.operands(2)(0) := fpuLoadData(means(io.input.bits.tag))
  val subTag = ShiftReg(io.input.bits.tag, subLatency, clock)

  // acc = acc + q * q
  mulFPU.io.req.valid := subFPU.io.resp.valid
  when(subFPU.io.resp.valid) {
    assert(mulFPU.io.req.ready)
  }
  mulFPU.io.req.bits := DontCare
  mulFPU.io.req.bits.op := FPOperation.FMADD
  mulFPU.io.req.bits.opModifier := 0.U
  mulFPU.io.req.bits.operands(0)(0) := subFPU.io.resp.bits.result(0)
  mulFPU.io.req.bits.operands(1)(0) := subFPU.io.resp.bits.result(0)
  mulFPU.io.req.bits.operands(2)(0) := accs(subTag)
  val mulTag = ShiftReg(subTag, fmaLatency, clock)

  io.input.ready := accIdle(io.input.bits.tag) && (accMode === s_acc)
  assert(!(io.computeVariances_ready && io.computeVariances_valid && io.input.valid))
  when(io.input.fire) {
    accIdle(io.input.bits.tag) := false.B
  }

  mulFPU.io.resp.ready := true.B
  subFPU.io.resp.ready := true.B


  io.output.bits.variances.zip(accs).foreach { case (variances, acc) =>
    variances := fpuUnpackResult(acc)
  }
  io.output.bits.means := means

  val sqrtLUT = Module(new LookupTableWithLatency("InvSqrt", sqrtLUTLatency))
  sqrtLUT.io.in := (fpuDataType match {
    case FPFloatFormat.Fp16Alt => fpuUnpackResult (mulFPU.io.resp.bits.result(0))
    case FPFloatFormat.Fp32 => mulFPU.io.resp.bits.result(0)(31, 16)
  })
  val sqrtLUTvalid = Wire(Bool())
  sqrtLUTvalid := false.B
  val sqrtLUTvalidOut = ShiftReg(sqrtLUTvalid, sqrtLUTLatency, clock)

  when(accMode === s_acc) {
    when(mulFPU.io.resp.valid) {
      accIdle(mulTag) := true.B
      accs(mulTag) := mulFPU.io.resp.bits.result(0)
    }
    when(io.computeVariances_ready && io.computeVariances_valid) {
      accMode := s_emitNorm
      normCntr := 0.U
      inflight := false.B
    }
  }.elsewhen(accMode === s_emitNorm) {
    mulFPU.io.req.valid := !inflight
    when(mulFPU.io.req.fire) {
      inflight := true.B
    }
    mulFPU.io.resp.ready := true.B
    mulFPU.io.req.bits.operands(0)(0) := Mux(io.norm_ty === NormType.LayerNorm.U, accs(normCntr), fpuLoadData(means(normCntr)))
    mulFPU.io.req.bits.operands(1)(0) := fpuLoadData(norm) // 1/N
    mulFPU.io.req.bits.operands(2)(0) := fpuLoadData(0x3727.U) // 1e-5
    sqrtLUTvalid := mulFPU.io.resp.valid

    val normShift = ShiftReg(normCntr, fmaLatency + sqrtLUTLatency, clock)
    when(sqrtLUTvalidOut) {
      accs(normShift) := fpuLoadData(sqrtLUT.io.out)
      normCntr := normCntr + 1.U
      inflight := false.B
      when(normCntr === (concurrency - 1).U) {
        accMode := s_fin
      }

    }
  }.elsewhen(accMode === s_fin) {
    io.output.valid := true.B
    when(io.output.fire) {
      accs.foreach(_ := 0.U)
      accMode := s_idle
    }
  }
}
