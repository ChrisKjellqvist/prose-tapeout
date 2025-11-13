package prose.toplevel

import beethoven.common.Misc
import org.chipsalliance.cde.config._
import beethoven.MemoryStreams._

import chisel3.util._
import chisel3._
import prose.activations.{ProseSIMD, Reciprocal}
import beethoven.common.Misc.{left_assign, multByIntPow2}
import prose.SpecialFunction

class EType(stype_latency: Int)(
    kMax: Int,
    Nmin: Int,
    N: Int,
    SRAMLatency: Int,
    fpuLatency: Int,
    simdLatency: Int,
    maxBatch: Int,
    maxNTileCols: Int,
    supportWideBias: Boolean,
    n_arrays: Int
)(implicit p: Parameters)
    extends BaseCore(
      kMax,
      Nmin,
      N,
      SRAMLatency,
      fpuLatency,
      simdLatency,
      maxBatch,
      supportWideBias,
      maxNTileCols,
      n_arrays,
      Some((SpecialFunction.EXP, stype_latency))
    ) {

  SpecialFunction.generate(SpecialFunction.EXP)

  val sm_idle :: sm_reduce :: sm_wait :: sm_div :: sm_stall :: sm_stream :: Nil =
    Enum(6)
  val softmax_state = RegInit(sm_idle)

  /** We need to storage accumulation of each batch separates but also not cause
    * top-level stalls so we may need to store multiple intermediates PER-BATCH.
    * What's the minimum storage?
    *
    * Max(2*fpuLatency, maxBatch) let F = fpulatency, B = maxBatch, b =
    * chosenBatch Proof: We store separate intermediates in different registers
    * cycle-after-cycle until we exceed the fpu latency (evenly across batches,
    * do not wrap differently for different batches). This requires x =
    * minarg_{i\in\mathcal{Z+}}{b * i \geq F} storage.
    *
    * We know that x \leq 2F because if bx>2F, then b(x-1)>F by b < F which is a
    * contradiction in the definition of x.
    *
    * If b > F, then the storage solution is trivial. The worst case storage is
    * b=F-1, requiring 2(F-1) regs.
    */

  val accumulatorMax = Math.max(maxBatch + 1, 2 * fpuLatency)
  val accumulators: Vec[Vec[UInt]] = RegInit(
    Vec(accumulatorMax, Vec(N * n_arrays, UInt(32.W))),
    VecInit(Seq.fill(accumulatorMax)(VecInit(Seq.fill(N * n_arrays)(0.U))))
  )
  val fpuLatencyCntrIn, fpuLatencyCntrOut, accumulatorBatchCounter,
      accumulatorBatchCounterOut = RegInit(UInt(log2Up(accumulatorMax).W), 0.U)
  // (multByIntPow2(1.U +& active_cores_MO, N) - 1.U)
  val smaxReducCounterWidth = log2Up(
    Seq(fpuLatency + maxBatch + 1, N, 3 * maxBatch - 3, n_arrays * N).max
  )
  val softmaxReduceCounter = RegInit(UInt(smaxReducCounterWidth.W), 0.U)

  when(state === s_start_row) {
    accumulators.flatten.foreach(_ := 0.U)
    accumulatorBatchCounter := 0.U
    accumulatorBatchCounterOut := 0.U
    fpuLatencyCntrIn := 0.U
    fpuLatencyCntrOut := 0.U

  }
  // take 1 off the latency since we're not doing a multiply?
  val accumulatorEngine = Module(
    new ProseSIMD(N * n_arrays, fpuLatency, false, 32, None)
  )
  val SIMD_outFire = simd_engine.io.readyOut && simd_engine.io.validOut
  accumulatorEngine.io.validIn := SIMD_outFire
  accumulatorEngine.io.accumulator zip accumulators(
    fpuLatencyCntrIn
  ) foreach left_assign
  accumulatorEngine.io.operand zip simd_engine.io.out foreach left_assign
  accumulatorEngine.io.readyOut := softmax_state === sm_idle // this can be over-set later

  when(SIMD_outFire) {
    fpuLatencyCntrIn := fpuLatencyCntrIn + 1.U
    val nextABC = accumulatorBatchCounter + 1.U
    accumulatorBatchCounter := nextABC
    when(nextABC === activationBatch) {
      accumulatorBatchCounter := 0.U
    }

    when(
      fpuLatencyCntrIn >= (fpuLatency - 1).U && (nextABC === activationBatch)
    ) {
      fpuLatencyCntrIn := 0.U
    }
  }

  when(
    accumulatorEngine.io.readyOut &&
      accumulatorEngine.io.validOut &&
      (softmax_state === sm_idle)
  ) {
    fpuLatencyCntrOut := fpuLatencyCntrOut + 1.U
    val nextABCOut = accumulatorBatchCounterOut + 1.U
    accumulatorBatchCounterOut := nextABCOut

    when(
      fpuLatencyCntrOut >= (fpuLatency - 1).U && (nextABCOut === activationBatch)
    ) {
      fpuLatencyCntrOut := 0.U
    }
    when(nextABCOut === activationBatch) {
      accumulatorBatchCounterOut := 0.U
    }
    accumulators(fpuLatencyCntrOut).zip(
      accumulatorEngine.io.out
    ) foreach left_assign
  }

  val (List(sm_req), List(sm_dat)) = getWriterWrapper("softmax_writeout")
  sm_req.valid := state === s_start_row
  val sm_req_len =
    Misc.multByIntPow2(CTRL_cmdCopy.batchSize * (1.U +& active_cores_MO), N * 2)
  sm_req.bits.len := sm_req_len
  sm_req.bits.addr := CTRL_cmdCopy.softmax_out.get
  when(state === s_start_row) {
    CTRL_cmdCopy.softmax_out.get := CTRL_cmdCopy.softmax_out.get + sm_req_len
  }
  sm_dat.data.valid := false.B
  val write_out = RegInit(Bool(), false.B)

  val reciprocal = Module(new Reciprocal)
  reciprocal.io.in.valid := false.B
  reciprocal.io.in.bits := DontCare

  val sm_norm_fifo = Module(
    new Queue(UInt(16.W), N * maxBatch * n_arrays, false, false, false, false)
  )
  sm_norm_fifo.io.enq.valid := reciprocal.io.out.valid
  sm_norm_fifo.io.enq.bits := reciprocal.io.out.bits
  reciprocal.io.out.ready := sm_norm_fifo.io.enq.ready
  sm_norm_fifo.io.deq.ready := false.B
  sm_dat.data.bits := sm_norm_fifo.io.deq.bits

  // prevent the greater core from returning until we're done with softmax
  can_move_to_finish := false.B
  when(softmax_state === sm_idle) {
    // the core has moved into it's auxiliary state
    when(state === s_aux) {
      when(activationBatch < fpuLatency.U) {
        softmaxReduceCounter := activationBatch
        accumulatorBatchCounter := 0.U
        softmax_state := sm_reduce
      }.otherwise {
        softmax_state := sm_div
        softmaxReduceCounter := 0.U
      }
    }
  }.elsewhen(softmax_state === sm_reduce) {
    accumulatorEngine.io.accumulator.zip(
      accumulators(accumulatorBatchCounter)
    ) foreach left_assign
    accumulatorEngine.io.operand.zip(
      accumulators(softmaxReduceCounter).map(_(31, 16))
    ) foreach left_assign
    accumulatorEngine.io.validIn := true.B
    softmax_state := sm_wait
  }.elsewhen(softmax_state === sm_wait) {
    accumulatorEngine.io.readyOut := true.B
    when(accumulatorEngine.io.validOut) {
      softmaxReduceCounter := softmaxReduceCounter + activationBatch
      accumulators(accumulatorBatchCounter).zip(
        accumulatorEngine.io.out
      ) foreach left_assign
      softmax_state := sm_reduce
      when(softmaxReduceCounter >= fpuLatency.U) {
        softmaxReduceCounter := activationBatch + 1.U + accumulatorBatchCounter
        accumulatorBatchCounter := accumulatorBatchCounter + 1.U
        when((accumulatorBatchCounter + 1.U) === activationBatch) {
          accumulatorBatchCounter := 0.U
          softmaxReduceCounter := 0.U
          softmax_state := sm_div
        }
      }
    }
  }.elsewhen(softmax_state === sm_div) {
    reciprocal.io.in.valid := true.B
    reciprocal.io.in.bits := accumulators(accumulatorBatchCounter)(
      softmaxReduceCounter
    )(31, 16)
    when(reciprocal.io.in.ready) {
      accumulatorBatchCounter := accumulatorBatchCounter + 1.U
      when((accumulatorBatchCounter + 1.U) === activationBatch) {
        accumulatorBatchCounter := 0.U
        softmaxReduceCounter := softmaxReduceCounter + 1.U
        when(
          softmaxReduceCounter === (multByIntPow2(
            1.U +& active_cores_MO,
            N
          ) - 1.U)
        ) {
          softmaxReduceCounter := 0.U
          softmax_state := sm_stall
        }
      }
    }
  }.elsewhen(softmax_state === sm_stall) {
    when(!reciprocal.io.out.valid) {
      softmax_state := sm_stream
    }
  }.elsewhen(softmax_state === sm_stream) {
    sm_dat.data.valid := sm_norm_fifo.io.deq.valid
    sm_norm_fifo.io.deq.ready := sm_dat.data.ready
    when(sm_dat.isFlushed && sm_req.ready) {
      can_move_to_finish := true.B
      // this shouldn't be necessary because moving into this state already implies state===s_aux
//      when(matrixOpCmd.resp.fire) {
      softmax_state := sm_idle
//      }
    }
  }
}
