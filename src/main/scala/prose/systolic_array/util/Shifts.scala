package prose.systolic_array.util

import beethoven.common.ShiftReg
import chipsalliance.rocketchip.config._
import chisel3._

/**
 * Note: We keep these modules separate so we can measure their power more easily from synthesis
 */
class ShiftArray (n: Int, dwidth: Int)(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(n, UInt(dwidth.W)))
    val out = Output(Vec(n, UInt(dwidth.W)))
  })

  io.in.zip(io.out).zipWithIndex foreach { case ((i, o), idx) =>
    o := ShiftReg(i, idx, clock)
  }
}

class ShiftScan (n: Int, dwidth: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(dwidth.W))
    val out = Output(Vec(n, UInt(dwidth.W)))
  })

  io.out.zipWithIndex.foreach { case (o, idx) =>
    o := {
      if (idx == 0) io.in
      else RegNext(io.out(idx-1))
    }
  }
}
