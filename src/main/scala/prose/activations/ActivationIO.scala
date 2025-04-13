package prose.activations

import chisel3._

class ActivationIO extends Bundle {
  val in = Input(UInt(16.W))
  val out = Output(UInt(16.W))
}

trait HasBlackBoxActivationIO {
  val io = IO(new Bundle {
    val x = Input(UInt(16.W))
    val y = Output(UInt(16.W))
    val clock = Input(Clock())
    val reset = Input(Reset())
    val enable = Input(Bool())
  })
}

trait HasActivationIO {
  val io = IO(new ActivationIO())
}