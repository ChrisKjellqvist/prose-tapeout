package prose.random

import beethoven._
import chisel3._
import prose.random.ShiftRegEnableReset.counter
object ShiftRegEnableReset {
  var counter = 0
  def apply(en: Bool, rst: Bool, in: UInt, depth: Int, defaultValue: Long, clock: Clock): UInt = {
    val srs = Module(new ShiftRegEnableReset(in.getWidth, depth, defaultValue))
    srs.io.clk := clock.asBool
    srs.io.en := en
    srs.io.rst := rst
    srs.io.in := in
    srs.io.out
  }
}

class ShiftRegEnableReset(width: Int,
                          depth: Int,
                          defaultValue: Long) extends BlackBox {
  val io = IO(new Bundle {
    val clk = Input(Bool())
    val en = Input(Bool())
    val rst = Input(Bool())
    val in = Input(UInt(width.W))
    val out = Output(UInt(width.W))
  })
  val id = counter
  counter += 1
  require(depth >= 1)

  os.makeDir.all(os.pwd / "srs")
  val fname = os.pwd / "srs" / s"ShiftRegEnableReset_${id}.v"
  override val desiredName = s"ShiftRegEnableReset_${id}"
  os.write.over(fname,
    f"""
       |module ShiftRegEnableReset_${id} (
       |  input clk,
       |  input en,
       |  input rst,
       |  input [${width - 1}:0] in,
       |  output [${width - 1}:0] out
       | );
       |reg [${width - 1}:0] mem [0:${depth - 1}];
       |
       |integer i;
       |always @(posedge clk) begin
       |  if (rst) begin
       |    for (i = 0; i < ${depth}; i = i + 1) begin
       |      mem[i] <= ${defaultValue};
       |    end
       |  end else if (en) begin
       |    for (i = 0; i < ${depth} - 1; i = i + 1) begin
       |      mem[i + 1] <= mem[i];
       |    end
       |    mem[0] <= in;
       |  end
       |end
       |
       |assign out = mem[${depth - 1}];
       |endmodule
       |""".stripMargin)
  BeethovenBuild.addSource(fname)
}
