package prose.random

import beethoven._
import chisel3._

class LinearFeedbackShiftRegister(l: Int, taps: List[Int]) extends BlackBox {
  override val desiredName = f"LFSR${l}_" + taps.mkString("_")

  val io = IO(new Bundle {
    val clock = Input(Bool())

    val set_valid = Input(Bool())
    val set_data = Input(UInt(l.W))

    val increment = Input(Bool())
    val out = Output(Bool())
  })

  val tap_map = "1'b1 ^ " + taps.map(a => "state[" + a + "]").mkString(" ^ ")

  val f =
    s"""
      |module ${desiredName} (
      |  input clock,
      |  input set_valid,
      |  input [${l-1}:0] set_data,
      |  input increment,
      |  output out);
      |
      |reg [${l-1}:0] state;
      |assign out = state[0];
      |wire next_bit = $tap_map;
      |
      |always @(posedge clock) begin
      |  if (set_valid) begin
      |    state <= set_data;
      |  end else if (increment) begin
      |    state <= {next_bit, state[${l-1}: 1]};
      |  end
      |end
      |endmodule
      |""".stripMargin


  os.makeDir.all(os.pwd / "lfsrs")
  val fname = os.pwd / "lfsrs" / (desiredName + ".v")

  if (!os.exists(fname)) {
    os.write(fname, f)
  }
  BeethovenBuild.addSource(fname)
}
