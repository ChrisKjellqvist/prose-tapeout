package prose.activations

import chisel3._
import beethoven._
import beethoven.common.{ShiftReg, ShiftRegEnable}
import chipsalliance.rocketchip.config._

/**
 * Look in resource folder for Lookup Table with provided name and wrap it in a BlackBox
 *
 * @param name name of the lookup table to find in the resouce folder
 */
class ResourceLookupTable(name: String) extends BlackBox with HasBlackBoxActivationIO {
  override val desiredName = name
  BeethovenBuild.addSource(os.pwd / "luts" / s"${name}.v")
}

class LookupTableWithLatency(name: String, latency: Int)(implicit p: Parameters) extends Module with HasActivationIO {
  override val desiredName = name + "_wl_module"
  val lut = Module(new ResourceLookupTable(name))
  lut.io.x := io.in
  lut.io.clock := clock
  lut.io.reset := reset
  lut.io.enable := true.B
  require(latency >= 1, "ResourceLookupTables have a minimum latency of 1, cannot do anything less")
  io.out := ShiftReg(lut.io.y, latency - 1, clock)
}

trait HasEnableIO {
  val enable = IO(Input(Bool()))
}

class LookupTableWithLatencyWithEnable(name: String, latency: Int)(implicit p: Parameters) extends LookupTableWithLatency(name, latency) with HasActivationIO with HasEnableIO {
  override val desiredName = name + "_wle_module"
  // override the set to io.out in parent
  io.out := ShiftRegEnable(lut.io.y, latency - 1, enable, clock)
  lut.io.enable := enable
}
