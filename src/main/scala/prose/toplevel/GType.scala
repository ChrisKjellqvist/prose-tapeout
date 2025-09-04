package prose.toplevel

import chipsalliance.rocketchip.config.Parameters
import prose.nn_util.SpecialFunction

class GType(gtype_latency: Int)(
  kMax: Int,
  Nmin: Int,
  N: Int,
  SRAMLatency: Int,
  fpuLatency: Int,
  maxBatch: Int,
  maxNColsTiles: Int,
  supportWideBias: Boolean
)(implicit p: Parameters) extends BaseCore(kMax, Nmin, N, SRAMLatency, fpuLatency, maxBatch, supportWideBias, maxNColsTiles, Some((SpecialFunction.GELU, gtype_latency))) {
  SpecialFunction.generate(SpecialFunction.GELU)
}
