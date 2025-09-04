package prose.toplevel

import chipsalliance.rocketchip.config.Parameters

class MType()(kMax: Int,
              Nmin: Int,
              N: Int,
              SRAMLatency: Int,
              fpuLatency: Int,
              maxBatch: Int,
              maxNColsTiles: Int,
              supportWideBias: Boolean
)(implicit p: Parameters) extends BaseCore(kMax, Nmin, N, SRAMLatency, fpuLatency, maxBatch, supportWideBias, maxNColsTiles, None)

