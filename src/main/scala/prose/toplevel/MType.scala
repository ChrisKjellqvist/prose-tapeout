package prose.toplevel

import chipsalliance.rocketchip.config.Parameters

class MType()(kMax: Int,
              Nmin: Int,
              N: Int,
              SRAMLatency: Int,
              fpuLatency: Int,
              simdLatency: Int,
              maxBatch: Int,
              maxNColsTiles: Int,
              supportWideBias: Boolean,
              n_arrays: Int
)(implicit p: Parameters) extends BaseCore(kMax, Nmin, N, SRAMLatency, fpuLatency, simdLatency, maxBatch, supportWideBias, maxNColsTiles, n_arrays, None)

