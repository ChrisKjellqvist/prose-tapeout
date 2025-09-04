package prose.nn_util

import beethoven.Generation.CppGeneration
import beethoven._
import prose.nn_util.SpecialFunction.SpecialFunction
import prose.toplevel.{EType, GType, MType}

case class SystolicType(nCores: Int,
                        namePrefix: String,
                        N: Int,
                        supportWideBias: Boolean,
                        specialFunc: Option[(SpecialFunction, Int)] = None) {
  def toConfig(maxMatrixLen: Int,
               Nmin: Int,
               SRAMLatency: Int,
               fpuLatency: Int,
               maxBatchSize: Int,
               maxNColsTiles: Int): AcceleratorSystemConfig = {
    CppGeneration.addPreprocessorDefinition(s"NCORES_${namePrefix}", nCores)
    AcceleratorSystemConfig(nCores = nCores,
      name = f"${namePrefix}Core",
      moduleConstructor = ModuleBuilder({p =>
        specialFunc match {
          case Some((SpecialFunction.EXP, latency)) => new EType(latency)(maxMatrixLen, Nmin, N, SRAMLatency, fpuLatency, maxBatchSize, maxNColsTiles, supportWideBias)(p)
          case Some((SpecialFunction.GELU, latency)) => new GType(latency)(maxMatrixLen, Nmin, N, SRAMLatency, fpuLatency, maxBatchSize, maxNColsTiles, supportWideBias)(p)
          case None => new MType()(maxMatrixLen, Nmin, N, SRAMLatency, fpuLatency, maxBatchSize, maxNColsTiles, supportWideBias)(p)
        }
      }),
      memoryChannelConfig = List(
        ReadChannelConfig(
          name = "weight_stream",
          dataBytes = 2 * Nmin,
          nChannels = N / Nmin,
//          maxInFlightTxs = Some(16)
        ),
        ReadChannelConfig(
          name = "activation_stream",
          dataBytes = 2 * Nmin,
          nChannels = N / Nmin,
//          maxInFlightTxs = Some(8)
        ),
        ReadChannelConfig(
          name = "norm_stream",
          dataBytes = 2,
//          maxInFlightTxs = Some(8)
        ),
        ReadChannelConfig(
          name = "bias_stream",
          dataBytes = 2 * Nmin,
          nChannels = if (supportWideBias) N / Nmin else 1
        ),
        WriteChannelConfig(
          name = "activation_out",
          dataBytes = 2 * Nmin,
          nChannels = N / Nmin,
          maxInFlightTxs = Some(1)
        )) ++ (if (specialFunc.exists(_._1 == SpecialFunction.EXP)) List(
        WriteChannelConfig(
          name = "softmax_writeout",
          dataBytes = 2,
          maxInFlightTxs = Some(1)
        )) else Nil)

    )
  }
}
