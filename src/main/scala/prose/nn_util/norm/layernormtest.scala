package prose.nn_util.norm

import beethoven._
import fpwrapper.FPFloatFormat
import prose.nn_util.norm.layernormtest._

object layernormtest {

  case class fpuLatencyBlock(fmaLatency: Int, sqrtLUTLatency: Int, subLatency: Int)

  class WithLayerNorm(nCores: Int,
                      maxSet: Int,
                      maxIntermediateEles: Int,
                      maxRepeats: Int,
                      floatFormat: FPFloatFormat.Type,
                      fpuLats: fpuLatencyBlock) extends AcceleratorConfig({
    AcceleratorSystemConfig(
      nCores = nCores,
      name = "Norm",
      moduleConstructor = ModuleBuilder(p => new LayerNormCore(maxSet, maxIntermediateEles, maxRepeats,
        floatFormat, fpuLats.subLatency, fpuLats.fmaLatency, fpuLats.sqrtLUTLatency)(p)),
      memoryChannelConfig = {
        val dataWidth = floatFormat match {
          case FPFloatFormat.Fp16Alt => 2
          case FPFloatFormat.Fp32 => 4
          case _ => throw new Exception("Unsupported float format")
        }
        List(
          ReadChannelConfig(name = "mean", dataBytes = dataWidth),
          ReadChannelConfig(name = "variance", dataBytes = dataWidth),
          ReadChannelConfig(name = "lnorm", dataBytes = dataWidth),
          ScratchpadConfig(name = "gamma_beta",
            dataWidthBits = 2 * dataWidth * 8,
            nDatas = maxSet,
            nPorts = 1,
            latency = 1),
          WriteChannelConfig(name = "output", dataBytes = dataWidth))
      })
  })
}

object MyLayerNormAccelerator extends BeethovenBuild(new WithLayerNorm(nCores = 1,
  maxSet = 32,
  maxIntermediateEles = 4,
  maxRepeats = 1, floatFormat = FPFloatFormat.Fp16Alt,
  fpuLatencyBlock(3, 1, 2)),
  buildMode = BuildMode.Simulation,
  platform = KriaPlatform())
