package prose.nn_util.residual

import org.chipsalliance.cde.config._
import beethoven._

class MatAddConfig(nCores: Int, fpuLatency: Int, N: Int, maxLength: Int) extends AcceleratorConfig({
    List(AcceleratorSystemConfig(
      nCores = nCores,
      name = "MatrixAdd",
      moduleConstructor = ModuleBuilder(p => new MatAdd(N, maxLength, fpuLatency)(p)),
      memoryChannelConfig = List(
        ReadChannelConfig("A", 2 * N),
        ReadChannelConfig("B", 2 * N),
        WriteChannelConfig("C", 2 * N)
      )
    ))
})
