package prose

import prose.nn_util.SystolicType
import beethoven._

class WithProse(cores: List[SystolicType],
                fpuLatency: Int,
                SRAMLatency: Int,
                maxBatchSize: Int,
                maxMatrixLen: Int,
                maxNDim: Int) extends AcceleratorConfig({
  val smallestN = cores.map(_.N).min
  // require that E is the smallest core
  cores.find(_.namePrefix == "E") match {
    case Some(e) => require(e.N == smallestN)
    case None => ;
  }
  cores.map(a => a.toConfig(maxMatrixLen, smallestN, SRAMLatency, fpuLatency, maxBatchSize, (maxNDim.toFloat / a.N).ceil.toInt))
})

