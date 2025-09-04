package prose

import chipsalliance.rocketchip.config.{Config, Field}
import beethoven._
import fpwrapper.{FPFloatFormat, FPUSourceType}
import fpwrapper.FPUSourceType.FPUSourceType
import prose.nn_util.norm.layernormtest.{WithLayerNorm, fpuLatencyBlock}
import prose.nn_util.residual.MatAddConfig
import prose.nn_util.{SpecialFunction, SystolicType}
import tsmc._
import tsmc.n16_harvard.N16_LIB


/**
 * Sorry in advance to this mess. It's somewhat complicated to be doing testing on so many platforms
 * If you run any of the benchmarks, try to keep the core count low, otherwise verilator takes like
 * 10 years to compile...
 *
 * Update: VCS is much better.
 */

object FPUBuildMode extends Field[FPUSourceType]

class TEST_kria_final_config extends AcceleratorConfig(
  new WithProse(List(
    SystolicType(
      nCores = 2,
      namePrefix = "E",
      N = 2,
      supportWideBias = true,
      Some(SpecialFunction.EXP, 2)),
    SystolicType(
      nCores = 1,
      namePrefix = "M",
      N = 4,
      supportWideBias = true,
      None),
    SystolicType(
      nCores = 1,
      namePrefix = "G",
      N = 2,
      supportWideBias = false,
      Some(SpecialFunction.GELU, 2))
  ),
    fpuLatency = 4,
    SRAMLatency = 2,
    maxBatchSize = 2,
    maxMatrixLen = 768, /* 768 in production*/
    maxNDim = 4096) ++
    new WithLayerNorm(1, 768, 2 * 4, 4096, FPFloatFormat.Fp16Alt, fpuLats = fpuLatencyBlock(fmaLatency = 4, sqrtLUTLatency = 4, subLatency = 4)) ++
    new MatAddConfig(1, 4, 2, 4096 * 4096 * 8))

class TestProSE_Sweep_Config extends WithProse(List(
  SystolicType(
    nCores = 1,
    namePrefix = "M8",
    N = 8,
    supportWideBias = true,
    None),
  SystolicType(
    nCores = 1,
    namePrefix = "M16",
    N = 16,
    supportWideBias = true,
    None),
  SystolicType(
    nCores = 1,
    namePrefix = "M32",
    N = 32,
    supportWideBias = true,
    specialFunc = None
  )
),
  fpuLatency = 4,
  SRAMLatency = 2,
  maxBatchSize = 2,
  maxMatrixLen = 768 /* 768 in production*/,
  maxNDim = 4096)


object TEST_kria_final_config extends BeethovenBuild(
  new TEST_kria_final_config,
  buildMode = BuildMode.Synthesis,
  platform = KriaPlatform(), additional_parameter = Some({
    case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
    case BQuiet => true
  }))

object tapeout extends BeethovenBuild(
  new TestProSE_Sweep_Config,
  buildMode = BuildMode.Synthesis,
  platform = new DirectTopTestChipPlatform(450, 4, new n16_harvard.N16_LIB(Some(Seq("tt_1p00v_1p00v_25c")))),
  additional_parameter = Some({
    case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
    case BQuiet => true
  })
)
