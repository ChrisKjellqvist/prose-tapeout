package prose

import beethoven.Generation.CppGeneration
import chipsalliance.rocketchip.config._
import beethoven._
import fpwrapper.{FPFloatFormat, FPUSourceType}
import fpwrapper.FPUSourceType.FPUSourceType
import prose.SpecialFunction.{EXP, SpecialFunction}
import prose.TapeoutConfig._
import prose.nn_util.norm.layernormtest.{WithLayerNorm, fpuLatencyBlock}
import prose.nn_util.residual.MatAddConfig
import prose.toplevel.{EType, GType, MType}
import tsmc._
import beethoven.BuildMode.Synthesis

object SpecialFunction extends Enumeration {
  val EXP, GELU = Value
  type SpecialFunction = Value

  def generate(ty: SpecialFunction.SpecialFunction): Unit = {
    val base_path = os.pwd / "src" / "main" / "c" / "generate_verilog"
    val lut_dir = os.pwd / "luts"
    // delete and remake
    os.remove.all(lut_dir / "Makefile")
    os.makeDir.all(lut_dir)
    val makefile_present = os.exists(lut_dir / "Makefile")
    if (!makefile_present) {
      try {
        os.proc("cmake", base_path).call(cwd = lut_dir)
      } catch {
        case e: os.SubprocessException =>
          os.remove(lut_dir / "CMakeCache.txt")
          os.proc("cmake", base_path).call(cwd = lut_dir)
      }
    }
    val executable = ty match {
      case SpecialFunction.EXP  => "generate_exp_rom"
      case SpecialFunction.GELU => "generate_gelu_rom"
    }
    os.proc("make").call(cwd = lut_dir)
    os.proc(lut_dir / executable, os.pwd / "luts").call(cwd = lut_dir)
  }
}

case class SystolicType(
    nCores: Int,
    namePrefix: String,
    N: Int,
    supportWideBias: Boolean,
    maxMatrixLen: Int,
    specialFunc: Option[(SpecialFunction, Int)] = None
) {
  def toConfig(
      Nmin: Int,
      SRAMLatency: Int,
      fpuLatency: Int,
      maxBatchSize: Int,
      maxNColsTiles: Int
  ): AcceleratorSystemConfig = {
    CppGeneration.addPreprocessorDefinition(s"NCORES_${namePrefix}", nCores)
    AcceleratorSystemConfig(
      nCores = 1,
      name = f"${namePrefix}Core",
      moduleConstructor = ModuleBuilder({ p =>
        specialFunc match {
          case Some((SpecialFunction.EXP, latency)) =>
            new EType(latency)(
              maxMatrixLen,
              Nmin,
              N,
              SRAMLatency,
              fpuLatency,
              fpuLatency + 1,
              maxBatchSize,
              maxNColsTiles,
              supportWideBias,
              nCores
            )(p)
          case Some((SpecialFunction.GELU, latency)) =>
            new GType(latency)(
              maxMatrixLen,
              Nmin,
              N,
              SRAMLatency,
              fpuLatency,
              fpuLatency + 1,
              maxBatchSize,
              maxNColsTiles,
              supportWideBias,
              nCores
            )(p)
          case None =>
            new MType()(
              maxMatrixLen,
              Nmin,
              N,
              SRAMLatency,
              fpuLatency,
              fpuLatency + 1,
              maxBatchSize,
              maxNColsTiles,
              supportWideBias,
              nCores
            )(p)
        }
      }),
      memoryChannelConfig = List(
        ReadChannelConfig(
          name = "weight_stream",
          dataBytes = 2 * Nmin,
          nChannels = N / Nmin
          //          maxInFlightTxs = Some(16)
        ),
        ReadChannelConfig(
          name = "activation_stream",
          dataBytes = 2 * Nmin,
          nChannels = nCores * N / Nmin
          //          maxInFlightTxs = Some(8)
        ),
        ReadChannelConfig(
          name = "norm_stream",
          dataBytes = 2
          //          maxInFlightTxs = Some(8)
        ),
        ReadChannelConfig(
          name = "bias_stream",
          dataBytes = 2 * Nmin,
          nChannels = if (supportWideBias) nCores * N / Nmin else 1
        ),
        WriteChannelConfig(
          name = "activation_out",
          dataBytes = 2 * Nmin,
          nChannels = nCores * N / Nmin,
          maxInFlightTxs = Some(1)
        )
      ) ++ (if (specialFunc.exists(_._1 == SpecialFunction.EXP))
              List(
                WriteChannelConfig(
                  name = "softmax_writeout",
                  dataBytes = 2,
                  maxInFlightTxs = Some(1)
                )
              )
            else Nil)
    )
  }
}

class WithProse(
    cores: List[SystolicType],
    fpuLatency: Int,
    SRAMLatency: Int,
    maxBatchSize: Int,
    maxNDim: Int
) extends AcceleratorConfig({
      val smallestN = cores.map(_.N).min
      // require that E is the smallest core
      cores.find(_.namePrefix == "E") match {
        case Some(e) => require(e.N == smallestN)
        case None    => ;
      }
      cores.map(a =>
        a.toConfig(
          smallestN,
          SRAMLatency,
          fpuLatency,
          maxBatchSize,
          (maxNDim.toFloat / a.N).ceil.toInt
        )
      )
    })

/** Sorry in advance to this mess. It's somewhat complicated to be doing testing
  * on so many platforms If you run any of the benchmarks, try to keep the core
  * count low, otherwise verilator takes like 10 years to compile...
  *
  * Update: VCS is much better.
  */

object ProSEPARAMS {
  val MAX_BATCH = 2

}

object GPTNEO {
  val K_DIM = 768
  val N_HEADS = 12
}

object FPUBuildMode extends Field[FPUSourceType]

class TEST_kria_final_config
    extends AcceleratorConfig(
      new WithProse(
        List(
          SystolicType(
            nCores = 1,
            namePrefix = "E",
            N = 2,
            supportWideBias = true,
            maxMatrixLen = GPTNEO.K_DIM / GPTNEO.N_HEADS,
            Some(SpecialFunction.EXP, 2)
          ),
          SystolicType(
            nCores = 1,
            namePrefix = "M",
            N = 4,
            supportWideBias = true,
            maxMatrixLen = GPTNEO.K_DIM * 4,
            None
          ),
          SystolicType(
            nCores = 1,
            namePrefix = "G",
            N = 2,
            supportWideBias = false,
            maxMatrixLen = GPTNEO.K_DIM,
            Some(SpecialFunction.GELU, 2)
          )
        ),
        fpuLatency = 3,
        SRAMLatency = 2,
        maxBatchSize = 2,
        maxNDim = 4096
      ) ++
        new WithLayerNorm(
          1,
          GPTNEO.K_DIM,
          ProSEPARAMS.MAX_BATCH * 4,
          4096,
          FPFloatFormat.Fp16Alt,
          fpuLats =
            fpuLatencyBlock(fmaLatency = 4, sqrtLUTLatency = 4, subLatency = 4)
        ) ++
        new MatAddConfig(1, 4, 2, 4096 * 4096 * 8)
    )

object TEST_kria_final_config
    extends BeethovenBuild(
      new TEST_kria_final_config,
      buildMode = BuildMode.Synthesis,
      platform = KriaPlatform(hasDebugAXICACHEPROT = true),
      additional_parameter = Some({
        case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
        case BQuiet       => true
      })
    )

object TapeoutConfig {
  val M_cores = 2
  val M_sz = 8

  val E_cores = 8
  val E_sz = 2

  val G_cores = 4
  val G_sz = 4

  val min_sz = Math.min(G_sz, Math.min(E_sz, M_sz))
  val max_batch_sz = 2
  val fma_latency = 2
  val max_output_matrix_dim = 4096
  val lut_latency = 3
}

class TapeoutConfig
    extends AcceleratorConfig(
      new WithProse(
        List(
          SystolicType(
            nCores = M_cores,
            namePrefix = "M",
            N = M_sz,
            supportWideBias = true,
            maxMatrixLen = GPTNEO.K_DIM * 4,
            None
          ),
          SystolicType(
            nCores = E_cores,
            namePrefix = "E",
            N = E_sz,
            supportWideBias = true,
            maxMatrixLen = GPTNEO.K_DIM / GPTNEO.N_HEADS * 2,
            Some(SpecialFunction.EXP, lut_latency)
          ),
          SystolicType(
            nCores = G_cores,
            namePrefix = "G",
            N = G_sz,
            supportWideBias = false,
            maxMatrixLen = GPTNEO.K_DIM,
            specialFunc = Some(SpecialFunction.GELU, lut_latency)
          )
        ),
        fpuLatency = fma_latency,
        SRAMLatency = 2,
        maxBatchSize = max_batch_sz,
        maxNDim = max_output_matrix_dim
      ) ++
        new WithLayerNorm(
          1,
          GPTNEO.K_DIM,
          max_batch_sz * min_sz,
          max_output_matrix_dim,
          FPFloatFormat.Fp16Alt,
          fpuLats = fpuLatencyBlock(
            fmaLatency = fma_latency,
            sqrtLUTLatency = lut_latency,
            subLatency = fma_latency
          )
        ) ++
        new MatAddConfig(
          1,
          fma_latency,
          min_sz,
          max_output_matrix_dim * max_output_matrix_dim * 2
        )
    )

object f2tapeoutconfig
    extends BeethovenBuild(
      new TapeoutConfig(),
      buildMode = BuildMode.Simulation,
      platform =
        new DirectTopTestChipPlatform(1000, 4, new n16_harvard.N16_LIB()),
      additional_parameter = Some({
        case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
        case BQuiet       => true
      })
    )

object tapeout
    extends BeethovenBuild(
      new TapeoutConfig(),
      buildMode = BuildMode.Synthesis,
      platform = new DirectTopTestChipPlatform(
        400,
        4,
        new n16_harvard.N16_LIB(Some(Seq("ssgnp_0p72v_0p72v_m40c")))
      ),
      additional_parameter = Some({
        case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
        case BQuiet       => true
      })
    )

object tapeoutsim
    extends BeethovenBuild(
      new TapeoutConfig(),
      buildMode = BuildMode.Simulation,
      platform = new DirectTopTestChipPlatform(
        400,
        4,
        new n16_harvard.N16_LIB(Some(Seq("ssgnp_0p72v_0p72v_m40c")))
      ),
      additional_parameter = Some({
        case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
        case BQuiet       => true
      })
    )

class NormCoreConfig
    extends AcceleratorConfig(
      new WithLayerNorm(
        1,
        GPTNEO.K_DIM,
        ProSEPARAMS.MAX_BATCH * 4,
        4096,
        FPFloatFormat.Fp16Alt,
        fpuLats =
          fpuLatencyBlock(fmaLatency = 4, sqrtLUTLatency = 4, subLatency = 4)
      )
    )

object NormCoreConfig
    extends BeethovenBuild(
      config = new NormCoreConfig,
      platform = KriaPlatform(),
      buildMode = Synthesis,
      additional_parameter = Some({
        case FPUBuildMode => FPUSourceType.NonSelfContainedSystemVerilog
        case BQuiet       => true
      })
    )
