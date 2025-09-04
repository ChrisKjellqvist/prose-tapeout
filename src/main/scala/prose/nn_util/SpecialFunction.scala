package prose.nn_util

object SpecialFunction extends Enumeration {
  val EXP, GELU = Value
  type SpecialFunction = Value
  def generate(ty: SpecialFunction.SpecialFunction): Unit = {
    val base_path = os.pwd / "src" / "main" / "c" / "generate_verilog"
    val lut_dir = os.pwd / "luts"
    // delete and remake
    os.remove.all(lut_dir / "Makefile")
    os.makeDir.all(lut_dir)
    val makefile_present = os.exists(lut_dir/"Makefile")
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
      case SpecialFunction.EXP => "generate_exp_exact"
      case SpecialFunction.GELU => "generate_gelu_exact"
    }
    if (!os.exists(lut_dir/executable)) {
      os.proc("make").call(cwd = lut_dir)
    }
    os.proc(lut_dir / executable).call(cwd = lut_dir)
  }
}