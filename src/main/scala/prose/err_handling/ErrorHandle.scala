package prose.err_handling

import beethoven.Platforms.BuildModeKey
import beethoven._
import chipsalliance.rocketchip.config.Parameters
import chisel3._

object ErrorHandle {
  def apply(conditionForError: Bool,
            assertMessage: String,
            errCode: ProseErr.Type)(implicit p: Parameters, errReg: ProseErr.Type, returnFlag: Bool): Unit = {
    p(BuildModeKey) match {
      case BuildMode.Synthesis =>
        when(conditionForError) {
          returnFlag := true.B
          errReg := errCode
        }
      case _ =>
        assert(!conditionForError, assertMessage)
    }
  }
}