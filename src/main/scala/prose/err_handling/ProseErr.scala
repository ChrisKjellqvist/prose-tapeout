package prose.err_handling

object ProseErr extends chisel3.ChiselEnum {
  val
  NoError,
  // batch size of 0?
  InvalidBatch,
  // activation buffer has not been activated
  InvalidActivation,
  // Specified stream activations but did not provide an activation stream initialization cmd
  ActivationStreamIdle,
  // Specified activation stream initialization twice without proceeding to consume it via another command.
  // Suggest flush to fix.
  OverlappingActivationInit,
  // Unexpected error. Not user error. This is a design error and should never occur.
  DesignError
  = Value
}
