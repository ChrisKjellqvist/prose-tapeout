// package prose.deprecated

// import beethoven.Generation.CppGeneration
// import beethoven._
// import beethoven.MemoryStreams.Memory
// import beethoven.common.Misc.left_assign
// import beethoven.common._
// import chipsalliance.rocketchip.config.{Config, Parameters}
// import chisel3._
// import chisel3.util._
// import fpwrapper.impl.fpnew.FPUNewImplementation
// import fpwrapper.{FPFloatFormat, FPOperation, FPRoundingMode, FPU}
// import prose.deprecated.AccumulationMode.{FlushAccumulates, ReadAndAccumulate}

// object AccumulationMode extends chisel3.ChiselEnum {
//   val ReadAndAccumulate, FlushAccumulates = Value
// }

// class AccumulationCore(stripeWidth: Int,
//                        maxBatch: Int,
//                        maxK: Int,
//                        fpuLatency: Int,
//                        nAccumulators: Int,
//                        exportDefines: Boolean = false)(implicit p: Parameters) extends AcceleratorCore {
//   if (exportDefines) {
//     CppGeneration.addUserCppDefinition(Seq(
//       ("int", "accumulator_stripe_width", stripeWidth),
//       ("int", "max_batch", maxBatch),
//       ("int", "max_k", maxK)
//     ))
//   }
//   require(nAccumulators >= stripeWidth * maxBatch)
//   val io = BeethovenIO(new AccelCommand("ProSEAccumulate") {
//     val addr = Address()
//     val mode = UInt(AccumulationMode().getWidth.W)
//     val batchCount = UInt(log2Up(maxBatch + 1).W)
//     val offset = UInt(log2Up(nAccumulators).W)
//     val len = UInt(log2Up(maxBatch * maxK + 1).W)
//   })

//   CppGeneration.exportChiselEnum(AccumulationMode)

//   val SRAM_latency = 1
//   val accumulators = Memory(latency = SRAM_latency, dataWidth = 32 * stripeWidth, nRows = nAccumulators, 0, 0, nReadWritePorts = 1)
//   val read_out_valid = ShiftReg(accumulators.read_enable(0) && accumulators.chip_select(0), SRAM_latency)
//   val acc_regs = Reg(Vec(stripeWidth, Vec(maxBatch, UInt(32.W))))


//   accumulators.initLow(clock)

//   val (Seq(read_tx), Seq(read_data)) = getReaderModules("reader")
//   val (Seq(write_tx), Seq(write_data)) = getWriterModules("writer")
//   Seq(read_tx, write_tx) foreach { q => q.valid := false.B; q.bits := DontCare }
//   read_data.data.ready := false.B
//   write_data.data.valid := false.B
//   write_data.data.bits := DontCare

//   // s_reset can only be reached by full system reset for the time being.
//   val s_reset :: s_idle :: s_init :: s_iterate :: s_writeback :: s_stream_to_mem :: s_stream_rewind :: s_finish :: Nil = Enum(8)
//   val state = RegInit(s_reset)
//   val mode = Reg(AccumulationMode())
//   val offset = RegInit(0.U(log2Up(nAccumulators).W))
//   val offsetSave = Reg(UInt(log2Up(nAccumulators + 1).W))
//   val len = Reg(UInt(log2Up(maxK + 1).W))
//   val batchCountRead, batchCountWrite, batchCount = Reg(UInt(log2Up(maxBatch).W))
//   val loopCnt, loopMax = Reg(UInt(log2Up(Math.max(fpuLatency + 2, maxBatch)).W))
//   io.req.ready := false.B
//   io.resp.valid := false.B
//   io.resp.bits := DontCare

//   val fpu = Module(new FPU(
//     implementation = FPUNewImplementation(ADDMUL = Some(fpuLatency)),
//     floattype = FPFloatFormat.Fp32,
//     lanes = stripeWidth))
//   fpu.io.req.bits := DontCare
//   fpu.io.req.bits.roundingMode := FPRoundingMode.RNE
//   fpu.io.req.bits.op := FPOperation.ADD
//   fpu.io.req.bits.opModifier := 0.U // addition, modifier=1 is subtraction
//   fpu.io.req.bits.operands(1).zip(acc_regs).foreach { case (fpu_in, ar) =>
//     fpu_in := ar(loopCnt)
//   }
//   val operandsRead = splitIntoChunks(read_data.data.bits, 16)
//   fpu.io.req.bits.operands(2).zip(operandsRead).foreach { case (dst, src) => dst := Cat(src, 0.U(16.W)) }
//   val block = loopCnt > batchCount || state =/= s_iterate
//   fpu.io.req.valid := read_data.data.valid && !block
//   read_data.data.ready := fpu.io.req.ready && !block

//   fpu.io.resp.ready := true.B
//   when(fpu.io.resp.fire) {
//     batchCountWrite := batchCountWrite + 1.U
//     when(batchCountWrite === batchCount) {
//       batchCountWrite := 0.U
//     }
//     acc_regs.map(_(batchCountWrite)).zip(fpu.io.resp.bits.result).foreach { case (dst, src) => dst := src }
//   }

//   when(state === s_reset) {
//     accumulators.addr(0) := offset
//     accumulators.chip_select(0) := true.B
//     accumulators.write_enable(0) := true.B
//     accumulators.read_enable(0) := false.B
//     accumulators.data_in(0) := 0.U
//     offset := offset + 1.U
//     when(offset === (nAccumulators - 1).U) {
//       state := s_idle
//     }
//   }.elsewhen(state === s_idle) {
//     io.req.ready := true.B
//     when(io.req.fire) {
//       read_tx.bits.addr := io.req.bits.addr.toUInt
//       write_tx.bits.addr := io.req.bits.addr.toUInt
//       batchCountRead := 0.U
//       batchCountWrite := 0.U
//       val l = (io.req.bits.len << log2Up(2 * stripeWidth)).asUInt
//       read_tx.bits.len := l
//       write_tx.bits.len := (io.req.bits.batchCount << log2Up(stripeWidth * 4)).asUInt
//       read_tx.valid := AccumulationMode(io.req.bits.mode) === ReadAndAccumulate
//       write_tx.valid := AccumulationMode(io.req.bits.mode) === FlushAccumulates

//       mode := AccumulationMode(io.req.bits.mode)
//       when(AccumulationMode(io.req.bits.mode) === ReadAndAccumulate) {
//         state := s_init
//       }.otherwise {
//         state := s_stream_to_mem
//       }
//       offsetSave := io.req.bits.offset
//       offset := io.req.bits.offset
//       len := io.req.bits.len
//       loopMax := Mux(io.req.bits.batchCount > (fpuLatency + 1).U, io.req.bits.batchCount - 1.U, (fpuLatency + 1).U)
//       batchCount := io.req.bits.batchCount - 1.U
//     }
//   }.elsewhen(state === s_init) {
//     accumulators.chip_select(0) := true.B
//     accumulators.read_enable(0) := true.B
//     accumulators.addr(0) := offset + batchCountRead
//     batchCountRead := batchCountRead + 1.U
//     when(read_out_valid) {
//       val r_out = splitIntoChunks(accumulators.data_out(0), 32)
//       acc_regs.map(_(batchCountWrite)).zip(r_out) foreach left_assign
//       batchCountWrite := batchCountWrite + 1.U
//     }
//     when(batchCountRead === batchCount) {
//       state := s_iterate
//       loopCnt := 0.U
//       batchCountWrite := 0.U
//     }
//   }.elsewhen(state === s_iterate) {
//     when(read_data.data.fire) {
//       loopCnt := loopCnt + 1.U
//       when(loopCnt === loopMax) {
//         loopCnt := 0.U
//       }
//     }
//     when(read_tx.ready) {
//       state := s_writeback
//       batchCountWrite := 0.U

//     }
//   }.elsewhen(state === s_writeback) {
//     accumulators.addr(0) := offset + batchCountWrite
//     batchCountWrite := batchCountWrite + 1.U
//     accumulators.chip_select(0) := true.B
//     accumulators.write_enable(0) := true.B
//     accumulators.data_in(0) := Cat(acc_regs.map(_(batchCountWrite)).reverse)
//     when(batchCountWrite === batchCount) {
//       state := s_finish
//     }
//   }.elsewhen(state === s_finish) {
//     io.resp.bits := DontCare
//     io.resp.valid := read_tx.ready && write_tx.ready
//     when(io.resp.fire) {
//       state := s_idle
//     }
//   }.elsewhen(state === s_stream_to_mem) {
//     write_data.data.valid := read_out_valid
//     write_data.data.bits := accumulators.data_out(0)
//     accumulators.read_enable(0) := true.B
//     accumulators.chip_select(0) := true.B
//     accumulators.addr(0) := offset
//     offset := offset + 1.U

//     when(write_data.data.fire) {
//       offsetSave := offsetSave + 1.U
//       when(offsetSave === (batchCount * stripeWidth.U - 1.U)) {
//         state := s_finish
//       }
//     }
//     when(!write_data.data.ready) {
//       state := s_stream_rewind
//       loopCnt := SRAM_latency.U
//     }
//   }.elsewhen(state === s_stream_rewind) {
//     when(loopCnt =/= 0.U) {
//       loopCnt := loopCnt - 1.U
//     }
//     when(loopCnt === 0.U && write_data.data.ready) {
//       state := s_stream_to_mem
//       offset := offsetSave
//     }
//   }
// }

// class WithAccumulators(nCores: Int,
//                        stripeWidth: Int,
//                        maxBatch: Int,
//                        maxK: Int,
//                        fpuLatency: Int,
//                        nAccumulators: Int) extends AcceleratorConfig(
//   List(AcceleratorSystemConfig(
//     nCores = nCores,
//     name = "accumulator",
//     moduleConstructor = ModuleBuilder(p => new AccumulationCore(
//       stripeWidth, maxBatch, maxK, fpuLatency, nAccumulators, true)(p)
//     ),
//     memoryChannelConfig = List(
//       ReadChannelConfig("reader", dataBytes = 2 * stripeWidth),
//       WriteChannelConfig("writer", dataBytes = 4 * stripeWidth)
//     ))))

// object AccumulatorTest extends BeethovenBuild(new WithAccumulators(1, 4, 8, 128, 1, 128),
//   platform = KriaPlatform(),
//   buildMode = BuildMode.Synthesis)