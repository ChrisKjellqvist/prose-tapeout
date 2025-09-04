//package prose.deprecated
//
//import beethoven.Generation.BeethovenBuild
//import beethoven.Platforms.ASIC.TechLib
//import chipkit.ChipKitPlatform
//import chipsalliance.rocketchip.config.Parameters
//import freechips.rocketchip.diplomacy.LazyModule
//import mcu.sys.MCULM
//
//class ChipKitWithM0(techlib: TechLib, clockRateMHz: Int) extends ChipKitPlatform({
//  (p: Parameters) =>
//    val mculm = LazyModule(new MCULM()(p))
//    mcu.sources foreach BeethovenBuild.addSource
//    mculm
//}, techlib, clockRateMHz)
