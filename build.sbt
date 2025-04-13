
// See README.md for license details.

ThisBuild / scalaVersion := "2.13.10"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "edu.duke.cs.apex"

val chiselVersion = "3.5.6"

/** If you are getting stack overflows when trying to run this project, increase the stack size of your compile
 * server
 * https://stackoverflow.com/questions/56066899/how-to-fix-the-error-org-jetbrains-jps-incremental-scala-remote-serverexceptio
 */
lazy val prose = {
  (project in file("."))
    .settings(
      name := "prose",
      libraryDependencies ++= Seq(
        "edu.berkeley.cs" %% "chisel3" % chiselVersion,
        "edu.duke.cs.apex" %% "fpnew-wrapper" % "0.2.32",
        "edu.duke.cs.apex" %% "beethoven-hardware" % "0.0.21",
        "edu.duke.cs.apex" %% "asic_lib" % "0.0.0",
      ),
      resolvers += ("reposilite-repository-releases" at "http://54.165.244.214:8080/releases").withAllowInsecureProtocol(true),
      addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % chiselVersion cross CrossVersion.full),
    )
}
