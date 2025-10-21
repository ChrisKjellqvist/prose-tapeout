// See README.md for license details.

// ThisBuild / scalaVersion := "2.13.10"
ThisBuild / scalaVersion := "2.13.16"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "edu.duke.cs.apex"

// val chiselVersion = "3.5.6"
val chiselVersion = "7.1.0"

/** If you are getting stack overflows when trying to run this project, increase
  * the stack size of your compile server
  * https://stackoverflow.com/questions/56066899/how-to-fix-the-error-org-jetbrains-jps-incremental-scala-remote-serverexceptio
  */
lazy val prose = {
  (project in file("."))
    .settings(
      name := "prose",
      libraryDependencies ++= Seq(
        "edu.duke.cs.apex" %% "beethoven-hardware" % "0.1.2",
        "edu.duke.cs.apex" %% "fpnew-wrapper" % "0.3.0",
        "edu.duke.cs.apex" %% "diplomacy" % "0.0.2",
        "edu.duke.cs.apex" %% "asic_lib" % "0.0.0"
      ),
      resolvers += ("reposilite-repository-releases" at "http://54.165.244.214:8080/releases")
        .withAllowInsecureProtocol(true),
      addCompilerPlugin(
        "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
      )
    )
}
