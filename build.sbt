
// See README.md for license details.

ThisBuild / scalaVersion := "2.13.16"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "edu.duke.cs.apex"

val chiselVersion = "7.1.0"

lazy val prose = {
  (project in file("."))
    .settings(
      name := "prose",
      libraryDependencies ++= Seq(
        "org.chipsalliance" %% "chisel" % chiselVersion,
        "edu.duke.cs.apex" %% "fpnew-wrapper" % "0.3.0",
        "edu.duke.cs.apex" %% "beethoven-hardware" % "0.1.3-dev5",
        "edu.duke.cs.apex" %% "asic_lib" % "0.0.0",
      ),
      resolvers += ("reposilite-repository-releases" at "http://54.165.244.214:8080/releases").withAllowInsecureProtocol(true),
      addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)
    )
}