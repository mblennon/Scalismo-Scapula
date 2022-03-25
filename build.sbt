name := "ScalismoScapula"

version := "0.1"

scalaVersion := "2.13.8"

resolvers ++= Seq(
  Resolver.bintrayRepo("unibas-gravis", "maven"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies  ++= Seq(
  "ch.unibas.cs.gravis" % "scalismo-native-all" % "4.0.0",
  "ch.unibas.cs.gravis" %% "scalismo-ui" % "develop-644894a3f93b9c61203c6d6c85b5d4e408a33317-SNAPSHOT",
  "io.github.cibotech" %% "evilplot" % "0.8.1"
)

dependencyOverrides += ("ch.unibas.cs.gravis" %% "scalismo" % "develop-245655f2f7d7abb25c6624bf84d945b6bdd7bcd6-SNAPSHOT")
