object ModelfittingwithMCMCusingScapulaSSM extends App {

  import breeze.linalg.{DenseMatrix, DenseVector}
  import scalismo.common.{PointId, ScalarMeshField, UnstructuredPointsDomain}
  import scalismo.geometry._
  import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
  import scalismo.mesh.{MeshMetrics, TriangleMesh}
  import scalismo.sampling.algorithms.MetropolisHastings
  import scalismo.sampling.evaluators.ProductEvaluator
  import scalismo.sampling.loggers.AcceptRejectLogger
  import scalismo.sampling.proposals.MixtureProposal
  import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
  import scalismo.statisticalmodel.{MultivariateNormalDistribution, PointDistributionModel}
  import scalismo.transformations.{Rotation3D, Translation3D, TranslationAfterRotation, TranslationAfterRotation3D}
  import scalismo.ui.api._
  import scalismo.ui.event.ScalismoPublisher
  import scalismo.ui.view.util.FancySlider
  import scalismo.ui.control._
  import scalismo.utils.Memoize
  import scalismo.registration.LandmarkRegistration
  import scalismo.transformations.RigidTransformation
  import scalismo.common.interpolation.TriangleMeshInterpolator3D
  import scala.concurrent.Future
  import java.awt.Color
  import scala.swing.event.ButtonClicked
  import scala.swing.{Button, FlowPanel}



  implicit val rng = scalismo.utils.Random(42)
  scalismo.initialize()


    case class Parameters(translationParameters: EuclideanVector[_3D],
                          rotationParameters: (Double, Double, Double),
                          modelCoefficients: DenseVector[Double],
                          noiseStddev : Double
                         )


    case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point[_3D]) {
      def poseTransformation: TranslationAfterRotation[_3D] = {
        val translation = Translation3D(parameters.translationParameters)
        val rotation = Rotation3D(
          parameters.rotationParameters._1,
          parameters.rotationParameters._2,
          parameters.rotationParameters._3,
          rotationCenter
        )
        TranslationAfterRotation3D(translation, rotation)
      }
    }

    //Evaluators: Modelling the target density


    case class PriorEvaluator(model: PointDistributionModel[_3D, TriangleMesh]) extends DistributionEvaluator[Sample] {

      val translationPrior = breeze.stats.distributions.Gaussian(0.0, 10.0)
      val rotationPrior = breeze.stats.distributions.Gaussian(0, 1.0)
      val noisePrior = breeze.stats.distributions.LogNormal(0, 1.0)

      override def logValue(sample: Sample): Double = {
        model.gp.logpdf(sample.parameters.modelCoefficients) +
          translationPrior.logPdf(sample.parameters.translationParameters.x) +
          translationPrior.logPdf(sample.parameters.translationParameters.y) +
          translationPrior.logPdf(sample.parameters.translationParameters.z) +
          rotationPrior.logPdf(sample.parameters.rotationParameters._1) +
          rotationPrior.logPdf(sample.parameters.rotationParameters._2) +
          rotationPrior.logPdf(sample.parameters.rotationParameters._3) +
          noisePrior.logPdf(sample.parameters.noiseStddev)
      }
    }

    case class SimpleCorrespondenceEvaluator(model: PointDistributionModel[_3D, TriangleMesh], correspondences: Seq[(PointId, Point[_3D])]) extends DistributionEvaluator[Sample] { override def logValue(sample: Sample): Double = {

        val currModelInstance = model.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)
        val lmUncertainty = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * sample.parameters.noiseStddev)
        val likelihoods = correspondences.map(correspondence => {
          val (id, targetPoint) = correspondence
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint
          lmUncertainty.logpdf(observedDeformation.toBreezeVector)
        })
        val loglikelihood = likelihoods.sum
        loglikelihood
      }
    }


    def marginalizeModelForCorrespondences(model: PointDistributionModel[_3D, TriangleMesh], correspondences: Seq[(PointId, Point[_3D])]): (PointDistributionModel[_3D, UnstructuredPointsDomain], Seq[(PointId, Point[_3D])]) = {
      val (modelIds, _) = correspondences.unzip
      val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
      val newCorrespondences = correspondences.map(idWithTargetPoint => {
        val (id, targetPoint) = idWithTargetPoint
        val modelPoint = model.reference.pointSet.point(id)
        val newId = marginalizedModel.reference.pointSet.findClosestPoint(modelPoint).id
        (newId, targetPoint)
      })
      (marginalizedModel, newCorrespondences)
    }

    case class CorrespondenceEvaluator(model: PointDistributionModel[_3D, TriangleMesh], correspondences: Seq[(PointId, Point[_3D])]) extends DistributionEvaluator[Sample] {
      val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)
      override def logValue(sample: Sample): Double = {
        val lmUncertainty = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * sample.parameters.noiseStddev)
        val currModelInstance = marginalizedModel
          .instance(sample.parameters.modelCoefficients)
          .transform(sample.poseTransformation)
        val likelihoods = newCorrespondences.map(correspondence => {
          val (id, targetPoint) = correspondence
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint
          lmUncertainty.logpdf(observedDeformation.toBreezeVector)
        })
        val loglikelihood = likelihoods.sum
        loglikelihood
      }
    }

    case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends DistributionEvaluator[A] {
      val memoizedLogValue = Memoize(evaluator.logValue, 10)
      override def logValue(sample: A): Double = {
        memoizedLogValue(sample)
      }
    }


    // The proposal generator

//    case class ShapeUpdateProposal(paramVectorSize: Int, stddev: Double) extends ProposalGenerator[Sample] with TransitionProbability[Sample] {
//      val perturbationDistr = new MultivariateNormalDistribution(
//        DenseVector.zeros(paramVectorSize),
//        DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev
//      )
//      override def propose(sample: Sample): Sample = {
//        val perturbation = perturbationDistr.sample()
//        val newParameters =
//          sample.parameters.copy(modelCoefficients = sample.parameters.modelCoefficients + perturbationDistr.sample)
//        sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
//      }
//      override def logTransitionProbability(from: Sample, to: Sample) = {
//        val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
//        perturbationDistr.logpdf(residual)
//      }
//    }

  case class ShapeUpdateProposal(model: PointDistributionModel[_3D, TriangleMesh], stddev: Double, numCoefficientsToChange : Int)(implicit rng: scalismo.utils.Random)
    extends ProposalGenerator[Sample]
      with TransitionProbability[Sample] {

    private val effectiveNumCoefficientsToChange = Math.min(model.rank, numCoefficientsToChange)

    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros(effectiveNumCoefficientsToChange),
      DenseMatrix.eye[Double](effectiveNumCoefficientsToChange) * stddev * stddev
    )

    override def propose(sample: Sample): Sample = {
      val shapeCoefficients = sample.parameters.modelCoefficients
      val newCoefficients = shapeCoefficients.copy
      newCoefficients(0 until effectiveNumCoefficientsToChange) := shapeCoefficients(0 until effectiveNumCoefficientsToChange) + perturbationDistr.sample
      val newParameters = sample.parameters.copy(modelCoefficients = newCoefficients)
      sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.modelCoefficients(0 until effectiveNumCoefficientsToChange) - from.parameters.modelCoefficients(0 until effectiveNumCoefficientsToChange)
      perturbationDistr.logpdf(residual)
    }
  }

    case class RotationUpdateProposal(stddev: Double) extends ProposalGenerator[Sample] with TransitionProbability[Sample] {
      val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * stddev * stddev)
      def propose(sample: Sample): Sample = {
        val perturbation = perturbationDistr.sample
        val newRotationParameters = (
          sample.parameters.rotationParameters._1 + perturbation(0),
          sample.parameters.rotationParameters._2 + perturbation(1),
          sample.parameters.rotationParameters._3 + perturbation(2)
        )
        val newParameters = sample.parameters.copy(rotationParameters = newRotationParameters)
        sample.copy(generatedBy = s"RotationUpdateProposal ($stddev)", parameters = newParameters)
      }
      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = DenseVector(
          to.parameters.rotationParameters._1 - from.parameters.rotationParameters._1,
          to.parameters.rotationParameters._2 - from.parameters.rotationParameters._2,
          to.parameters.rotationParameters._3 - from.parameters.rotationParameters._3
        )
        perturbationDistr.logpdf(residual)
      }
    }

    case class TranslationUpdateProposal(stddev: Double) extends ProposalGenerator[Sample] with TransitionProbability[Sample] {
      val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3), DenseMatrix.eye[Double](3) * stddev * stddev)
      def propose(sample: Sample): Sample = {
        val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector.fromBreezeVector(
          perturbationDistr.sample()
        )
        val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
        sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
      }
      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = to.parameters.translationParameters - from.parameters.translationParameters
        perturbationDistr.logpdf(residual.toBreezeVector)
      }
    }

    case class NoiseStddevUpdateProposal(stddev: Double)(implicit rng : scalismo.utils.Random) extends ProposalGenerator[Sample] with TransitionProbability[Sample] {
      val perturbationDistr = breeze.stats.distributions.Gaussian(0, stddev)(rng.breezeRandBasis)
      def propose(sample: Sample): Sample = {
        val newSigma = sample.parameters.noiseStddev +  perturbationDistr.sample()
        val newParameters = sample.parameters.copy(noiseStddev = newSigma)
        sample.copy(generatedBy = s"NoiseStddevUpdateProposal ($stddev)", parameters = newParameters)
      }
      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = to.parameters.noiseStddev - from.parameters.noiseStddev
        perturbationDistr.logPdf(residual)
      }
    }


    // Building the markov chain

    class Logger extends AcceptRejectLogger[Sample] {
      private val numAccepted = collection.mutable.Map[String, Int]()
      private val numRejected = collection.mutable.Map[String, Int]()
      override def accept(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]): Unit = {
        val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
        numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
      }
      override def reject(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]): Unit = {
        val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
        numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
      }
      def acceptanceRatios(): Map[String, Double] = {
        val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
        val acceptanceRatios = for (generatorName <- generatorNames) yield {
          val total = (numAccepted.getOrElse(generatorName, 0)
            + numRejected.getOrElse(generatorName, 0)).toDouble
          (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
        }
        acceptanceRatios.toMap
      }
    }

    def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
      val normFactor = 1.0 / mesh.pointSet.numberOfPoints
      mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
    }


  def Compare(mesh1: TriangleMesh[_3D], mesh2: TriangleMesh[_3D]): String = {
    val avgdist:Double = MeshMetrics.avgDistance(mesh1, mesh2)
    val hausdist:Double = MeshMetrics.hausdorffDistance(mesh1, mesh2)
    val dice:Double = MeshMetrics.diceCoefficient(mesh1,mesh2)
    // for every point of m1, find its closest distance to m2
    val distanceValues = mesh1.pointSet.points.map{ p => mesh2.operations.shortestDistanceToSurfaceSquared(p) }
    val dva = distanceValues.toArray
    // val distanceValues = reference.pointSet.points.map{ p => meshOps.shortestDistanceToSurfaceSquared(p) }
    // build a Scalar mesh field with the distance associated to each vertex
    val scalarMesh = ScalarMeshField(mesh1, dva)
    // Write to file.
    MeshIO.writeScalarMeshField[Double](scalarMesh, new java.io.File("meshWithDistances.vtk"))
    val compareGroup = ui.createGroup("compare")
    ui.show(compareGroup, scalarMesh, "distances")
    val av = f"$avgdist%1.2f"; val hd = f"$hausdist%1.2f"; val dc = f"$dice%1.2f"
    return (f"avg distance = " + av + " & hausdorff distance = " + hd + " & Dice coeff = " + dc)
  }


  def findtransform(): RigidTransformation[_3D] = {
    val modellms = ui.filter(modelGroup, (v : LandmarkView) => true)
    val modellms_pts = modellms.map(l => l.landmark.point)
    val targetlms = ui.filter(targetGroup, (v : LandmarkView) => true)
    val targetlms_pts = targetlms.map(l => l.landmark.point)
    val ptset = modellms_pts zip targetlms_pts
    val bestTransform : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(ptset, center = Point(0, 0, 0))
    return bestTransform
  }

  def MCMC(correspondences:Seq[(PointId, Point[_3D])], model: scalismo.statisticalmodel.PointDistributionModel[scalismo.geometry._3D,scalismo.mesh.TriangleMesh] ,
           mView: scalismo.ui.api.ShowInScene.ShowInScenePointDistributionModelTriangleMesh3D.View , burnin:Int, num_iterations:Int,initialParameters: Parameters) = {



    // The posterior evaluator
    val likelihoodEvaluator = CachedEvaluator(CorrespondenceEvaluator(model, correspondences))
    val priorEvaluator = CachedEvaluator(PriorEvaluator(model))
    val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)
    val shapeUpdateProposal = ShapeUpdateProposal(model, 0.1, numCoefficientsToChange = 5)
    val rotationUpdateProposal = RotationUpdateProposal(0.005)
    val translationUpdateProposal = TranslationUpdateProposal(5.0)
    val noiseStddevUpdateProposal = NoiseStddevUpdateProposal(0.1)
    val generator = MixtureProposal.fromProposalsWithTransition(
      (0.1, shapeUpdateProposal),
      (0.1, rotationUpdateProposal),
      (0.7, translationUpdateProposal),
      (0.1, noiseStddevUpdateProposal)
    )



    val initialSample = Sample("initial", initialParameters, computeCenterOfMass(model.mean))
    val chain = MetropolisHastings(generator, posteriorEvaluator)
    val logger = new Logger()
    val mhIterator = chain.iterator(initialSample, logger)

    val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
        if (iteration % 500 == 0) {
        mView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
        mView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
      }
      sample
    }
    val samples = samplingIterator.drop(burnin).take(num_iterations).toIndexedSeq

    println(logger.acceptanceRatios())

    val bestSample = samples.maxBy(posteriorEvaluator.logValue)
    val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    val bestfitmodel = ui.show(resultGroup, bestFit, "best fit")
    bestfitmodel.color = Color.GREEN
    val currModelInstance2 = model1.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
//    ui.setVisibility(mView.referenceView, Viewport.none)




    def computeMean(model: PointDistributionModel[_3D, UnstructuredPointsDomain], id: PointId): Point[_3D] = {
      var mean = EuclideanVector(0, 0, 0)
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        mean += pointForInstance.toVector
      }
      (mean * 1.0 / samples.size).toPoint
    }

    def computeCovarianceFromSamples(model: PointDistributionModel[_3D, UnstructuredPointsDomain],
                                     id: PointId,
                                     mean: Point[_3D]): SquareMatrix[_3D] = {
      var cov = SquareMatrix.zeros[_3D]
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        val v = pointForInstance - mean
        cov += v.outer(v)
      }
      cov * (1.0 / samples.size)
    }

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    for ((id, _) <- newCorrespondences) {
      val meanPointPosition = computeMean(marginalizedModel, id)
      println(s"expected position for point at id $id  = $meanPointPosition")
      val cov = computeCovarianceFromSamples(marginalizedModel, id, meanPointPosition)
      println(
        s"posterior variance computed  for point at id (shape and pose) $id  = ${cov(0, 0)}, ${cov(1, 1)}, ${cov(2, 2)}"
      )
    }

    bestSample: Sample
  }

  def MCMC2(correspondences:Seq[(PointId, Point[_3D])], model: scalismo.statisticalmodel.PointDistributionModel[scalismo.geometry._3D,scalismo.mesh.TriangleMesh] ,
           mView: scalismo.ui.api.ShowInScene.ShowInScenePointDistributionModelTriangleMesh3D.View , burnin:Int, num_iterations:Int,initialParameters: Parameters, com: Point[_3D]) = {


    // The posterior evaluator
    val likelihoodEvaluator = CachedEvaluator(CorrespondenceEvaluator(model, correspondences))
    val priorEvaluator = CachedEvaluator(PriorEvaluator(model))
    val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)
    val shapeUpdateProposal = ShapeUpdateProposal(model, 0.05, numCoefficientsToChange = 100)
    val rotationUpdateProposal = RotationUpdateProposal(0.001)
    val translationUpdateProposal = TranslationUpdateProposal(0.5)
    val noiseStddevUpdateProposal = NoiseStddevUpdateProposal(0.01)
    val generator = MixtureProposal.fromProposalsWithTransition(
      (0.7, shapeUpdateProposal),
      (0.1, rotationUpdateProposal),
      (0.1, translationUpdateProposal),
      (0.01, noiseStddevUpdateProposal)
    )

    val initialSample = Sample("initial", initialParameters, com)

    val chain = MetropolisHastings(generator, posteriorEvaluator)
    val logger = new Logger()
    val mhIterator = chain.iterator(initialSample, logger)

    val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 500 == 0) {
        mView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
        mView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
      }
      sample
    }
    val samples = samplingIterator.drop(burnin).take(num_iterations).toIndexedSeq

    println(logger.acceptanceRatios())

    val bestSample = samples.maxBy(posteriorEvaluator.logValue)
    val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    val bestfitmodel = ui.show(resultGroup, bestFit, "best fit")
    bestfitmodel.color = Color.GREEN
//    ui.setVisibility(mView.referenceView, Viewport.none)



    def computeMean(model: PointDistributionModel[_3D, UnstructuredPointsDomain], id: PointId): Point[_3D] = {
      var mean = EuclideanVector(0, 0, 0)
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        mean += pointForInstance.toVector
      }
      (mean * 1.0 / samples.size).toPoint
    }

    def computeCovarianceFromSamples(model: PointDistributionModel[_3D, UnstructuredPointsDomain],
                                     id: PointId,
                                     mean: Point[_3D]): SquareMatrix[_3D] = {
      var cov = SquareMatrix.zeros[_3D]
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        val v = pointForInstance - mean
        cov += v.outer(v)
      }
      cov * (1.0 / samples.size)
    }

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    for ((id, _) <- newCorrespondences) {
      val meanPointPosition = computeMean(marginalizedModel, id)
      println(s"expected position for point at id $id  = $meanPointPosition")
      val cov = computeCovarianceFromSamples(marginalizedModel, id, meanPointPosition)
      println(
        s"posterior variance computed  for point at id (shape and pose) $id  = ${cov(0, 0)}, ${cov(1, 1)}, ${cov(2, 2)}"
      )
    }

    bestSample: Sample

  }

//  val initialParameters2 = Parameters(
//    translationParameters = EuclideanVector(0, 0, 0),
//    rotationParameters = (0.0, 0.0, 0.0),
//    modelCoefficients = DenseVector.zeros[Double](5),
//    noiseStddev = 0.5
//  )
//
//var bfp:Parameters = initialParameters2

  class CustomUI(val ui : ScalismoUI) extends SimplePluginAPI with ScalismoPublisher {
    // Create a panel of buttons using Scala Swing
    val panel = new FlowPanel() {
      val button1 = new Button("Run")
      val button2 = new Button("Dist")
      val button3 = new Button("compare")
      val button4 = new Button("Landmarks")
      val button5 = new Button("Run2")
      val button6 = new Button("reset")
      val button7 = new Button(" ")
      val burn = new FancySlider { min = 1000; max = 50000; value = 1000}
      val its = new FancySlider {min = 0; max = 250000; value = 10000}

      contents += (button1, button5, button3,button2,button4, burn, its, button6)

    }
    // Tell the plugin to listen to events from the button and install the handlers
    listenTo(panel.button1)
    listenTo(panel.button2)
    listenTo(panel.button3)
    listenTo(panel.button3)
    listenTo(panel.button4)
    listenTo(panel.button5)
    listenTo(panel.button6)
    listenTo(panel.button7)
    listenTo(panel.its)
    listenTo(panel.burn)


    reactions += {


            // This RUN button works out point correspondences, gets settings for burnin in and iterations, and then runs MCMC function
      case ButtonClicked(panel.`button1`) => {

        // get point correspondances
        val modellms = ui.filter(modelGroup, (v : LandmarkView) => true)
        val reflms = modellms.map(l => l.landmark.point)
        val reflms2  = reflms.map(l => model1.reference.pointSet.findClosestPoint(l).id).toSeq
        val targetlms = ui.filter(targetGroup, (v : LandmarkView) => true)
        for (landmarkView <- targetlms) {
          landmarkView.color = Color.GREEN
          landmarkView.scalingFactor = 2.5
                }
        val tlms = targetlms.map(l => l.landmark.point)
        val corr = reflms2
          .zip(tlms)
          .map(modelIdWithTargetPoint => {
            val (modelId, targetPoint) = modelIdWithTargetPoint
            (modelId, targetPoint)
          })

          val burn_in = panel.burn.value
          val iterations = panel.its.value

        val z1 = findtransform().parameters
        val translationz = EuclideanVector3D(z1(0), z1(1), z1(2))
        val rotationz = (z1(3), z1(4), z1(5))

        val initialParameters1 = Parameters(
          //          translationParameters = EuclideanVector(0, 0, 0),
          //          rotationParameters = (0.0, 0.0, 0.0),
          translationParameters = translationz,
          rotationParameters = rotationz,
          modelCoefficients = DenseVector.zeros[Double](model1.rank),
          noiseStddev = 1.0
        )

        implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
        val bf = Future (MCMC(corr,model1, modelView, burn_in, iterations, initialParameters1))
        //return parameters to UI
        message("FINISHED. burnin = " + burn_in.toString + " & iterations = " + iterations.toString)

      }

// this button will calculate distance from landmark on bestfit to targetshape
      case ButtonClicked(panel.`button2`) => {

        val gtMesh2 = ui.filter(targetGroup, (v : TriangleMeshView ) => true).last.triangleMesh

        val modellms = ui.filter(resultGroup, (v : LandmarkView) => true)
        val reflms = modellms.map(l => l.landmark.point)
        val pid  = reflms.map(l => gtMesh2.pointSet.findClosestPoint(l).id)

        for (j <- 0 to(pid.length- 1)) {
          val ptc = gtMesh2.pointSet.point(pid(j))
          val disttogt = ((ptc - reflms(j)).norm).toString
          message(disttogt)
        }

        val distances = for (x <- reflms.map(l => gtMesh2.pointSet.findClosestPoint(l).id)) yield (gtMesh2.pointSet.point(x))
        val tupleofpts = reflms.zip(distances)
        val x = for (i <- tupleofpts) yield (i._1 - i._2).norm
        message("average distances = " + (x.sum / x.length).toString)

      }
      // This button is used to save the best fit STL and the target shape in a folder so that a comparison can be made
      // This button compares the best fit shape with the groundtruth shape using function called result which also
      //creates a color map of the differences
      case ButtonClicked(panel.`button3`) => {

        val a = ui.filter(targetGroup, (v : TriangleMeshView ) => true)
        if (a.length > 0) {
          MeshIO.writeMesh(a(0).triangleMesh, new java.io.File("C:/Tutorial/datasets/groundtruth.stl"))
          val b = ui.filter(resultGroup, (v : TriangleMeshView ) => true)
          MeshIO.writeMesh(b.last.triangleMesh, new java.io.File("C:/Tutorial/datasets/bestfit.stl"))
          message("saved target humerus as groundtruth.stl and best fit as bestfit.stl")
          val targetMesh = MeshIO.readMesh(new java.io.File("C:/Tutorial/datasets/bestfit.stl")).get
          val refMesh = MeshIO.readMesh(new java.io.File("C:/Tutorial/datasets/groundtruth.stl")).get
          val result = Compare(targetMesh,refMesh)
          message(result)
        }
        else message("missing target shape")

      }

      // This button loads in preset landmarks for the humerus
      case ButtonClicked(panel.`button4`) => {
        val LMs = LandmarkIO.readLandmarksJson3D(new java.io.File("C:/Tutorial/datasets/scapulaLM.json")).get
        val HL = ui.show(modelGroup, LMs, "Humerus landmarks")
        val lv = ui.filter[LandmarkView](modelGroup, (v : LandmarkView) => true)
        for (landmarkView <- lv) {
          landmarkView.color = Color.red
          landmarkView.scalingFactor = 2.5
//          println(landmarkView.landmark.point)
        }
      }

      // This button will so a 2nd run to finetune
      case ButtonClicked(panel.`button5`) => {

//        val lastBestfit = ui.filter(resultGroup, (v : TriangleMeshView ) => true).last.triangleMesh
//
//        val new_com = computeCenterOfMass(lastBestfit)

          val new_com = computeCenterOfMass(model1.mean)


        // get point correspondances
        val modellms = ui.filter(modelGroup, (v : LandmarkView) => true)
        val reflms = modellms.map(l => l.landmark.point)
        val reflms2  = reflms.map(l => model1.reference.pointSet.findClosestPoint(l).id).toSeq
        val targetlms = ui.filter(targetGroup, (v : LandmarkView) => true)

        val tlms = targetlms.map(l => l.landmark.point)
        val corr = reflms2
          .zip(tlms)
          .map(modelIdWithTargetPoint => {
            val (modelId, targetPoint) = modelIdWithTargetPoint
            (modelId, targetPoint)
          })

        val burn_in = panel.burn.value
        val iterations = panel.its.value

        val z = modelView.shapeModelTransformationView.shapeTransformationView.coefficients
        val z1 = modelView.shapeModelTransformationView.poseTransformationView.transformation.parameters

        val translationz = EuclideanVector3D(z1(0), z1(1), z1(2))
        val rotationz = (z1(3), z1(4), z1(5))

        val initialParametersft = Parameters(
          translationParameters = translationz,
          rotationParameters = rotationz,
          modelCoefficients = z,
          noiseStddev = 0.1
        )

        implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
        Future (MCMC2(corr,model1, modelView, burn_in, iterations, initialParametersft, new_com))
        //return parameters to UI
        message("burnin = " + burn_in.toString + " & iterations = " + iterations.toString)


      }

      //this button resets the shape model
      case ButtonClicked(panel.`button6`) => {

        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = DenseVector.zeros[Double](model1.rank)
        val translation = Translation3D(EuclideanVector3D(0, 0, 0))
        val rotationCenter = Point3D(0.0, 0.0, 0.0)
        val rotation = Rotation3D(0f, 0f, 0f, rotationCenter)
        modelView.shapeModelTransformationView.poseTransformationView.transformation = TranslationAfterRotation(translation,rotation)

      }

      case ButtonClicked(panel.`button7`) => {



      }
    }



    // this method is called when the plugin is activated
    override def onActivated(): Unit = {
      addToToolbar(panel)
      message("plugin activated")
    }
    // This method is called when the plugin is deactivated
    override def onDeactivated(): Unit = {
      removeFromToolbar(panel)
      message("plugin deactivated")
    }

  }



  val ui = ScalismoUI()
  val model1 = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("C:/Tutorial/datasets/ScapulaSSMtrain/models/scapulaSSM_aug100.h5")).get

  val model_dec = model1.reference.operations.decimate(5000)
  val lowresmodel = model1.newReference(model_dec,TriangleMeshInterpolator3D())

  val modelGroup = ui.createGroup("model")
  val targetGroup = ui.createGroup("target")
  val resultGroup = ui.createGroup("result")
  val modelView = ui.show(modelGroup, model1, "model")
  modelView.referenceView.opacity = 0.5


  val plugin = new CustomUI(ui)
  plugin.activate()



}
