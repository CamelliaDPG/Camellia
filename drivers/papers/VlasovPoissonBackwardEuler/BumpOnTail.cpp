//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "ConfigurationSpaceFormulation.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"
#include "VlasovBackwardEulerFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

class Maxwellian1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1V(double u, double sigma, double n)
  :
  _u(u),
  _sigma(sigma),
  _n(n)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
//    std::cout << "weight: " << weight << std::endl;
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
};

static const std::string S_BUMP_ON_TAIL = "bump-on-tail";

enum ProblemChoice
{
  BUMP_ON_TAIL
};

RefinementStrategyPtr constructRefStrategy(VlasovBackwardEulerFormulation &form, double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator)
{
  auto vlasovSoln  = form.vlasovSolution();
  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(vlasovSoln, refinementThreshold);

  refStrategy->setMaxElements(maxElements);
  refStrategy->setMaximumRefinementLevel(maxRefinementLevel);
  refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = vlasovSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
  return refStrategy;
}

static const std::map<std::string, ProblemChoice> stringToProblemChoice {{S_BUMP_ON_TAIL,BUMP_ON_TAIL}};
static const std::map<ProblemChoice, std::string> problemChoiceToString {{BUMP_ON_TAIL,S_BUMP_ON_TAIL}};

ProblemChoice getProblemChoice(const std::string &problemChoiceString)
{
  const auto &entry = stringToProblemChoice.find(problemChoiceString);
  if (entry == stringToProblemChoice.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized problem choice string");
  }
  else
  {
    return entry->second;
  }
}

std::string getProblemChoiceString(ProblemChoice problem)
{
  const auto &entry = problemChoiceToString.find(problem);
  if (entry == problemChoiceToString.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized problem choice");
  }
  else
  {
    return entry->second;
  }
}

void performRefinements(double & lastRefinementMaxElementError, VlasovBackwardEulerFormulation &form, RefinementStrategyPtr refStrategy, bool gateRefinements, double refinementThreshold, int maxElements)
{
  if (refStrategy == Teuchos::null) return;
  bool shouldRefine;
  if (gateRefinements)
  {
    // we compute error indicator first; if no refinement-eligible element has error greater than what would have been refined last time, then we don't do a refinement step now.
    
    auto localCellsToExclude = refStrategy->localCellsIneligibleForRefinement();
    
    ErrorIndicatorPtr errorIndicator = refStrategy->errorIndicator();
    errorIndicator->measureError();
    const double maxError = errorIndicator->maxError(localCellsToExclude);
    
    shouldRefine = maxError > refinementThreshold * lastRefinementMaxElementError;
    
    if (shouldRefine)
    {
      lastRefinementMaxElementError = maxError;
    }
    else
    {
      // let's still record the measured error in RefinementStrategy's records
      const double totalEnergyError = errorIndicator->totalError();
      const int numElements = form.vlasovMesh()->numActiveElements();
      const int numDofs     = form.vlasovMesh()->numGlobalDofs();
      refStrategy->recordResults(numElements, numDofs, totalEnergyError);
    }
  }
  else
  {
    shouldRefine = true;
  }
  
  if (shouldRefine)
  {
    // for now, print refinement info to console on rank 0.
    const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
    const bool doUnrefinements = (maxElements < std::numeric_limits<int>::max());
    refStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
    
    // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
    form.configurationSpaceMesh()->repartitionAndRebuild();
  }
}

void setPoissonBC(VlasovBackwardEulerFormulation &form, const double &x0)
{
  // we have periodic BCs; we just need to pin the potential at a point -- we choose the x0 (left) point.
  const int poissonTagLeft = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0));
  
  auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
  form.addConfigurationSpaceBC(poissonTagLeft, VE_hat, Function::constant(0.0));
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  const int rank = mpiSession.getRank();
  
  // block off variables so that everything gets destructed before the finalize() call at the end.
  {
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    mpiSession.barrier();
    
    bool implicit = true;
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    
    const double DEFAULT_V0     = 0.50;
    const double DEFAULT_VWIDTH = 1.00;
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = true;
    
    bool useAdaptivity = false;
    double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    bool gateRefinements = true;
    
    int xElements = 8;
    int vElements = 8;
    int numUniformRefinements = 0;
    int delta_k   = 2;
    double dt_max = -1.0;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int numTimeSteps = 500;
    int outputFrequency = 10; // number of time steps to take before outputting viz.
    bool useL2Trace = true;
    std::string problemChoiceString = "bump-on-tail";
    const ProblemChoice problemChoice = BUMP_ON_TAIL;
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool runOverkillVersusAdaptiveComparison = false;
    bool useEnergyForUnrefinements = false;
    int uniformMeshRefinementLevel = 3; // would take a coarse 8x8 mesh to 64x64 (this is used for overkill experiment)
    bool saveViz = true;
    std::string outputDir = "/tmp";
    int solutionOutputFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 20000;
    
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("u0", &u0);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("gateRefinements", "dontGateRefinements", &gateRefinements);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("refinementThreshold", &refinementThreshold);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("uniformMeshRefinementLevel", &uniformMeshRefinementLevel);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    // for cold diode, we non-dimensionalize; we modify the below accordingly so that we can output dimensional values to results file.
    double velocityScaling = 1.;
    double timeScaling     = 1.;
    double spatialScaling  = 1.;
    double EScaling        = 1.;
    double nScaling         = 1.;
    double potentialScaling = 1.;
    
    if (vWidth == std::numeric_limits<double>::infinity())
    {
      vWidth = 20.0;
    }
    if (v0 == std::numeric_limits<double>::infinity())
    {
      v0 = -10.0;
    }
    
    bool usePeriodicBCs = true;
    k_x = 0.3;
    xWidth =  2 * M_PI / k_x;
    u0 = 4.5;
    n0 = 0.1;
    usePeriodicBCs = true;
    
    if (dt_max < 0)
    {
      dt_max = 1e-1;
    }
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    
    double dt = dt_max;
    VlasovBackwardEulerFormulation form(q, eps0, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
//    poissonConfig->configurationSpaceMesh()->bilinearForm()->printTrialTestInteractions();
    form.refineUniformly(numUniformRefinements);
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    setPoissonBC(form, x0);
    
    RefinementStrategyPtr formRefStrategy;
    
    form.setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    form.setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
      form.setIterativeSolveTolerances(cgTols);
    }
    form.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    form.setFixedPointTolerance(fixedPointTol);
    
    FunctionPtr f0;
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    
    // the following flags are considered only if useAdaptivity is true:
    bool adaptToCaptureInitialData = true;
    if (problemChoice == BUMP_ON_TAIL)
    {
      FunctionPtr fM_0 = Teuchos::rcp(new Maxwellian1V( 0.0, 1.0, 1.0-n0) );
      FunctionPtr fM_b = Teuchos::rcp(new Maxwellian1V(  u0, 0.5,     n0) );
      f0               = (fM_0 + 0.5 * fM_b) * (1. + gamma * cos_kx);
      std::cout << "f0 for bump-on-tail: " << f0->displayString() << "\n";
      adaptToCaptureInitialData = true;
    }
    
    using std::vector;
    using std::pair;
    
    form.setInitialConditions(f0);
    
    // adapt to capture initial data:
    if (useAdaptivity && adaptToCaptureInitialData)
    {
      FunctionPtr f_soln = Function::solution(form.f(), form.vlasovSolution());
      
      FunctionPtr f0_err = f0 - f_soln;
      int quadratureEnrichment = 4;
      
      auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.vlasovMesh(), f0_err, quadratureEnrichment);
      
      double refinementThresholdForInitialData = 0.20;
      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
      refStrategy->setMaxElements(maxElements);
      ErrorIndicatorPtr unrefinementErrorIndicator;
      if (!useEnergyForUnrefinements)
      {
        unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
      }
      refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
      refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
      refStrategy->setMaxP(H1Order);
      
      double f0_norm = f0->l2norm(form.vlasovMesh(), quadratureEnrichment);
      double relErr = f0_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / f0_norm;
      
      int numInitialRefinements = 0;
      while (form.vlasovMesh()->numActiveElements() < maxElements * 0.90)
      {
        if (rank==0)
        {
          std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
        }
        const bool unrefine = true;
        const bool printToConsole = (rank == 0);
        refStrategy->refineAndUnrefine(unrefine, printToConsole);
        form.setInitialConditions(f0); // do projection again
        numInitialRefinements++;
        relErr = f0_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / f0_norm;
        
        if (form.vlasovMesh()->numActiveElements() > maxElements)
        {
          // do one more refine + unrefine, and then switch to unrefinements only
          refStrategy->refineAndUnrefine(unrefine, printToConsole);
          while (form.vlasovMesh()->numActiveElements() > maxElements)
          {
            // repeat until unrefinement gets us down to maxElements or below
            if (rank==0)
            {
              std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and relErr is " << relErr << ".\n";
              std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
            }
            
            refStrategy->setRelativeErrorThreshold(1.1); // ensure one unrefinement-only pass
            refStrategy->refineAndUnrefine(unrefine, printToConsole);
            refStrategy->setRelativeErrorThreshold(refinementThresholdForInitialData);
            
            form.setInitialConditions(f0); // do projection again
            numInitialRefinements++;
            relErr = f0_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / f0_norm;
            
            if (numInitialRefinements > 25)
            {
              // that seems like too many; break
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not get initial mesh down to maxElements through refine/unrefine after 25 adaptive steps");
            }
          }
        }
      }
      if (rank==0)
      {
        std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
      }
      // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
      form.configurationSpaceMesh()->repartitionAndRebuild();
      
      formRefStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
    }
    
    std::ostringstream suffix;
    suffix << "_u0" << u0;
    if (useAdaptivity)
    {
      suffix << "_AMR_threshold" << refinementThreshold;
    }
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << vElements;

    std::string problemPrefix;
    if (problemChoice == BUMP_ON_TAIL)
    {
      problemPrefix = "BumpOnTail";
    }

    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();

    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << outputDir << "/";
      if (resultsFilePrefix == "")
      {
        resultsFileStream << "results_" << problemPrefix << "_" << suffix.str();
      }
      else
      {
        resultsFileStream << resultsFilePrefix;
      }
      
      resultsFileStream << "_" << numTimeSteps << "_timeSteps";
      resultsFileStream << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    auto getSpatialMeshFilePath = [&]() {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << "mesh1D_" << problemPrefix << suffix.str();
      
      spatialMeshInfoPathStringStream << numTimeSteps << "_timeSteps";
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    auto getTimeStepReportFilePath = [&]() {
      std::ostringstream timeStepReportPathStringStream;
      timeStepReportPathStringStream << outputDir << "/";
      timeStepReportPathStringStream << "timeStepReport_" << problemPrefix << suffix.str();
      
      timeStepReportPathStringStream << numTimeSteps << "_timeSteps";
      timeStepReportPathStringStream << ".dat";
      
      return timeStepReportPathStringStream.str();
    };
    
    auto refReportFilePath = getTimeStepReportFilePath();
    const std::string directoryPath = outputDir;
    
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double energyError) {
      auto refReportFilePath = getTimeStepReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    auto reportTimeStepComplete = [&](const int &timeStepNumber, const double &energyError) {
      auto numVlasovDofs       = form.vlasovMesh()->numGlobalDofs();
      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
      auto poissonElementCount = form.configurationSpaceMesh()->numActiveElements();
      auto poissonDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, energyError);
        std::cout << "Time step " << timeStepNumber << " complete.\n";
      }
    };
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << directoryPath << "/" << problemPrefix << "_results" << suffix.str() << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
    if (saveViz)
    {
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
      poissonViz = Teuchos::rcp(new HDF5Exporter (form.configurationSpaceMesh(), poissonSS.str(), directoryPath) );
    }
    
    double t = 0.0;
    if (saveViz)
    {
      poissonViz->exportSolution(form.configurationSpaceSolution(),t);
      vlasovViz->exportSolution(form.vlasovSolution(),t);
    }
    
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
    FunctionPtr E_squared = E_soln * E_soln;
    FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form.vlasovSolution(), form.f(), form.configurationSpaceMesh()) );
        
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1);
    auto velocity = momentum / density;
            
    bool previousSolveUsedGeometricMultigrid = false;
    bool previousSolveUsedDirect = false;
    
    double lastRefinementMaxElementError = 0.0;
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
      if (numVlasovDofs > maxDofsForDirectSolve)
      {
        if ((rank == 0) && (!previousSolveUsedGeometricMultigrid))
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
        }
        form.setUseGeometricMultigrid(true);
        previousSolveUsedGeometricMultigrid = true;
        previousSolveUsedDirect = false;
      }
      else
      {
        if ((rank == 0) && (previousSolveUsedGeometricMultigrid))
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
        }
        form.setUseGeometricMultigrid(false);
        previousSolveUsedGeometricMultigrid = false;
        previousSolveUsedDirect = true;
      }
      
      form.takeTimeStep();
      t += dt;
      
      performRefinements(lastRefinementMaxElementError, form, formRefStrategy, gateRefinements, refinementThreshold, maxElements);
      
      const double energyError = (formRefStrategy != Teuchos::null) ? formRefStrategy->getEnergyError(formRefStrategy()->getNumRefinements() - 1) : -1;
      
      const bool computeMinimum_bn = false; // average value on a side
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          vlasovViz->exportSolution(form.vlasovSolution(),t);
          poissonViz->exportSolution(form.configurationSpaceSolution(), {density,velocity}, {"n_e","v"},t);
        }
        
        reportTimeStepComplete(timeStepNumber,energyError);
      }
    } // end time step loop
    
    FunctionPtr potential = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), form.configurationSpaceSolution());
        
    // output to results file:
    {
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
      // the di() methods are 1-based:
      static const int SPATIAL_INDEX  = 1;
      static const int VELOCITY_INDEX = 1;
      
      // SpatialFilter's components are zero-based
      static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
      static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
      
      auto allActivePhysicalCells = form.configurationSpaceMesh()->getActiveCellIDsGlobal();
      
      form.configurationSpaceSolution()->importSolutionForOffRankCells(allActivePhysicalCells);
      
      // update solutionMoments once more and gather data to rank 0, so we can access it in the block below
      solutionMoments->update();
      solutionMoments->gatherOnRankZero();
      
      if (rank == 0)
      {
        std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> ne_values; // map from x coordinate to final density value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> velocity_values; // map from x coordinate to final velocity value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> potential_values; // map from x coordinate to final potential value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
        GlobalIndexType firstPhysicalCellID = *allActivePhysicalCells.begin();
        
        const int cubatureEnrichmentDegree = 5; // over-integrate to make up for undermeasurement on coarse initial meshes
        auto basisCache = BasisCache::basisCacheForCell(form.configurationSpaceMesh(), firstPhysicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
        std::vector<GlobalIndexType> physicalCellIDsVector(allActivePhysicalCells.begin(),allActivePhysicalCells.end());
        
        const int numPoints = basisCache->getRefCellPoints().extent_int(0);
        auto & cubWeights = basisCache->getCubatureWeightsFC();
        
        auto physicalCellNodes = form.configurationSpaceMesh()->physicalCellNodes(physicalCellIDsVector);
        basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
        
        FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), numPoints);
        E_soln->values(E_values_FC, basisCache);
        
        FieldContainer<double> potential_values_FC(physicalCellIDsVector.size(), numPoints);
        potential->values(potential_values_FC, basisCache);
        
        FieldContainer<double> density_values_FC(physicalCellIDsVector.size(), numPoints);
        density->values(density_values_FC, basisCache);
        
        FieldContainer<double> velocity_values_FC(physicalCellIDsVector.size(), numPoints);
        velocity->values(velocity_values_FC, basisCache);
        
        const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
        
        const int numCells         = physicalQuadraturePoints.dimension(0);
        
        for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
        {
          const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            const double & x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,0);
            const double pointWeight = cubWeights(pointOrdinal) * cellWidth / 2.0; // 2.0: width of the reference line.
            
            E_values[x]           = E_values_FC(cellOrdinal,pointOrdinal);
            potential_values[x]   = potential_values_FC(cellOrdinal,pointOrdinal);
            ne_values[x]          = density_values_FC(cellOrdinal,pointOrdinal);
            velocity_values[x]    = velocity_values_FC(cellOrdinal,pointOrdinal);
            pointWeight_values[x] = pointWeight;
          }
        }
        
        std::ofstream resultsFileStream(resultsFile, std::ios::out);
        
        resultsFileStream << "x\tquadrature weight\tE\tphi\tne\tvx\n";
        resultsFileStream << setprecision(15);
        
        for (const auto & pointEntry : pointWeight_values)
        {
          const auto x           = pointEntry.first * spatialScaling;
          const auto pointWeight = pointEntry.second * spatialScaling;
          const auto E           = E_values[pointEntry.first]         * EScaling;
          const auto potential   = potential_values[pointEntry.first] * potentialScaling;
          const auto ne          = ne_values[pointEntry.first]        * nScaling;
          const auto velocity    = velocity_values[pointEntry.first]  * velocityScaling;
          
          resultsFileStream << x           << "\t";
          resultsFileStream << pointWeight << "\t";
          resultsFileStream << E           << "\t";
          resultsFileStream << potential   << "\t";
          resultsFileStream << ne          << "\t";
          resultsFileStream << velocity    << "\n";
        }
        
        resultsFileStream.close();
        
        auto spatialMeshFile = getSpatialMeshFilePath();
        std::ofstream spatialMeshFileStream(spatialMeshFile, std::ios::out);
        // let's collect the spatial nodes and sort them
        std::set<double> spatialNodes;
        for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
        {
          spatialNodes.insert(physicalCellNodes(cellOrdinal,0,0)); // (min x, min t)
          spatialNodes.insert(physicalCellNodes(cellOrdinal,1,0)); // (max x, min t)
        }
        resultsFileStream << "x\n";
        for (const auto & node : spatialNodes)
        {
          spatialMeshFileStream << node << "\n";
        }
        spatialMeshFileStream.close();
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
