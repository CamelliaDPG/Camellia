//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

/*
 
 This code is copied and pasted from TwoStreamInstability.  See Issue #118 for what should happen here.
 
 */

#include "AbsFunction.h"
#include "BasisSumFunction.h"
//#include "Boltzmann1D1VFormulation.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "FixedCountRefinementStrategy.h"
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

static const std::string S_COLD_DIODE   = "cold-diode";

enum ProblemChoice
{
  COLD_DIODE
};

static const std::map<std::string, ProblemChoice> stringToProblemChoice {{S_COLD_DIODE,COLD_DIODE}};
static const std::map<ProblemChoice, std::string> problemChoiceToString {{COLD_DIODE,S_COLD_DIODE}};

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

void checkMesh( MeshPtr mesh)
{
  const auto & myCellIDs = mesh->cellIDsInPartition();
  
  auto globalDofAssignment = mesh->globalDofAssignment();
  
  for (auto & cellID : myCellIDs)
  {
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      auto neighborInfo = cell->getNeighborInfo(sideOrdinal, mesh->getTopology());
      const IndexType & neighborCellID = neighborInfo.first;
      const auto & mySideOrdinalInNeighbor = neighborInfo.second;
      if (neighborCellID != -1)
      {
        int myParity       = globalDofAssignment->sideParity(cellID, sideOrdinal);
        int neighborParity = globalDofAssignment->sideParity(neighborCellID, mySideOrdinalInNeighbor);
        TEUCHOS_TEST_FOR_EXCEPTION(myParity != -neighborParity, std::invalid_argument, "inconsistent parities");
      }
    }
  }
}

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
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    double x1     = x0 + xWidth;
    
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = true;
    
    bool useAdaptivity = false;
  //  double refinementThreshold   = 0.999999; // essentially, refine just the maximum error element (and any that thanks to solution symmetry have same error)
  //  double unrefinementThreshold = 1.000001; // similarly, unrefine just the minimum-error parent element
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
    std::string problemChoiceString = "cold-diode";
    ProblemChoice problemChoice = COLD_DIODE;
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = false;
    int uniformMeshRefinementLevel = 3; // would take a coarse 8x8 mesh to 64x64 (this is used for overkill experiment)
    bool saveViz = true;
    std::string outputDir = "/tmp";
    int solutionOutputFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 20000;
    int fixedCountRefinementTarget = -1; // set to positive number to used fixed-count ref strategy instead of greedy strategy
    
    int vlasovTraceEnrichment = -1; // if left unset by user, will default to 0 for all problems but cold diode, where it is 1 (at least for now).
    double coldDiodeSigma = -1.0; // will be set automatically below if not specified by user
    double bcResolutionTolerance = -1;  // only applies if doing an adaptive solve (and so far only for cold diode); -1 indicates to spend the whole element budget to resolve; unrefinements at later time steps will tend to reduce the BC resolution over time, though experimentally the loss is often negligible (but this will depend on precise settings!).
    bool useCrankNicolson = false;
    
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("implicit", "explicit", &implicit);
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("sigma", &coldDiodeSigma); // sigma for cold diode -- ignored for other problems
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
    cmdp.setOption("problem", &problemChoiceString);
    cmdp.setOption("refinementThreshold", &refinementThreshold);
    cmdp.setOption("fixedCountRefTarget", &fixedCountRefinementTarget);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("uniformMeshRefinementLevel", &uniformMeshRefinementLevel);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("vlasovTraceEnrichment", &vlasovTraceEnrichment);
    cmdp.setOption("bcResolutionTolerance", &bcResolutionTolerance);
    cmdp.setOption("useCrankNicolson", "useBackwardEuler", &useCrankNicolson);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    problemChoice = getProblemChoice(problemChoiceString);
    
    // for cold diode, we non-dimensionalize; we modify the below accordingly so that we can output dimensional values to results file.
    double velocityScaling = 1.;
    double timeScaling     = 1.;
    double spatialScaling  = 1.;
    double EScaling        = 1.;
    double nScaling         = 1.;
    double potentialScaling = 1.;
    
    if (vlasovTraceEnrichment == -1)
    {
      if (problemChoice == COLD_DIODE)
      {
        // we no longer think it's a good idea to enrich the traces separately; refine to resolve the BCs; don't change the polynomial order of traces globally
        vlasovTraceEnrichment = 0;
      }
      else
      {
        // theory says that H^1 order is enough for convergence rates (if BCs are resolved)
        vlasovTraceEnrichment = 0;
      }
    }
    
    if (problemChoice != COLD_DIODE)
    {
      if (vWidth == std::numeric_limits<double>::infinity())
      {
        vWidth = 20.0;
      }
      if (v0 == std::numeric_limits<double>::infinity())
      {
        v0 = -10.0;
      }
    }
    
    bool usePeriodicBCs = true;
    if (problemChoice == COLD_DIODE)
    {
      const double e               = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e             =  9.10938356e-31;  // m_e, mass of electron in kg
      
      const double W               = 1e4;       // beam intensity from SAND report [eV]
      const double J               = 9.33582e4; // current density from SAND report [A/m^2]
      
      // v_beam is defined by 1/2 m v^2 = W ==> v_beam = sqrt( 2 W /m ):
      const double v_beam          = sqrt(2 * W * e / m_e ); // e converts from eV to Joules
      
      const double q0              = e;
      const double m0              = m_e;
      
      velocityScaling  = v_beam;
      timeScaling      = 1.97485663509e-9; // final time from SAND report [s]
      spatialScaling   = velocityScaling * timeScaling; // [m]
      EScaling         = m0 * velocityScaling / (q0 * timeScaling);
      potentialScaling = EScaling * spatialScaling;
      
      if (rank == 0)
      {
        std::cout << "Characteristic E-field strength E_0 : " << EScaling << std::endl;
        std::cout << "spatial scaling : " << spatialScaling << std::endl;
      }
      
      const double permittivity = 8.85418782e-12; // permittivity of free space
      
      nScaling     = J / (e * v_beam); // number density of plasma
      
      const double omega_p = sqrt( q0 * q0 * nScaling / (m0 * permittivity));
      
      xWidth = 0.01 / spatialScaling;
      x1     = x0 + xWidth;
      u0     = 1.0;
      n0     = 1.0;
      q      = -1.0;
      
      if (coldDiodeSigma == -1)
      {
        sigma = 0.05; // default sigma value
      }
      else
      {
        sigma = coldDiodeSigma;
      }
      
      // eps0 is a weight that enters RHS of Poisson; we use this to scale the problem:
      eps0   = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
      usePeriodicBCs = false;
      enforceQuasiNeutrality = false;
      
      if (vWidth == std::numeric_limits<double>::infinity())
      {
        vWidth = 1.00;
      }
      if (v0 == std::numeric_limits<double>::infinity())
      {
        v0 = 0.50;
      }
      
      if (rank == 0)
      {
        std::cout << "sigma = " << sigma << std::endl;
        std::cout << "u0 = " << u0 << std::endl;
        std::cout << "n0 = " << n0 << std::endl;
      }
    }
    
    if (dt_max < 0)
    {
      dt_max = 1e-1;
    }
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation({q}, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    
    double dt = dt_max;
    const bool useSerendipityBasis = false;
    const int maxMomentNumber = 0;
    VlasovBackwardEulerFormulation form({q}, {m}, {"e"}, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber, useCrankNicolson);
    
    form.refineUniformly(numUniformRefinements);
    form.setRunQuietly(true);
    
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
    bool adaptToCaptureInitialBCs  = false; // true for cold diode
    bool adaptToCaptureInitialData = true;
    
    if (problemChoice == COLD_DIODE)
    {
      f0 = Function::zero(); // start with vacuum
      adaptToCaptureInitialData = false; // zero solution is always exactly captured…
      adaptToCaptureInitialBCs  = true;
    }
    
    using std::vector;
    using std::pair;
    vector<pair<pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true, and only for cold diode
    
    if (!usePeriodicBCs)
    {
      if (problemChoice == COLD_DIODE)
      {
        // pin potential to zero at left and right:
        const int poissonTagLeft  = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0));
        const int poissonTagRight = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x1));
        
        auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
        // homogeneous on left and right:
        form.addConfigurationSpaceBC(poissonTagLeft,  VE_hat, Function::constant(0.0));
        form.addConfigurationSpaceBC(poissonTagRight, VE_hat, Function::constant(0.0));
        
        // set a Maxwellian at the inflow:
        FunctionPtr fM_u0 = Teuchos::rcp(new Maxwellian1V( u0, sigma, n0) );
        form.setInitialAndInflowConditions(f0, fM_u0);
        
        const bool weightFluxesBySideParity = true;
        FunctionPtr tn_soln = Function::solution(form.t_n_hat(), form.vlasovSolution(), weightFluxesBySideParity);
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        int inflowTag = form.tagVlasovSpatialInflow();
        FunctionPtr inflowRestriction = Function::meshSideSetCharacteristic(tagName, inflowTag);
              
        FunctionPtr sgn = Function::sideParity();
        FunctionPtr n = Function::normal();
        FunctionPtr n_x = n->spatialComponent(1); // 1-based
        FunctionPtr vx = form.v_i(0);
        
        FunctionPtr tn_inflowExact;
        if (useCrankNicolson)
        {
          auto priorTimeSoln = form.vlasovSolutionPriorTime();
          auto f_k = Function::solution(form.f(), form.vlasovSolutionPriorTime(), weightFluxesBySideParity);
          tn_inflowExact = 0.5 * (fM_u0 + f_k) * sgn * n_x * vx;
        }
        else
        {
          tn_inflowExact = fM_u0 * sgn * n_x * vx;
        }
        bcFunctionsToResolve.push_back({{tn_inflowExact, tn_soln},inflowTag});
      }
    }
    form.setInitialConditions(f0);
    
    RefinementStrategyPtr formRefStrategy;
    
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
      
      if (fixedCountRefinementTarget > 0)
      {
        ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
        formRefStrategy = Teuchos::rcp(new FixedCountRefinementStrategy(errorIndicator, fixedCountRefinementTarget) );
        gateRefinements = false;
      }
      else
      {
        formRefStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
      }
    }
    else if (useAdaptivity && adaptToCaptureInitialBCs)
    {
      for (auto bcPair : bcFunctionsToResolve)
      {
        FunctionPtr bcFxnExact = bcPair.first.first;
        FunctionPtr bcFxnSoln  = bcPair.first.second;
        int bcTag = bcPair.second;
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
        
        // project onto vlasovSolution:
        std::map<int,FunctionPtr> bcMap;
        bcMap[form.t_n_hat()->ID()] = bcFxnExact;
        form.vlasovSolution()->projectOntoMesh(bcMap);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureEnrichment = 4;
        
        auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.vlasovMesh(), bc_err, quadratureEnrichment);
        
        double refinementThresholdForInitialData = 0.80;
        auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
  //      refStrategy->setMaxElements(maxElements);
        refStrategy->setMaxP(H1Order);
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        double relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
        
        int numBCRefinements = 0;
        int numVlasovElements = form.vlasovMesh()->numActiveElements();
        while ((relErr > bcResolutionTolerance) && (numVlasovElements < maxElements))
        {
          if (rank==0)
          {
            std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
            std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
          }
          
          refStrategy->refine();
          
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form.configurationSpaceMesh()->repartitionAndRebuild();
          
          // project the BC onto the mesh again
          form.vlasovSolution()->projectOntoMesh(bcMap);
          
          bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
          relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
          
          numVlasovElements = form.vlasovMesh()->numActiveElements();
        }
        if (rank==0)
        {
          std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
        }
      }
      
      ErrorIndicatorPtr unrefinementErrorIndicator;
      if (!useEnergyForUnrefinements)
      {
        unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
      }
      
      if (fixedCountRefinementTarget > 0)
      {
        ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
        formRefStrategy = Teuchos::rcp(new FixedCountRefinementStrategy(errorIndicator, fixedCountRefinementTarget) );
        gateRefinements = false;
      }
      else
      {
        formRefStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
      }
    }
    
    std::ostringstream suffix;
    if (implicit)
    {
      suffix << "Implicit";
    }
    if (useL2Trace)
    {
      suffix << "_L2Trace";
    }
    else
    {
      suffix << "_H1Trace";
    }
    if (problemChoice != COLD_DIODE)
    {
      suffix << "_u0" << u0;
    }
    else
    {
      suffix << "_sigma" << sigma;
    }
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
    if (problemChoice == COLD_DIODE)
    {
      problemPrefix = "ColdDiode";
    }

    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix <<"Vlasov" << suffix.str();

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
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tBC err\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double bcErr, const double energyError) {
      auto refReportFilePath = getTimeStepReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << bcErr << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    auto getBCResolution = [&]() {
      double bcRelErr = -1;
      for (auto bcPair : bcFunctionsToResolve)
      {
        FunctionPtr bcFxnExact = bcPair.first.first;
        FunctionPtr bcFxnSoln  = bcPair.first.second;
        int bcTag = bcPair.second;
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureEnrichment = 4;
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        bcRelErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
      }
      return bcRelErr;
    };
    
    auto reportBCResolutionAndTimeStepComplete = [&](const int &timeStepNumber, const double &bcRelErr, const double &energyError) {
      auto numVlasovDofs       = form.vlasovMesh()->numGlobalDofs();
      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
      auto poissonElementCount = form.configurationSpaceMesh()->numActiveElements();
      auto poissonDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        std::cout << "BC resolution, relative error in L^2 norm: " << bcRelErr << std::endl;
        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, bcRelErr, energyError);
        std::cout << "Time step " << timeStepNumber << " complete.\n";
      }
    };
   
    
  //  std::cout << "vlasov name: " << vlasovSS.str() << std::endl;
  //  std::cout << "poisson name: " << poissonSS.str() << std::endl;
    
    const std::string directoryPath = outputDir;
    
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
    
//    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(1, form.vlasovSolution(), form.f(), form.configurationSpaceMesh()) );
        
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
      
      checkMesh(form.vlasovMesh());
      
      const double bcRelErr = getBCResolution(); // returns -1 if BCs to resolve weren't specified
      performRefinements(lastRefinementMaxElementError, form, formRefStrategy, gateRefinements, refinementThreshold, maxElements);
      const double energyError = (formRefStrategy != Teuchos::null) ? formRefStrategy->getEnergyError(formRefStrategy()->getNumRefinements() - 1) : -1;
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          vlasovViz->exportSolution(form.vlasovSolution(),t);
          poissonViz->exportSolution(form.configurationSpaceSolution(), {density,velocity}, {"n_e","v"},t);
        }
        
        reportBCResolutionAndTimeStepComplete(timeStepNumber,bcRelErr,energyError);
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
    
  //  // output to results file:
  //  auto activePhysicalCells = form.configurationSpaceMesh()->getActiveCellIDsGlobal();
  //  form.configurationSpaceSolution()->importSolutionForOffRankCells(activePhysicalCells);
  //
  //  if (rank == 0)
  //  {
  //    std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
  //    std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
  //    GlobalIndexType firstPhysicalCellID = *activePhysicalCells.begin();
  //
  //    auto basisCache = BasisCache::basisCacheForCell(form.configurationSpaceMesh(), firstPhysicalCellID);
  //    std::vector<GlobalIndexType> physicalCellIDsVector(activePhysicalCells.begin(),activePhysicalCells.end());
  //
  //    auto physicalCellNodes = form.configurationSpaceMesh()->physicalCellNodes(physicalCellIDsVector);
  //    basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
  //
  //    const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
  //    const FieldContainer<double> & cubatureWeights = basisCache->getCubatureWeightsFC();
  //
  //    const int numCells  = physicalQuadraturePoints.dimension(0);
  //    const int numPoints = physicalQuadraturePoints.dimension(1);
  //
  //    FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), numPoints);
  //    E_soln->values(E_values_FC, basisCache);
  //
  //    for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
  //    {
  //      const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
  //      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
  //      {
  //        const double & x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,0);
  //        const double & E = E_values_FC(cellOrdinal,pointOrdinal);
  //        const double pointWeight = cubatureWeights(pointOrdinal) * cellWidth;
  //
  //        E_values[x]           = E;
  //        pointWeight_values[x] = pointWeight;
  //      }
  //    }
  //
  //    std::ofstream resultsFileStream(resultsFile, std::ios::out);
  //
  //    resultsFileStream << "x\tquadrature weight\tE\n";
  //    resultsFileStream << setprecision(15);
  //
  //    for (const auto & pointEntry : pointWeight_values)
  //    {
  //      const auto x           = pointEntry.first * spatialScaling;
  //      const auto pointWeight = pointEntry.second * spatialScaling;
  //      const auto E           = E_values[pointEntry.first] * EScaling;
  //
  //      resultsFileStream << x           << "\t";
  //      resultsFileStream << pointWeight << "\t";
  //      resultsFileStream << E           << "\n";
  //    }
  //
  //    resultsFileStream.close();
  //
  //    std::cout << "Wrote results to " << resultsFile << std::endl;
  //  }
      
    const bool outputInflowValues = false; // note that "true" may cause a crash on distributed runs
    if (outputInflowValues && (problemChoice == COLD_DIODE))
    {
      // output t_n / v along inflow boundary (which is x=0)
      const int X_COMP_ZERO_BASED  = 0;
      const int VX_COMP_ZERO_BASED = 1;
      SpatialFilterPtr inflowFilter = SpatialFilter::matchingComponent(X_COMP_ZERO_BASED, x0);
      
      auto vlasovMeshTopo = form.vlasovMesh()->getTopology()->baseMeshTopology();
      auto inflowSideIndices = vlasovMeshTopo->getBoundarySidesThatMatch(inflowFilter);
      std::set<GlobalIndexType> phaseInflowCells;
      
      const unsigned leftSideOrdinal = 2;
      const unsigned sideDim = vlasovMeshTopo->getDimension() - 1;
      
      for (const auto & sideEntityIndex : inflowSideIndices)
      {
        IndexType phaseCellID = vlasovMeshTopo->getFirstCellForSide(sideEntityIndex).first;
        if (!vlasovMeshTopo->isParent(phaseCellID))
        {
          phaseInflowCells.insert(phaseCellID);
        }
        // assert that the sideEntityIndex we found is in fact the leftSideOrdinal
        assert( sideEntityIndex == vlasovMeshTopo->getCell(phaseCellID)->entityIndex(sideDim, leftSideOrdinal) );
      }
      
      form.vlasovSolution()->importSolutionForOffRankCells(phaseInflowCells);
      
      if (rank == 0)
      {
        std::map<double,double> f_values; // map from v_x coordinate to a computed f = t_n / v value; we use a map so it's sorted by v_x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        GlobalIndexType firstPhaseCellID = *phaseInflowCells.begin();
        
        const bool testVsTest = false;
        const int cubatureEnrichment = 10; // get a bunch of extra points
        auto basisCache = BasisCache::basisCacheForCell(form.vlasovMesh(), firstPhaseCellID, testVsTest, cubatureEnrichment)->getSideBasisCache(leftSideOrdinal);
        std::vector<GlobalIndexType> phaseCellIDsVector(phaseInflowCells.begin(),phaseInflowCells.end());
        
        const auto & cubPointsFC  = basisCache->getRefCellPointsFC();
        const int numPoints = cubPointsFC.dimension(0); // (P,D)
        
        auto physicalCellNodes = form.vlasovMesh()->physicalCellNodes(phaseCellIDsVector);
        basisCache->setPhysicalCellNodes(physicalCellNodes, phaseCellIDsVector, false);
        
        FunctionPtr tn_soln = Function::solution(form.t_n_hat(), form.vlasovSolution(), false);
        
        FieldContainer<double> tn_values_FC(phaseCellIDsVector.size(), numPoints);
        tn_soln->values(tn_values_FC, basisCache);
        
        const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
        
        const int numCells  = physicalQuadraturePoints.dimension(0);
        
        std::ostringstream gnuPlotStringStream;
        gnuPlotStringStream << directoryPath << "/" << problemPrefix << "_inflowBC" << suffix.str() << ".gnuplot";
        std::string gnuPlotFile = gnuPlotStringStream.str();
        
        std::ofstream gnuPlotFileStream(gnuPlotFile, std::ios::out);
        
        gnuPlotFileStream << setprecision(15);
        gnuPlotFileStream << "set term png\n";
        gnuPlotFileStream << "set output \"" << problemPrefix << "_inflowBC" << suffix.str() << ".png\"\n";
        gnuPlotFileStream << "plot '-' using 1:2 t \"f inflow\" smooth unique\n";
        gnuPlotFileStream << "set term png\n";
        
        /*
         set term png
         set output "inflowBCs.png"
         plot '-' using 1:2 t "f inflow" smooth unique
         */
        gnuPlotFileStream << "plot '-' using 1:2 smooth unique\n"; // inline data
        for (int cellOrdinal=0; cellOrdinal<phaseCellIDsVector.size(); cellOrdinal++)
        {
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            const double & v_x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,VX_COMP_ZERO_BASED);
            const double & t_n = tn_values_FC(cellOrdinal,pointOrdinal);
            
            const double f = t_n / v_x;
            gnuPlotFileStream << "  " << v_x << " " << f << std::endl;
          }
        }
        gnuPlotFileStream << "  e\n";
        gnuPlotFileStream.close();
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
