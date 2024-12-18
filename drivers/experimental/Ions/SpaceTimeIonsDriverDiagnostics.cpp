//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDebugUtility.h"
#include "ConfigurationSpaceFormulation.h"
#include "EnergyErrorFunction.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "FixedCountRefinementStrategy.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MaxwellianFunction.h"
#include "MonomialFunctions.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TimeRampFunction.h"
#include "TrigFunctions.h"
#include "VlasovSpaceTimeFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

//! TODO: put this into the main Camellia source (i.e. part of the library)
class CellIDFunction : public Function
{
public:
  CellIDFunction() : Function(0) {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    int numCells = cellIDs.size();
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
      }
    }
  }
};

class TemporalRamp1D1T : public TFunction<double>
{
  double _rampWidth; // amount of time before the value is 1.0
public:
  TemporalRamp1D1T(double rampWidth)
  :
  _rampWidth(rampWidth)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & spaceTimePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = spaceTimePoints.dimension(0);
    const int numPoints = spaceTimePoints.dimension(1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double t = spaceTimePoints(cellOrdinal,pointOrdinal,1); // t is the second dimension for 1D1T
        double value;
        if (t < _rampWidth)
        {
          value = t / _rampWidth;
        }
        else
        {
          value = 1.0;
        }
        values(cellOrdinal,pointOrdinal) = value;
      }
    }
  }
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    auto heavisideT = Function::heavisideY(_rampWidth); // 1 for y < _rampWidth, 0 otherwise
    return (1.0 / _rampWidth) * heavisideT;
  }
  
  TFunctionPtr<double> dz() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dt() override
  {
    return dy();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 2) // 1-based
    {
      return dt();
    }
    else
    {
      return Function::zero();
    }
  }
};

class Maxwellian1T1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1T1V(double u, double sigma, double n)
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
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,2); // vx is the third dimension for 1D1T1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dz() override
  {
    auto vx = MonomialFunction::xi_n(2, 1);
    FunctionPtr f_M = Teuchos::rcp( new Maxwellian1T1V(_u,_sigma,_n));
    return (_u - vx) / (_sigma * _sigma) * f_M;
  }
  
  TFunctionPtr<double> dt() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 3) // 1-based
    {
      return dz();
    }
    else
    {
      return Function::zero();
    }
  }
};

// this communication strategy is wasteful, but we don't expect this to be a major bottleneck…
double getMaxMagnitudeOnCell(FunctionPtr fxn, MeshPtr mesh, GlobalIndexType &cellID)
{
  auto meshTopo = mesh->getTopology();
  auto cell = meshTopo->getCell(cellID);
  auto activeCells = cell->getDescendants(meshTopo);
  double maxValue = 0;
  
//  {
//    // DEBUGGING
//    int rank = mesh->Comm()->MyPID();
//    if (rank == 0)
//    {
//      std::cout << "Physical cell " << cellID;
//      if (activeCells.size() > 1)
//      {
//        std::cout << " has descendants: ";
//        Camellia::print("", activeCells);
//      }
//      else
//      {
//        std::cout << " is active.\n";
//      }
//    }
//  }
  for (auto activeCellID : activeCells)
  {
    double myMax = 0;
    if (mesh->myCellsInclude(activeCellID))
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, activeCellID);
      int numPoints = basisCache->getRefCellPointsFC().dimension(0);
      FieldContainer<double> values(1, numPoints);
      fxn->values(values, basisCache);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double pointValue = values(0, pointOrdinal);
        myMax = std::max(std::abs(pointValue), myMax);
      }
    }
    double globalMax = MPIWrapper::sum(*mesh->Comm(), myMax);
//    std::cout << "max value on cell " << activeCellID << ": " << globalMax << std::endl;
    maxValue = std::max(maxValue, globalMax);
  }
  return maxValue;
}

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

RefinementStrategyPtr constructRefStrategy(VlasovSpaceTimeFormulation &form, double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator)
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

void performRefinements(double & lastRefinementMaxElementError, VlasovSpaceTimeFormulation &form, RefinementStrategyPtr refStrategy, RefinementStrategyPtr unrefinementStrategy,
                        int maxElements)
{
  if (refStrategy == Teuchos::null) return;
  
  const bool doUnrefinements = false;
  if (doUnrefinements)
  {
    // for now, print refinement info to console on rank 0.
    const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
    unrefinementStrategy->setRelativeErrorThreshold(1.1); // ensure that we only do unrefinements
    int maxUnrefinementSteps = 15; // In case we end up unable to unrefine to maxElements for some reason, cap the number of unrefinements
    int numUnrefinementStepsTaken = 0;
    int unrefinementTarget = maxElements * .90;
    unrefinementStrategy->setMaxElements(unrefinementTarget);
    while ((form.vlasovMesh()->numActiveElements() > unrefinementTarget) && (numUnrefinementStepsTaken < maxUnrefinementSteps))
    {
      // update solution to ensure that the errors measured are up to date post-refinement
      form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
      form.vlasovSolution()->solve();
      form.configurationSpaceSolve();
      
      unrefinementStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
      form.configurationSpaceMesh()->repartitionAndRebuild();
      numUnrefinementStepsTaken++;
    }
        
    if (numUnrefinementStepsTaken > 0)
    {
      // update solution to ensure that the errors measured are up to date post-refinement
      form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
      form.vlasovSolution()->solve();
      form.configurationSpaceSolve();
    }
  }
  
  // for now, print refinement info to console on rank 0.
  const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
  refStrategy->refineAndUnrefine(false, printToConsole); // false: don't unrefine
  
  // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
  form.configurationSpaceMesh()->repartitionAndRebuild();
  
  auto configElementCount = form.configurationSpaceMesh()->numActiveElements();
  auto configDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
  if (printToConsole)
  {
    std::cout << "Configuration-space mesh has " << configElementCount << " active elements and " << configDofCount << " global degrees of freedom.\n";
  }
}

//! Sets a homogeneous BC at the x0 point, and parameter-based BC at the x1 point.  Returns a pointer to the parameter function; the caller may then set it to an appropriate value before (any/each) Poisson solve.
ParameterFunctionPtr setPoissonBC(VlasovSpaceTimeFormulation &form, const double &x0, const double &x1)
{
  bool restrictToBoundary = true;
  
  const int poissonTagLeft  = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0), restrictToBoundary);
  const int poissonTagRight = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x1), restrictToBoundary);
  
  auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
  // homogeneous on left:
  form.addConfigurationSpaceBC(poissonTagLeft, VE_hat, Function::constant(0.0));
  
  // parameter on right:
  ParameterFunctionPtr rightBCFunction = ParameterFunction::parameterFunction(0.0);
  form.addConfigurationSpaceBC(poissonTagRight, VE_hat, rightBCFunction);
  
  return rightBCFunction;
}

//! MPI-collective method for 1D meshes.  Evaluates f_soln on whatever rank owns the point x, and communicates the result to every rank.  Returns the value.
double getSolutionFunctionValue(MeshPtr solnMesh, FunctionPtr f_soln, const double x)
{
  FieldContainer<double> points(1,1);
  points(0,0) = x;
  
  auto cellIDs = solnMesh->cellIDsForPoints(points);
  
  double value = 0.0;
  if (cellIDs.size() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != 1, std::invalid_argument, "Found more than one cellID for point x.");
    const auto cellID = cellIDs[0];
    
    if (solnMesh->myCellsInclude(cellID))
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(solnMesh, cellID);
      
      if (f_soln->boundaryValueOnly())
      {
        // i.e., a trace
        // 1D trace: sides are just points.  Does x match one of the sides?
        const auto & nodes = basisCache->getPhysicalCellNodes();
        double leftNode  = nodes(0,0,0); // (C,N,D)
        double rightNode = nodes(0,1,0);
        BasisCachePtr sideCache;
        if (std::abs(x-leftNode) < 1e-14)
        {
          sideCache = basisCache->getSideBasisCache(0);
        }
        else if (std::abs(x-rightNode) < 1e-14)
        {
          sideCache = basisCache->getSideBasisCache(1);
        }
        else
        {
          std::cout << "ERROR: x is not a mesh vertex, but f_soln is a trace solution.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found");
        }
        
        FieldContainer<double> fValue(1,1); // (C,P)
        
        f_soln->values(fValue, sideCache);
        
        value = fValue(0,0);
      }
      else
      {
        // volume variable
        FieldContainer<double> refPoint(1,1,1);
        FieldContainer<double> physPoint(1,1,1);
        physPoint(0,0,0) = x;
        
        int cubatureDegree = 1; // not curvilinear…
        
        CamelliaCellTools::mapToReferenceFrame( refPoint, physPoint, solnMesh->getTopology(), cellID, cubatureDegree);
        
        refPoint.resize(1,1);
        
        basisCache->setRefCellPointsFC(refPoint);
        
        FieldContainer<double> fValue(1,1); // (C,P)
        
        f_soln->values(fValue, basisCache);
        
        value = fValue(0,0);
      }
    }
  }
  
  // all but the MPI-owner have a zero value, so we can sum to communicate the value to all ranks.
  value = MPIWrapper::sum(value);
  return value;
}

double inflowErrorOnBoundary(VlasovSpaceTimeFormulation &form, std::vector<FunctionPtr> f_in, const int bcTag, const int p_max)
{
  const std::string & tagName = DIRICHLET_SET_TAG_NAME;
  FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
  
  const int delta_k = form.vlasovMesh()->globalDofAssignment()->getTestOrderEnrichment();
  const int quadratureDegree = p_max * 2 + delta_k;
  const int quadratureEnrichment = (4 + quadratureDegree > Intrepid2::Parameters::MaxCubatureDegreeEdge) ? (Intrepid2::Parameters::MaxCubatureDegreeEdge - quadratureDegree) : 4;
  
  double errSquared = 0;
  double bcNormSquared = 0;
  for (int speciesOrdinal=0; speciesOrdinal<int(f_in.size()); speciesOrdinal++)
  {
    FunctionPtr f_soln = Function::solution(form.f(speciesOrdinal), form.vlasovSolution());
    FunctionPtr bc_err = (f_in[speciesOrdinal] - f_soln) * bcRestriction;
    
    double bc_norm = (f_in[speciesOrdinal] * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
    bcNormSquared += bc_norm * bc_norm;
    double err = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment);
    errSquared += err * err;
  }
  
  return sqrt(errSquared / bcNormSquared);
}

void setUseGMG(VlasovSpaceTimeFormulation &form, int maxDofsForDirectSolve, bool &previousSolveUsedGeometricMultigrid, bool &previousSolveUsedDirect, bool logChangeToConsole)
{
  auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
  if (numVlasovDofs > maxDofsForDirectSolve)
  {
    if (logChangeToConsole && (!previousSolveUsedGeometricMultigrid))
    {
      std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
    }
    form.setUseGeometricMultigrid(true);
    previousSolveUsedGeometricMultigrid = true;
    previousSolveUsedDirect = false;
  }
  else
  {
    if (logChangeToConsole && (previousSolveUsedGeometricMultigrid))
    {
      std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
    }
    form.setUseGeometricMultigrid(false);
    previousSolveUsedGeometricMultigrid = false;
    previousSolveUsedDirect = true;
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
    
    double q_e_scaled    = -1.0;
    double eps0          =  1.0; // overwritten below
    double q_ion_scaled  = -q_e_scaled;
    double x0     =  0.0;
    double xWidth =  1.0; // non-dimensionalized in some appropriate way
    double potentialBC = 12000; // in Volts (physical units); overridden to 0 if disableElectricField is set to true
    double voltageBCDamping = 1.; // a multiplier for the applied voltage; using this for diagnosis
    
    const double permittivity = 8.85418782e-12; // permittivity of free space

    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    double tFinal = 0.25; // TODO: add time slabbing.  This tFinal gets us to 1/144 of the desired tFinal.
    
    bool enforceQuasiNeutrality = false;
    
    int maxElements = 8000;
    
    std::string normChoice = "graph"; // alternatives so far are "L2" and "natural"
    
    int xElements = 4;
    int vElements = 8;
    int tElements = 1;
    int numUniformRefinements = 0;
    int numEnergyRefinements = 0;
    int delta_k   = 2;
    double timeRampWidth = 2.0; // assuming domain width of 1.0, velocity of 1.0, this fully turns on BCs after two transit times.  Modestly aggressive; could increase to 3.0 if there appear to be artifacts.
    int H1Order = 3; // this gives f, E (which are in L^2) a polynomial order of 2
    int minSubsteps = 2;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-5;
    double bcResolutionTol = 1e-1;
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = false;
    bool saveViz = true;
    std::string outputDir = "/private/viz";
    int solutionOutputFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-6;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 80000;
    bool useSerendipityBasis = false;
    bool includeIons = true;
    bool includeElectrons = true;
    bool disableElectricField = false;
    bool disableMagneticField = false;
    double overrideEps0 = -1.; // if set from CLI, will override the computed eps0 value.
    bool useCrankNicolson = false;
    double energyErrorTarget = 0.01; // try to refine to get below this threshold
    
    double physicalGapWidth = .00216; // [m], from Christenson and Lau (1994)
    double criticalCurrentRatio = 0.95; // I/I_C.  0.95 is case C2; 1.1 is case C3
    
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("numEnergyRefinements", &numEnergyRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("timeRampWidth", &timeRampWidth);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("tFinal", &tFinal, "final time");
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("norm", &normChoice);
    cmdp.setOption("bcResolutionTol", &bcResolutionTol);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
    cmdp.setOption("energyErrorTarget", &energyErrorTarget);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("overrideEps0", &overrideEps0);
    cmdp.setOption("physicalGapWidth", &physicalGapWidth);
    cmdp.setOption("ions", "no-ions", &includeIons);
    cmdp.setOption("electrons", "no-electrons", &includeElectrons);
    cmdp.setOption("disableElectricField", "includeElectricField", &disableElectricField);
    cmdp.setOption("disableMagneticField", "includeMagneticField", &disableMagneticField);
    cmdp.setOption("criticalCurrentRatio", &criticalCurrentRatio);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    // Trying a velocity scaling that goes from -0.3c to +0.3c.
    const double DEFAULT_V0     = -1.00;
    const double DEFAULT_VWIDTH =  2.00;
    const bool useAdaptivity = (numEnergyRefinements > 0);
    
    const double c         = 299792458;
    double velocityScaling = 0.30 * c;
//    const double v_beam    = 0.00139891 * c;
    // scale so that our domain width is 1.0
    double spatialScaling  = physicalGapWidth;
    double timeScaling     = spatialScaling / velocityScaling;
    
    double EScaling        = 1.;
    double n0              = 1.;
    double potentialScaling = 1.;
    double currentScaling   = 1.;
    double BScaling         = 1.;
    double currentDensityScaling = 1.;
    
    double B = 0.27; // z-direction magnetic field
    
    FunctionPtr extraAccelerationFormula; // function of x
    
    double m_e_scaled;
    double m_ion_scaled;
    
    // determine scalings
    {
      const double e   = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e =  9.10938356e-31;  // m_e, mass of electron in kg
      const double Omega = B * e / m_e; // cyclotron frequency in Hz
      const double OmegaScaling = 1.0 / timeScaling;
      potentialScaling = potentialBC;
      
      const double qScaling = - e / q_e_scaled; // charge scaling
      EScaling = potentialScaling / spatialScaling;
      if (rank == 0)
        cout << "potentialScaling: " << potentialScaling << std::endl;
      
      const double mScaling = EScaling * qScaling * timeScaling / velocityScaling;
      
      m_e_scaled   = m_e / mScaling;
      m_ion_scaled = m_e_scaled * 1836.15;
      
      xWidth = physicalGapWidth / spatialScaling;
      BScaling = potentialScaling * timeScaling / (spatialScaling * spatialScaling);
      
      FunctionPtr x = Function::xn(1);  // x^1
      const double B_scaled = B / BScaling;
      const double Omega_scaled = Omega / OmegaScaling;
      if (rank == 0)
        cout << "extra acceleration term: " << B_scaled * Omega_scaled << " x.\n";
      extraAccelerationFormula = B_scaled * Omega_scaled * x;
      
      const double J_C = 2.1e5; // A / m^2; critical value from Christenson & Lau
      n0 = J_C / (qScaling * velocityScaling); // number density of plasma
      
      const double omega_p = qScaling * sqrt( n0 / (mScaling * permittivity));
//      std::cout << "omega_p = sqrt(n0) * " << q0 * sqrt( 1.0 / (m0 * permittivity)) << std::endl;
      
      // eps0 is a weight that enters PoissonFormulation; we use this to scale the problem:
      eps0 = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
      
      if (overrideEps0 != -1.0)
      {
        if (rank == 0)
        {
          std::cout << "NOTE: overriding eps0 (which was " << eps0 << ") with value " << overrideEps0 << std::endl;
        }
        eps0 = overrideEps0;
      }
      
      // we know that total current (density) is Iem + eps0 * dE / dt
      // we have a scaling for eps0: it's our eps0 divided by the physical permittivity of free space
      // we also have a scaling for E: EScaling
      // and for t: timeScaling
      // so the currentScaling should be (permittivity / eps0) * EScaling / timeScaling
      
      currentDensityScaling = (permittivity / eps0) * EScaling / timeScaling; // should match J_C
      currentScaling        = currentDensityScaling * spatialScaling * spatialScaling;
      
      if (rank == 0)
      {
        std::cout << "test norm:           " << normChoice << std::endl;
        std::cout << "E-field scaling:     " << EScaling << std::endl;
        std::cout << "B-field scaling:     " << BScaling << std::endl;
        std::cout << "V scaling:           " << potentialScaling << std::endl;
        std::cout << "physical gap width:  " << physicalGapWidth << std::endl;
        std::cout << "spatial scaling:     " << spatialScaling << std::endl;
        std::cout << "temporal scaling:    " << timeScaling << std::endl;
        std::cout << "domain width:        " << xWidth << std::endl;
        std::cout << "n0:                  " << n0 << std::endl;
        std::cout << "current scaling:     " << currentScaling << std::endl;
        std::cout << "J scaling:           " << currentDensityScaling << std::endl;
        std::cout << "eps0:                " << eps0 << std::endl;
      }
      enforceQuasiNeutrality = false;
    }
    
    if (vWidth == std::numeric_limits<double>::infinity())
    {
      vWidth = DEFAULT_VWIDTH;
    }
    if (v0 == std::numeric_limits<double>::infinity())
    {
      v0 = DEFAULT_V0;
    }
    
    bool usePeriodicBCs = false;
    
    double x1 = x0 + xWidth;
    
    std::vector<double> q_values;
    std::vector<double> m_values;
    std::vector<string> speciesLabels;
    if (includeElectrons)
    {
      q_values.push_back(q_e_scaled);
      m_values.push_back(m_e_scaled);
      speciesLabels.push_back("e");
    }
    if (includeIons)
    {
      q_values.push_back(q_ion_scaled);
      m_values.push_back(m_ion_scaled);
      speciesLabels.push_back("i");
    }
    const int numSpecies = int(q_values.size());
    
    if (disableElectricField)
    {
      potentialBC = 0;
    }
    
    // use parameter function to allow us to turn magnetic field on with a delay.
    ParameterFunctionPtr extraAccelerationParameter = ParameterFunction::parameterFunction(extraAccelerationFormula);
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q_values, eps0, SPACE_TIME);
    if (!disableMagneticField)
    {
      poissonConfig->setExtraAcceleration({extraAccelerationParameter});
    }
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x1, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    
    std::vector<double> tVertices;
    double t = 0;
    const double t0 = 0;
    for (int i=0; i<=tElements; i++)
    {
      t = t0 + (i * tFinal) / tElements;
      tVertices.push_back(t);
    }
    
    int maxMomentNumber = 1; // we want to compute bulk velocity
    
    const EBasisFamily basisFamily = useSerendipityBasis ? BASIS_FAMILY_SERENDIPITY : BASIS_FAMILY_HIERARCHICAL;
    
    VlasovSpaceTimeFormulation form(q_values, m_values, speciesLabels, poissonConfig, spatialMeshTopo, tVertices, vxVertices, std::vector<double>(), std::vector<double>(), H1Order, delta_k, numUniformRefinements, basisFamily, maxMomentNumber);
    
    IPPtr ip;
    if (normChoice == "graph")
    {
      ip = form.vlasovBF()->graphNorm();
    }
    else if (normChoice == "L2")
    {
      ip = form.vlasovBF()->l2Norm();
    }
    else if (normChoice == "natural")
    {
      ip = form.vlasovBF()->naiveNorm(3); // 3 dimensions (x,t,v)
    }
    else if (normChoice == "graph01")
    {
      // identical to graph norm, except written in terms of dx() rather than di(), and drops dy() and dz() terms
      ip = IP::ip();
      const int e_species = 0;
      auto w_e = form.w(e_species);
      auto v_x = form.v_i(0);
      ip->addTerm(w_e); // L^2 term
      ip->addTerm(- v_x * w_e->dx()); // drops:  - w_e->dy() 0.260927 * (E_x + 4.98512 * x) \frac{\partial}{\partial z} w_e
    }
    else if (normChoice == "graph1")
    {
      // identical to graph norm, except written in terms of dx() and dy() rather than di(), and drops dz() term
      ip = IP::ip();
      const int e_species = 0;
      auto w_e = form.w(e_species);
      auto v_x = form.v_i(0);
      ip->addTerm(w_e); // L^2 term
      ip->addTerm(- v_x * w_e->dx() - w_e->dy()); // drops: 0.260927 * (E_x + 4.98512 * x) \frac{\partial}{\partial z} w_e
    }
    else if (normChoice == "graph10")
    {
      // identical to graph norm, except written in terms of dy() rather than di(), and drops dx() and dz() terms
      ip = IP::ip();
      const int e_species = 0;
      auto w_e = form.w(e_species);
      auto v_x = form.v_i(0);
      ip->addTerm(w_e); // L^2 term
      ip->addTerm(- w_e->dy()); // drops: - v_x * w_e->dx() + 0.260927 * (E_x + 4.98512 * x) \frac{\partial}{\partial z} w_e
    }
    else if (normChoice == "graph2")
    {
      // identical to graph norm, except written in terms of dx(), dy(), dz() rather than di(), and drops the E_x part of the weight on the dz() term
      ip = IP::ip();
      const int e_species = 0;
      auto w_e = form.w(e_species);
      auto v_x = form.v_i(0);
      ip->addTerm(w_e); // L^2 term
      ip->addTerm(- v_x * w_e->dx() - w_e->dy() + 0.260927 * (extraAccelerationFormula) * w_e->dz()); // drops: 0.260927 * (E_x + 4.98512 * x) \frac{\partial}{\partial z} w_e
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported normChoice");
    }
    if (rank == 0)
    {
      std::cout << "Test space IP:\n";
      ip->printInteractions();
    }
    form.vlasovSolution()->setIP(ip);
    
//    if (useConservativeNorm)
//    {
//      double theta = useCrankNicolson ? 0.5 : 1.0;
//      auto w  = form.w();
//      auto vx = form.v_i(0);
//      auto ax = form.a()[0];
//
//      auto w_dx = w->di(1);
//      auto w_dv = w->di(2);
//
//      IPPtr ip = IP::ip();
//      ip->addTerm(w);
//      ip->addTerm(w - (2 * theta * dt * vx) * w_dx - (2*theta*dt*ax)* w_dv);
//
//      form.vlasovSolution()->setIP(ip);
//    }

    if (disableElectricField)
    {
      // set the RHS to 0 -- in combination with the zero potential BC we set above, this forces the electric field to be zero.
      // (This needs to be done after the VBEFormulation has been constructed, since it calls poissonConfig->setMoments(), which
      //  is where the RHS for Poisson gets set up.)
      poissonConfig->solution()->setRHS(RHS::rhs());
    }
    
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    auto voltageParameter = setPoissonBC(form, x0, x1);
        
    // voltageParameter is a "ParameterFunction" -- we need fill it in with an appropriate value
    double voltageValue = potentialBC / potentialScaling * voltageBCDamping;  // 12000 V
    voltageParameter->setValue(voltageValue);
      
    FunctionPtr velocityExtremaRestriction =   Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovMaxVelocity(0))
                                             + Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovMinVelocity(0));
    
    std::vector<FunctionPtr> fAtExtrema(numSpecies);
    for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
    {
      fAtExtrema[speciesOrdinal] = Function::solution(form.f(speciesOrdinal), form.vlasovSolution()) * velocityExtremaRestriction;
    }
      
    if (rank == 0)
    {
      if (useCrankNicolson) std::cout << "Using Crank-Nicolson in time.\n";
      std::cout << "Setting non-dimensional voltage at x=1 to " << potentialBC / potentialScaling << std::endl;
    }
    
    auto configMesh = poissonConfig->configurationSpaceMesh();
    
    // electrons flow in from left
    double u0_e;
    {
      const double e    = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e  =  9.10938356e-31;  // m_e, mass of electron in kg
      const double W    = 0.5; // [eV] beam intensity from Christenson & Lau
      
      // v_e is defined by 1/2 m v^2 = W ==> v_beam = sqrt( 2 W /m ):
      const double v_e = sqrt(2 * W * e / m_e ); // e converts from eV to Joules
      u0_e = v_e / velocityScaling;
      if (rank == 0)
      {
        std::cout << "u0_e: " << u0_e << std::endl;
      }
    }
    
    double sigmaSquared_e = 1.0 * u0_e * u0_e; // let thermal velocity be 100% of drift velocity
    double n = 1.0;
    const double firstMomentOfMaxwellian_e = MaxwellianFunction::momentNormalized1D(u0_e, sigmaSquared_e, 1); // will return u0_e
    const double J_inflow_physical_e = criticalCurrentRatio * 2.1e5; // Amps // criticalCurrentRatio: CLI parameter (default of 0.95 is Case C2 from C & L; use 1.1 for Case C3)
    const double J_inflow_scaled_e   = J_inflow_physical_e / currentDensityScaling;
    const double inflow_weight_e     = J_inflow_scaled_e / firstMomentOfMaxwellian_e;
    if (rank == 0)
    {
      std::cout << "inflow_weight_e: " << inflow_weight_e << std::endl;
    }
    
    const int firstVelocityDimension = configMesh->getDimension();
    const int numVelocityDimensions  = form.velocityDim();
    FunctionPtr positiveRestriction = Function::heaviside(0, firstVelocityDimension); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
    FunctionPtr fM_u0_e = inflow_weight_e * MaxwellianFunction::maxwellian(firstVelocityDimension, numVelocityDimensions, {u0_e}, sigmaSquared_e, n) * positiveRestriction;
    
    // ions flow in from right
    double u0_ion = -u0_e / sqrt(1836.15); // ratio of electron mass to ion mass (assigning them both the same energy)
    double sigmaSquared_ion = 0.005 * 0.005;
    FunctionPtr negativeRestriction = 1 - positiveRestriction; // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
    FunctionPtr fM_u0_ion = MaxwellianFunction::maxwellian(firstVelocityDimension, numVelocityDimensions, {u0_ion}, sigmaSquared_ion, n) * negativeRestriction;
    
    FunctionPtr f_e_in = fM_u0_e;
    FunctionPtr f_i_in = fM_u0_ion;
    
    ErrorIndicatorPtr unrefinementErrorIndicator;
    if (!useEnergyForUnrefinements)
    {
      unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
    }
    double refinementThreshold = 1.1; // > 1 ==> no refinements will take place
    RefinementStrategyPtr unrefinementStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
    
    form.setFixedPointMaxSubsteps(maxSubsteps);
    form.setFixedPointMinSubsteps(minSubsteps);
    form.setFixedPointTolerance(fixedPointTol);
    
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
    
    using std::vector;
    using std::tuple;
    vector<tuple<VarPtr,FunctionPtr,FunctionPtr,int> > bcFunctionsToResolve; // (var, exact function, solution function, tag)
    
    int speciesOrdinal = 0;
    std::vector<FunctionPtr> f0, f_in;
    if (includeElectrons)
    {
      f0.push_back(Function::zero()); // vacuum initial conditions for both species
      f_in.push_back(f_e_in);
      
      VarPtr tn_e = form.t_n_hat(speciesOrdinal);
      
      auto inflowTag = form.tagVlasovSpatialInflow();
      
      FunctionPtr sgn = Function::sideParity();
      FunctionPtr n = Function::normal();
      FunctionPtr n_x = n->spatialComponent(1); // 1-based
      FunctionPtr vx = form.v_i(0);
      
      FunctionPtr tn_inflowExact = f_e_in * sgn * n_x * vx;
      const bool weightFluxesBySideParity = true;
      FunctionPtr tn_soln = Function::solution(tn_e, form.vlasovSolution(), weightFluxesBySideParity);
      bcFunctionsToResolve.push_back({tn_e,tn_inflowExact,tn_soln,inflowTag});
      speciesOrdinal++;
    }
    if (includeIons)
    {
      f0.push_back(Function::zero()); // vacuum initial conditions for both species
      f_in.push_back(f_i_in);
      
      VarPtr tn_i = form.t_n_hat(speciesOrdinal);
      
      auto inflowTag = form.tagVlasovSpatialInflow();
      
      FunctionPtr sgn = Function::sideParity();
      FunctionPtr n = Function::normal();
      FunctionPtr n_x = n->spatialComponent(1); // 1-based
      FunctionPtr vx = form.v_i(0);
      
      FunctionPtr tn_inflowExact = f_i_in * sgn * n_x * vx;
      const bool weightFluxesBySideParity = true;
      FunctionPtr tn_soln = Function::solution(tn_i, form.vlasovSolution(), weightFluxesBySideParity);
      bcFunctionsToResolve.push_back({tn_i,tn_inflowExact,tn_soln,inflowTag});
    }
    form.setInitialAndInflowConditions(f0, f_in, timeRampWidth);
    
    FunctionPtr Ex = Function::solution(poissonConfig->E_fieldVar(0), form.configurationSpaceSolution());
    
    FunctionPtr vxAcceleration = Ex + extraAccelerationFormula;
    double maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
    if (rank == 0)
    {
      std::cout << "Max acceleration on initial solution is " << maxAcceleration_vx << std::endl;
    }
    int    p_max = H1Order;  // for now, no p refinements.  (May want to revisit; if we do, can set p_max = 10 - delta_k -- p is limited by Intrepid2 quadrature rules at the moment.)
    
    ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
    const int numElementsToRefine = std::max(int(.05 / 3. * maxElements), 1);
    RefinementStrategyPtr fixedCountRefStrategy = Teuchos::rcp( new FixedCountRefinementStrategy(errorIndicator, numElementsToRefine) );
    
    refinementThreshold = 0.20; // refine any cell that has more than 20% of the maximum element error
    RefinementStrategyPtr greedyRefStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, Teuchos::null);
    greedyRefStrategy->setEnforceOneIrregularity(true);
    
    const bool useGreedyStrategy = true;
    RefinementStrategyPtr refStrategy = useGreedyStrategy ? greedyRefStrategy : fixedCountRefStrategy;
    
//    for (auto bcTuple : bcFunctionsToResolve)
//    {
//      using std::get;
//      VarPtr var = get<0>(bcTuple);
//      FunctionPtr bcFxnExact = get<1>(bcTuple);
//      FunctionPtr bcFxnSoln  = get<2>(bcTuple);
//      int bcTag = get<3>(bcTuple);
//
//      const std::string & tagName = DIRICHLET_SET_TAG_NAME;
//      FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
//
//      // project onto vlasovSolution:
//      std::map<int,FunctionPtr> bcMap;
//      bcMap[var->ID()] = bcFxnExact;
//      form.vlasovSolution()->projectOntoMesh(bcMap);
//
//      FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
//      int quadratureDegree = p_max * 2 + delta_k;
//      int quadratureEnrichment = (4 + quadratureDegree > Intrepid2::Parameters::MaxCubatureDegreeEdge) ? (Intrepid2::Parameters::MaxCubatureDegreeEdge - quadratureDegree) : 4;
//
//      auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.vlasovMesh(), bc_err, quadratureEnrichment);
//
//      double refinementThresholdForInitialData = 0.80;
//      auto refStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, Teuchos::null);
//      refStrategy->setEnforceOneIrregularity(true);
////      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
////      refStrategy->setMaxElements(maxElements);
////      refStrategy->setMaxP(p_max);
////      refStrategy->setMinH(h_min);
//
//      double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
//      double relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
//
//      int numBCRefinements = 0;
//      int numVlasovElements = form.vlasovMesh()->numActiveElements();
//      int prevNumVlasovElements = numVlasovElements;
//      int numGlobalDofs     = form.vlasovMesh()->globalDofCount();
//      int prevNumGlobalDofs = 0;
////      while ((relErr > bcResolutionTol) && (numVlasovElements < maxElements))
////      {
////        if (rank==0)
////        {
////          std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
////          std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
//////          std::cout << "h_min is " << h_min << std::endl;
////        }
////
////        refStrategy->refine(rank==0);
////
////        // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
////        form.configurationSpaceMesh()->repartitionAndRebuild();
////
////        // project the BC onto the mesh again
////        form.vlasovSolution()->projectOntoMesh(bcMap);
////
////        bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
////        relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
////
////        numGlobalDofs = form.vlasovMesh()->globalDofCount();
////        numVlasovElements = form.vlasovMesh()->numActiveElements();
////
////        if ((numVlasovElements == prevNumVlasovElements) && (numGlobalDofs == prevNumGlobalDofs))
////        {
////          if (rank == 0)
////          {
////            std::cout << "No refinements performed on this pass; exiting refinement loop early.\n";
////          }
////          break;
////        }
////
////        prevNumVlasovElements = numVlasovElements;
////        prevNumGlobalDofs = numGlobalDofs;
////      }
//      if (rank==0)
//      {
//        std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
//        std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
//      }
//    }
    
    std::ostringstream suffix;
    if (useAdaptivity)
    {
      suffix << "_budget" << maxElements;
    }
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;
    suffix << "_gap" << physicalGapWidth;
    
    if (disableElectricField)
    {
      suffix << "_noEField";
    }
    if (disableMagneticField)
    {
      suffix << "_noBField";
    }

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << tElements << "x" << vElements;
    suffix << "_I_IC_" << criticalCurrentRatio;
    
    suffix << "_" << normChoice << "Norm";

    std::string problemPrefix = "Diagnostic";
    
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
      
      resultsFileStream << "_" << tFinal << "_tFinal";
      resultsFileStream << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    auto getSpatialMeshFilePath = [&]() {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << "mesh1D_" << problemPrefix << suffix.str();
      
      spatialMeshInfoPathStringStream << tFinal << "_tFinal";
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
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
    
    const int num1DPointsPoisson = 3; // default is 4
    
    FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.vlasovSolution());
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
            
    auto solutionMoments = form.solutionMoments();
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1); // aka current
    auto velocity = momentum / density;
            
    bool previousSolveUsedGeometricMultigrid = false;
    bool previousSolveUsedDirect = false;
    
    double lastRefinementMaxElementError = 0.0;
        
//    refStrategy->setMinH(h_min);
//    refStrategy->setMaxP(p_max);
//
//    auto solnRep = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolution()));
    
    using std::vector;
    vector<vector<GlobalIndexType>> manualRefinements;
    manualRefinements.push_back({16,17,18,19});
    manualRefinements.push_back({32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59});
    manualRefinements.push_back({64,65,66,67,72,73,74,75,80,81,82,83,88,89,90,91,96,97,98,99,104,105,106,107,112,113,114,115,120,121,122,123,128,129,130,131,136,137,138,139,144,145,146,147,152,153,154,155,160,161,162,163,168,169,170,171,176,177,178,179,184,185,186,187});
    manualRefinements.push_back({224,225,226,227,232,233,234,235,240,241,242,243,248,249,250,251,256,257,258,259,264,265,266,267,272,273,274,275,280,281,282,283,288,289,290,291,296,297,298,299,304,305,306,307,312,313,314,315,320,321,322,323,328,329,330,331,336,337,338,339,344,345,346,347,352,353,354,355,360,361,362,363,368,369,370,371,376,377,378,379,384,385,386,387,392,393,394,395,400,401,402,403,408,409,410,411,416,417,418,419,424,425,426,427,432,433,434,435,440,441,442,443,448,449,450,451,456,457,458,459,464,465,466,467,472,473,474,475,480,481,482,483,488,489,490,491,496,497,498,499,504,505,506,507,512,513,514,515,520,521,522,523,528,529,530,531,536,537,538,539,544,545,546,547,552,553,554,555,560,561,562,563,568,569,570,571,576,577,578,579,584,585,586,587,592,593,594,595,600,601,602,603,608,609,610,611,616,617,618,619,624,625,626,627,632,633,634,635,640,641,642,643,648,649,650,651,656,657,658,659,664,665,666,667,672,673,674,675,680,681,682,683,688,689,690,691,696,697,698,699,704,705,706,707,712,713,714,715,720,721,722,723,728,729,730,731});
    manualRefinements.push_back({880,882,928,930,944,946,1120,1122,1124,1126,1136,1138,1140,1142,1184,1186,1188,1190,1200,1202,1204,1206,2932,2934,2980,2982,2996});
    manualRefinements.push_back({866,948,950,2912,2914,2928,3424,3426,3432,3434,3438,3440,3442,3444,3446,3448,3450,3452,3454,3456,3458,3460,3462,3464,3466,3468,3470,3472,3474,3476,3478,3480,3482,3484,3486,3488,3490,3496,3498,3504,3506,3508,3510,3512,3514,3516,3518,3520,3522,3528,3530,3536,3538,3540,3542,3544,3546,3548,3550,3552,3554,3560,3562,3568,3570,3572,3574,3576,3578,3580,3582,3584,3586,3592,3594,3598,3600,3602,3604,3606,3608,3610,3612,3614,3616,3618,3620,3622,3628,3630,3636,3638,3682,3684,3686,3688,3690,3692,3694});
//    manualRefinements.push_back({3428,3430,3436,3534,3556,3558,3564,3566,3588,3590,3596,3624,3680,3752,3754,3758,3760,3762,3768,3770,3776,3778,3780,3782,3788,3790,3796,3798,3804,3806,3812,3814,3820,3822,3828,3830,3832,3834,3840,3842,3844,3846,3848,3850,3852,3854,3856,3858,3862,3864,3866,3868,3870,3872,3874,3876,3878,3880,3882,3884,3886,3888,3890,3892,3894,3896,3898,3900,3902,3904,3906,3908,3910,3912,3914,3916,3918,3920,3922,3924,3926,3928,3930,3932,3934,3936,3938,3940,3942,3944,3946,3948,3950,3952,3954,3956,3958,3960,3962,3964,3966,3968,3970,3972,3974,3976,3978,3980,3982,3984,3986,3988,3990,3992,3994,3996,3998,4000,4002,4004,4006,4008,4010,4012,4014,4016,4018,4020,4022,4024,4026,4028,4030,4032,4034,4040,4042,4048,4050,4056,4058,4062,4064,4066,4068,4070,4072,4074,4076,4078,4080,4082,4084,4086,4088,4090,4092,4094,4096,4098,4100,4102,4104,4106,4108,4110,4112,4114,4116,4118,4120,4122,4124,4126,4128,4130,4132,4134,4136,4138,4140,4142,4144,4146,4148,4150,4152,4154,4156,4158,4160,4162,4164,4166,4168,4170,4172,4174,4176,4178,4180,4182,4184,4186,4188,4190,4192,4194,4196,4198,4200,4202,4204,4206,4208,4210,4212,4214,4216,4218,4220,4222,4224,4226,4228,4230,4232,4234,4236,4238,4240,4242,4244,4246,4248,4250,4252,4254,4256,4258,4260,4262,4264,4266,4268,4270,4272,4274,4276,4278,4280,4282,4284,4286,4288,4290,4292,4294,4296,4298,4300,4302,4304,4306,4308,4310,4312,4314,4316,4318,4320,4322,4324,4326,4328,4330,4332,4334,4336,4338,4340,4342,4344,4346,4348,4350,4352,4354,4356,4360,4362,4364,4366,4368,4370,4372,4374,4376,4378,4380,4382,4384,4386,4388,4390,4392,4394,4396,4398,4400,4402,4404,4406,4408,4410,4412,4414,4416,4418,4420,4422,4428,4430,4436,4438,4440,4442,4444,4446,4448,4450,4452,4454,4456,4458,4460,4462,4464,4466,4468,4470,4472,4474,4476,4478,4484,4486,4488,4490,4502,4506,4508,4512,4514,4516,4518,4520,4522,4524,4526,4528,4530,4532,4534,4536,4538,4540,4542,4584,4586,4628,4630});
    
    for (auto refinements : manualRefinements)
    {
      form.vlasovMesh()->hRefine(refinements);
      form.vlasovMesh()->enforceOneIrregularity();
      form.configurationSpaceMesh()->repartitionAndRebuild();
      
      bool printToConsole = (rank == 0);
      auto vlasovElementCount = form.vlasovMesh()->numActiveElements();
      auto vlasovDofCount     = form.vlasovMesh()->numGlobalDofs();
      auto configElementCount = form.configurationSpaceMesh()->numActiveElements();
      auto configDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
      if (printToConsole)
      {
        std::cout << "Vlasov mesh has " << vlasovElementCount << " active elements and " << vlasovDofCount << " global degrees of freedom.\n";
        std::cout << "Configuration-space mesh has " << configElementCount << " active elements and " << configDofCount << " global degrees of freedom.\n";
      }
    }
    
//    vector<int> cellIDs {6772,6774,6784,6786};
//
//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(form.vlasovMesh()->globalDofAssignment().get());
//    for (auto cellID : cellIDs)
//    {
//      if (form.vlasovMesh()->myCellsInclude(cellID))
//      {
//        std::cout << "dof indices for cell " << cellID << ":\n";
//        minRule->getGlobalDofIndices(cellID).print(std::cout);
//
//        Camellia::FieldContainer<double> physicalPoint(1,1,3);
//        physicalPoint[0] = 0;
//        physicalPoint[1] = 0.25 - 1./512;
//        physicalPoint[2] = 0.00390625;
//
//        auto basisCache = BasisCache::basisCacheForCell(form.vlasovMesh(), cellID);
//        auto physicalCellNodes = basisCache->getPhysicalCellNodes();
//
//        Camellia::FieldContainer<double> refPoint(1,3);
//        CamelliaCellTools::mapToReferenceFrame(refPoint, physicalPoint, form.vlasovMesh()->getTopology(), cellID, 3);
//
//        cout << "On cell " << cellID << ", reference-space preimage of (" << physicalPoint[0] << "," << physicalPoint[1] << "," << physicalPoint[2] << ") = ";
//        cout << "(" << refPoint[0] << "," << refPoint[1] << "," << refPoint[2] << ").\n";
//
//        auto cellTopo = basisCache->cellTopology();
//        const int sideDim = 2;
//        const int sideIndex = 4;
//        Camellia::FieldContainer<double> refPoint2D(1,2);
//        if (cellID == 6772) { refPoint2D[0] =  1; refPoint2D[1] =  1;}
//        if (cellID == 6774) { refPoint2D[0] = -1; refPoint2D[1] =  1;}
//        if (cellID == 6784) { refPoint2D[0] =  1; refPoint2D[1] = -1;}
//        if (cellID == 6786) { refPoint2D[0] = -1; refPoint2D[1] = -1;}
//        Camellia::FieldContainer<double> refPoint3D(1,3);
//        CamelliaCellTools::mapToReferenceSubcell(refPoint3D, refPoint2D, sideDim, sideIndex, cellTopo);
//        cout << "On cell " << cellID << ", mapped point from side: (" << refPoint3D[0] << "," << refPoint3D[1] << "," << refPoint3D[2] << ").\n";
//      }
//      form.vlasovMesh()->Comm()->Barrier();
//    }
//    if (rank == 0)
//    {
//      form.vlasovMesh()->getTopology()->printAllEntitiesInBaseMeshTopology();
//    }
    form.vlasovMesh()->Comm()->Barrier();
   
//    exit(0); // TODO: delete this line
    // solve block:
    {
      setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
      
//      if (useAdaptivity)
//      {
//        for (int refNumber=0; refNumber<numEnergyRefinements; refNumber++)
//        {
//          form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
//          form.solve();
//
//          if (saveViz)
//          {
//            poissonViz->exportSolution(form.configurationSpaceSolution(),refNumber,num1DPointsPoisson);
//            vlasovViz->exportSolution(form.vlasovSolution(),{energyErrorFunction, cellIDFunction}, {"Vlasov energy error", "Vlasov cell ID"},refNumber);
//          }
//
//          std::vector<double> f_disagreement(numSpecies);
//          for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
//          {
//            double min_f_at_extrema = fAtExtrema[speciesOrdinal]->minimumValue(form.vlasovMesh());
//            double max_f_at_extrema = fAtExtrema[speciesOrdinal]->maximumValue(form.vlasovMesh());
//            f_disagreement[speciesOrdinal] = std::max(std::abs(min_f_at_extrema), std::abs(max_f_at_extrema));
//          }
//
//          double solnNorm = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolution()))->getNorm();
//          if (rank == 0)
//          {
//            std::cout << "Solution, energy norm: " << solnNorm << std::endl;
//          }
//
//          double totalEnergyError = refStrategy->computeTotalEnergyError();
//          if (rank == 0)
//          {
//            std::cout << "Solution's energy norm: " << solnNorm << std::endl;
//            std::cout << "Relative energy error: " << totalEnergyError / solnNorm << std::endl;
//            for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
//            {
//              std::cout << "Species " << speciesOrdinal << ", max disagreement with homogeneous BCs at velocity extrema: " << f_disagreement[speciesOrdinal] << std::endl;
//            }
//          }
//
//          maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
//          performRefinements(lastRefinementMaxElementError, form, refStrategy, unrefinementStrategy, maxElements);
//
//          setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
//        }
//      }
      form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
      form.solve();
      
      if (saveViz)
      {
        poissonViz->exportSolution(form.configurationSpaceSolution(),numEnergyRefinements,num1DPointsPoisson);
        vlasovViz->exportSolution(form.vlasovSolution(),{energyErrorFunction, cellIDFunction}, {"Vlasov energy error", "Vlasov cell ID"},numEnergyRefinements);
      }
      
      std::vector<double> f_disagreement(numSpecies);
      for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
      {
        double min_f_at_extrema = fAtExtrema[speciesOrdinal]->minimumValue(form.vlasovMesh());
        double max_f_at_extrema = fAtExtrema[speciesOrdinal]->maximumValue(form.vlasovMesh());
        f_disagreement[speciesOrdinal] = std::max(std::abs(min_f_at_extrema), std::abs(max_f_at_extrema));
      }
      
      double solnNorm = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolution()))->getNorm();
      double totalEnergyError = refStrategy->computeTotalEnergyError();
      if (rank == 0)
      {
        std::cout << "Solution's energy norm: " << solnNorm << std::endl;
        std::cout << "Relative energy error: " << totalEnergyError / solnNorm << std::endl;
        for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
        {
          std::cout << "Species " << speciesOrdinal << ", max disagreement with homogeneous BCs at velocity extrema: " << f_disagreement[speciesOrdinal] << std::endl;
        }
      }
      
      if (rank == 0)
      {
        std::cout << "Max acceleration is " << maxAcceleration_vx << std::endl;
        //        std::cout << "h_min: " << h_min << std::endl;
      }
      
      {
        // compute, report BC error for inflow:
        const double relErr = inflowErrorOnBoundary(form, f_in, form.tagVlasovSpatialInflow(), p_max);
        if (rank == 0)
        {
          std::cout << "Relative L^2 error in f compared to f_in at inflow: " << relErr << std::endl;
        }
      }
      
      double energyError = -1.;
      if ((refStrategy != Teuchos::null) && (refStrategy->getNumRefinements() > 0))
      {
        energyError = refStrategy->getEnergyError(refStrategy->getNumRefinements() - 1);
      }
    }
    
    auto f_soln = Function::solution(form.f(), form.vlasovSolution());
    
//    for (auto cellID: cellIDs)
//    {
//      if (form.vlasovMesh()->myCellsInclude(cellID))
//      {
//        Camellia::FieldContainer<double> physicalPoint(1,1,3);
//        physicalPoint[0] = 0;
//        physicalPoint[1] = 0.25 - 1./512;
//        physicalPoint[2] = 0.00390625;
//
//        auto basisCache = BasisCache::basisCacheForCell(form.vlasovMesh(), cellID);
//        auto physicalCellNodes = basisCache->getPhysicalCellNodes();
//
//        Camellia::FieldContainer<double> refPoint(1,3);
//        CamelliaCellTools::mapToReferenceFrame(refPoint, physicalPoint, form.vlasovMesh()->getTopology(), cellID, 3);
//
//        basisCache->setRefCellPointsFC(refPoint);
//
//        Camellia::FieldContainer<double> valueAtPoint(1,1);
//
//        f_soln->values(valueAtPoint, basisCache);
//        cout << "On cell " << cellID << ", value at (" << physicalPoint[0] << "," << physicalPoint[1] << "," << physicalPoint[2] << ") = " << valueAtPoint[0] << std::endl;
//
//        extraAccelerationFormula->values(valueAtPoint, basisCache);
//        cout << "On cell " << cellID << ", extra acceleration at (" << physicalPoint[0] << "," << physicalPoint[1] << "," << physicalPoint[2] << ") = " << valueAtPoint[0] << std::endl;
//
//
//        auto dofOrdering = form.vlasovMesh()->getElementType(cellID)->trialOrderPtr;
//        auto basisCoefficients = form.vlasovSolution()->allCoefficientsForCellID(cellID);
//
//        cout << "basisCoefficients:\n" << basisCoefficients;
//
//        cout << "dofOrdering:\n";
//        dofOrdering->print(std::cout);
//
//      }
//    }
        
    FunctionPtr potential = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), form.configurationSpaceSolution());
        
    // output to results file:
    {
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
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
        
        const int numCells = physicalCellIDsVector.size();
        
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
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
          const auto ne          = ne_values[pointEntry.first]        * n0; // TODO: determine if this scaling is correct; we probably need to divide by spatialScaling^3.
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
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
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
