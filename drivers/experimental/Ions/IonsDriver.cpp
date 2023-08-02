//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "ConfigurationSpaceFormulation.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "FixedCountRefinementStrategy.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MaxwellianFunction.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"
#include "VlasovBackwardEulerFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

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

void performRefinements(double & lastRefinementMaxElementError, VlasovBackwardEulerFormulation &form, RefinementStrategyPtr refStrategy, RefinementStrategyPtr unrefinementStrategy,
                        int maxElements)
{
  if (refStrategy == Teuchos::null) return;
  
  const bool doUnrefinements = (maxElements < std::numeric_limits<int>::max());
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
      if (form.gmgSolver() != Teuchos::null)
      {
        form.vlasovSolution()->solve(form.gmgSolver());
      }
      else
      {
        form.vlasovSolution()->solve();
      }
      form.configurationSpaceSolve();
      
      unrefinementStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
      form.configurationSpaceMesh()->repartitionAndRebuild();
      form.reconstructGMGSolver();
      numUnrefinementStepsTaken++;
    }
        
    if (numUnrefinementStepsTaken > 0)
    {
      // update solution to ensure that the errors measured are up to date post-refinement
      form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
      if (form.gmgSolver() != Teuchos::null)
      {
        form.vlasovSolution()->solve(form.gmgSolver());
      }
      else
      {
        form.vlasovSolution()->solve();
      }
      form.configurationSpaceSolve();
    }
  }
  
  // for now, print refinement info to console on rank 0.
  const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
  refStrategy->refineAndUnrefine(false, printToConsole); // false: don't unrefine
  
  // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
  form.configurationSpaceMesh()->repartitionAndRebuild();
  
  // also reconstruct GMG solver, if one is in use:
  form.reconstructGMGSolver();
}

//! Sets a homogeneous BC at the x0 point, and parameter-based BC at the x1 point.  Returns a pointer to the parameter function; the caller may then set it to an appropriate value before (any/each) Poisson solve.
ParameterFunctionPtr setPoissonBC(VlasovBackwardEulerFormulation &form, const double &x0, const double &x1)
{
  // we have periodic BCs; we just need to pin the potential at a point -- we choose the x1 (right) point.
  const int poissonTagLeft  = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0));
  const int poissonTagRight = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x1));
  
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

double inflowErrorOnBoundary(VlasovBackwardEulerFormulation &form, std::vector<FunctionPtr> f_in, const int bcTag)
{
  const std::string & tagName = DIRICHLET_SET_TAG_NAME;
  FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
  
  double errSquared = 0;
  double bcNormSquared = 0;
  for (int speciesOrdinal=0; speciesOrdinal<int(f_in.size()); speciesOrdinal++)
  {
    FunctionPtr f_soln = Function::solution(form.f(speciesOrdinal), form.vlasovSolution());
    FunctionPtr bc_err = (f_in[speciesOrdinal] - f_soln) * bcRestriction;
    
    const int quadratureEnrichment = 4;
    double bc_norm = (f_in[speciesOrdinal] * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
    bcNormSquared += bc_norm * bc_norm;
    double err = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment);
    errSquared += err * err;
  }
  
  return sqrt(errSquared / bcNormSquared);
}

void setUseGMG(VlasovBackwardEulerFormulation &form, int maxDofsForDirectSolve, bool &previousSolveUsedGeometricMultigrid, bool &previousSolveUsedDirect, bool logChangeToConsole)
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
    
    const double permittivity = 8.85418782e-12; // permittivity of free space

    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = false;
    
    bool useAdaptivity = false;
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    
    int xElements = 16;
    int vElements = 20;
    int numUniformRefinements = 2;
    int delta_k   = 2;
    double dt_max = -1.0;
    int H1Order = 3; // this gives f, E (which are in L^2) a polynomial order of 2
    int minSubsteps = 2;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int numTimeSteps = 80;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = false;
    bool saveViz = true;
    std::string outputDir = "/private/viz";
    int solutionOutputFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 80000;
    bool useSerendipityBasis = false;
    bool includeIons = true;
    bool includeElectrons = true;
    bool disableElectricField = false;
    bool disableMagneticField = false;
    double overrideEps0 = -1.; // if set from CLI, will override the computed eps0 value.
    bool useCrankNicolson = false;
    
    double physicalGapWidth = .00216; // [m], from Christenson and Lau (1994)
    
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
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
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
    cmdp.setOption("useCrankNicolson", "useBackwardEuler", &useCrankNicolson);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    // use velocity domain (-3, 6), by default, with the idea that our velocity scaling will place the centers of the Maxwellians around ±1.
    const double DEFAULT_V0     = -3.00;
    const double DEFAULT_VWIDTH =  9.00;
    
    const double v_beam    = 0.00139891 * 299792458;
    double velocityScaling = v_beam; // c times (u0_e as fraction of c) --> u0_e* = 1
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
    
    FunctionPtr extraAcceleration; //
    
    double m_e_scaled;
    double m_ion_scaled;
    
    // determine scalings
    {
      const double e   = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e =  9.10938356e-31;  // m_e, mass of electron in kg
      const double Omega = B * e / m_e; // cyclotron frequency in Hz
      const double OmegaScaling = 1.0 / timeScaling;
      potentialScaling = potentialBC;
      
      const double qScaling = - e / q_e_scaled; // CORRECT charge scaling
//      const double qScaling = - e / q_e_scaled / 10000; // INCORRECT charge scaling, but which lets us solve...
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
      extraAcceleration = B_scaled * Omega_scaled * x;
      
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
    
    if (dt_max < 0)
    {
      dt_max = 1e-8; // experimentally, this appears to be about the biggest (initial) time step we can get away with, for adaptive mesh with backward Euler.  Could be that we could increase this by using Crank-Nicolson.  Could also be we should do something more sophisticated with the test space inner product.  Something to consider: What would a *Newton* iteration look like?  What does the linearized system look like?  Can we specify a test space that reflects that underlying system, rather than one step of the fixed point iteration that we're actually solving??
      // (Here's something very weird: during initial mesh refinement, the error increases with every refinement, seemingly without bound.  Is this a feature of the problem setup, or indicative of a general problem with the error measurement?  Notably, we do *not* generally see this once the actual run starts.)
    }
    if (disableElectricField)
    {
      q_e_scaled = 0.0;
      q_ion_scaled = 0.0;
    }
    
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
    
    if (disableElectricField)
    {
      potentialBC = 0;
    }
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q_values, eps0, TIME_MARCHING);
    if (!disableMagneticField)
    {
      poissonConfig->setExtraAcceleration({extraAcceleration});
    }
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x1, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    
    double dt = dt_max;
    int maxMomentNumber = 1; // we want to compute current for circuit solve.
    
    VlasovBackwardEulerFormulation form(q_values, m_values, speciesLabels, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber, useCrankNicolson);
    
    if (disableElectricField)
    {
      // set the RHS to 0 -- in combination with the zero potential BC we set above, this forces the electric field to be zero.
      // (This needs to be done after the VBEFormulation has been constructed, since it calls poissonConfig->setMoments(), which
      //  is where the RHS for Poisson gets set up.)
      poissonConfig->solution()->setRHS(RHS::rhs());
    }
    
    form.refineUniformly(numUniformRefinements);
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    auto voltageParameter = setPoissonBC(form, x0, x1);
    
    auto moments_k   = form.solutionMomentsPriorTime(); // we need to call update() on this after each time step
    auto moments_kp1 = form.solutionMoments();
        
    // voltageParameter is a "ParameterFunction" -- we need fill it in with an appropriate value
    voltageParameter->setValue(potentialBC / potentialScaling); // 12000 V
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
    
    double sigmaSquared_e = 0.01 * u0_e * u0_e; // let thermal velocity be 10% of drift velocity
    double n = 1.0;
    const double firstMomentOfMaxwellian_e = MaxwellianFunction::momentNormalized1D(u0_e, sigmaSquared_e, 1); // will return u0_e
    const double J_inflow_physical_e = 0.95 * 2.1e5; // Amps // TODO: make the 0.95 weight subject to CLI parameter (0.95 is Case C2 from C & L)
    const double J_inflow_scaled_e   = J_inflow_physical_e / currentDensityScaling;
    const double inflow_weight_e     = J_inflow_scaled_e / firstMomentOfMaxwellian_e;
    if (rank == 0)
    {
      std::cout << "inflow_weight_e: " << inflow_weight_e << std::endl;
    }
    
    FunctionPtr positiveRestriction = Function::heavisideY(0); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
    FunctionPtr fM_u0_e = inflow_weight_e * MaxwellianFunction::maxwellian(1, 1, {u0_e}, sigmaSquared_e, n) * positiveRestriction;
    
    // ions flow in from right
    double u0_ion = -u0_e / sqrt(1836.15); // ratio of electron mass to ion mass (assigning them both the same energy)
    double sigmaSquared_ion = 0.005 * 0.005;
    FunctionPtr negativeRestriction = 1 - Function::heavisideY(0); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
    FunctionPtr fM_u0_ion = MaxwellianFunction::maxwellian(1, 1, {u0_ion}, sigmaSquared_ion, n) * negativeRestriction;
    
    FunctionPtr f_e_in = fM_u0_e;
    FunctionPtr f_i_in = fM_u0_ion;
    
    ErrorIndicatorPtr unrefinementErrorIndicator;
    if (!useEnergyForUnrefinements)
    {
      unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
    }
    double refinementThreshold = 1.1; // > 1 ==> no refinements will take place
    RefinementStrategyPtr unrefinementStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
    
    ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
    const int numElementsToRefine = std::max(int(.05 / 3. * maxElements), 1);
    auto fixedCountRefStrategy = Teuchos::rcp( new FixedCountRefinementStrategy(errorIndicator, numElementsToRefine) );
    
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
    form.setInitialAndInflowConditions(f0, f_in);
    
    double bcResolutionTolerance = 0.01; // 1.0; // 0.01; // DEBUGGING: turn off bc refinements
    for (auto bcTuple : bcFunctionsToResolve)
    {
      using std::get;
      VarPtr var = get<0>(bcTuple);
      FunctionPtr bcFxnExact = get<1>(bcTuple);
      FunctionPtr bcFxnSoln  = get<2>(bcTuple);
      int bcTag = get<3>(bcTuple);
      
      const std::string & tagName = DIRICHLET_SET_TAG_NAME;
      FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
      
      // project onto vlasovSolution:
      std::map<int,FunctionPtr> bcMap;
      bcMap[var->ID()] = bcFxnExact;
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
    
    std::ostringstream suffix;
    if (useAdaptivity)
    {
      suffix << "_budget" << maxElements;
    }
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;
    suffix << "_gap" << physicalGapWidth;

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << vElements;
    if (useCrankNicolson)
    {
      suffix << "_CN";
    }

    std::string problemPrefix = "Ions";
    
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
      const int num1DPointsPoisson = 10; // default is 4
      poissonViz->exportSolution(form.configurationSpaceSolution(),t,num1DPointsPoisson);
      vlasovViz->exportSolution(form.vlasovSolution(),t,num1DPointsPoisson);
    }
    
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
            
    auto solutionMoments = form.solutionMoments();
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1); // aka current
    auto velocity = momentum / density;
            
    bool previousSolveUsedGeometricMultigrid = false;
    bool previousSolveUsedDirect = false;
    
    double lastRefinementMaxElementError = 0.0;
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
      
      form.startTimeStep();
      
      if (useAdaptivity)
      {
        int maxRefinementSteps = 1; // just do 1 pass on every time step after the first
        int minElements = 0;   // don't gate on the element count except for the first time step
        if (timeStepNumber == 1)
        {
          maxRefinementSteps = 15; // a somewhat arbitary number; aim for this to be enough to reach minElements
          minElements = maxElements * .90; // in first time step, refine until we get within 10% of the maximum
          if (rank == 0)
          {
            std::cout << "\n\nAdapting initial mesh to meet budget of " << maxElements << " elements.\n";
          }
        }
        for (int refNumber=0; refNumber<maxRefinementSteps; refNumber++)
        {
          form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
          auto gmgSolver = form.gmgSolver();
          if (gmgSolver != Teuchos::null)
          {
            form.vlasovSolution()->solve(gmgSolver);
          }
          else
          {
            form.vlasovSolution()->solve();
          }
          form.configurationSpaceSolve();
                    
          performRefinements(lastRefinementMaxElementError, form, fixedCountRefStrategy, unrefinementStrategy, maxElements);
          
          setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
          
          if (form.vlasovMesh()->numActiveElements() >= minElements)
          {
            break;
          }
        }
      }
      form.finishTimeStep();
      
      t += dt;
      
//      {
//        // DEBUGGING
//        if ((rank == 0) && (timeStepNumber == 1))
//        {
//          std::cout << "****************************************************************\n";
//          std::cout << "* EXPERIMENT: increasing dt by 2 at each step until dt = 1e-2 *\n";
//          std::cout << "****************************************************************\n";
//        }
//        if ((dt < 1e-2) && ((timeStepNumber % 10) == 1))
//        {
//          dt *= 2;
//          form.setTimeStep(dt);
//        }
//      }
      
      {
        // compute, report BC error for inflow:
        const double relErr = inflowErrorOnBoundary(form, f_in, form.tagVlasovSpatialInflow());
        if (rank == 0)
        {
          std::cout << "Relative L^2 error in f compared to f_in at inflow: " << relErr << std::endl;
        }
      }
      
      moments_kp1->update();
      if (rank == 0) std::cout << "moments_kp1 updated.\n";
      moments_k->update();
      if (rank == 0) std::cout << "moments_k   updated.\n";
      
      double energyError = -1.;
      if ((fixedCountRefStrategy != Teuchos::null) && (fixedCountRefStrategy->getNumRefinements() > 0))
      {
        energyError = fixedCountRefStrategy->getEnergyError(fixedCountRefStrategy->getNumRefinements() - 1);
      }
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          const int num1DPointsPoisson = 10; // default is 4
          vlasovViz->exportSolution(form.vlasovSolution(),t,num1DPointsPoisson);
          poissonViz->exportSolution(form.configurationSpaceSolution(), {density,velocity}, {"n_e","v"},t,num1DPointsPoisson);
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
