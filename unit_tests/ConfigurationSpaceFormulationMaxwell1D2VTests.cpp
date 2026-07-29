//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ConfigurationSpaceFormulationMaxwell1D2VTests
//  Camellia
//
//  Created by Nate Roberts with Sandia AI ChatGPT 5.2 on 3/5/26.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "ConfigurationSpaceFormulationMaxwell1D2V.h"

#include "Camellia.h"

#include "BC.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "RHS.h"
#include "Solution.h"
#include "SpatialFilter.h"

using namespace Camellia;

namespace {

static const int TAG_XMIN = 11;
static const int TAG_XMAX = 12;
static const int TAG_TMIN = 13;

struct MaxwellMS
{
  // exact fields on configuration space (x,t):
  FunctionPtr ExExact  = Function::zero();
  FunctionPtr ByExact  = Function::zero();
  FunctionPtr phiExact = Function::zero();
  FunctionPtr AExact   = Function::zero();

  // RHS terms in the formulation (already divided by eps0, matching code RHS usage)
  FunctionPtr rhoOverEps0 = Function::zero();
  FunctionPtr JxOverEps0  = Function::zero();
  FunctionPtr JzOverEps0  = Function::zero();

  // BC choices
  bool imposePhiHatOnSpatialBoundary = false;
  FunctionPtr phiHatBC = Function::zero();

  bool imposeAHatOnSpatialBoundary = false;
  FunctionPtr AHatBC = Function::zero();

  bool imposeExHatOnInitialTime = false;
  FunctionPtr ExHatInitBC = Function::zero();
  
  bool imposeExHatOnSpatialBoundary = false;
  FunctionPtr ExHatBC = Function::zero();
  
  bool imposeByHatOnSpatialBoundary = false;
  FunctionPtr ByHatBC = Function::zero();
};

struct MaxwellErrors
{
  double errEx  = 0;
  double errBy  = 0;
  double errPhi = 0;
  double errA   = 0;
};

void addBoundaryTag(MeshTopologyPtr meshTopo, int tag,
                    SpatialFilterPtr filter, int sideDim)
{
  auto sideSet = meshTopo->createEntitySet();
  std::vector<IndexType> sideIndices = meshTopo->getBoundarySidesThatMatch(filter);
  for (auto sideIndex : sideIndices)
  {
    sideSet->addEntity(sideDim, sideIndex);
  }
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, tag, sideSet);
}

void setConfigBC(SolutionPtr soln, int tag, VarPtr var, FunctionPtr value)
{
  if (soln->bc() == Teuchos::null)
  {
    soln->setBC(BC::bc());
  }
  soln->bc()->addDirichlet(var, tag, value);
}

double l2Error(FunctionPtr computed, FunctionPtr exact, MeshPtr mesh)
{
  FunctionPtr diff = computed - exact;
  return diff->l2norm(mesh);
}

MaxwellErrors computeMaxwellManufacturedErrors(const MaxwellMS &ms,
                                               int H1Order, int delta_k,
                                               int numXElems, int numTElems,
                                               double x0, double x1,
                                               double t0, double t1,
                                               Teuchos::FancyOStream &out, bool &success)
{
  if (ms.imposePhiHatOnSpatialBoundary && ms.imposeExHatOnSpatialBoundary)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
      "Overconstrained: impose either phi_hat or Ex_hat on spatial boundary, not both.");
  
  if (ms.imposeAHatOnSpatialBoundary && ms.imposeByHatOnSpatialBoundary)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
      "Overconstrained: impose either A_hat or By_hat on spatial boundary, not both.");
  
  // --- build 1D spatial topology, then extrude in time to get configuration-space (space-time) topology
  std::vector<double> xDimensions = {x1 - x0};
  std::vector<int>    xElemCounts = {numXElems};
  std::vector<double> xOrigin     = {x0};

  MeshTopologyPtr spatialTopo = MeshFactory::rectilinearMeshTopology(xDimensions, xElemCounts, xOrigin);
  std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(t0, t1 - t0, numTElems);

  MeshTopologyPtr configTopo = MeshFactory::extrudedMeshTopology(spatialTopo, {temporalVertices});

  // --- create Maxwell configuration formulation
  // q is unused for RHS-for-testing; eps0 appears only through your manufacturing (rhoOverEps0 etc.)
  double q = 1.0, eps0 = 1.0;
  auto maxwell = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, eps0));
  maxwell->initializeMesh(configTopo, H1Order, delta_k);

  MeshPtr configMesh = maxwell->configurationSpaceMesh();
  SolutionPtr soln   = maxwell->solution();

  // --- create boundary tags (x=x0, x=x1, t=t0)
  // For configuration-space mesh, spaceDim=1 => coords are (x,t); time is component 1 (0-based).
  const int xComp = 0;
  const int tComp = 1;
  const int sideDim = 1; // sides of 2D (x,t) cells are 1D

  SpatialFilterPtr xMinFilter = SpatialFilter::matchingComponent(xComp, x0);
  SpatialFilterPtr xMaxFilter = SpatialFilter::matchingComponent(xComp, x1);
  SpatialFilterPtr tMinFilter = SpatialFilter::matchingComponent(tComp, t0);

  addBoundaryTag(configTopo, TAG_XMIN, xMinFilter, sideDim);
  addBoundaryTag(configTopo, TAG_XMAX, xMaxFilter, sideDim);
  addBoundaryTag(configTopo, TAG_TMIN, tMinFilter, sideDim);

  // --- apply BCs (Dirichlet on traces/fluxes)
  // phi hat is VE_hat()
  if (ms.imposePhiHatOnSpatialBoundary)
  {
    setConfigBC(soln, TAG_XMIN, maxwell->VE_hat(), ms.phiHatBC);
    setConfigBC(soln, TAG_XMAX, maxwell->VE_hat(), ms.phiHatBC);
  }
  // A hat: in your Maxwell1D2V implementation only comp 1 is non-null (represents A_z in writeup)
  if (ms.imposeAHatOnSpatialBoundary)
  {
    VarPtr Ahat = maxwell->A_hat(1);
    TEUCHOS_TEST_FOR_EXCEPTION(Ahat == Teuchos::null, std::invalid_argument, "A_hat(1) returned null");
    setConfigBC(soln, TAG_XMIN, Ahat, ms.AHatBC);
    setConfigBC(soln, TAG_XMAX, Ahat, ms.AHatBC);
  }
  // Ex hat initial time:
  if (ms.imposeExHatOnInitialTime)
  {
    VarPtr ExHat = maxwell->E_hat(0);
    TEUCHOS_TEST_FOR_EXCEPTION(ExHat == Teuchos::null, std::invalid_argument, "E_hat(0) returned null");
    setConfigBC(soln, TAG_TMIN, ExHat, ms.ExHatInitBC);
  }
  if (ms.imposeExHatOnSpatialBoundary)
  {
    VarPtr ExHat = maxwell->E_hat(0);
    setConfigBC(soln, TAG_XMIN, ExHat, ms.ExHatBC);
    setConfigBC(soln, TAG_XMAX, ExHat, ms.ExHatBC);
  }
  if (ms.imposeByHatOnSpatialBoundary)
  {
    VarPtr ByHat = maxwell->B_hat(1);
    setConfigBC(soln, TAG_XMIN, ByHat, ms.ByHatBC);
    setConfigBC(soln, TAG_XMAX, ByHat, ms.ByHatBC);
  }

  // --- RHS
  maxwell->setRHSForTesting(ms.rhoOverEps0, ms.JxOverEps0, ms.JzOverEps0);

  // --- solve
  maxwell->solve(/*updateMoments=*/false);

  // --- build computed functions
  auto solnWeak = Teuchos::rcp(soln.get(), false);

  FunctionPtr ExComp  = Function::solution(maxwell->E_fieldVar(0), solnWeak);
  FunctionPtr ByComp  = Function::solution(maxwell->B_fieldVar(1), solnWeak);
  FunctionPtr phiComp = Function::solution(maxwell->VE_fieldVar(), solnWeak);
  FunctionPtr AComp   = Function::solution(maxwell->A_fieldVar(1), solnWeak);

  // --- compute L2 errors
  double errEx  = l2Error(ExComp,  ms.ExExact,  configMesh);
  double errBy  = l2Error(ByComp,  ms.ByExact,  configMesh);
  double errPhi = l2Error(phiComp, ms.phiExact, configMesh);
  double errA   = l2Error(AComp,   ms.AExact,   configMesh);

  out << "L2 errors: "
      << " Ex="  << errEx
      << " By="  << errBy
      << " phi=" << errPhi
      << " A="   << errA
      << "\n";

  MaxwellErrors errors;
  errors.errEx  = errEx;
  errors.errBy  = errBy;
  errors.errPhi = errPhi;
  errors.errA   = errA;
  return errors;
}

double maxwellDiscreteConsistencyEnergy(const MaxwellMS &ms,
                                        int H1Order, int delta_k,
                                        int numXElems, int numTElems,
                                        double x0, double x1,
                                        double t0, double t1)
{
  std::vector<double> xDimensions = {x1 - x0};
  std::vector<int>    xElemCounts = {numXElems};
  std::vector<double> xOrigin     = {x0};

  MeshTopologyPtr spatialTopo = MeshFactory::rectilinearMeshTopology(xDimensions, xElemCounts, xOrigin);
  std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(t0, t1 - t0, numTElems);

  MeshTopologyPtr configTopo = MeshFactory::extrudedMeshTopology(spatialTopo, {temporalVertices});

  double q = 1.0, eps0 = 1.0;
  auto maxwell = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, eps0));
  maxwell->initializeMesh(configTopo, H1Order, delta_k);

  SolutionPtr soln = maxwell->solution();

  maxwell->setRHSForTesting(ms.rhoOverEps0, ms.JxOverEps0, ms.JzOverEps0);

  std::map<int,FunctionPtr> proj;
  proj[maxwell->E_fieldVar(0)->ID()] = ms.ExExact;
  proj[maxwell->B_fieldVar(1)->ID()] = ms.ByExact;
  proj[maxwell->VE_fieldVar()->ID()] = ms.phiExact;
  proj[maxwell->A_fieldVar(1)->ID()] = ms.AExact;

  proj[maxwell->E_hat(0)->ID()] = ms.ExExact;
  proj[maxwell->B_hat(1)->ID()] = ms.ByExact;
  proj[maxwell->VE_hat()->ID()] = ms.phiExact;
  proj[maxwell->A_hat(1)->ID()] = ms.AExact;

  soln->projectOntoMesh(proj);
  soln->clearComputedResiduals();
  return soln->energyErrorTotal();
}

void testMaxwellManufactured(const MaxwellMS &ms,
                             int H1Order, int delta_k,
                             int numXElems, int numTElems,
                             double x0, double x1,
                             double t0, double t1,
                             double tolL2,
                             Teuchos::FancyOStream &out, bool &success)
{
  MaxwellErrors e = computeMaxwellManufacturedErrors(ms, H1Order, delta_k, numXElems, numTElems,
                                                     x0, x1, t0, t1,
                                                     out, success);
  TEST_COMPARE(e.errEx,  <, tolL2);
  TEST_COMPARE(e.errBy,  <, tolL2);
  TEST_COMPARE(e.errPhi, <, tolL2);
  TEST_COMPARE(e.errA,   <, tolL2);
}

void testMaxwellDiscreteConsistency(const MaxwellMS &ms,
                                   int H1Order, int delta_k,
                                   int numXElems, int numTElems,
                                   double x0, double x1, bool useGradedSpatialMesh,
                                   double t0, double t1,
                                   double tolEnergy,
                                   Teuchos::FancyOStream &out, bool &success)
{
  // build configTopo, make maxwell formulation as in computeMaxwellManufacturedErrors()
  
  // --- build 1D spatial topology, then extrude in time to get configuration-space (space-time) topology
  std::vector<double> xDimensions = {x1 - x0};
  std::vector<int>    xElemCounts = {numXElems};
  std::vector<double> xOrigin     = {x0};

  MeshTopologyPtr spatialTopo = MeshFactory::rectilinearMeshTopology(xDimensions, xElemCounts, xOrigin);
  std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(t0, t1 - t0, numTElems);

  MeshTopologyPtr configTopo = MeshFactory::extrudedMeshTopology(spatialTopo, {temporalVertices});

  // --- create Maxwell configuration formulation
  // q is unused for RHS-for-testing; eps0 appears only through your manufacturing (rhoOverEps0 etc.)
  double q = 1.0, eps0 = 1.0;
  auto maxwell = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, eps0));
  maxwell->initializeMesh(configTopo, H1Order, delta_k);

  MeshPtr configMesh = maxwell->configurationSpaceMesh();
  SolutionPtr soln   = maxwell->solution();

  // --- RHS
  maxwell->setRHSForTesting(ms.rhoOverEps0, ms.JxOverEps0, ms.JzOverEps0);
  
  // project onto all trial vars:
  std::map<int,FunctionPtr> proj;
  proj[maxwell->E_fieldVar(0)->ID()] = ms.ExExact;
  proj[maxwell->B_fieldVar(1)->ID()] = ms.ByExact;
  proj[maxwell->VE_fieldVar()->ID()] = ms.phiExact;
  proj[maxwell->A_fieldVar(1)->ID()] = ms.AExact;

  proj[maxwell->E_hat(0)->ID()] = ms.ExExact;   // WARNING: only correct if E_hat represents the trace value, not normal flux
  proj[maxwell->B_hat(1)->ID()] = ms.ByExact;
  proj[maxwell->VE_hat()->ID()] = ms.phiExact;
  proj[maxwell->A_hat(1)->ID()] = ms.AExact;

  soln->projectOntoMesh(proj);

  soln->clearComputedResiduals();
  double energyError = soln->energyErrorTotal();
  out << "energyError = " << energyError << "\n";
  TEST_COMPARE(energyError, <, tolEnergy);
}

// ------------------- concrete manufactured solutions -------------------

MaxwellMS ms_zero()
{
  MaxwellMS ms;
  ms.imposePhiHatOnSpatialBoundary = true;
  ms.phiHatBC = Function::zero();
  ms.imposeAHatOnSpatialBoundary = true;
  ms.AHatBC = Function::zero();
  ms.imposeExHatOnInitialTime = true;
  ms.ExHatInitBC = Function::zero();
  return ms;
}

MaxwellMS ms_gauss_only(double x0, double x1)
{
  (void)x0; (void)x1;
  auto x = Function::xn(1);
  double pi = M_PI;

  MaxwellMS ms;

  ms.phiExact = sin(pi * x);
  ms.ExExact  = -pi * cos(pi * x);

  ms.AExact   = Function::zero();
  ms.ByExact  = Function::zero();

  ms.rhoOverEps0 = pi * pi * sin(pi * x);
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();

  ms.imposePhiHatOnSpatialBoundary = true;
  ms.phiHatBC = ms.phiExact; // Dirichlet phi hat
  
  return ms;
}

MaxwellMS ms_magnetic_only(double x0, double x1)
{
  (void)x0; (void)x1;
  auto x = Function::xn(1);
  double pi = M_PI;

  MaxwellMS ms;
  ms.AExact  = sin(pi * x);
  ms.ByExact = -pi * cos(pi * x);

  ms.phiExact = Function::zero();
  ms.ExExact  = Function::zero();

  ms.JzOverEps0  = pi * pi * sin(pi * x);
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();

  ms.imposeAHatOnSpatialBoundary = true;
  ms.AHatBC = ms.AExact;
  return ms;
}

MaxwellMS ms_poly_mixed_nonzeroAll(double t0)
{
  MaxwellMS ms;
  auto x = Function::xn(1);
  auto t = Function::yn(1);

  // Exact fields
  ms.ExExact  = x + t;
  ms.phiExact = -0.5 * x * x - x * t;

  ms.ByExact = x + t;
  ms.AExact  = -0.5 * x * x - x * t;

  // RHS (already divided by eps0)
  ms.rhoOverEps0 = Function::constant(1.0);
  ms.JxOverEps0  = Function::constant(-1.0);
  ms.JzOverEps0  = Function::constant(1.0);

  // Minimal BCs:
  // - pin phi_hat and A_hat on spatial boundary
  // - pin Ex_hat at initial time (anchors the time equation)
  ms.imposePhiHatOnSpatialBoundary = true;
  ms.phiHatBC = ms.phiExact;

  ms.imposeAHatOnSpatialBoundary = true;
  ms.AHatBC = ms.AExact;

  ms.imposeExHatOnInitialTime = true;
  ms.ExHatInitBC = Function::constant(t0); // Ex(x,t0)=x+t0; but Ex_hat is a trace var; if you want exact, set it to x+t0
  // Better: exact initial-time Ex_hat as a function of x:
  ms.ExHatInitBC = x + Function::constant(t0);

  return ms;
}

MaxwellMS ms_time_only(double omega)
{
  auto x = Function::xn(1);
  auto t = Function::yn(1);
  MaxwellMS ms;

  ms.ExExact  = sin(omega * t);
  ms.phiExact = -x * sin(omega * t);

  ms.ByExact = Function::zero();
  ms.AExact  = Function::zero();

  ms.JxOverEps0  = -omega * cos(omega * t);
  ms.rhoOverEps0 = Function::zero();
  ms.JzOverEps0  = Function::zero();

  // Need phi_hat BCs now to pin phi (otherwise nullspace / drift)
  ms.imposePhiHatOnSpatialBoundary = true;
  ms.phiHatBC = ms.phiExact;

  // Still good to anchor the time equation:
  ms.imposeExHatOnInitialTime = true;
  ms.ExHatInitBC = Function::zero();
  return ms;
}

MaxwellMS ms_poly_gauss_quadratic_nonzeroRho()
{
  MaxwellMS ms;
  auto x = Function::xn(1);

  ms.ExExact  = x;
  ms.phiExact = -0.5 * x * x;

  ms.ByExact  = Function::zero();
  ms.AExact   = Function::zero();

  ms.rhoOverEps0 = Function::constant(1.0);
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();

  ms.imposePhiHatOnSpatialBoundary = true;
  ms.phiHatBC = ms.phiExact;

  return ms;
}

MaxwellMS ms_poly_magnetic_quadratic_nonzeroJz()
{
  MaxwellMS ms;
  auto x = Function::xn(1);

  ms.ByExact = x;
  ms.AExact  = -0.5 * x * x;

  ms.ExExact  = Function::zero();
  ms.phiExact = Function::zero();

  ms.JzOverEps0  = Function::constant(1.0);
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();

  ms.imposeAHatOnSpatialBoundary = true;
  ms.AHatBC = ms.AExact;

  return ms;
}

MaxwellMS ms_poly_time_linear_nonzeroJx(double t0)
{
  MaxwellMS ms;
  auto x = Function::xn(1);
  auto t = Function::yn(1);

  ms.ExExact  = t;
  ms.phiExact = -x * t;

  ms.ByExact = Function::zero();
  ms.AExact  = Function::zero();

  ms.JxOverEps0  = Function::constant(-1.0);
  ms.rhoOverEps0 = Function::zero();
  ms.JzOverEps0  = Function::zero();

  ms.imposePhiHatOnSpatialBoundary = true;
  ms.phiHatBC = ms.phiExact;

  ms.imposeExHatOnInitialTime = true;
  ms.ExHatInitBC = Function::constant(t0); // E(t0)=t0

  return ms;
}

// ------------------- unit tests -------------------

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyGaussLinear)
{
  MaxwellMS ms;
  auto x = Function::xn(1);
  ms.phiExact = x;
  ms.ExExact  = Function::constant(-1.0);
  ms.AExact   = Function::zero();
  ms.ByExact  = Function::zero();
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();
  
  const bool useGradedSpatialMesh = false;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/2, /*delta_k=*/1,
                                 /*numXElems=*/2, /*numTElems=*/1,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyGaussLinear_GradedSpatialMesh)
{
  MaxwellMS ms;
  auto x = Function::xn(1);
  ms.phiExact = x;
  ms.ExExact  = Function::constant(-1.0);
  ms.AExact   = Function::zero();
  ms.ByExact  = Function::zero();
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();
  
  const bool useGradedSpatialMesh = true;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/2, /*delta_k=*/1,
                                 /*numXElems=*/4, /*numTElems=*/1,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyGaussQuadratic_NonzeroRho)
{
  auto ms = ms_poly_gauss_quadratic_nonzeroRho();

  const bool useGradedSpatialMesh = false;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/3, /*delta_k=*/1,
                                 /*numXElems=*/2, /*numTElems=*/1,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyGaussQuadratic_NonzeroRho_GradedSpatialMesh)
{
  auto ms = ms_poly_gauss_quadratic_nonzeroRho();

  const bool useGradedSpatialMesh = true;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/3, /*delta_k=*/1,
                                 /*numXElems=*/4, /*numTElems=*/1,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyMagneticLinear)
{
  MaxwellMS ms;
  auto x = Function::xn(1);
  ms.AExact   = x;
  ms.ByExact  = Function::constant(-1.0);
  ms.ExExact  = Function::zero();
  ms.phiExact = Function::zero();
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();

  const bool useGradedSpatialMesh = false;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/2, /*delta_k=*/1,
                                 /*numXElems=*/2, /*numTElems=*/1,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyMixed_JxSignMatters)
{
  auto msExpected = ms_poly_mixed_nonzeroAll(/*t0=*/0.0);
  auto msOpposite = msExpected;
  msOpposite.JxOverEps0 = -msOpposite.JxOverEps0;

  const double energyExpected =
    maxwellDiscreteConsistencyEnergy(msExpected, 3, 1, 2, 2, -1.0, 1.0, 0.0, 1.0);
  const double energyOpposite =
    maxwellDiscreteConsistencyEnergy(msOpposite, 3, 1, 2, 2, -1.0, 1.0, 0.0, 1.0);

  out << "energyExpected = " << energyExpected << "\n";
  out << "energyOpposite = " << energyOpposite << "\n";

  TEST_COMPARE(energyExpected, <, 1e-12);
  TEST_COMPARE(energyOpposite, >, 1e-8);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyTimeLinear)
{
  // E=t, phi=-x t, Jx=1
  auto ms = ms_poly_time_linear_nonzeroJx(/*t0=*/0.0);

  const bool useGradedSpatialMesh = false;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/2, /*delta_k=*/1,
                                 /*numXElems=*/2, /*numTElems=*/2,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyTimeLinear_JxSignMatters)
{
  auto msExpected = ms_poly_time_linear_nonzeroJx(/*t0=*/0.0);
  auto msOpposite = msExpected;
  msOpposite.JxOverEps0 = -msOpposite.JxOverEps0;

  const int H1Order = 2;
  const int delta_k = 1;
  const int numXElems = 2;
  const int numTElems = 2;
  const double x0 = -1.0, x1 = 1.0;
  const double t0 = 0.0,  t1 = 1.0;

  const double energyExpected =
    maxwellDiscreteConsistencyEnergy(msExpected,
                                     H1Order, delta_k,
                                     numXElems, numTElems,
                                     x0, x1, t0, t1);

  const double energyOpposite =
    maxwellDiscreteConsistencyEnergy(msOpposite,
                                     H1Order, delta_k,
                                     numXElems, numTElems,
                                     x0, x1, t0, t1);

  out << "energyExpected = " << energyExpected << "\n";
  out << "energyOpposite = " << energyOpposite << "\n";

  TEST_COMPARE(energyExpected, <, 1e-12);
  TEST_COMPARE(energyOpposite, >, 1e-8);
  TEST_COMPARE(energyOpposite, >, 1e4 * energyExpected);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyMagneticQuadratic_NonzeroJz)
{
  auto ms = ms_poly_magnetic_quadratic_nonzeroJz();

  const bool useGradedSpatialMesh = false;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/3, /*delta_k=*/1,
                                 /*numXElems=*/2, /*numTElems=*/1,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, DiscreteConsistency_PolyMixed_NonzeroAll)
{
  auto ms = ms_poly_mixed_nonzeroAll(/*t0=*/0.0);

  const bool useGradedSpatialMesh = false;
  
  testMaxwellDiscreteConsistency(ms,
                                 /*H1Order=*/3, /*delta_k=*/1,
                                 /*numXElems=*/2, /*numTElems=*/2,
                                 -1.0, 1.0, useGradedSpatialMesh,
                                 0.0, 1.0,
                                 /*tolEnergy=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Patch_Zero)
{
  testMaxwellManufactured(ms_zero(),
                          /*H1Order=*/2, /*delta_k=*/1,
                          /*numXElems=*/2, /*numTElems=*/1,
                          /*x0=*/-1.0, /*x1=*/1.0,
                          /*t0=*/0.0,  /*t1=*/1.0,
                          /*tolL2=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Manufactured_GaussOnly_Converges_h)
{
  auto ms = ms_gauss_only(-1.0, 1.0);

  const int H1Order = 2, delta_k = 1;
  const double x0=-1.0, x1=1.0, t0=0.0, t1=1.0;

  auto e1 = computeMaxwellManufacturedErrors(ms, H1Order, delta_k,
                                            /*numXElems=*/4, /*numTElems=*/1,
                                            x0,x1,t0,t1,out,success);
  auto e2 = computeMaxwellManufacturedErrors(ms, H1Order, delta_k,
                                            /*numXElems=*/8, /*numTElems=*/1,
                                            x0,x1,t0,t1,out,success);

  out << "GaussOnly: errEx coarse=" << e1.errEx << " fine=" << e2.errEx << "\n";
  out << "GaussOnly: errPhi coarse=" << e1.errPhi << " fine=" << e2.errPhi << "\n";

  // robust “must decrease” thresholds (tune if needed):
  TEST_COMPARE(e2.errEx,  <, 0.8 * e1.errEx);
  TEST_COMPARE(e2.errPhi, <, 0.8 * e1.errPhi);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Manufactured_MagneticOnly_Converges_h)
{
  auto ms = ms_magnetic_only(-1.0, 1.0);

  const int H1Order = 2, delta_k = 1;
  const double x0=-1.0, x1=1.0, t0=0.0, t1=1.0;

  auto e1 = computeMaxwellManufacturedErrors(ms, H1Order, delta_k,
                                            /*numXElems=*/4, /*numTElems=*/1,
                                            x0,x1,t0,t1,out,success);
  auto e2 = computeMaxwellManufacturedErrors(ms, H1Order, delta_k,
                                            /*numXElems=*/8, /*numTElems=*/1,
                                            x0,x1,t0,t1,out,success);

  out << "MagneticOnly: errBy coarse=" << e1.errBy << " fine=" << e2.errBy << "\n";
  out << "MagneticOnly: errA  coarse=" << e1.errA  << " fine=" << e2.errA  << "\n";

  TEST_COMPARE(e2.errBy, <, 0.8 * e1.errBy);
  TEST_COMPARE(e2.errA,  <, 0.8 * e1.errA);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Manufactured_TimeOnly_Converges_dt)
{
  auto ms = ms_time_only(/*omega=*/2.0);

  const int H1Order = 2, delta_k = 1;
  const double x0=-1.0, x1=1.0, t0=0.0, t1=1.0;

  auto e1 = computeMaxwellManufacturedErrors(ms, H1Order, delta_k,
                                            /*numXElems=*/2, /*numTElems=*/4,
                                            x0,x1,t0,t1,out,success);
  auto e2 = computeMaxwellManufacturedErrors(ms, H1Order, delta_k,
                                            /*numXElems=*/2, /*numTElems=*/8,
                                            x0,x1,t0,t1,out,success);

  out << "TimeOnly: errEx coarse=" << e1.errEx << " fine=" << e2.errEx << "\n";
  out << "TimeOnly: errPhi coarse=" << e1.errPhi << " fine=" << e2.errPhi << "\n";

  TEST_COMPARE(e2.errEx,  <, 0.8 * e1.errEx);
  TEST_COMPARE(e2.errPhi, <, 0.8 * e1.errPhi);
}

// poly-Gauss: phi(x)=x, Ex=-1, rho=0
TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Poly_GaussLinear)
{
  // Build ms:
  MaxwellMS ms;
  auto x = Function::xn(1);
  // exact
  ms.phiExact = x;
  ms.ExExact  = Function::constant(-1);
  ms.AExact   = Function::zero();
  ms.ByExact  = Function::zero();
  // RHS: no charge, no currents
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();
  // BCs: pin phi-hat at x=±1
  ms.imposePhiHatOnSpatialBoundary   = true;
  ms.phiHatBC                       = ms.phiExact;

  // Solve on one element in x, one in t:
  testMaxwellManufactured(ms,
    /*H1Order=*/2, /*delta_k=*/0,
    /*numXElems=*/1, /*numTElems=*/1,
    /*x0=*/-1.0, /*x1=*/1.0,
    /*t0=*/ 0.0, /*t1=*/1.0,
    /*tolL2=*/1e-12, out, success);
}

// poly-magnetic: A(x)=x, By=-1, Jz=0
TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Poly_MagneticLinear)
{
  MaxwellMS ms;
  auto x = Function::xn(1);
  ms.AExact   = x;
  ms.ByExact  = Function::constant(-1.0);
  ms.phiExact = Function::zero();
  ms.ExExact  = Function::zero();
  ms.rhoOverEps0 = Function::zero();
  ms.JxOverEps0  = Function::zero();
  ms.JzOverEps0  = Function::zero();
  // BCs: pin A-hat at x=±1
  ms.imposeAHatOnSpatialBoundary   = true;
  ms.AHatBC                        = ms.AExact;

  testMaxwellManufactured(ms,
    /*H1Order=*/2, /*delta_k=*/0,
    /*numXElems=*/1, /*numTElems=*/1,
    /*x0=*/-1.0, /*x1=*/1.0,
    /*t0=*/ 0.0, /*t1=*/1.0,
    /*tolL2=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Poly_GaussQuadratic_NonzeroRho)
{
  auto ms = ms_poly_gauss_quadratic_nonzeroRho();

  testMaxwellManufactured(ms,
                          /*H1Order=*/3, /*delta_k=*/0,
                          /*numXElems=*/1, /*numTElems=*/1,
                          -1.0, 1.0, 0.0, 1.0,
                          /*tolL2=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Poly_MagneticQuadratic_NonzeroJz)
{
  auto ms = ms_poly_magnetic_quadratic_nonzeroJz();

  testMaxwellManufactured(ms,
                          /*H1Order=*/3, /*delta_k=*/0,
                          /*numXElems=*/1, /*numTElems=*/1,
                          -1.0, 1.0, 0.0, 1.0,
                          /*tolL2=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Poly_TimeLinear_NonzeroJx)
{
  auto ms = ms_poly_time_linear_nonzeroJx(/*t0=*/0.0);

  testMaxwellManufactured(ms,
                          /*H1Order=*/2, /*delta_k=*/0,
                          /*numXElems=*/1, /*numTElems=*/1,
                          -1.0, 1.0, 0.0, 1.0,
                          /*tolL2=*/1e-12, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationMaxwell1D2V, Poly_Mixed_NonzeroAll)
{
  auto ms = ms_poly_mixed_nonzeroAll(/*t0=*/0.0);

  testMaxwellManufactured(ms,
                          /*H1Order=*/3, /*delta_k=*/0,
                          /*numXElems=*/1, /*numTElems=*/1,
                          -1.0, 1.0, 0.0, 1.0,
                          /*tolL2=*/1e-12, out, success);
}

} // namespace

