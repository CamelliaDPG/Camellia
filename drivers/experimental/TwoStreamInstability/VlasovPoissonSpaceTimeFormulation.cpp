//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  VlasovPoissonSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/3/2020.
//
//

#include "VlasovPoissonSpaceTimeFormulation.h"
#include "CamelliaMemoryTools.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "MonomialFunctions.h"
#include "ParameterFunction.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

#include "CamelliaDataHelpers.h"

using namespace Camellia;

const string VlasovPoissonSpaceTimeFormulation::S_F = "f";
const string VlasovPoissonSpaceTimeFormulation::S_E = "E";
const string VlasovPoissonSpaceTimeFormulation::S_VE = "V_E";

const string VlasovPoissonSpaceTimeFormulation::S_F_HAT = "\\widehat{f}";
const string VlasovPoissonSpaceTimeFormulation::S_T_N_HAT = "\\widehat{t}_n";
const string VlasovPoissonSpaceTimeFormulation::S_E_HAT = "\\widehat{E}";
const string VlasovPoissonSpaceTimeFormulation::S_VE_HAT = "\\widehat{V}_E";

const string VlasovPoissonSpaceTimeFormulation::S_W = "w";

// Poisson test functions
const string VlasovPoissonSpaceTimeFormulation::S_V_POISSON = "v_{\rm Poisson}";
const string VlasovPoissonSpaceTimeFormulation::S_TAU_POISSON = "\\tau_{\rm Poisson}";

VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulation(VlasovPoissonSpaceTimeFormulationChoice formulationChoice,
                                                                     double q, double eps0, double m,
                                                                     const double x0, const double xWidth, const int xElementsCoarse,
                                                                     const double t0, const double tFinal, const int tElementsCoarse,
                                                                     const double v0, const double vWidth, const int vElementsCoarse,
                                                                     const int H1Order, const int delta_k,
                                                                     const bool usePeriodicBCs, const bool enforceQuasiNeutrality,
                                                                     const int vlasovTraceEnrichment,
                                                                     const double firstTimeDivisionSize)
:
_formulationChoice(formulationChoice),
_usePeriodicBCs(usePeriodicBCs),
_enforceQuasiNeutrality(enforceQuasiNeutrality)
{
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At start of VlasovPoissonSpaceTimeFormulation construction, memory HWM:      " << memoryUsageInGigabytes << " GB.\n";
//  }
  
  const int spaceDim = 1;
  const int velocityDim = 1;
  _delta_k = delta_k;
  _tn_enrichment = vlasovTraceEnrichment;
  
  _q    = ParameterFunction::parameterFunction(q);
  _eps0 = ParameterFunction::parameterFunction(eps0);
  _m    = ParameterFunction::parameterFunction(m);
  
  _n0   = ParameterFunction::parameterFunction(0.0); // will be set/updated during solve if quasi-neutrality enforced
  
  auto intervalMeshTopology = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElementsCoarse, usePeriodicBCs);
  
  TEUCHOS_TEST_FOR_EXCEPTION(firstTimeDivisionSize < 0, std::invalid_argument, "first time division must be non-negative (may be 0 if no extra time division is required)");
  int numTimeElements = (firstTimeDivisionSize <= 0) ? tElementsCoarse : tElementsCoarse + 1;
  std::vector<double> timeVertices(numTimeElements + 1);
  
  int iOffset = 0;
  double remainingTime = tFinal - t0;
  timeVertices[0] = t0;
  if (firstTimeDivisionSize > 0)
  {
    // extra "time step" at initial time
    remainingTime -= firstTimeDivisionSize;
    timeVertices[1] = firstTimeDivisionSize + t0;
    iOffset = 1;
  }
  for (int i=1; i<tElementsCoarse+1; i++)
  {
    timeVertices[i+iOffset] = t0 + firstTimeDivisionSize + (i * remainingTime) / tElementsCoarse;
  }
  
  auto intervalMeshTopologySpaceTime = MeshFactory::extrudedMeshTopology(intervalMeshTopology, {timeVertices});
  
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(intervalMeshTopologySpaceTime, {v0}, {vWidth}, {vElementsCoarse});
  
  const bool l2Trace   = true; // non-conforming
  const bool ultraweak = _formulationChoice == ULTRAWEAK;
  
  if (ultraweak)
  {
    Space tauSpace    = (spaceDim > 1) ? HDIV : HGRAD;
    Space E_hat_space = L2; // non-conforming
    Space ESpace      = (spaceDim > 1) ? VECTOR_L2 : L2;

    // fields
    VarPtr f; // Vlasov
    VarPtr E; // Poisson
    VarPtr VE; // potential (Poisson)

    // traces
    VarPtr E_hat, VE_hat, t_n_hat, f_hat; // t_n_hat: L2 trace, f_hat: H^1 trace

    // tests
    VarPtr w;   // Vlasov test function

    VarFactoryPtr poissonVF = VarFactory::varFactory();
    E  = poissonVF->fieldVarSpaceTime(S_E,  ESpace, L2); // L2 in time, too
    VE = poissonVF->fieldVarSpaceTime(S_VE, L2,     L2); // L2 in time, too
        
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();

    if (spaceDim > 1)
      E_hat = poissonVF->fluxVarSpaceOnly(S_E_HAT, E * (n * parity));
    else
      E_hat = poissonVF->fluxVarSpaceOnly(S_E_HAT, E * (Function::normal_1D() * parity));
    
    if (spaceDim > 1)
      VE_hat = poissonVF->traceVarSpaceOnly(S_VE_HAT, VE, L2);
    else
      VE_hat = poissonVF->fluxVarSpaceOnly(S_VE_HAT, VE * (Function::normal_1D() * parity), L2); // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
    
    VarPtr v_Poisson   = poissonVF->testVarSpaceTime(S_V_POISSON, HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
    VarPtr tau_Poisson = poissonVF->testVarSpaceTime(S_TAU_POISSON, tauSpace, HGRAD); // HDIV or HGRAD in space, HGRAD in time
    
    _poissonBF = BF::bf(poissonVF);
    
    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
      _poissonBF->addTerm(VE, tau_Poisson->di(1));
      _poissonBF->addTerm( -E, tau_Poisson);
      _poissonBF->addTerm(-VE_hat, tau_Poisson);

      _poissonBF->addTerm(-E, v_Poisson->di(1));
      _poissonBF->addTerm(E_hat, v_Poisson);
    }
    else
    {
      _poissonBF->addTerm(VE, tau_Poisson->div());
      _poissonBF->addTerm(-E, tau_Poisson);
      _poissonBF->addTerm(-VE_hat, tau_Poisson->dot_normal());

      _poissonBF->addTerm(-E, v_Poisson->grad());
      _poissonBF->addTerm(E_hat, v_Poisson);
      
      // the above won't *quite* work for space-time: note the use of (spatial) div(), grad(), and dot_normal() operators.  We'll want to separate out the components of E, and do (-Ex, v_Poisson->di(1)), etc.  So if we get here, let's throw an exception:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Incorrect implementation; need to fix the spatial operators to work in a space-time context…");
    }
    
    Epetra_CommPtr comm = Teuchos::null;
    const bool allowPruning = false;
    map<int,int> trialOrderEnhancements;
    map<int,int> testOrderEnhancements;
    trialOrderEnhancements[ E_hat->ID()] = 1;
    trialOrderEnhancements[VE_hat->ID()] = 1;
    MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
    MeshPtr poissonMesh = Teuchos::rcp( new Mesh(intervalMeshTopologySpaceTime, _poissonBF, H1Order, delta_k,
                                                 trialOrderEnhancements, testOrderEnhancements, partitionPolicy,
                                                 comm, allowPruning) );
    
    _poissonSoln     = Solution::solution(_poissonBF, poissonMesh);
    _poissonSolnPrev = Solution::solution(_poissonBF, poissonMesh); // will copy out after fixed point solve
    auto poissonSolnWeakPtr     = Teuchos::rcp(_poissonSoln.get(),     false);
    auto poissonSolnPrevWeakPtr = Teuchos::rcp(_poissonSolnPrev.get(), false);
    
    _ESoln     = Function::solution(E, poissonSolnWeakPtr);
    FunctionPtr E_solnPrev = Function::solution(E, poissonSolnPrevWeakPtr);
    
    _EDiff = _ESoln - E_solnPrev;
    
    FunctionPtr v = this->vx();
    
    // Vlasov equation/mesh will not "know" about "space-time" as such -- no variables will be marked as space-time.
    // We can do this because the flux variable (t_n_hat) is defined on every interface, so we don't need the weird
    // only-supported-on-spatial-interfaces thing that we need for space-time Poisson.
    VarFactoryPtr vlasovVF = VarFactory::varFactory();
    
    f = vlasovVF->fieldVar(S_F);
    w = vlasovVF->testVar(S_W, HGRAD); // w lies in HGRAD of the whole 1D1T1V space
    
    if (l2Trace)
    {
      t_n_hat = vlasovVF->fluxVar(S_T_N_HAT);
    }
    else
    {
      f_hat = vlasovVF->traceVar(S_F_HAT);
    }
    
    _vlasovBF = BF::bf(vlasovVF);
    
    trialOrderEnhancements.clear();
    testOrderEnhancements.clear();
    if (l2Trace)
    {
      trialOrderEnhancements[t_n_hat->ID()] = 1 + _tn_enrichment; // +1 for non-conforming, + whatever user requested
    }
    
    MeshPtr vlasovMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _vlasovBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
    
    {
      // to avoid an exception being thrown for an unset solution when _E_extruded->updateValues() is first called, we project zero onto the Poisson solution first
      std::map<int,FunctionPtr> zeroPoisson;
      FunctionPtr zero = Function::zero();
      zeroPoisson[this->E()->ID()] = zero;
      zeroPoisson[this->E_hat()->ID()] = zero;
      _poissonSoln->projectOntoMesh(zeroPoisson);
    }
    auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
    _E_extruded = Teuchos::rcp( new PhysicalSolutionExtrusion(_poissonSoln, E, vlasovMeshWeakPtr) );

    // the di() methods are 1-based:
    static const int SPATIAL_INDEX  = 1;
    static const int TEMPORAL_INDEX = 2;
    static const int VELOCITY_INDEX = 3;
    
    // SpatialFilter's components are zero-based
    static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
    static const int TEMPORAL_INDEX_ZERO_BASED = TEMPORAL_INDEX - 1;
    static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
    
    _vlasovBF->addTerm(              -v * f, w->di(SPATIAL_INDEX));  // dx
    _vlasovBF->addTerm(                  -f, w->di(TEMPORAL_INDEX)); // dt
    _vlasovBF->addTerm(-(q/m)*_E_extruded*f, w->di(VELOCITY_INDEX)); // dv
    
    if (l2Trace)
    {
      _vlasovBF->addTerm(t_n_hat, w);
    }
    else
    {
      FunctionPtr n = Function::normal(); // TODO: make the normal function use Intrepid2::Data to store only a single value on each side (rather than one per point), and to "know" about the tensor product structures present (i.e., make it fast on the Vlasov mesh).
      
      // b: convective direction
      FunctionPtr b_dot_normal = Function::zero();
      b_dot_normal = b_dot_normal + n->spatialComponent(SPATIAL_INDEX) * v;
      b_dot_normal = b_dot_normal + n->spatialComponent(TEMPORAL_INDEX);
      b_dot_normal = b_dot_normal + n->spatialComponent(VELOCITY_INDEX) * (q/m) * _E_extruded;
      
      _vlasovBF->addTerm(b_dot_normal * f_hat, w);
    }

    _vlasovSoln     = Solution::solution(_vlasovBF,vlasovMesh);
    _vlasovSolnPrev = Solution::solution(_vlasovBF,vlasovMesh); // will copy out after fixed-point solve
    
    auto vlasovSolnWeakPtr     = Teuchos::rcp(_vlasovSoln.get(),     false);
    auto vlasovSolnPrevWeakPtr = Teuchos::rcp(_vlasovSolnPrev.get(), false);
    
    _fSoln     = Function::solution(f, vlasovSolnWeakPtr);
    FunctionPtr f_solnPrev = Function::solution(f, vlasovSolnPrevWeakPtr);
    
    _fDiff = _fSoln - f_solnPrev;
    
    const int maxMomentNumber = 0; // for now, we only do the zeroth moment
    _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln, f, poissonMesh) );
    
    // note: although the temporal dimension here is also an extrusion dimension, we do NOT indicate it as a
    //       "surviving" extrusion dimension; the extrusions referred to in SolutionMomentFunction are extrusions
    //       in the higher-dimensional *relative* to the lower-dimensional one.  Here, there is only one such dimension,
    //       the velocity dimension.
    _f_velocityIntegral = _solutionMoments->solutionMomentFunction(0);
    
    // set RHS for the two problems
    // casts to allow FunctionPtr overloads to be used:
    auto qFunction = FunctionPtr(_q);
    auto mFunction = FunctionPtr(_m);
    auto epsFxn = FunctionPtr(_eps0);
    auto n0Fxn  = _enforceQuasiNeutrality ? FunctionPtr(_n0) : Function::zero(); // _n0 will be initialized when initial values are specified, if quasi-neutrality is enforced
    auto f_velocityIntegralFxn = FunctionPtr(_f_velocityIntegral);
    auto poissonRHS = RHS::rhs();
    poissonRHS->addTerm( qFunction / epsFxn * f_velocityIntegralFxn * v_Poisson);
    _poissonSoln->setRHS(poissonRHS);
    
    auto vlasovRHS = RHS::rhs();
    _vlasovSoln->setRHS(vlasovRHS); // zero RHS for vlasov
    
    // set the test-space inner products.
    _poissonSoln->setIP(_poissonBF->graphNorm());
    
    poissonMesh->registerSolution(_poissonSoln);
    vlasovMesh->registerSolution(_vlasovSoln);
    
    IPPtr vlasovTestNorm = _vlasovBF->graphNorm();
    _vlasovSoln->setIP(vlasovTestNorm);
    
    SpatialFilterPtr leftSideFilter    = SpatialFilter::matchingX(x0);
    SpatialFilterPtr rightSideFilter   = SpatialFilter::matchingX(x0+xWidth);
    SpatialFilterPtr initialTimeFilter = SpatialFilter::matchingComponent(TEMPORAL_INDEX_ZERO_BASED, t0);
    SpatialFilterPtr minVelocityFilter = SpatialFilter::matchingComponent(VELOCITY_INDEX_ZERO_BASED, v0);
    SpatialFilterPtr maxVelocityFilter = SpatialFilter::matchingComponent(VELOCITY_INDEX_ZERO_BASED, v0+vWidth);
    
    // create entity sets for BCs:
    {
      // physical (space-time) mesh entity sets:
      // create an empty entity set.
      EntitySetPtr leftSideSet  = intervalMeshTopologySpaceTime->createEntitySet();
      EntitySetPtr rightSideSet = intervalMeshTopologySpaceTime->createEntitySet();
      
      // here we use getSidesThatMatch instead of getBoundarySidesThatMatch -- this allows matches even when periodic BCs are present.
      std::vector<IndexType> leftSideIndices  = intervalMeshTopologySpaceTime->getSidesThatMatch(leftSideFilter);
      std::vector<IndexType> rightSideIndices = intervalMeshTopologySpaceTime->getSidesThatMatch(rightSideFilter);
      
      const int sideDim = spaceDim; // sides in space-time mesh have dimension equal to spatial mesh
      
      for (const auto & leftSideIndex : leftSideIndices)
      {
        leftSideSet->addEntity(sideDim, leftSideIndex);
      }
      intervalMeshTopologySpaceTime->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, leftSideSet);
      
      for (const auto & rightSideIndex : rightSideIndices)
      {
        rightSideSet->addEntity(sideDim, rightSideIndex);
      }
      intervalMeshTopologySpaceTime->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X1, rightSideSet);
    }
    {
      // Vlasov side sets
      const int sideDim = spaceDim + velocityDim; // phase space, space-time: sides have dimension of (fixed-time) phase space
      auto initialTimeSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> initialTimeSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(initialTimeFilter);
      for (const auto & initialTimeSideIndex : initialTimeSideIndices)
      {
        initialTimeSideSet->addEntity(sideDim, initialTimeSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_TMIN, initialTimeSideSet);
      
      auto bottomSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> bottomSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(minVelocityFilter);
      for (const auto & bottomSideIndex : bottomSideIndices)
      {
        bottomSideSet->addEntity(sideDim, bottomSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_VMIN, bottomSideSet);
      auto topSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> topSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(maxVelocityFilter);
      for (const auto & topSideIndex : topSideIndices)
      {
        topSideSet->addEntity(sideDim, topSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_VMAX, topSideSet);
      
      SpatialFilterPtr inflowLeftFilter  = SpatialFilter::matchingX(x0)          & SpatialFilter::greaterThanOrEqualToZ(0.0);
      SpatialFilterPtr inflowRightFilter = SpatialFilter::matchingX(x0 + xWidth) & SpatialFilter::lessThanOrEqualToZ   (0.0);
      
      auto inflowLeftSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> inflowLeftSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowLeftFilter);
      for (const auto & leftSideIndex : inflowLeftSideIndices)
      {
        inflowLeftSideSet->addEntity(sideDim, leftSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_INFLOW_LEFT, inflowLeftSideSet);
      auto inflowRightSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> inflowRightSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowRightFilter);
      for (const auto & inflowRightSideIndex : inflowRightSideIndices)
      {
        inflowRightSideSet->addEntity(sideDim, inflowRightSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_INFLOW_RIGHT, inflowRightSideSet);
    }

    // set BCs
    // set homogeneous conditions on the v boundaries
    // if we are not periodic in x, then the user must add conditions for left and right boundaries
    auto vlasovBC = BC::bc();
    auto traceVar = l2Trace ? t_n_hat : f_hat;
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMIN, Function::zero());
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMAX, Function::zero());
    _vlasovSoln->setBC(vlasovBC);
    
    auto poissonBC = BC::bc();
    
    if (usePeriodicBCs)
    {
      poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X0, Function::constant(0.0));
    }
    else
    {
      poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
    }
    _poissonSoln->setBC(poissonBC);
    
    // static condensation does not work properly with GMG due to details of VelocitySpaceExtrusion; it also does not appear to speed things up for the direct solve
//    if (!_haveTableau)
//    {
//      _vlasovSoln->setUseCondensedSolve(true); // should work without issue given no tableau; we need to test with tableaus before enabling with those
//    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported VlasovPoissonSpaceTimeFormulationChoice");
  }
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At conclusion of VlasovPoissonSpaceTimeFormulation construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//  }
  PhysicalMeshUpdater::registerForUpdates(_vlasovSoln->mesh(), _poissonSoln->mesh());
}

MeshPtr VlasovPoissonSpaceTimeFormulation::vlasovMesh()
{
  return _vlasovSoln->mesh();
}

SolutionPtr VlasovPoissonSpaceTimeFormulation::vlasovSolution()
{
  return _vlasovSoln;
}

void VlasovPoissonSpaceTimeFormulation::clearSolution()
{
  _vlasovSoln->clear();
  _poissonSoln->clear();
  
  
}

Teuchos::RCP<GMGSolver> VlasovPoissonSpaceTimeFormulation::constructGMGSolver()
{
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "Before GMG construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//    }
    
    int kCoarse = 0;

    bool useCondensedSolve = _vlasovSoln->usesCondensedSolve();

    auto multigridStrategy = GMGOperator::V_CYCLE;
//    auto multigridStrategy = GMGOperator::SMOOTHER_ONLY;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    GMGOperator::SmootherChoice smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ; // CAMELLIA_ADDITIVE_SCHWARZ defines cells as blocks; Ifpack's defines all cells on the MPI rank as a block, making MPI scaling less predictable.  (Camellia's implementation is likely less well-optimized, though.)
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL; // as configured, this is very slow; I think it's treating all data on an MPI rank as a block…
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    vector<MeshPtr> meshes;
    
    if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
    {
      meshes = GMGSolver::meshesForMultigrid(_vlasovSoln->mesh(), kCoarse, 1);
    }
    else
    {
      meshes = vector<MeshPtr> {_vlasovSoln->mesh()};
    }
    
    auto gmgSolver = Teuchos::rcp( new GMGSolver(_vlasovSoln, meshes, _maxIters, _cgTol, multigridStrategy,
                                                 Solver::getDirectSolver(true), useCondensedSolve) );
    
    gmgSolver->setSmootherType(smootherChoice);
    
    gmgSolver->setAztecOutput(_aztecOutputLevel);
    
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "After GMG construction, memory HWM:  " << memoryUsageInGigabytes << " GB.\n";
//    }
  return gmgSolver;
}

FunctionPtr VlasovPoissonSpaceTimeFormulation::E_extruded()
{
  return _E_extruded;
}

VarPtr VlasovPoissonSpaceTimeFormulation::f()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr VlasovPoissonSpaceTimeFormulation::f_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

MeshPtr VlasovPoissonSpaceTimeFormulation::poissonMesh()
{
  return _poissonSoln->mesh();
}

SolutionPtr VlasovPoissonSpaceTimeFormulation::poissonSolution()
{
  return _poissonSoln;
}

bool VlasovPoissonSpaceTimeFormulation::hasL2Trace() const {
  return true;
}

VarPtr VlasovPoissonSpaceTimeFormulation::E()
{
  return _poissonBF->varFactory()->trialVar(S_E);
}

VarPtr VlasovPoissonSpaceTimeFormulation::E_hat()
{
  return _poissonBF->varFactory()->trialVar(S_E_HAT);
}

void VlasovPoissonSpaceTimeFormulation::setFixedPointMaxSubsteps(const int maxSubsteps)
{
  _maxSubsteps = maxSubsteps;
}

void VlasovPoissonSpaceTimeFormulation::setFixedPointMinSubsteps(const int minSubsteps)
{
  _minSubsteps = minSubsteps;
}

void VlasovPoissonSpaceTimeFormulation::setFixedPointTolerance(const double fTol)
{
  _fixedPointTolerance = fTol;
}

void VlasovPoissonSpaceTimeFormulation::setPotentialBCs(FunctionPtr VE_left, FunctionPtr VE_right)
{
  auto poissonBC = BC::bc();
  
  TEUCHOS_TEST_FOR_EXCEPTION(_usePeriodicBCs, std::invalid_argument, "setting left and right potential BCs is not allowed for periodic meshes");
  
  poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X0, VE_left);  // set the left potential
  poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X1, VE_right); // set the right potential
  
  _poissonSoln->setBC(poissonBC);
}

void VlasovPoissonSpaceTimeFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void VlasovPoissonSpaceTimeFormulation::setIterativeSolveMaxIterations(const int maxIters)
{
  _maxIters = maxIters;
}

void VlasovPoissonSpaceTimeFormulation::setIterativeSolveTolerance(const double cgTol)
{
  _cgTol = cgTol;
}

void VlasovPoissonSpaceTimeFormulation::setIterativeSolveTolerances(const std::vector<double> cgTols)
{
  _cgTols = cgTols;
}

void VlasovPoissonSpaceTimeFormulation::setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator, bool gateRefinements)
{
  _refinementThreshold = refinementThreshold;
  _maxElements = maxElements;
  _maxRefinementLevel = maxRefinementLevel;
  _lastRefinementMaxElementError = -1.0;
  _gateRefinements = gateRefinements;
  
  if (_refStrategy == Teuchos::null)
  {
    _refStrategy = RefinementStrategy::energyErrorRefinementStrategy(_vlasovSoln, _refinementThreshold);
  }
  _refStrategy->setRelativeErrorThreshold(_refinementThreshold);
  _refStrategy->setMaxElements(_maxElements);
  _refStrategy->setMaximumRefinementLevel(_maxRefinementLevel);
  _refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = _vlasovSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  _refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
}

void VlasovPoissonSpaceTimeFormulation::solve()
{
  const bool useAdaptivity = _refStrategy != Teuchos::null;
  const int rank = MPIWrapper::CommWorld()->MyPID();
  
  Teuchos::RCP<GMGSolver> gmgSolver;
  
  std::vector<int>    substepIterationCounts(_maxSubsteps);
  std::vector<double> substepStepSizesVlasov;
  std::vector<double> substepStepSizesPoisson;
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
  const bool exportViz = true; // for debugging of fixed-point iteration, set to true.  TODO: make this subject to external control.
  
  if (exportViz)
  {
    static int solveNumber = 0;
    std::string prefix = "VlasovPoissonSpaceTimeFormulation_";
    std::ostringstream vlasovSS;
    vlasovSS << prefix << "Vlasov" << solveNumber;
    
    std::ostringstream poissonSS;
    poissonSS << prefix << "Poisson" << solveNumber;
      
    const std::string directoryPath = "/tmp";
    
    vlasovViz = Teuchos::rcp(new HDF5Exporter(this->vlasovMesh(), vlasovSS.str(), directoryPath));
    poissonViz = Teuchos::rcp(new HDF5Exporter (this->poissonMesh(), poissonSS.str(), directoryPath) );
    solveNumber++;
  }
  
  updateElectricField(); // doing the Poisson solve first may help with fixed point convergence.  (And this is far cheaper than the Vlasov solve, so well worth it if it does help…)
  
  // get the PhysicalSolutionExtrusion ready to evaluate (in context of Vlasov BF)
  // updateValues() will do MPI communication of physical solution values so they're where they need to be for phase space mesh:
  static_cast<PhysicalSolutionExtrusion*>(_E_extruded.get())->updateValues();
  for (int substepOrdinal=0; substepOrdinal < _maxSubsteps; substepOrdinal++)
  {
    if (rank == 0)
    {
      std::cout << "\n\nSolving substep " << substepOrdinal+1 << " of up to " << _maxSubsteps << " (with early stop when f update is below " << _fixedPointTolerance << ")" << std::endl;
    }
    // solve for f:
    if (!_useGeometricMultigrid)
    {
      _vlasovSoln->solve();
      substepIterationCounts[substepOrdinal] = -1; // -1 to indicate not an iterative solver
    }
    else
    {
      Teuchos::RCP<GMGSolver> gmgSolver = constructGMGSolver();
      if (_cgTols.size() > 0) // then we should have one per substep
      {
        gmgSolver->setTolerance(_cgTols[substepOrdinal]);
      }
      _vlasovSoln->solve(gmgSolver);
      substepIterationCounts[substepOrdinal] = gmgSolver->iterationCount();
    }
    updateElectricField();
    
    // Copy vlasov, Poisson solution to Prev solutions.  For substepOrdinal > 0, measure difference between the two solutions first.
    
    if (substepOrdinal > 0)
    {
      auto vlasovMesh  = _vlasovSoln->mesh();
      auto poissonMesh = _poissonSoln->mesh();
      
      const double fDiff_L2 = _fDiff->l2norm(vlasovMesh);
      const double EDiff_L2 = _EDiff->l2norm(poissonMesh);
      
      const double f_L2 = _fSoln->l2norm(vlasovMesh);
      const double E_L2 = _ESoln->l2norm(poissonMesh);
      
      const double fRelDiff = fDiff_L2 / f_L2;
      const double ERelDiff = EDiff_L2 / E_L2;
      
      substepStepSizesVlasov.push_back(fRelDiff);
      substepStepSizesPoisson.push_back(ERelDiff);
      
      if (rank == 0)
      {
        std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in f: " << fRelDiff << std::endl;
        std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in E: " << ERelDiff << std::endl << std::endl;
      }
      
      if ((fRelDiff < _fixedPointTolerance) && (substepOrdinal >= _minSubsteps-1))
      {
        // tolerance reached, min substeps taken
        break;
      }
    }
    
    _vlasovSolnPrev->setSolution(_vlasovSoln);
    _poissonSolnPrev->setSolution(_poissonSoln);
    
    if (exportViz)
    {
      poissonViz->exportSolution(poissonSolution(),substepOrdinal);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal);
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  _fixedPointL2StepSizesVlasov.push_back(substepStepSizesVlasov);
  _fixedPointL2StepSizesPoisson.push_back(substepStepSizesPoisson);
}

void VlasovPoissonSpaceTimeFormulation::updateElectricField()
{
  // update f integral, n0
  _solutionMoments->update();
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    double f_integral = _f_velocityIntegral->integrate(_poissonSoln->mesh());
    n0 = f_integral / _poissonMeshMeasure;
//    std::cout << "1D1V n0 = " << n0 << std::endl;
  }
  else
  {
    n0 = 0;
  }
  _n0->setValue(n0);
//  {
//    // DEBUGGING
//    std::cout << "f_integral = " << f_integral << std::endl;
//    std::cout << "n0 = " << n0 << std::endl;
//    
//    std::cout << "L^2 norm of _f_velocityIntegral: " << _f_velocityIntegral->l2norm(_poissonSoln->mesh()) << std::endl;
//    
//    const auto physicalMesh = _poissonSoln->mesh();
//    const auto & myCells = physicalMesh->cellIDsInPartition();
//    for (const auto & physicalCellID : myCells)
//    {
//      BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
//      bool success = true;
//      Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
//      testFunctionDataPathMatchesViewPath1T1V(_f_velocityIntegral, basisCache, 1e-14, out, success);
//      if (!success)
//      {
//        std::cout << "testFunctionDataPathMatchesViewPath1T1V(): failed on physical cell " << physicalCellID << std::endl;
//      }
//    }
//    
//    _poissonSoln->setWriteRHSToMatrixMarketFile(true, "poissonRHS.dat");
//    _poissonSoln->setWriteMatrixToFile(true, "poissonStiffness.dat");
//  }
  
  // solve for E, E_hat:
  _poissonSoln->solve();
    
  static_cast<PhysicalSolutionExtrusion*>(_E_extruded.get())->updateValues();
}

void VlasovPoissonSpaceTimeFormulation::setVlasovInitialConditions(FunctionPtr f0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_usePeriodicBCs, std::invalid_argument, "setting initial conditions without setting inflow BCs is not allowed for non-periodic meshes");

  auto vlasovBC = BC::bc();
  const bool l2Trace = hasL2Trace();
  VarPtr traceVar = l2Trace ? this->t_n_hat() : this->f_hat();
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMIN, Function::zero());
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMAX, Function::zero());
  
  if (l2Trace)
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN, -f0); // minus sign thanks to outward normal on temporal interface
  }
  else
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN,  f0);
  }
  _vlasovSoln->setBC(vlasovBC);
}

void VlasovPoissonSpaceTimeFormulation::setVlasovInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_left, FunctionPtr f_right)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_usePeriodicBCs, std::invalid_argument, "setting inflow BCs is not allowed for periodic meshes");

  auto vlasovBC = BC::bc();
  const bool l2Trace = hasL2Trace();
  VarPtr traceVar = l2Trace ? this->t_n_hat() : this->f_hat();
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMIN, Function::zero());
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMAX, Function::zero());
  
  if (l2Trace)
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN, -f0); // minus sign thanks to outward normal on temporal interface
    
    const int VELOCITY_INDEX_ZERO_BASED = 2; // 1D1T1V: velocity is the final dimension
    
    FunctionPtr v = MonomialFunction::xi_n(VELOCITY_INDEX_ZERO_BASED, 1);
    FunctionPtr leftTrace  = - f_left * v; // minus sign is thanks to outward normal
    FunctionPtr rightTrace =  f_right * v;
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_LEFT,   leftTrace);
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_RIGHT, rightTrace);
  }
  else
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN, f0);
    
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_LEFT,   f_left);
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_RIGHT, f_right);
  }
  _vlasovSoln->setBC(vlasovBC);
}

Teuchos::RCP<SolutionMoments> VlasovPoissonSpaceTimeFormulation::solutionMoments()
{
  return _solutionMoments;
}

// test variables:
VarPtr VlasovPoissonSpaceTimeFormulation::w()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

void VlasovPoissonSpaceTimeFormulation::setUseGeometricMultigrid(bool useGeometricMultigrid)
{
  _useGeometricMultigrid = useGeometricMultigrid;
}

VarPtr VlasovPoissonSpaceTimeFormulation::tau()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->testVar(S_TAU_POISSON);
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovInitialTime() const
{
  return VLASOV_TAG_TMIN;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovLeftInflow() const
{
  return VLASOV_TAG_INFLOW_LEFT;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovRightInflow() const
{
  return VLASOV_TAG_INFLOW_RIGHT;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovMaxVelocity() const
{
  return VLASOV_TAG_VMAX;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovMinVelocity() const
{
  return VLASOV_TAG_VMIN;
}

VarPtr VlasovPoissonSpaceTimeFormulation::t_n_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

VarPtr VlasovPoissonSpaceTimeFormulation::VE()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_VE);
}

VarPtr VlasovPoissonSpaceTimeFormulation::VE_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_VE_HAT);
}

FunctionPtr VlasovPoissonSpaceTimeFormulation::vx() const
{
  // the vx component is the first one after the dimension of the physical mesh
  int vx_component_zero_based = _poissonSoln->mesh()->getDimension();
  return MonomialFunction::xi_n(vx_component_zero_based, 1);
}

FunctionPtr VlasovPoissonSpaceTimeFormulation::zerothMoment()
{
  return _f_velocityIntegral;
}

// below, we set up unit tests against the private classes above
#include "Teuchos_UnitTestHarness.hpp"

void testVelocitySpace1D1TIntegral(const bool &doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  double q      = 1.0;
  double eps0   = 1.0;
  double m      = 1.0;
  double x0     = 0.0;
  double xWidth = 1.0;
  double t0     = 0.0;
  double tWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 1.0;
  
  const int xElements = 4;
  const int vElements = 4;
  const int tElements = 1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  using FormulationChoice = Camellia::VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK
  };
  
  set<GlobalIndexType> vlasovCellsToRefine;
  if (doRefinement)
  {
    vlasovCellsToRefine = {0,5,8};
  }
  
  for (auto formulationChoice : formulationChoices)
  {
    const int vlasovTraceEnrichment = 0;
    VlasovPoissonSpaceTimeFormulation form(formulationChoice,
                                           q, eps0, m,
                                           x0, xWidth, xElements,
                                           t0, tWidth, tElements,
                                           v0, vWidth, vElements,
                                           H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality,
                                           vlasovTraceEnrichment);
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr z  = Function::zn(1);
    FunctionPtr xz = x * z;
    // integral of xz dz = x * z^2 / 2
    // we integrate from v0 to v0 + vWidth
    const double expectedIntegralZ = (0.5 * (v0 + vWidth) * (v0 + vWidth) - 0.5 * v0 * v0);
    FunctionPtr expectedIntegral = x * expectedIntegralZ;
    const double expectedIntegralXZ = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0) * expectedIntegralZ;
    const double actualIntegralXZ   = expectedIntegral->integrate(form.poissonMesh());
    
    TEST_FLOATING_EQUALITY(expectedIntegralXZ, actualIntegralXZ, tol);
    
    if (vlasovCellsToRefine.size() > 0)
    {
      auto vlasovMesh = form.vlasovMesh();
      auto line_x_line_x_line = CellTopology::cellTopology(CellTopology::line(), 2);
      auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line_x_line->getKey());
      vlasovMesh->hRefine(vlasovCellsToRefine, refPattern);
    }
    
    auto f = form.f();
    std::map<int, FunctionPtr> projectionMap;
    projectionMap[f->ID()] = xz;
    
    auto vlasovSoln = form.vlasovSolution();
    const int solutionOrdinal = 0;
    vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    auto poissonMesh = form.poissonMesh();
    
    SolutionMoments solutionMoments(0, vlasovSoln, f, poissonMesh);
    auto velocitySpaceIntegral = solutionMoments.solutionMomentFunction(0);
    FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedIntegral;
    
    solutionMoments.update();
    
    double magnitudeExpected = expectedIntegral->l2norm(poissonMesh);
    double err_l2 = err->l2norm(poissonMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionVlasovPoissonSpaceTimeFormulation, AnalyticIntegralMatches )
{
  const bool doRefinement = false;
  testVelocitySpace1D1TIntegral(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionVlasovPoissonSpaceTimeFormulation, AnalyticIntegralMatchesWithRefinement )
{
  const bool doRefinement = true;
  testVelocitySpace1D1TIntegral(doRefinement, out, success);
}

class SideRestriction : public TFunction<double>
{
  std::vector<int> _sidesToMatch;
public:
  SideRestriction(std::vector<int> sidesToMatch)
  :
  Function("SideRestriction",0),
  _sidesToMatch(sidesToMatch)
  {}
  
  bool boundaryValueOnly() override
  {
    return true;
  }
  
  using TFunction<double>::values;
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    const int sideOrdinal = basisCache->getSideIndex();
    bool sideMatches = false;
    for (auto sideToMatch : _sidesToMatch)
    {
      if (sideToMatch == sideOrdinal)
      {
        sideMatches = true;
        break;
      }
    }
    const double value = sideMatches ? 1.0 : 0.0;
    values.initialize(value);
  }
};

void testVelocityExtrusion1D1T(const int &xMeshWidth, const int &tMeshWidth, const int &vMeshWidth,
                               const std::vector< std::vector<GlobalIndexType> > &refinements,
                               Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-14;
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  double t0     = 0.0;
  double tWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 20.0;
  
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  using FormulationChoice = Camellia::VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK
  };
  
  for (auto formulationChoice : formulationChoices)
  {
    const int vlasovTraceEnrichment = 0;
    VlasovPoissonSpaceTimeFormulation form(formulationChoice,
                                           q, eps0, m,
                                           x0, xWidth, xMeshWidth,
                                           t0, tWidth, tMeshWidth,
                                           v0, vWidth, vMeshWidth,
                                           H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality,
                                           vlasovTraceEnrichment);
    
    FunctionPtr xt = Function::xn(1) * Function::yn(1);
    
    FunctionPtr physicalSoln;
    
    auto poissonSoln = form.poissonSolution();
    std::map<int, FunctionPtr> projectionMap;
    
    auto E = form.E();
    projectionMap[E->ID()] = xt;
    physicalSoln = Function::solution(E, poissonSoln);
    
    auto poissonMesh = form.poissonMesh();
    auto vlasovMesh = form.vlasovMesh();
    
    auto extrusion = Teuchos::rcp( new PhysicalSolutionExtrusion(poissonSoln, E, vlasovMesh) );
    
    auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
    
    if (refinements.size() > 0)
    {
      for (auto &cellsToRefine : refinements)
      {
        auto refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
        vlasovMesh->hRefine(cellsToRefine, refPattern, false); // false: don't rebuild yet (wait for 1-irregularity enforcement)
        vlasovMesh->enforceOneIrregularity(true);
        
        poissonMesh->repartitionAndRebuild();
      }
    }
    
    const int solutionOrdinal = 0;
    poissonSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    extrusion->updateValues();
    
    FunctionPtr expected = xt;
    
    FunctionPtr err = FunctionPtr(extrusion) - expected;
    
  //  double magnitudeActual = extrusion->l2norm(vlasovMesh);
  //  std::cout << "magnitudeActual = " << magnitudeActual << std::endl;
    
    double magnitudeExpected = expected->l2norm(vlasovMesh);
  //  std::cout << "magnitudeExpected = " << magnitudeExpected << std::endl;
    double err_l2 = err->l2norm(vlasovMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
    
//    if (!success)
//    {
//      Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
//      std::string prefix = "VlasovPoissonSpaceTimeFormulation_";
//      std::ostringstream vlasovSS;
//      vlasovSS << prefix << "Vlasov";
//
//      std::ostringstream poissonSS;
//      poissonSS << prefix << "Poisson";
//
//      const std::string directoryPath = "/tmp";
//
//      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
//      poissonViz = Teuchos::rcp(new HDF5Exporter (form.poissonMesh(), poissonSS.str(), directoryPath) );
//
//      poissonViz->exportSolution(form.poissonSolution(),0);
//      // right now, can't export E extrusion because it only implements a Data-based values(), which HDF5Exporter does not support.
////      vlasovViz->exportSolution(form.vlasovSolution(),{expected, extrusion, err},{"exact E extrusion","actual E extrusion", "err"},0);
//    }
    
    // we now only implement data path, so we can't test that View path matches it…
//    const auto & phaseCellIDs = vlasovMesh->cellIDsInPartition();
//    for (auto phaseCellID : phaseCellIDs)
//    {
//      BasisCachePtr basisCache = BasisCache::basisCacheForCell(vlasovMesh, phaseCellID);
//      if (volumeVariable)
//      {
//        testFunctionDataPathMatchesViewPath1T1V(extrusion, basisCache, tol * magnitudeExpected, out, success);
//      }
//      else
//      {
//        const int numSides = basisCache->cellTopology()->getSideCount();
//        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
//        {
//          auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
//          testFunctionDataPathMatchesViewPath1T1V(extrusion, sideCache, tol * magnitudeExpected, out, success);
//        }
//      }
//    }
  }
}

void testVelocityExtrusion1D1T(bool doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  using std::vector;
  vector< vector<GlobalIndexType> > refinements;
  
  if (doRefinement)
  {
    refinements = vector< vector<GlobalIndexType> >({{0,1}}); // one refinement, of cells 0 and 1.
  }
  
  const int xElements = 2;
  const int tElements = 2;
  const int vElements = 2;
    
  testVelocityExtrusion1D1T(xElements, tElements, vElements, refinements, out, success);
}


TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonSpaceTimeFormulation, VolumeExtrusion )
{
  const bool doRefinement = false;
  testVelocityExtrusion1D1T(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonSpaceTimeFormulation, VolumeExtrusionWithRefinement )
{
  const bool doRefinement = true;
  testVelocityExtrusion1D1T(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonSpaceTimeFormulation, VolumeExtrusionWithSpecificRefinements )
{
  // this matches a refinement sequence for space-time cold diode; I want to check whether we have errors in the extrusion for this case.
  // (I got a ways into writing this before I realized I was not in the space-time Vlasov-Poisson tests; for now, I'm going forward with
  //  it, just letting the last dimension be 1 element wide; probably behavior will be similar.)
  
  const bool volumeVariable = true;
  
  using std::vector;
  vector<vector<GlobalIndexType> > refinements;
  
  refinements.push_back(vector<GlobalIndexType> {4, 5, 6, 7, 8, 9});
  refinements.push_back(vector<GlobalIndexType> {21, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 57, 59});
  refinements.push_back(vector<GlobalIndexType> {97, 99, 100, 101, 102, 103, 104, 106, 107, 110, 128, 129, 130, 131, 132, 134, 160, 161, 162, 163, 165, 167, 168, 170, 172, 174, 175, 192, 193, 194, 195, 199, 208, 210, 227});
  
  const int xElements = 2;
  const int tElements = 1;
  const int vElements = 8;
  
  testVelocityExtrusion1D1T(xElements, tElements, vElements, refinements, out, success);
}
