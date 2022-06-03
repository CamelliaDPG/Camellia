//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  VlasovBackwardEulerFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 1/11/22.
//
//

#include "VlasovBackwardEulerFormulation.h"
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

const string VlasovBackwardEulerFormulation::S_F = "f";

const string VlasovBackwardEulerFormulation::S_F_HAT = "\\widehat{f}";
const string VlasovBackwardEulerFormulation::S_T_N_HAT = "\\widehat{t}_n";

const string VlasovBackwardEulerFormulation::S_W = "w";

namespace
{
  //! private class determines regions in phase-space mesh where v \cdot n_x < 0
  class SpatialFilterSpatialInflow : public SpatialFilter
  {
    int _spaceDim;
    int _vDimStart;
  public:
    SpatialFilterSpatialInflow(int spaceDim, int vDimStart)
    :
    _spaceDim(spaceDim),
    _vDimStart(vDimStart)
    {}
    
    bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) override
    {
      // TODO: Kokkos implementation
      const auto & normals = basisCache->getSideNormals(); // shape: C,P,D
      
      const auto & points = basisCache->getPhysicalCubaturePoints();
      
//      cout << "points:\n" << basisCache->getPhysicalCubaturePointsFC();
//      cout << "normals:\n" << normals;
      
      const int numCells  = normals.dimension(0);
      const int numPoints = normals.dimension(1);
      bool somePointMatches = false;
      for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
        {
          double v_dot_n = 0.0;
          for (int d=0; d<_spaceDim; d++)
          {
            double v_d = points(cellOrdinal,pointOrdinal,_vDimStart + d);
            double n_d = normals(cellOrdinal,pointOrdinal,d);
            v_dot_n += v_d * n_d;
          }
          if (v_dot_n < 0)
          {
            pointsMatch(cellOrdinal,pointOrdinal) = true;
            somePointMatches = true;
          }
          else
          {
            pointsMatch(cellOrdinal,pointOrdinal) = false;
          }
        }
      }
      
//      cout << "pointsMatch:\n" << pointsMatch;
      
      return somePointMatches;
    }
    
    void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Not supported");
    }
  };
}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               std::vector<double> velocitySpaceVertices_y,
                                                               std::vector<double> velocitySpaceVertices_z,
                                                               const int H1Order, const int delta_k)
:
_spaceDim(spatialMeshTopology->getDimension()),
_delta_k(delta_k),
_configurationSpaceFormulation(configurationSpaceFormulation),
_q(q),
_m(m)
{
  _configurationSpaceMeshTopology = spatialMeshTopology;
  _configurationSpaceFormulation->initializeMesh(_configurationSpaceMeshTopology, H1Order, delta_k);
  
  _dt = ParameterFunction::parameterFunction(dt);
  
  std::vector<std::vector<double> > velocitySpaceVertices;
  
  TEUCHOS_TEST_FOR_EXCEPTION(velocitySpaceVertices_x.size() == 0, std::invalid_argument, "velocitySpaceVertices_x may not be empty");
  velocitySpaceVertices.push_back(velocitySpaceVertices_x);
  if (velocitySpaceVertices_y.size() > 0)
  {
    velocitySpaceVertices.push_back(velocitySpaceVertices_y);
    if (velocitySpaceVertices_z.size() > 0)
    {
      velocitySpaceVertices.push_back(velocitySpaceVertices_z);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(velocitySpaceVertices_y.size() > 0, std::invalid_argument, "if velocitySpaceVertices_y is empty, then velocitySpaceVertices_z must also be empty");
  }
  
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(_configurationSpaceMeshTopology, velocitySpaceVertices);
  
  _spaceDim = spatialMeshTopology->getDimension();
  _velocityDim = velocitySpaceVertices.size();
  
  // fields
  VarPtr f; // Vlasov
  
  // traces
  VarPtr t_n_hat; // t_n_hat: L^2 (non-conforming) trace
  VarPtr f_hat;   // f_hat:   conforming trace (unused in present implementation -- see l2Trace flag below)

  // tests
  VarPtr w;   // Vlasov test function

  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();
  
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  f = vlasovVF->fieldVar(S_F);
  w = vlasovVF->testVar(S_W, HGRAD); // w lies in HGRAD of phase space
  
  t_n_hat = vlasovVF->fluxVar(S_T_N_HAT);
  
  const bool l2Trace = true;
  if (l2Trace)
  {
    t_n_hat = vlasovVF->fluxVar(S_T_N_HAT);
  }
  else
  {
    f_hat = vlasovVF->traceVar(S_F_HAT);
  }
  
  _vlasovBF = BF::bf(vlasovVF);
  
  map<int,int> trialOrderEnhancements;
  trialOrderEnhancements[t_n_hat->ID()] = 1; // +1 for non-conforming, + whatever user requested

  map<int,int> testOrderEnhancements;
  
  const bool allowPruning = true;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  Epetra_CommPtr comm = MPIWrapper::CommWorld();
  MeshPtr vlasovMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _vlasovBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
  
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  auto thisWeakPtr = Teuchos::rcp(this, false);
  vlasovMesh->registerObserver(thisWeakPtr);
  
  _vlasovSoln          = Solution::solution(_vlasovBF,vlasovMesh);
  _vlasovSolnPrev      = Solution::solution(_vlasovBF,vlasovMesh); // will copy out after fixed-point solve
  _vlasovSolnPriorTime = Solution::solution(_vlasovBF,vlasovMesh);
  
  // default BCs for Vlasov set zero at velocity extrema; suitable when you have periodic BCs in config space
  auto vlasovBC = BC::bc();
  VarPtr t_n = this->t_n_hat();
  for (int vd=0; vd<_velocityDim; vd++)
  {
    const int tagMin = this->tagVlasovMinVelocity(vd);
    const int tagMax = this->tagVlasovMaxVelocity(vd);
    vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
    vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
  }
  _vlasovSoln->setBC(vlasovBC);
  
  const int maxMomentNumber = _configurationSpaceFormulation->maxMomentNumber();
  _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln, f, _configurationSpaceFormulation->configurationSpaceMesh()) );
  
  _n0 = ParameterFunction::parameterFunction(0); // we'll update this when initial conditions are set, if _enforceQuasiNeutrality is true
  _configurationSpaceFormulation->setMoments(_solutionMoments, _n0);
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetVelocityOneBased = _spaceDim + 1;
  
  // SpatialFilter's components are zero-based
  const int indexOffsetSpaceZeroBased    = indexOffsetSpaceOneBased  - 1;
  const int indexOffsetVelocityZeroBased = indexOffsetVelocityOneBased - 1;
  
  // The below assumes that _spaceDim <= _velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim > _velocityDim, std::invalid_argument, "_velocityDim must be at least as large as _spaceDim");
  std::vector<FunctionPtr> v;
  for (int i=0; i<_velocityDim; i++)
  {
    v.push_back(this->v_i(i));
  }
  
  _vlasovBF->addTerm(f / _dt, w);
  
  if (l2Trace)
  {
    _vlasovBF->addTerm(t_n_hat, w);
  }
  else
  {
    // TODO: add support for conforming traces
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Conforming traces not yet supported.");
  }

  for (int i=0; i<_spaceDim; i++)
  {
    _vlasovBF->addTerm( -v[i] * f, w->di(indexOffsetSpaceOneBased + i));  // dx_i
  }
  std::vector<FunctionPtr> grad_v_weight(_velocityDim); // the thing that gets dotted with grad_v of f (in the strong form)
  
  std::vector<FunctionPtr> E(3,Function::zero());
  std::vector<FunctionPtr> B(3,Function::zero());
  
  for (int i=0; i<_velocityDim; i++)
  {
    E[i] = _configurationSpaceFormulation->E_i(i);
    B[i] = _configurationSpaceFormulation->B_i(i);
  }
  
  // we only include the v_cross_B term for 3V case
  // TODO: determine whether v_cross_B ever makes sense for 1V or 2V cases; if so, add something here.
  std::vector<FunctionPtr> v_cross_B(3,Function::zero());
  if (_velocityDim == 3)
  {
    v_cross_B[0] = v[1] * B[2] - v[2] * B[1];
    v_cross_B[1] = v[2] * B[0] - v[0] * B[2];
    v_cross_B[2] = v[0] * B[1] - v[1] * B[0];
  }
  
  for (int i=0; i<_velocityDim; i++)
  {
    grad_v_weight[i] = E[i] + v_cross_B[i];
  }
  
  for (int i=0; i<_velocityDim; i++)
  {
    auto weight = grad_v_weight[i];
    _vlasovBF->addTerm(-(q/m) * weight * f, w->di(indexOffsetVelocityOneBased + i)); // d/dvi
  }

  auto vlasovSolnWeakPtr          = Teuchos::rcp(_vlasovSoln.get(),          false);
  auto vlasovSolnPriorTimeWeakPtr = Teuchos::rcp(_vlasovSolnPriorTime.get(), false);
  auto vlasovSolnPrevWeakPtr      = Teuchos::rcp(_vlasovSolnPrev.get(),      false);
  
  _fSoln     = Function::solution(f, vlasovSolnWeakPtr);
  FunctionPtr f_solnPrev = Function::solution(f, vlasovSolnPrevWeakPtr);
  
  _fDiff = _fSoln - f_solnPrev;
    
  auto vlasovRHS = RHS::rhs();
  const bool weightFluxesBySideParity = true;
  const SolutionComponent solnComponent = STANDARD_SOLUTION;
  const int stageOrdinal = 0; // would be -1 if we had an ERKTableau
  auto f_previous = Function::solution(f, vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity,
                                       solnComponent, stageOrdinal, std::string("k"));
  
  vlasovRHS->addTerm(f_previous / FunctionPtr(_dt) * w);
  _vlasovSoln->setRHS(vlasovRHS);
    
  vlasovMesh->registerSolution(_vlasovSoln);
    
  IPPtr vlasovTestNorm = _vlasovBF->graphNorm();
  _vlasovSoln->setIP(vlasovTestNorm);
  
  SpatialFilterPtr inflowFilter = Teuchos::rcp(new SpatialFilterSpatialInflow(_spaceDim,indexOffsetVelocityZeroBased) );
    
  std::vector<double> v0,vMax;
  
  std::vector<SpatialFilterPtr> minVelocityFilters, maxVelocityFilters;
  for (int i=0; i<_velocityDim; i++)
  {
    auto & dimVertices = velocitySpaceVertices[i];
    const int numVertices = dimVertices.size();
    v0.push_back(dimVertices[0]);
    vMax.push_back(dimVertices[numVertices-1]);
    
    SpatialFilterPtr minVelocityFilter = SpatialFilter::matchingComponent(indexOffsetVelocityZeroBased + i, v0[i]  );
    SpatialFilterPtr maxVelocityFilter = SpatialFilter::matchingComponent(indexOffsetVelocityZeroBased + i, vMax[i]);
    
    minVelocityFilters.push_back(minVelocityFilter);
    maxVelocityFilters.push_back(maxVelocityFilter);
  }
  
  std::vector<int> minVelocityTags {VLASOV_TAG_VXMIN, VLASOV_TAG_VYMIN, VLASOV_TAG_VZMIN};
  std::vector<int> maxVelocityTags {VLASOV_TAG_VXMAX, VLASOV_TAG_VYMAX, VLASOV_TAG_VZMAX};
  
  // create entity sets for BCs:
  {
    // Vlasov side sets
    
    const int sideDim = _spaceDim + _velocityDim - 1; // phase space sides have dimension one less than that of phase space
    
    // inflow conditions
    auto inflowSideSet = extrudedMeshTopology->createEntitySet();
    std::vector<IndexType> inflowSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowFilter, vlasovMesh);
    for (const auto & inflowSideIndex : inflowSideIndices)
    {
      inflowSideSet->addEntity(sideDim, inflowSideIndex);
    }
    extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_SPATIAL_INFLOW, inflowSideSet);

    for (int i=0; i<_velocityDim; i++)
    {
      auto minSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> minSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(minVelocityFilters[i]);
      for (const auto & minSideIndex : minSideIndices)
      {
        minSideSet->addEntity(sideDim, minSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, minVelocityTags[i], minSideSet);
      auto maxSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> maxSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(maxVelocityFilters[i]);
      for (const auto & maxSideIndex : maxSideIndices)
      {
        maxSideSet->addEntity(sideDim, maxSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, maxVelocityTags[i], maxSideSet);
    }
  }
  
  PhysicalMeshUpdater::registerForUpdates(_vlasovSoln->mesh(), _configurationSpaceFormulation->configurationSpaceMesh());
}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               std::vector<double> velocitySpaceVertices_y,
                                                               const int H1Order, const int delta_k)
:
VlasovBackwardEulerFormulation(q, m, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, velocitySpaceVertices_y, std::vector<double>(),
                               H1Order, delta_k) {}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               const int H1Order, const int delta_k)
:
VlasovBackwardEulerFormulation(q, m, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, std::vector<double>(), std::vector<double>(),
                               H1Order, delta_k) {}

MeshPtr VlasovBackwardEulerFormulation::vlasovMesh()
{
  return _vlasovSoln->mesh();
}

SolutionPtr VlasovBackwardEulerFormulation::vlasovSolution()
{
  return _vlasovSoln;
}

void VlasovBackwardEulerFormulation::addConfigurationSpaceBC(const int tag, VarPtr configVariable, FunctionPtr dirichletValue)
{
  if (_configurationSpaceFormulation->solution()->bc() == Teuchos::null)
  {
    // create new BC object
    _configurationSpaceFormulation->solution()->setBC(BC::bc());
  }
  _configurationSpaceFormulation->solution()->bc()->addDirichlet(configVariable, tag, dirichletValue);
}

VarPtr VlasovBackwardEulerFormulation::B_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->B_hat(zeroBasedComp);
}

void VlasovBackwardEulerFormulation::clearConfigurationSpaceBCs()
{
  _configurationSpaceFormulation->solution()->setBC(BC::bc());
}

void VlasovBackwardEulerFormulation::clearSolution()
{
  _vlasovSoln->clear();
}

Teuchos::RCP<ConfigurationSpaceFormulation> VlasovBackwardEulerFormulation::configurationSpaceFormulation()
{
  return _configurationSpaceFormulation;
}

MeshPtr VlasovBackwardEulerFormulation::configurationSpaceMesh()
{
  return _configurationSpaceFormulation->configurationSpaceMesh();
}

MeshTopologyPtr VlasovBackwardEulerFormulation::configurationSpaceMeshTopology()
{
  return _configurationSpaceMeshTopology;
}

SolutionPtr VlasovBackwardEulerFormulation::configurationSpaceSolution()
{
  return _configurationSpaceFormulation->solution();
}

void VlasovBackwardEulerFormulation::configurationSpaceSolve()
{
  _solutionMoments->update();
  {
    double n0;
    if (_enforceQuasiNeutrality)
    {
      FunctionPtr zeroMoment   = _solutionMoments->solutionMomentFunction(0);
      auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
      const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
      const double f_integral  = zeroMoment->integrate(configurationMesh);
      n0 = f_integral / meshMeasure;
    }
    else
    {
      n0 = 0;
    }
    _n0->setValue(n0);
  }
  _configurationSpaceFormulation->solve(false); // false: don't update solution moments (we just did so, above)
  _configurationSpaceFormulation->updateConfigurationSpaceFunctions();
}

Teuchos::RCP<GMGSolver> VlasovBackwardEulerFormulation::constructGMGSolver()
{
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

int VlasovBackwardEulerFormulation::createConfigurationMeshSideSetTag(SpatialFilterPtr regionToMatch)
{
  const int thisTag = _nextConfigurationSpaceTag++;
  
  const int sideDim = _spaceDim-1;
  auto matchingSideSet = _configurationSpaceMeshTopology->createEntitySet();
  std::vector<IndexType> initialTimeSideIndices = _configurationSpaceMeshTopology->getSidesThatMatch(regionToMatch);
  for (const auto & initialTimeSideIndex : initialTimeSideIndices)
  {
    matchingSideSet->addEntity(sideDim, initialTimeSideIndex);
  }
  _configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, thisTag, matchingSideSet);
  
  return thisTag;
}

void VlasovBackwardEulerFormulation::didHRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool didRepartitionAndRebuild)
{
  _gmgSolver = Teuchos::null;
}

void VlasovBackwardEulerFormulation::didHUnrefine(const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
{
  _gmgSolver = Teuchos::null;
}

VarPtr VlasovBackwardEulerFormulation::E_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->E_hat(zeroBasedComp);
}

VarPtr VlasovBackwardEulerFormulation::f()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr VlasovBackwardEulerFormulation::f_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

bool VlasovBackwardEulerFormulation::hasL2Trace() const {
  return true;
}

const std::vector< std::vector<int> > & VlasovBackwardEulerFormulation::iterationCounts() const
{
  return _iterationCounts;
}

void VlasovBackwardEulerFormulation::refineUniformly(const int numRefinements)
{
  auto configMesh = configurationSpaceMesh();
  auto configActiveCellCount = configMesh->numActiveElements();
  
  int countMultiplier = 1;
  for (int refinementNumber = 0; refinementNumber<numRefinements; refinementNumber++)
  {
    // it should be the case that refinements in Vlasov induce refinements in configuration space mesh
    RefinementStrategy::hRefineUniformly(_vlasovSoln->mesh());
    configMesh->repartitionAndRebuild();
    countMultiplier *= 2;
  }
  
  // The assert below is only valid for 1D configuration space: it asserts that each uniform refinement multiplies element count by two.
  if (configMesh->getDimension() == 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(configMesh->numActiveElements() != configActiveCellCount * countMultiplier, std::invalid_argument, "Vlasov refinement should induce configuration-space refinements.");
  }
}

void VlasovBackwardEulerFormulation::setEnforceQuasiNeutrality(bool enforceQuasiNeutrality)
{
  _enforceQuasiNeutrality = enforceQuasiNeutrality;
}

void VlasovBackwardEulerFormulation::setFixedPointMaxSubsteps(const int maxSubsteps)
{
  _maxSubsteps = maxSubsteps;
}

void VlasovBackwardEulerFormulation::setFixedPointMinSubsteps(const int minSubsteps)
{
  _minSubsteps = minSubsteps;
}

void VlasovBackwardEulerFormulation::setFixedPointTolerance(const double fTol)
{
  _fixedPointTolerance = fTol;
}

void VlasovBackwardEulerFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void VlasovBackwardEulerFormulation::setIterativeSolveMaxIterations(const int maxIters)
{
  _maxIters = maxIters;
}

void VlasovBackwardEulerFormulation::setIterativeSolveTolerance(const double cgTol)
{
  _cgTol = cgTol;
}

void VlasovBackwardEulerFormulation::setIterativeSolveTolerances(const std::vector<double> cgTols)
{
  _cgTols = cgTols;
}

void VlasovBackwardEulerFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

void VlasovBackwardEulerFormulation::takeTimeStep()
{
  const int rank = MPIWrapper::CommWorld()->MyPID();
  _vlasovSolnPriorTime->setSolution(_vlasovSoln);
  
//  {
//    // DEBUGGING
//    std::cout << "Vlasov BF: " << _vlasovSoln->bf()->displayString() << std::endl;
//    std::cout << "Poisson BF: " << _configurationSpaceFormulation->solution()->bf()->displayString() << std::endl;
//    
//    _vlasovSoln->setWriteMatrixToFile(true, "/tmp/A_vlasov.dat");
//    _vlasovSoln->setWriteRHSToMatrixMarketFile(true, "/tmp/b_vlasov.dat");
//    _configurationSpaceFormulation->solution()->setWriteMatrixToFile(true, "/tmp/A_poisson.dat");
//    _configurationSpaceFormulation->solution()->setWriteRHSToMatrixMarketFile(true, "/tmp/b_poisson.dat");
//  }
  
  Teuchos::RCP<GMGSolver> gmgSolver;
  
  std::vector<int>    substepIterationCounts(_maxSubsteps);
  std::vector<double> substepStepSizesVlasov;
  
  if (_useGeometricMultigrid && (_gmgSolver == Teuchos::null))
  {
    _gmgSolver = constructGMGSolver();
  }
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, configurationViz;
  const bool exportViz = true; // for debugging of fixed-point iteration, set to true.  TODO: make this subject to external control.
  
  if (exportViz)
  {
    static int solveNumber = 0;
    std::string prefix = "VlasovBackwardEulerFormulation_";
    std::ostringstream vlasovSS;
    vlasovSS << prefix << "Vlasov" << solveNumber;
    
    std::ostringstream configurationSS;
    configurationSS << prefix << _configurationSpaceFormulation->name() << solveNumber;
      
    const std::string directoryPath = "/tmp";
    
    vlasovViz = Teuchos::rcp(new HDF5Exporter(this->vlasovMesh(), vlasovSS.str(), directoryPath));
    configurationViz = Teuchos::rcp(new HDF5Exporter (_configurationSpaceFormulation->configurationSpaceMesh(), configurationSS.str(), directoryPath) );
    
    configurationViz->exportSolution(_configurationSpaceFormulation->solution(),0);
    vlasovViz->exportSolution(vlasovSolution(),0);
    
    solveNumber++;
  }
    
  // get the PhysicalSolutionExtrusions ready to evaluate (in context of Vlasov BF)
  // updateConfigurationSpaceFunctions() will do MPI communication of configuration-space solution values so they're where they need to be for phase space mesh:
  _configurationSpaceFormulation->updateConfigurationSpaceFunctions();
  for (int substepOrdinal=0; substepOrdinal < _maxSubsteps; substepOrdinal++)
  {
    _vlasovSolnPrev->setSolution(_vlasovSoln);
    _configurationSpaceFormulation->copySolutionToPrevious();
    
    if (rank == 0)
    {
      std::cout << "\n\nSolving substep " << substepOrdinal+1 << " of up to " << _maxSubsteps << " (with early stop when f update is below " << _fixedPointTolerance << ")" << std::endl;
    }
    // solve for f:
    if (_gmgSolver == Teuchos::null)
    {
      _vlasovSoln->solve();
      substepIterationCounts[substepOrdinal] = -1; // -1 to indicate not an iterative solver
    }
    else
    {
      if (_cgTols.size() > 0) // then we should have one per substep
      {
        _gmgSolver->setTolerance(_cgTols[substepOrdinal]);
      }
      _vlasovSoln->solve(_gmgSolver);
      substepIterationCounts[substepOrdinal] = _gmgSolver->iterationCount();
    }
    configurationSpaceSolve();
    
    // Compute difference between current and previous substep solutions.  For substepOrdinal == 0, this is the difference with prior time step.
    
    auto vlasovMesh  = _vlasovSoln->mesh();
    const double fDiff_L2 = _fDiff->l2norm(vlasovMesh);
    const double f_L2 = _fSoln->l2norm(vlasovMesh);
    const double fRelDiff = fDiff_L2 / f_L2;
    
    const double configurationSpaceRelativeDiff = _configurationSpaceFormulation->l2NormOfSolutionUpdate();
    
    substepStepSizesVlasov.push_back(fRelDiff);
    
    if (rank == 0)
    {
      std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in f: " << fRelDiff << std::endl;
      std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in E and/or B: " << configurationSpaceRelativeDiff << std::endl;
    }
    
    if ((fRelDiff < _fixedPointTolerance) && (substepOrdinal >= _minSubsteps-1))
    {
      // tolerance reached, min substeps taken
      break;
    }
        
    if (exportViz)
    {
      configurationViz->exportSolution(_configurationSpaceFormulation->solution(),substepOrdinal+1);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal+1);
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  _fixedPointL2StepSizesVlasov.push_back(substepStepSizesVlasov);
}

void VlasovBackwardEulerFormulation::setInitialConditions(FunctionPtr f0)
{
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[this->f()->ID()] = f0;
  
  const int solutionOrdinal = 0;
  _vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  _vlasovSolnPrev->setSolution(_vlasovSoln);
  _vlasovSolnPriorTime->setSolution(_vlasovSoln);
  _t = 0;
  
  configurationSpaceSolve();
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in)
{
  auto vlasovBC = BC::bc();
  
  VarPtr t_n = this->t_n_hat();
  for (int vd=0; vd<_velocityDim; vd++)
  {
    const int tagMin = this->tagVlasovMinVelocity(vd);
    const int tagMax = this->tagVlasovMaxVelocity(vd);
    vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
    vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr v_dot_nx = Function::zero();
  for (int vd=0; vd<std::min(_velocityDim,_spaceDim); vd++)
  {
    FunctionPtr v_d = this->v_i(vd);
    FunctionPtr n_d = n->spatialComponent(vd+1); // 1-based index for Function::spatialComponent();
    v_dot_nx = v_dot_nx + v_d * n_d;
  }
  
  const int tagSpatialInflow = this->tagVlasovSpatialInflow();
  vlasovBC->addDirichlet(t_n, tagSpatialInflow, v_dot_nx * f_in);

  _vlasovSoln->setBC(vlasovBC);
  
  setInitialConditions(f0);
}

Teuchos::RCP<SolutionMoments> VlasovBackwardEulerFormulation::solutionMoments()
{
  return _solutionMoments;
}

// test variables:
VarPtr VlasovBackwardEulerFormulation::w()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

void VlasovBackwardEulerFormulation::setUseGeometricMultigrid(bool useGeometricMultigrid)
{
  _useGeometricMultigrid = useGeometricMultigrid;
}

int VlasovBackwardEulerFormulation::tagVlasovSpatialInflow() const
{
  return VLASOV_TAG_SPATIAL_INFLOW;
}

int VlasovBackwardEulerFormulation::tagVlasovMaxVelocity(int zeroBasedComp) const
{
  switch (zeroBasedComp)
  {
    case 0: return VLASOV_TAG_VXMAX;
    case 1: return VLASOV_TAG_VYMAX;
    case 2: return VLASOV_TAG_VZMAX;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zeroBasedComp must be 0, 1, 2")
  }
}

int VlasovBackwardEulerFormulation::tagVlasovMinVelocity(int zeroBasedComp) const
{
  switch (zeroBasedComp)
  {
    case 0: return VLASOV_TAG_VXMIN;
    case 1: return VLASOV_TAG_VYMIN;
    case 2: return VLASOV_TAG_VZMIN;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zeroBasedComp must be 0, 1, 2")
  }
}

VarPtr VlasovBackwardEulerFormulation::t_n_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

FunctionPtr VlasovBackwardEulerFormulation::v_i(int compZeroBased) const
{
  int vx_component_zero_based = _spaceDim;
  return MonomialFunction::xi_n(vx_component_zero_based + compZeroBased, 1);
}

VarPtr VlasovBackwardEulerFormulation::VE_configurationSpaceTrace()
{
  return _configurationSpaceFormulation->VE_hat();
}

FunctionPtr VlasovBackwardEulerFormulation::x_i(const int zeroBasedComp) const
{
  return MonomialFunction::xi_n(zeroBasedComp, 1);
}


// below, we set up unit tests against the private classes above
#include "Teuchos_UnitTestHarness.hpp"

namespace {
void testVelocitySpace1DIntegral(const bool &doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  double q      = 1.0;
  double eps0   = 1.0;
  double m      = 1.0;
  double x0     = 0.0;
  double xWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 1.0;
  
  const int xElements = 4;
  const int vElements = 4;
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  set<GlobalIndexType> vlasovCellsToRefine;
  if (doRefinement)
  {
    vlasovCellsToRefine = {0,5,8};
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
  VlasovBackwardEulerFormulation form(q, m, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
  auto poissonMesh = poissonConfig->configurationSpaceMesh();
  
  FunctionPtr x  = Function::xn(1);
  FunctionPtr y  = Function::yn(1);
  FunctionPtr xy = x * y;
  // integral of xy dy = x * y^2 / 2
  // we integrate from v0 to v0 + vWidth
  const double expectedIntegralY = (0.5 * (v0 + vWidth) * (v0 + vWidth) - 0.5 * v0 * v0);
  FunctionPtr expectedIntegral = x * expectedIntegralY;
  const double expectedIntegralXY = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0) * expectedIntegralY;
  const double actualIntegralXY   = expectedIntegral->integrate(poissonMesh);
  
  TEST_FLOATING_EQUALITY(expectedIntegralXY, actualIntegralXY, tol);
  
  if (vlasovCellsToRefine.size() > 0)
  {
    auto vlasovMesh = form.vlasovMesh();
    auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
    auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
    vlasovMesh->hRefine(vlasovCellsToRefine, refPattern);
  }
  
  auto f = form.f();
  std::map<int, FunctionPtr> projectionMap;
  projectionMap[f->ID()] = xy;
  
  auto vlasovSoln = form.vlasovSolution();
  const int solutionOrdinal = 0;
  vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
  const int maxMomentNumber = 0; // for now, we only do the zeroth moment
  SolutionMoments solutionMoments(maxMomentNumber, vlasovSoln, f, poissonMesh);
  auto velocitySpaceIntegral = solutionMoments.solutionMomentFunction(0); // one argument because 1V
  FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedIntegral;
  
  solutionMoments.update();
  
  double magnitudeExpected = expectedIntegral->l2norm(poissonMesh);
  double err_l2 = err->l2norm(poissonMesh);
  
  TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionVlasovPoissonBackwardEulerFormulation_1D1V, AnalyticIntegralMatches )
{
  const bool doRefinement = false;
  testVelocitySpace1DIntegral(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionVlasovPoissonBackwardEulerFormulation_1D1V, AnalyticIntegralMatchesWithRefinement )
{
  const bool doRefinement = true;
  testVelocitySpace1DIntegral(doRefinement, out, success);
}

void testVelocityExtrusion(bool volumeVariable, bool doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  double v0     = 0.0;
  double vWidth = 20.0;
  
  const int xElements = 2;
  const int vElements = 2;
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
  
  auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
  std::vector<double> vxVertices;
  for (int i=0; i<=vElements; i++)
  {
    const double vx = v0 + (i * vWidth) / vElements;
    vxVertices.push_back(vx);
  }
  VlasovBackwardEulerFormulation form(q, m, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
  auto poissonMesh = poissonConfig->configurationSpaceMesh();

  FunctionPtr x  = Function::xn(1);
  
  FunctionPtr soln_1D;
  
  auto poissonSoln = form.configurationSpaceSolution();
  std::map<int, FunctionPtr> projectionMap;
  const int X_COMP = 0;
  VarPtr E = poissonConfig->E_fieldVar(X_COMP);
  projectionMap[E->ID()] = x;
  soln_1D = Function::solution(E, poissonSoln);
  
  TEST_ASSERT(soln_1D->boundaryValueOnly() == false);
  
  auto vlasovMesh = form.vlasovMesh();
  
  auto extrusion = Teuchos::rcp( new PhysicalSolutionExtrusion(poissonSoln, E, vlasovMesh) );
  
  auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 1);
  
  if (doRefinement)
  {
    const set<GlobalIndexType> cellIDs_2D {0,1};
    auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
    auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
    vlasovMesh->hRefine(cellIDs_2D, refPattern);
  }
  
  const int solutionOrdinal =0;
  poissonSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  
  extrusion->updateValues();
  
  FunctionPtr expected = x;
  FunctionPtr err = FunctionPtr(extrusion) - expected;
  
//  double magnitudeActual = extrusion->l2norm(vlasovMesh);
//  std::cout << "magnitudeActual = " << magnitudeActual << std::endl;
  
  double magnitudeExpected = expected->l2norm(vlasovMesh);
//  std::cout << "magnitudeExpected = " << magnitudeExpected << std::endl;
  double err_l2 = err->l2norm(vlasovMesh);
  
  TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonBackwardEulerFormulation_1D1V, VolumeExtrusion )
{
  const bool volumeVariable = true;
  const bool doRefinement = false;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonBackwardEulerFormulation_1D1V, VolumeExtrusionWithRefinement )
{
  const bool volumeVariable = true;
  const bool doRefinement = true;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}
} // end anonymous namespace
