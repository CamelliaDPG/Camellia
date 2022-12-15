//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  VlasovSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 1/11/22.
//
//

#include "VlasovSpaceTimeFormulation.h"
#include "CamelliaMemoryTools.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "ParameterFunction.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SimpleSolutionFunction.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

#include "CamelliaDataHelpers.h"

using namespace Camellia;

const string VlasovSpaceTimeFormulation::S_F = "f";

const string VlasovSpaceTimeFormulation::S_F_HAT = "\\widehat{f}";
const string VlasovSpaceTimeFormulation::S_T_N_HAT = "\\widehat{t}_n";

const string VlasovSpaceTimeFormulation::S_W = "w";

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
      
      const int numCells  = normals.dimension(0);
      const int numPoints = normals.dimension(1);
      FieldContainer<double> convectionComponentValues(normals.dimension(0),normals.dimension(1)); // C,P
      FieldContainer<double> convectionDottedWithNormal(normals.dimension(0),normals.dimension(1)); // C,P
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
      
      return somePointMatches;
    }
    
    void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Not supported");
    }
  };
}


VlasovSpaceTimeFormulation::VlasovSpaceTimeFormulation(double q, double m,
                                                       bool useSerendipityBasis,
                                                       Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                       MeshTopologyPtr spatialMeshTopology,
                                                       std::vector<double> temporalVertices,
                                                       std::vector<double> velocitySpaceVertices_x,
                                                       std::vector<double> velocitySpaceVertices_y,
                                                       std::vector<double> velocitySpaceVertices_z,
                                                       const int H1Order, const int delta_k,
                                                       const int numUniformRefinements)
:
_spaceDim(spatialMeshTopology->getDimension()),
_useSerendipityBasis(useSerendipityBasis),
_H1Order(H1Order),
_delta_k(delta_k),
_configurationSpaceFormulation(configurationSpaceFormulation),
_q(q),
_m(m),
_velocitySpaceVertices_x(velocitySpaceVertices_x),
_velocitySpaceVertices_y(velocitySpaceVertices_y),
_velocitySpaceVertices_z(velocitySpaceVertices_z),
_spatialMeshTopology(spatialMeshTopology),
_numUniformRefinements(numUniformRefinements)
{
  _t0 = temporalVertices[0];
  _t1 = temporalVertices[temporalVertices.size()-1];
  
  _slabTemporalCoordinates = {_t0, _t1}; // will be overwritten by setSlabHistory(), if that's called (i.e., if this is not the initial slab).
  
  _configurationSpaceMeshTopology = MeshFactory::extrudedMeshTopology(spatialMeshTopology, {temporalVertices});
  _configurationSpaceFormulation->initializeMesh(_configurationSpaceMeshTopology, H1Order, delta_k);
  
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

  // tests
  VarPtr w;   // Vlasov test function

  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();
  
  // Vlasov equation/mesh will not "know" about "space-time" as such -- no variables will be marked as space-time.
  // We can do this because the flux variable (t_n_hat) is defined on every interface, so we don't need the weird
  // only-supported-on-spatial-interfaces thing that we need for space-time Poisson.
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  BasisFamilyPtr basisFamily;
  if (_useSerendipityBasis)
  {
    basisFamily = BasisFamily::serendipityBasisFamily();
  }
  else
  {
    basisFamily = BasisFamily::hierarchicalBasisFamily();
  }
  
  f = vlasovVF->fieldVar(S_F);
  w = vlasovVF->testVar(S_W, HGRAD); // w lies in HGRAD of the whole space-time phase space
  
  t_n_hat = vlasovVF->fluxVar(S_T_N_HAT);
  
  vlasovVF->setTrialBasisFamily(basisFamily);
  vlasovVF->setTestBasisFamily(basisFamily);
  _vlasovBF = BF::bf(vlasovVF);
  
  map<int,int> trialOrderEnhancements;
  trialOrderEnhancements[t_n_hat->ID()] = 1; // +1 for non-conforming, + whatever user requested

  map<int,int> testOrderEnhancements;
  
  const bool allowPruning = false;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  Epetra_CommPtr comm = MPIWrapper::CommWorld();
  MeshPtr vlasovMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _vlasovBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
  
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  _vlasovSoln     = Solution::solution(_vlasovBF,vlasovMesh);
  _vlasovSolnPrev = Solution::solution(_vlasovBF,vlasovMesh); // will copy out after fixed-point solve
  
  const int maxMomentNumber = _configurationSpaceFormulation->maxMomentNumber();
  _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln, f, _configurationSpaceFormulation->configurationSpaceMesh()) );
  
  _n0 = ParameterFunction::parameterFunction(0); // we'll update this when initial conditions are set, if _enforceQuasiNeutrality is true
  _configurationSpaceFormulation->setMoments(_solutionMoments,_n0);
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + _spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  // SpatialFilter's components are zero-based
//  const int indexOffsetSpaceZeroBased    = indexOffsetSpaceOneBased  - 1;
  const int indexOffsetTimeZeroBased     = indexOffsetTimeOneBased - 1;
  const int indexOffsetVelocityZeroBased = indexOffsetVelocityOneBased - 1;
  
  // The below assumes that _spaceDim <= _velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim > _velocityDim, std::invalid_argument, "_velocityDim must be at least as large as _spaceDim");
  std::vector<FunctionPtr> v;
  for (int i=0; i<_velocityDim; i++)
  {
    v.push_back(this->v_i(i));
  }
  for (int i=0; i<_spaceDim; i++)
  {
    _vlasovBF->addTerm( -v[i] * f, w->di(indexOffsetSpaceOneBased + i));  // dx_i
  }
  _vlasovBF->addTerm( -f, w->di(indexOffsetTimeOneBased)); // dt
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
    grad_v_weight[i] = (-q/m) * (E[i] + v_cross_B[i]);
  }
  
  for (int i=0; i<_velocityDim; i++)
  {
    // grad v of f
    _vlasovBF->addTerm(grad_v_weight[i] * f, w->di(indexOffsetVelocityOneBased + i));
//    _vlasovBF->addTerm(-(q/m)*_E_extruded*f, w->di(VELOCITY_INDEX)); // dv
  }
  
  _vlasovBF->addTerm(t_n_hat, w);

  auto vlasovSolnWeakPtr     = Teuchos::rcp(_vlasovSoln.get(),     false);
  auto vlasovSolnPrevWeakPtr = Teuchos::rcp(_vlasovSolnPrev.get(), false);
  
  _fSoln     = Function::solution(f, vlasovSolnWeakPtr);
  FunctionPtr f_solnPrev = Function::solution(f, vlasovSolnPrevWeakPtr);
  
  _fDiff = _fSoln - f_solnPrev;
    
  auto vlasovRHS = RHS::rhs();
  _vlasovSoln->setRHS(vlasovRHS); // zero RHS for vlasov, until we add collisions
    
  vlasovMesh->registerSolution(_vlasovSoln);
    
  IPPtr vlasovTestNorm = _vlasovBF->graphNorm();
  _vlasovSoln->setIP(vlasovTestNorm);
  
  SpatialFilterPtr inflowFilter = Teuchos::rcp(new SpatialFilterSpatialInflow(_spaceDim,indexOffsetVelocityZeroBased) );
  
  double t0 = temporalVertices[0];
  SpatialFilterPtr initialTimeFilter = SpatialFilter::matchingComponent(indexOffsetTimeZeroBased, t0);
  
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
    
    // initial conditions
    const int sideDim = _spaceDim + _velocityDim; // phase space, space-time: sides have dimension of (fixed-time) phase space
    auto initialTimeSideSet = extrudedMeshTopology->createEntitySet();
    std::vector<IndexType> initialTimeSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(initialTimeFilter);
    for (const auto & initialTimeSideIndex : initialTimeSideIndices)
    {
      initialTimeSideSet->addEntity(sideDim, initialTimeSideIndex);
    }
    extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_TMIN, initialTimeSideSet);
    
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
  // create initial-time tag for configuration-space mesh
  {
    const int sideDim = _spaceDim; // space-time configuration-space mesh has dimension _spaceDim + 1; sides have dimension _spaceDim
    auto initialTimeSideSet = _configurationSpaceMeshTopology->createEntitySet();
    std::vector<IndexType> initialTimeSideIndices = _configurationSpaceMeshTopology->getBoundarySidesThatMatch(initialTimeFilter);
    for (const auto & initialTimeSideIndex : initialTimeSideIndices)
    {
      initialTimeSideSet->addEntity(sideDim, initialTimeSideIndex);
    }
    _configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, CONFIGURATION_TAG_TMIN, initialTimeSideSet);
  }
  
  PhysicalMeshUpdater::registerForUpdates(_vlasovSoln->mesh(), _configurationSpaceFormulation->configurationSpaceMesh());
  
  for (int uniformRefinementNumber=0; uniformRefinementNumber < _numUniformRefinements; uniformRefinementNumber++)
  {
    const auto & activeCells  = _vlasovSoln->mesh()->getActiveCellIDsGlobal();
    _vlasovSoln->mesh()->hRefine(activeCells);
    // (the configuration-space mesh will automatically be refined when the Vlasov mesh is, by virtue of the registerForUpdates() call above)
  }
}

MeshPtr VlasovSpaceTimeFormulation::vlasovMesh()
{
  return _vlasovSoln->mesh();
}

SolutionPtr VlasovSpaceTimeFormulation::vlasovSolution()
{
  return _vlasovSoln;
}

void VlasovSpaceTimeFormulation::addConfigurationSpaceBC(const int tag, VarPtr configVariable, FunctionPtr dirichletValue)
{
  if (_configurationSpaceFormulation->solution()->bc() == Teuchos::null)
  {
    // create new BC object
    _configurationSpaceFormulation->solution()->setBC(BC::bc());
  }
  _configurationSpaceFormulation->solution()->bc()->addDirichlet(configVariable, tag, dirichletValue);
  
  _configurationSpaceBCs.push_back(std::make_tuple(tag,configVariable,dirichletValue));
}

VarPtr VlasovSpaceTimeFormulation::B_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->B_hat(zeroBasedComp);
}

void VlasovSpaceTimeFormulation::clearConfigurationSpaceBCs()
{
  _configurationSpaceFormulation->solution()->setBC(BC::bc());
}

void VlasovSpaceTimeFormulation::clearSolution()
{
  _vlasovSoln->clear();
}

void VlasovSpaceTimeFormulation::computeConservationDiagnostics(bool updateSolutionMoments)
{
  if (updateSolutionMoments)
  {
    _solutionMoments->update();
  }
  
  // record conservation diagnostics
  int slabNumber = this->slabOrdinal();
  
  // ensure that our containers are big enough:
  _massIntegrals.resize(slabNumber+1, Teuchos::null);
  _momentumIntegrals.resize(_spaceDim);
  for (int d=0; d<_spaceDim; d++)
  {
    _momentumIntegrals[d].resize(slabNumber+1, Teuchos::null);
  }
  _energyIntegrals.resize(slabNumber+1, Teuchos::null);
  
  if (_maxMomentForConservationDiagnostics >= 0)
  {
    // create Solution object
    VarFactoryPtr varFactory = VarFactory::varFactory();
    
    VarPtr massVar;
    std::vector<VarPtr> momentumVar(_spaceDim);
    VarPtr energyVar;
    
    std::map<int, FunctionPtr> projectionMap;
    if (_density != Teuchos::null)
    {
      massVar = varFactory->fieldVar("mass");
      projectionMap[massVar->ID()] = _density;
    }
    for (int d=0; d<_spaceDim; d++)
    {
      if (_momentum[d] != Teuchos::null)
      {
        momentumVar[d] = varFactory->fieldVar("momentum" + std::to_string(d));
        projectionMap[momentumVar[d]->ID()] = _momentum[d];
      }
    }
    if (_energy != Teuchos::null)
    {
      energyVar = varFactory->fieldVar("energy");
      projectionMap[energyVar->ID()] = _energy;
    }
    
    auto H1Order = _configurationSpaceFormulation->configurationSpaceMesh()->globalDofAssignment()->getInitialH1Order();
    auto delta_k = 0; // since we're not solving, no enrichment required.  (Also, we don't have any test variables.)
    MeshPtr conservationDiagnosticsMesh = Teuchos::rcp(new Mesh(_configurationSpaceMeshTopology, varFactory, H1Order, delta_k) );
    
    SolutionPtr diagnosticSolution = Solution::solution(conservationDiagnosticsMesh);
    diagnosticSolution->projectOntoMesh(projectionMap);
            
    FunctionPtr one = Function::constant(1.0);
    const int quadratureEnrichment = 0;
    _massIntegrals[slabNumber] = Teuchos::rcp( new SpatialIntegralOfSpaceTimeSolution(diagnosticSolution, massVar, one, quadratureEnrichment) );
    
    if (_maxMomentForConservationDiagnostics >= 1)
    {
      for (int d=0; d<_spaceDim; d++)
      {
        _momentumIntegrals[d][slabNumber] = Teuchos::rcp( new SpatialIntegralOfSpaceTimeSolution(diagnosticSolution, momentumVar[d], one, quadratureEnrichment) );
      }
    }
    
    if (_maxMomentForConservationDiagnostics >= 2)
    {
      _energyIntegrals[slabNumber] = Teuchos::rcp( new SpatialIntegralOfSpaceTimeSolution(diagnosticSolution, energyVar, one, quadratureEnrichment) );
    }
  }
}

Teuchos::RCP<ConfigurationSpaceFormulation> VlasovSpaceTimeFormulation::configurationSpaceFormulation()
{
  return _configurationSpaceFormulation;
}

MeshPtr VlasovSpaceTimeFormulation::configurationSpaceMesh()
{
  return _configurationSpaceFormulation->configurationSpaceMesh();
}

SolutionPtr VlasovSpaceTimeFormulation::configurationSpaceSolution()
{
  return _configurationSpaceFormulation->solution();
}

Teuchos::RCP<GMGSolver> VlasovSpaceTimeFormulation::constructGMGSolver()
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

int VlasovSpaceTimeFormulation::createConfigurationMeshSideSetTag(SpatialFilterPtr regionToMatch, const bool restrictToBoundary)
{
  const int thisTag = _nextConfigurationSpaceTag++;
  
  const int sideDim = _spaceDim; // space-time configuration-space mesh has dimension _spaceDim + 1; sides have dimension _spaceDim
  auto matchingSideSet = _configurationSpaceMeshTopology->createEntitySet();
  std::vector<IndexType> matchingSideIndices = restrictToBoundary ? _configurationSpaceMeshTopology->getBoundarySidesThatMatch(regionToMatch)
                                                                  : _configurationSpaceMeshTopology->getSidesThatMatch        (regionToMatch);
  for (const auto & matchingSideIndex : matchingSideIndices)
  {
    matchingSideSet->addEntity(sideDim, matchingSideIndex);
  }
  _configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, thisTag, matchingSideSet);
  
  _configurationSpaceSideSetFilters.push_back({regionToMatch,restrictToBoundary});
  
  return thisTag;
}

VarPtr VlasovSpaceTimeFormulation::E_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->E_hat(zeroBasedComp);
}

bool VlasovSpaceTimeFormulation::enforceQuasiNeutrality() const
{
  return _enforceQuasiNeutrality;
}

bool VlasovSpaceTimeFormulation::exportViz() const
{
  return _exportViz;
}

VarPtr VlasovSpaceTimeFormulation::f()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr VlasovSpaceTimeFormulation::f_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

int VlasovSpaceTimeFormulation::fixedPointMaxSubsteps() const
{
  return _maxSubsteps;
}

int VlasovSpaceTimeFormulation::fixedPointMinSubsteps() const
{
  return _minSubsteps;
}

double VlasovSpaceTimeFormulation::fixedPointTolerance() const
{
  return _fixedPointTolerance;
}

bool VlasovSpaceTimeFormulation::hasL2Trace() const {
  return true;
}

int VlasovSpaceTimeFormulation::iterativeSolveMaxIterations() const
{
  return _maxIters;
}

int VlasovSpaceTimeFormulation::iterativeSolveOutputLevel() const
{
  return _aztecOutputLevel;
}

double VlasovSpaceTimeFormulation::iterativeSolveTolerance() const
{
  return _cgTol;
}

std::vector<double> VlasovSpaceTimeFormulation::iterativeSolveTolerances() const
{
  return _cgTols;
}

int VlasovSpaceTimeFormulation::maxMomentForConservationDiagnostics() const
{
  return _maxMomentForConservationDiagnostics;
}

Teuchos::RCP<VlasovSpaceTimeFormulation> VlasovSpaceTimeFormulation::nextTimeSlab(double tWidth, int numTemporalElements)
{
  auto temporalVertices = MeshTools::verticesFor1DIntervalMesh(_t1, tWidth, numTemporalElements);
  
  // create a new spatial MeshTopology based on the root of the current _spatialMeshTopology
  auto rootSpatialMeshTopology = MeshFactory::rootMeshTopologyCopy(_spatialMeshTopology);
  
  auto nextSlabForm = Teuchos::rcp(new VlasovSpaceTimeFormulation(_q, _m, _useSerendipityBasis, _configurationSpaceFormulation, rootSpatialMeshTopology,
                                                                  temporalVertices, _velocitySpaceVertices_x, _velocitySpaceVertices_y, _velocitySpaceVertices_z,
                                                                  _H1Order, _delta_k, _numUniformRefinements));
  
  auto nextSlabVlasovMesh = nextSlabForm->vlasovMesh(); // "from" in cellIDMap
  auto thisSlabVlasovMesh = this->vlasovMesh();
  
  std::map<IndexType,IndexType> cellIDMap;
  std::map<int,int> sideOrdinalMap;
  
  MeshTools::hRefineToMatchAtBoundary(cellIDMap, sideOrdinalMap, nextSlabVlasovMesh, VLASOV_TAG_TMIN, thisSlabVlasovMesh);
  
  VarPtr tn = this->t_n_hat();
  bool weightFluxesBySideParity = false; // should not matter: boundary parity is 1
  SolutionPtr thisSlabVlasovSoln = this->vlasovSolution();
  auto tnInterfaceFxn = Teuchos::rcp( new SimpleSolutionFunction<double>(tn, thisSlabVlasovSoln, weightFluxesBySideParity) );
  tnInterfaceFxn->setCellIDMap(cellIDMap, sideOrdinalMap); // this map translates from the nextSlabVlasovMesh's numbering to thisSlabVlasovMesh's.
  // at temporal outflow, the normal in "tn" is positive, so tn = f.  This means we can simply use tnInterfaceFxn as "f0" for the next slab.
  FunctionPtr f0 = tnInterfaceFxn;
  
  if (_f_inflow != Teuchos::null)
  {
    nextSlabForm->setInitialAndInflowConditions(f0, _f_inflow);
  }
  else
  {
    nextSlabForm->setInitialConditions(f0);
  }
  
  for (auto sideSetTagFilter : _configurationSpaceSideSetFilters)
  {
    nextSlabForm->createConfigurationMeshSideSetTag(sideSetTagFilter.first,sideSetTagFilter.second);
  }
  
  for (auto entry : _configurationSpaceBCs)
  {
    int tag           = std::get<0>(entry);
    VarPtr var        = std::get<1>(entry);
    FunctionPtr value = std::get<2>(entry);
    
    nextSlabForm->addConfigurationSpaceBC(tag, var, value);
  }
  
  // copy my settings:
  nextSlabForm->setIterativeSolveOutputLevel(_aztecOutputLevel);
  nextSlabForm->setIterativeSolveTolerance(_cgTol);
  nextSlabForm->setIterativeSolveTolerances(_cgTols);
  nextSlabForm->setUseGeometricMultigrid(_useGeometricMultigrid);
  nextSlabForm->setEnforceQuasiNeutrality(_enforceQuasiNeutrality);
  nextSlabForm->setIterativeSolveMaxIterations(_maxIters);
  nextSlabForm->setFixedPointMaxSubsteps(_maxSubsteps);
  nextSlabForm->setFixedPointMinSubsteps(_minSubsteps);
  nextSlabForm->setFixedPointTolerance(_fixedPointTolerance);
  nextSlabForm->setExportViz(_exportViz,_vizOutputDir);
  nextSlabForm->setMaxMomentForConservationDiagnostics(_maxMomentForConservationDiagnostics);
  
  // copy slab/diagnostic history:
  nextSlabForm->setSlabHistory(_slabTemporalCoordinates, _massIntegrals, _momentumIntegrals, _energyIntegrals);
  
  // import solution data so that it's available for BC setting
  std::set<GlobalIndexType> currentCellIDsForRank;
  const auto &myNextCellIDs = nextSlabVlasovMesh->cellIDsInPartition();
  for (const auto &myNextCellID : myNextCellIDs)
  {
    auto mapEntry = cellIDMap.find(myNextCellID); // all rank-local cells which lie on the final-time boundary should have an entry
    if (mapEntry != cellIDMap.end())
    {
      currentCellIDsForRank.insert(mapEntry->second);
    }
  }
  _vlasovSoln->importSolutionForOffRankCells(currentCellIDsForRank);
  
  /*
   Note: the above will work if there is not cell migration or refinement after we return nextSlabForm.
         If nextSlabForm gets refined, then a new pass of hRefineToMatchAtBoundary() will be needed, along
         with the above import.  Probably the right solution going forward will be to separate out this
         functionality into a new helper method, and call it both from here and upon mesh refinement;
         doing the latter transparently will require that the next slab has a reference to the previous.
         So that we don't require all slabs to stay in memory, we'll want to nullify this reference during
         nextTimeSlab().  (Also need to nullify the BC for the current slab -- as we do below -- for the
         same reason: it refers to the previous slab's solution, which has a BC for the previous one, etc.
         So we have a linked list of Solution objects; need to break those links to free up the memory.)
   */
  // nullify the BC for current slab; if it's not the first slab, it will have reference to a prior slab's solution,
  // and on back to the first slab, unless we nullify.  We could also call setInitialConditions(Function::zero()) or
  // something like that, but setting to null has the advantage that we'll fail fast if the user tries to do a solve
  // on the current slab after calling nextTimeSlab() -- it will crash with a null pointer exception.
  _vlasovSoln->setBC(Teuchos::null);
  
  return nextSlabForm;
}

void VlasovSpaceTimeFormulation::setEnforceQuasiNeutrality(bool enforceQuasiNeutrality)
{
  _enforceQuasiNeutrality = enforceQuasiNeutrality;
}

void VlasovSpaceTimeFormulation::setExportViz(bool exportViz, const std::string outputDir)
{
  _exportViz = exportViz;
  _vizOutputDir = outputDir;
}

void VlasovSpaceTimeFormulation::setFixedPointMaxSubsteps(const int maxSubsteps)
{
  _maxSubsteps = maxSubsteps;
}

void VlasovSpaceTimeFormulation::setFixedPointMinSubsteps(const int minSubsteps)
{
  _minSubsteps = minSubsteps;
}

void VlasovSpaceTimeFormulation::setFixedPointTolerance(const double fTol)
{
  _fixedPointTolerance = fTol;
}

void VlasovSpaceTimeFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void VlasovSpaceTimeFormulation::setIterativeSolveMaxIterations(const int maxIters)
{
  _maxIters = maxIters;
}

void VlasovSpaceTimeFormulation::setIterativeSolveTolerance(const double cgTol)
{
  _cgTol = cgTol;
}

void VlasovSpaceTimeFormulation::setIterativeSolveTolerances(const std::vector<double> cgTols)
{
  _cgTols = cgTols;
}

void VlasovSpaceTimeFormulation::solve()
{
  const int rank = MPIWrapper::CommWorld()->MyPID();
  
  Teuchos::RCP<GMGSolver> gmgSolver;
  
  std::vector<int>    substepIterationCounts(_maxSubsteps);
  std::vector<double> substepStepSizesVlasov;
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, configurationViz;
  
  if (_exportViz)
  {
    static int solveNumber = 0;
    std::string prefix = "VlasovSpaceTimeFormulation_";
    std::ostringstream vlasovSS;
    vlasovSS << prefix << "Vlasov" << solveNumber;
    
    std::ostringstream configurationSS;
    configurationSS << prefix << _configurationSpaceFormulation->name() << solveNumber;
      
    const std::string directoryPath = "/tmp";
    
    vlasovViz = Teuchos::rcp(new HDF5Exporter(this->vlasovMesh(), vlasovSS.str(), directoryPath));
    configurationViz = Teuchos::rcp(new HDF5Exporter (_configurationSpaceFormulation->configurationSpaceMesh(), configurationSS.str(), directoryPath) );
    solveNumber++;
  }
  
  // get the PhysicalSolutionExtrusions ready to evaluate (in context of Vlasov BF)
  // updateConfigurationSpaceFunctions() will do MPI communication of configuration-space solution values so they're where they need to be for phase space mesh:
  _configurationSpaceFormulation->updateConfigurationSpaceFunctions();
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
    _configurationSpaceFormulation->updateConfigurationSpaceFunctions(); // update/MPI-communicate (Poisson/Maxwell) values to make available for phase-space solution
    
    // Copy vlasov, Poisson solution to Prev solutions.  For substepOrdinal > 0, measure difference between the two solutions first.
    if (substepOrdinal > 0)
    {
      auto vlasovMesh  = _vlasovSoln->mesh();
      const double fDiff_L2 = _fDiff->l2norm(vlasovMesh);
      const double f_L2 = _fSoln->l2norm(vlasovMesh);
      const double fRelDiff = fDiff_L2 / f_L2;
      
      substepStepSizesVlasov.push_back(fRelDiff);
      
      if (rank == 0)
      {
        std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in f: " << fRelDiff << std::endl;
      }
      
      if ((fRelDiff < _fixedPointTolerance) && (substepOrdinal >= _minSubsteps-1))
      {
        // tolerance reached, min substeps taken
        break;
      }
    }
    
    _vlasovSolnPrev->setSolution(_vlasovSoln);
    
    if (_exportViz)
    {
      configurationViz->exportSolution(_configurationSpaceFormulation->solution(),substepOrdinal);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal);
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  _fixedPointL2StepSizesVlasov.push_back(substepStepSizesVlasov);
  
  computeConservationDiagnostics(false); // false: don't update solutionMoments; it was just updated above
}

void VlasovSpaceTimeFormulation::setInitialConditions(FunctionPtr f0)
{
  _f0 = f0;
  
  auto vlasovBC = BC::bc();
  
  VarPtr t_n = this->t_n_hat();
  for (int vd=0; vd<_velocityDim; vd++)
  {
    int tagMin = this->tagVlasovMinVelocity(vd);
    int tagMax = this->tagVlasovMaxVelocity(vd);
    vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
    vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
  }
  
  int tagInitialTime = this->tagVlasovInitialTime();
  vlasovBC->addDirichlet(t_n, tagInitialTime, -f0); // minus sign thanks to outward normal on temporal interface
  
  _vlasovSoln->setBC(vlasovBC);
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
    const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
    const double f_integral  = f0->integrate(_vlasovSoln->mesh());
    n0 = f_integral / meshMeasure;
  }
  else
  {
    n0 = 0;
  }
  _n0->setValue(n0);
}

void VlasovSpaceTimeFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in)
{
  _f0 = f0;
  _f_inflow = f_in;
  
  auto vlasovBC = BC::bc();
  
  VarPtr t_n = this->t_n_hat();
  for (int vd=0; vd<_velocityDim; vd++)
  {
    const int tagMin = this->tagVlasovMinVelocity(vd);
    const int tagMax = this->tagVlasovMaxVelocity(vd);
    vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
    vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
  }
  
  const int tagInitialTime = this->tagVlasovInitialTime();
  vlasovBC->addDirichlet(t_n, tagInitialTime, -f0); // minus sign thanks to outward normal on temporal interface
  
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
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
    const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
    const double f_integral  = f0->integrate(_vlasovSoln->mesh());
    n0 = f_integral / meshMeasure;
  }
  else
  {
    n0 = 0;
  }
  _n0->setValue(n0);
}

void VlasovSpaceTimeFormulation::setMaxMomentForConservationDiagnostics(const int maxMomentForConservationDiagnostics)
{
  _maxMomentForConservationDiagnostics = maxMomentForConservationDiagnostics;
  
  if (_solutionMoments->maxMomentNumber() < maxMomentForConservationDiagnostics)
  {
    // need to re-create _solutionMoments
    _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentForConservationDiagnostics, _vlasovSoln, f(), _configurationSpaceFormulation->configurationSpaceMesh()) );
    _configurationSpaceFormulation->setMoments(_solutionMoments,_n0);
  }
  
  _momentum.resize(_spaceDim);
  
  if (_spaceDim == 1)
  {
    _density     = (maxMomentForConservationDiagnostics >= 0) ? _solutionMoments->solutionMomentFunction(0) : Teuchos::null;
    _momentum[0] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments->solutionMomentFunction(1) : Teuchos::null;
    _energy      = (maxMomentForConservationDiagnostics >= 2) ? _solutionMoments->solutionMomentFunction(2) : Teuchos::null;
  }
  else if (_spaceDim == 2)
  {
    _density     = (maxMomentForConservationDiagnostics >= 0) ? _solutionMoments->solutionMomentFunction(0,0) : Teuchos::null;
    _momentum[0] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments->solutionMomentFunction(1,0) : Teuchos::null;
    _momentum[1] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments->solutionMomentFunction(0,1) : Teuchos::null;
    if (maxMomentForConservationDiagnostics < 2)
    {
      _energy = Teuchos::null;
    }
    else
    {
      _energy = _solutionMoments->solutionMomentFunction(2,0) + _solutionMoments->solutionMomentFunction(0,2);
    }
  }
  else if (_spaceDim == 3)
  {
    _density     = (maxMomentForConservationDiagnostics >= 0) ? _solutionMoments->solutionMomentFunction(0,0,0) : Teuchos::null;
    _momentum[0] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments->solutionMomentFunction(1,0,0) : Teuchos::null;
    _momentum[1] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments->solutionMomentFunction(0,1,0) : Teuchos::null;
    _momentum[2] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments->solutionMomentFunction(0,0,1) : Teuchos::null;
    if (maxMomentForConservationDiagnostics < 2)
    {
      _energy = Teuchos::null;
    }
    else
    {
      _energy = _solutionMoments->solutionMomentFunction(2,0,0) + _solutionMoments->solutionMomentFunction(0,2,0) + _solutionMoments->solutionMomentFunction(0,0,2);
    }
  }
}

Teuchos::RCP<SolutionMoments> VlasovSpaceTimeFormulation::solutionMoments()
{
  return _solutionMoments;
}

void VlasovSpaceTimeFormulation::setSlabHistory(const std::vector<double> &slabTemporalCoordinates,
                                                const std::vector<Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution>> &massIntegrals,
                                                const std::vector<std::vector<Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution>>> &momentumIntegrals,
                                                const std::vector<Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution>> &energyIntegrals)
{
  _slabTemporalCoordinates = slabTemporalCoordinates;
  _massIntegrals = massIntegrals;
  _momentumIntegrals = momentumIntegrals;
  _energyIntegrals = energyIntegrals;
  
  // last listed temporal coordinate should match our start coordinate
  double tol = 1e-14;
  double t0Diff = std::abs(_slabTemporalCoordinates[_slabTemporalCoordinates.size()-1] - _t0);
  TEUCHOS_TEST_FOR_EXCEPTION(t0Diff > tol, std::invalid_argument, "slabTemporalCoordinates: final entry must match this slab's start time");
  _slabTemporalCoordinates.push_back(_t1);
}

int VlasovSpaceTimeFormulation::slabOrdinal() const
{
  return _slabTemporalCoordinates.size() - 2;
}

// test variables:
VarPtr VlasovSpaceTimeFormulation::w()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

void VlasovSpaceTimeFormulation::setUseGeometricMultigrid(bool useGeometricMultigrid)
{
  _useGeometricMultigrid = useGeometricMultigrid;
}

FunctionPtr VlasovSpaceTimeFormulation::t() const
{
  int tCompZeroBased = _spaceDim;
  return MonomialFunction::xi_n(tCompZeroBased, 1);
}

int VlasovSpaceTimeFormulation::tagConfigurationInitialTime() const
{
  return CONFIGURATION_TAG_TMIN;
}

int VlasovSpaceTimeFormulation::tagVlasovInitialTime() const
{
  return VLASOV_TAG_TMIN;
}

int VlasovSpaceTimeFormulation::tagVlasovSpatialInflow() const
{
  return VLASOV_TAG_SPATIAL_INFLOW;
}

int VlasovSpaceTimeFormulation::tagVlasovMaxVelocity(int zeroBasedComp) const
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

int VlasovSpaceTimeFormulation::tagVlasovMinVelocity(int zeroBasedComp) const
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

vector<pair<double,double>> VlasovSpaceTimeFormulation::timeSeriesForSpatialIntegral(Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution> integral,
                                                                                     const double &t0, const double &dt, const int &numValues) const
{
  auto temporalSolution         = integral->temporalSolution();
  auto temporalSolutionFunction = integral->temporalFunction();
  auto mesh = temporalSolution->mesh();
  auto activeCells = mesh->getTopology()->getLocallyKnownActiveCellIndices(); // we assume mesh geometry is not distributed
  temporalSolution->importSolutionForOffRankCells(activeCells);
  
  // confirm that the mesh geometry is not distributed:
  const int numActiveCells = activeCells.size();
  TEUCHOS_TEST_FOR_EXCEPTION(mesh->numActiveElements() != numActiveCells, std::invalid_argument, "Error: not all active cell indices are locally known");
  
  const double tol = 1e-10;
  
  std::map<IndexType, vector<double> > pointsForCell;
  
  const double t1 = t0 + dt * (numValues - 1);
  
  vector<double> allPoints;
  {
    double t = t0;
    int t_ordinal = 0;
    while (t <= t1 + tol)
    {
      allPoints.push_back(t);
      t_ordinal++;
      t = t0 + t_ordinal * dt;
    }
  }
  const int numPoints = allPoints.size();
  
  vector<pair<double,double>> fxnValues(numPoints);
  if (numPoints == 0) return fxnValues;

  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(CellTopology::line(), 1);
  basisCache->setMesh(mesh);

  auto meshTopo = mesh->getTopology();
  
  FieldContainer<double> vertices(2,1); // (N,D)
  for (const auto &cellID : activeCells)
  {
    meshTopo->verticesForCell(vertices, cellID);
    double t_left  = vertices(0,0);
    double t_right = vertices(1,0);
    if (t_right != t1)
    {
      // rightmost cell owns its final point; intermediate cells do not
      t_right -= tol * 2;
    }

    // determine the first matching point that intersects the cell (if any)
    int t_firstOrdinal = (t_left - t0 + tol) / dt; // in allPoints
    if (allPoints[t_firstOrdinal] + tol < t_left)
    {
      // then go to the next point
      t_firstOrdinal = (t_left - t0 + dt + tol) / dt; // in allPoints
    }
    if (t_firstOrdinal >= numPoints) continue; // can't have any matches in this cell, then
    TEUCHOS_TEST_FOR_EXCEPTION(allPoints[t_firstOrdinal] + tol < t_left, std::logic_error, "logic error determining t_firstOrdinal");

    vector<double> points;

    double t = dt * t_firstOrdinal + t0;
    while (t < t_right + tol)
    {
      points.push_back(t);
      t += dt;
    }
    pointsForCell[cellID] = points;

    double refCellScaling = 2.0 / (t_right - t_left);
    int numPointsForCell = points.size();
    
    if (numPointsForCell == 0) continue;
    
    FieldContainer<double> refCellPoints(numPointsForCell,1);
    for (int pointOrdinal=0; pointOrdinal<numPointsForCell; pointOrdinal++)
    {
      const auto &point = points[pointOrdinal];
      double refPoint = (point - t_left) * refCellScaling - 1.0; // -1.0 is the image of t_left on the reference cell
      refCellPoints(pointOrdinal,0) = refPoint;
    }

    FieldContainer<double> cellFxnValues(1,numPointsForCell);
    auto physicalCellNodes = mesh->physicalCellNodesForCell(cellID);

    basisCache->setRefCellPointsFC(refCellPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, {cellID}, false);
    temporalSolutionFunction->values(cellFxnValues, basisCache);

    for (int pointOrdinal=0; pointOrdinal<numPointsForCell; pointOrdinal++)
    {
      fxnValues[t_firstOrdinal + pointOrdinal] = {points[pointOrdinal],cellFxnValues(0,pointOrdinal)};
    }
  }
  return fxnValues;
}

vector<pair<double,double>> VlasovSpaceTimeFormulation::timeSeriesForSlabSpatialIntegrals(const std::vector<Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution>> &integrals,
                                                                                          const double &t0, const double &dt, const int &numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(integrals.size() != _slabTemporalCoordinates.size() - 1, std::invalid_argument, "integrals container is not the expected size.  computeConservationDiagnostics() may not have been called.");
  vector<pair<double,double>> allValues;
  const int numSlabs = _slabTemporalCoordinates.size() - 1;
  double t = t0;
  double tol = 1e-14;
  
  for (int slabNumber=0; slabNumber<numSlabs; slabNumber++)
  {
    double slabStart = _slabTemporalCoordinates[slabNumber];
    double slabEnd   = _slabTemporalCoordinates[slabNumber+1];
    double slabWidth = slabEnd - slabStart;
    
    double initialTimeOffset = t - slabStart;
    TEUCHOS_TEST_FOR_EXCEPTION(initialTimeOffset < -tol, std::invalid_argument, "Start time is prior to slab start");
//    while (initialTimeOffset < -tol) // t is before slab start (should only happen for initial slab, and then only if called with a t0 prior to start time)
//    {
//      // pad with zeros for unknown values prior to slab start
//      allValues.push_back({t,0});
//      t += dt;
//      initialTimeOffset = t - slabStart;
//    }
    if (initialTimeOffset >= slabWidth) continue; // we're into the next slab, at least
    
    int slabNumValues;
    if (slabNumber < numSlabs - 1)
    {
      slabNumValues = (slabWidth - initialTimeOffset) / dt;
    }
    else
    {
      slabNumValues = numValues - allValues.size();
    }
    
    auto slabValues = timeSeriesForSpatialIntegral(integrals[slabNumber], t, dt, slabNumValues);
    
    allValues.insert(allValues.end(), slabValues.begin(), slabValues.end());
    t += slabNumValues * dt;
  }
  return allValues;
}

VarPtr VlasovSpaceTimeFormulation::t_n_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalEnergyTimeSeries(const double t0, const double dt, const int numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 2, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set to 2 to enable totalEnergyTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_energyIntegrals, t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalMassTimeSeries(const double t0, const double dt, const int numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 0, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 0 to enable totalMassTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_massIntegrals, t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalXMomentumTimeSeries(const double t0, const double dt, const int numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 1, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 1 to enable totalXMomentumTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_momentumIntegrals[0], t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalYMomentumTimeSeries(const double t0, const double dt, const int numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim < 2, std::invalid_argument, "spatial dimension must be >= 2 for totalYMomentumTimeSeries()");
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 1, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 1 to enable totalYMomentumTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_momentumIntegrals[1], t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalZMomentumTimeSeries(const double t0, const double dt, const int numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim < 3, std::invalid_argument, "spatial dimension must be 3 for totalZMomentumTimeSeries()");
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 1, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 1 to enable totalZMomentumTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_momentumIntegrals[2], t0, dt, numValues);
}

bool VlasovSpaceTimeFormulation::useGeometricMultigrid() const
{
  return _useGeometricMultigrid;
}

FunctionPtr VlasovSpaceTimeFormulation::v_i(int compZeroBased) const
{
  int vx_component_zero_based = _spaceDim + 1; // plus 1 for temporal dimension
  return MonomialFunction::xi_n(vx_component_zero_based + compZeroBased, 1);
}

VarPtr VlasovSpaceTimeFormulation::VE_configurationSpaceTrace()
{
  return _configurationSpaceFormulation->VE_hat();
}

std::string VlasovSpaceTimeFormulation::vizOutputDir() const
{
  return _vizOutputDir;
}

FunctionPtr VlasovSpaceTimeFormulation::x_i(const int zeroBasedComp) const
{
  return MonomialFunction::xi_n(zeroBasedComp, 1);
}
