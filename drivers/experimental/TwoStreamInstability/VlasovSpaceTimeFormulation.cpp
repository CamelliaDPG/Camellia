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
                                                       Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                       MeshTopologyPtr spatialMeshTopology,
                                                       std::vector<double> temporalVertices,
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
  
  f = vlasovVF->fieldVar(S_F);
  w = vlasovVF->testVar(S_W, HGRAD); // w lies in HGRAD of the whole space-time phase space
  
  t_n_hat = vlasovVF->fluxVar(S_T_N_HAT);
  
  _vlasovBF = BF::bf(vlasovVF);
  
  map<int,int> trialOrderEnhancements;
  trialOrderEnhancements[t_n_hat->ID()] = 1; // +1 for non-conforming, + whatever user requested

  map<int,int> testOrderEnhancements;
  
  const bool allowPruning = true;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  Epetra_CommPtr comm = MPIWrapper::CommWorld();
  MeshPtr vlasovMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _vlasovBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
  
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  _vlasovSoln     = Solution::solution(_vlasovBF,vlasovMesh);
  _vlasovSolnPrev = Solution::solution(_vlasovBF,vlasovMesh); // will copy out after fixed-point solve
  
  const int maxMomentNumber = _configurationSpaceFormulation->maxMomentNumber();
  _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln, f, _configurationSpaceFormulation->configurationSpaceMesh()) );
  
  _configurationSpaceFormulation->setMoments(_solutionMoments);
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + _spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  // SpatialFilter's components are zero-based
  const int indexOffsetSpaceZeroBased    = indexOffsetSpaceOneBased  - 1;
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
    grad_v_weight[i] = E[i] + v_cross_B[i];
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

int VlasovSpaceTimeFormulation::createConfigurationMeshSideSetTag(SpatialFilterPtr regionToMatch)
{
  const int thisTag = _nextConfigurationSpaceTag++;
  
  const int sideDim = _spaceDim; // space-time configuration-space mesh has dimension _spaceDim + 1; sides have dimension _spaceDim
  auto matchingSideSet = _configurationSpaceMeshTopology->createEntitySet();
  std::vector<IndexType> initialTimeSideIndices = _configurationSpaceMeshTopology->getBoundarySidesThatMatch(regionToMatch);
  for (const auto & initialTimeSideIndex : initialTimeSideIndices)
  {
    matchingSideSet->addEntity(sideDim, initialTimeSideIndex);
  }
  _configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, thisTag, matchingSideSet);
  
  return thisTag;
}

VarPtr VlasovSpaceTimeFormulation::E_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->E_hat(zeroBasedComp);
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

bool VlasovSpaceTimeFormulation::hasL2Trace() const {
  return true;
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
  const bool exportViz = true; // for debugging of fixed-point iteration, set to true.  TODO: make this subject to external control.
  
  if (exportViz)
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
    _configurationSpaceFormulation->solve(true); // true: do update solution moments
    
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
    
    if (exportViz)
    {
      configurationViz->exportSolution(_configurationSpaceFormulation->solution(),substepOrdinal);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal);
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  _fixedPointL2StepSizesVlasov.push_back(substepStepSizesVlasov);
}

void VlasovSpaceTimeFormulation::setInitialConditions(FunctionPtr f0)
{
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
}

void VlasovSpaceTimeFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in)
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
}

Teuchos::RCP<SolutionMoments> VlasovSpaceTimeFormulation::solutionMoments()
{
  return _solutionMoments;
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

VarPtr VlasovSpaceTimeFormulation::t_n_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
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

FunctionPtr VlasovSpaceTimeFormulation::x_i(const int zeroBasedComp) const
{
  return MonomialFunction::xi_n(zeroBasedComp, 1);
}
