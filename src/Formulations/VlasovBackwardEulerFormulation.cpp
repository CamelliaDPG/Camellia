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

#include <sys/stat.h> // for mkdir, stat()

#include "CamelliaDataHelpers.h"

using namespace Camellia;

using std::string;
using std::vector;

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

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               std::vector<double> velocitySpaceVertices_y,
                                                               std::vector<double> velocitySpaceVertices_z,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson)
:
VlasovBackwardEulerFormulation({q},{m},{""},useSerendipityBasis,configurationSpaceFormulation,spatialMeshTopology,
                               dt, velocitySpaceVertices_x, velocitySpaceVertices_y, velocitySpaceVertices_z,
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson)
{}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               std::vector<double> velocitySpaceVertices_y,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson)
:
VlasovBackwardEulerFormulation(q, m, useSerendipityBasis, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, velocitySpaceVertices_y, std::vector<double>(),
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson) {}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson)
:
VlasovBackwardEulerFormulation(q, m, useSerendipityBasis, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, std::vector<double>(), std::vector<double>(),
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson) {}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(vector<double> q, vector<double> m, vector<string> speciesLabels,
                                                               bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson)
:
VlasovBackwardEulerFormulation(q, m, speciesLabels, useSerendipityBasis, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, std::vector<double>(), std::vector<double>(),
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson) {}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(vector<double> q, vector<double> m, vector<string> speciesLabels,
                                                               bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               vector<double> velocitySpaceVertices_x,
                                                               vector<double> velocitySpaceVertices_y,
                                                               vector<double> velocitySpaceVertices_z,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson)
:
_spaceDim(spatialMeshTopology->getDimension()),
_useSerendipityBasis(useSerendipityBasis),
_delta_k(delta_k),
_crankNicolson(useCrankNicolson),
_configurationSpaceFormulation(configurationSpaceFormulation),
_numSpecies(q.size()),
_q(q),
_m(m),
_speciesLabels(speciesLabels)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_q.size() != _m.size(), std::invalid_argument, "q and m arguments must agree in length");
  
  TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies == 0, std::invalid_argument, "Must have at least one species");
  
  if (speciesLabels.size() == 0)
  {
    if (_numSpecies == 1)
    {
      speciesLabels = {""}; // no suffix for single-species
    }
    else
    {
      for (int i=1; i<=_numSpecies; i++)
      {
        speciesLabels.push_back(std::to_string(i));
      }
    }
  }
  
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
  _configurationSpaceFormulation->setDimensions(_spaceDim, _velocityDim);
  
  // fields
  std::vector<VarPtr> f(_numSpecies); // Vlasov distributions
  
  // traces
  std::vector<VarPtr> t_n_hat(_numSpecies); // t_n_hat: L^2 (non-conforming) trace
  std::vector<VarPtr> f_hat(_numSpecies);   // f_hat:   conforming trace (unused in present implementation -- see l2Trace flag below)

  // tests
  std::vector<VarPtr> w(_numSpecies);   // Vlasov test function

  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();
  
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  map<int,int> trialOrderEnhancements;
  _f_names.resize(_numSpecies);
  _w_names.resize(_numSpecies);
  _f_hat_names.resize(_numSpecies,"");
  _tn_hat_names.resize(_numSpecies,"");
  for (int i=0; i<_numSpecies; i++)
  {
    std::string suffix = (speciesLabels[i].length() > 0) ? "_" + speciesLabels[i] : "";
    _f_names[i]      = S_F + suffix;
    _w_names[i]      = S_W + suffix;
    _tn_hat_names[i] = S_T_N_HAT + suffix;
    _f_hat_names[i]  = S_F_HAT + suffix;
    
    VarPtr f_i = vlasovVF->fieldVar(_f_names[i]);
    VarPtr w_i = vlasovVF->testVar(_w_names[i], HGRAD); // w lies in HGRAD of phase space
    
    VarPtr t_n_hat_i = vlasovVF->fluxVar(_tn_hat_names[i]);
    trialOrderEnhancements[t_n_hat_i->ID()] = 1; // +1 for non-conforming
    
    f[i]       = f_i;
    w[i]       = w_i;
    t_n_hat[i] = t_n_hat_i;
  }
  
  BasisFamilyPtr basisFamily;
  if (_useSerendipityBasis)
  {
    basisFamily = BasisFamily::serendipityBasisFamily();
  }
  else
  {
    basisFamily = BasisFamily::hierarchicalBasisFamily();
  }
  vlasovVF->setBasisFamily(basisFamily);
  _vlasovBF = BF::bf(vlasovVF);
  
  map<int,int> testOrderEnhancements;
  
  const bool allowPruning = false; // some day, we might want to have a fully distributed mesh.  But right now, ExtrudedMeshTopology relies on knowing all the phase-space cell IDs.  Would not be too hard to do the appropriate MPI communication, but right now the memory cost of the mesh is not prohibitive.
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
  for (int i=0; i<_numSpecies; i++)
  {
    VarPtr t_n = this->t_n_hat(i);
    for (int vd=0; vd<_velocityDim; vd++)
    {
      const int tagMin = this->tagVlasovMinVelocity(vd);
      const int tagMax = this->tagVlasovMaxVelocity(vd);
      vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
      vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
    }
  }
  _vlasovSoln->setBC(vlasovBC);
  
  const int maxMomentNumber = std::max(_configurationSpaceFormulation->maxMomentNumber(), maxMomentNumberForDriver);
  _solutionMoments.resize(_numSpecies);
  _solutionMomentsPriorTime.resize(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    _solutionMoments[i]          = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln,          f[i], _configurationSpaceFormulation->configurationSpaceMesh()) );
    _solutionMomentsPriorTime[i] = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSolnPriorTime, f[i], _configurationSpaceFormulation->configurationSpaceMesh()) );
  }
  
  _n0 = ParameterFunction::parameterFunction(0); // we'll update this when initial conditions are set, if _enforceQuasiNeutrality is true
  _configurationSpaceFormulation->setMoments(_solutionMoments, _n0);
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetVelocityOneBased = _spaceDim + 1;
  
  // SpatialFilter's components are zero-based
//  const int indexOffsetSpaceZeroBased    = indexOffsetSpaceOneBased  - 1; // unused
  const int indexOffsetVelocityZeroBased = indexOffsetVelocityOneBased - 1;
  
  // The below assumes that _spaceDim <= _velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim > _velocityDim, std::invalid_argument, "_velocityDim must be at least as large as _spaceDim");
  std::vector<FunctionPtr> v;
  for (int i=0; i<_velocityDim; i++)
  {
    v.push_back(this->v_i(i));
  }
  
  // revision 6-8-23: move weights on f[i] onto w[i], so that we get something like b(f[i], g(w[i])), where g incorporates diff ops and weights
  std::vector<LinearTermPtr> w_against_f_i(_numSpecies);
  
  // theta for time-stepping rule: 1.0 for backward Euler; 0.5 for Crank-Nicolson
  const double theta = _crankNicolson ? 0.5 : 1.0;
  
  const bool l2Trace = true;
  for (int i=0; i<_numSpecies; i++)
  {
    w_against_f_i[i] = w[i] / _dt;
//    _vlasovBF->addTerm(f[i] / _dt, w[i]);
    
    if (l2Trace)
    {
      _vlasovBF->addTerm(theta * t_n_hat[i], w[i]);
    }
    else
    {
      // TODO: add support for conforming traces
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Conforming traces not yet supported.");
    }
    
    for (int d=0; d<_spaceDim; d++)
    {
      w_against_f_i[i] = w_against_f_i[i] - theta * v[d] * w[i]->di(indexOffsetSpaceOneBased + d); // dx_d
    }
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
  
  auto extraAcceleration = _configurationSpaceFormulation->extraAcceleration();
  for (int i=0; i<_velocityDim; i++)
  {
    grad_v_weight[i] = E[i] + v_cross_B[i] + extraAcceleration[i];
  }
  
  for (int i=0; i<_numSpecies; i++)
  {
    for (int d=0; d<_velocityDim; d++)
    {
      auto weight = grad_v_weight[d];
      w_against_f_i[i] = w_against_f_i[i] - theta * (q[i]/m[i]) * weight * w[i]->di(indexOffsetVelocityOneBased + d); // d/dv_d
//      _vlasovBF->addTerm(-(q[i]/m[i]) * weight * f[i], w[i]->di(indexOffsetVelocityOneBased + d)); // d/dv_d
    }
  }
  // now add in the f terms:
  for (int i=0; i<_numSpecies; i++)
  {
    _vlasovBF->addTerm(f[i], w_against_f_i[i]);
  }

  auto vlasovSolnWeakPtr          = Teuchos::rcp(_vlasovSoln.get(),          false);
  auto vlasovSolnPriorTimeWeakPtr = Teuchos::rcp(_vlasovSolnPriorTime.get(), false);
  auto vlasovSolnPrevWeakPtr      = Teuchos::rcp(_vlasovSolnPrev.get(),      false);
  
  auto vlasovRHS = RHS::rhs();
  _fSoln.resize(_numSpecies);
  _fDiff.resize(_numSpecies);
  
  std::vector<FunctionPtr> grad_v_weight_priorTime(_velocityDim); // the thing that gets dotted with grad_v of f (in the strong form)
  
  if (theta != 1.0)
  {
    std::vector<FunctionPtr> E_priorTime(3,Function::zero());
    std::vector<FunctionPtr> B_priorTime(3,Function::zero());
    
    for (int vd=0; vd<_velocityDim; vd++)
    {
      E_priorTime[vd] = _configurationSpaceFormulation->E_i_priorTime(vd);
      B_priorTime[vd] = _configurationSpaceFormulation->B_i_priorTime(vd);
    }
    
    // we only include the v_cross_B term for 3V case
    // TODO: determine whether v_cross_B ever makes sense for 1V or 2V cases; if so, add something here.
    std::vector<FunctionPtr> v_cross_B_priorTime(3,Function::zero());
    if (_velocityDim == 3)
    {
      v_cross_B_priorTime[0] = v[1] * B_priorTime[2] - v[2] * B_priorTime[1];
      v_cross_B_priorTime[1] = v[2] * B_priorTime[0] - v[0] * B_priorTime[2];
      v_cross_B_priorTime[2] = v[0] * B_priorTime[1] - v[1] * B_priorTime[0];
    }
    for (int vd=0; vd<_velocityDim; vd++)
    {
      grad_v_weight_priorTime[vd] = E_priorTime[vd] + v_cross_B_priorTime[vd];
    }
  }
    
  for (int i=0; i<_numSpecies; i++)
  {
    _fSoln[i]     = Function::solution(f[i], vlasovSolnWeakPtr);
    FunctionPtr f_solnPrev = Function::solution(f[i], vlasovSolnPrevWeakPtr);
    
    _fDiff[i] = _fSoln[i] - f_solnPrev;
    
    const bool weightFluxesBySideParity = true;
    const SolutionComponent solnComponent = STANDARD_SOLUTION;
    const int stageOrdinal = 0; // would be -1 if we had an ERKTableau
    auto f_previous = Function::solution(f[i], vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity,
                                         solnComponent, stageOrdinal, std::string("k"));
    
    vlasovRHS->addTerm(f_previous / FunctionPtr(_dt) * w[i]);
  
    if (theta != 1.0)
    {
      if (l2Trace)
      {
        auto t_n_hat_previous_i = Function::solution(t_n_hat[i], vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity,
                                                     solnComponent, stageOrdinal, std::string("k"));
        vlasovRHS->addTerm(-(1-theta) * t_n_hat_previous_i * w[i]);
      }
      else
      {
        // TODO: add support for conforming traces
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Conforming traces not yet supported.");
      }
      
      for (int d=0; d<_spaceDim; d++)
      {
        vlasovRHS->addTerm( (1-theta) * v[d] * f_previous * w[i]->di(indexOffsetSpaceOneBased + d));  // dx_d
      }
      
      for (int vd=0; vd<_velocityDim; vd++)
      {
        auto weight = grad_v_weight_priorTime[vd];
        vlasovRHS->addTerm( (1-theta) * (q[i]/m[i]) * weight * f_previous * w[i]->di(indexOffsetVelocityOneBased + vd)); // d/dv_vd
      }
    }
  }
  
  _vlasovSoln->setRHS(vlasovRHS);
    
  vlasovMesh->registerSolution(_vlasovSoln);
  vlasovMesh->registerSolution(_vlasovSolnPriorTime);
    
  IPPtr vlasovTestNorm = _vlasovBF->graphNorm();
  _vlasovSoln->setIP(vlasovTestNorm);
  
  SpatialFilterPtr inflowFilter = Teuchos::rcp(new SpatialFilterSpatialInflow(_spaceDim,indexOffsetVelocityZeroBased) );
    
  std::vector<double> v0,vMax;
  
  std::vector<SpatialFilterPtr> minVelocityFilters, maxVelocityFilters;
  for (int vd=0; vd<_velocityDim; vd++)
  {
    auto & dimVertices = velocitySpaceVertices[vd];
    const int numVertices = dimVertices.size();
    v0.push_back(dimVertices[0]);
    vMax.push_back(dimVertices[numVertices-1]);
    
    SpatialFilterPtr minVelocityFilter = SpatialFilter::matchingComponent(indexOffsetVelocityZeroBased + vd, v0[vd]  );
    SpatialFilterPtr maxVelocityFilter = SpatialFilter::matchingComponent(indexOffsetVelocityZeroBased + vd, vMax[vd]);
    
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
  for (int i=0; i<_numSpecies; i++)
  {
    _solutionMoments[i]->update();
  }
  {
    double n0;
    if (_enforceQuasiNeutrality)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies > 1, std::invalid_argument, "quasi-neutrality enforcement assumes single species");
      FunctionPtr zeroMoment   = _solutionMoments[0]->solutionMomentFunction(0);
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
  std::vector<IndexType> matchingSideIndices = _configurationSpaceMeshTopology->getSidesThatMatch(regionToMatch);
  for (const auto & matchingSideIndex : matchingSideIndices)
  {
    matchingSideSet->addEntity(sideDim, matchingSideIndex);
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

VarPtr VlasovBackwardEulerFormulation::f(int speciesOrdinal)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(_f_names[speciesOrdinal]);
}

VarPtr VlasovBackwardEulerFormulation::f_hat(int speciesOrdinal)
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(_f_hat_names[speciesOrdinal]);
}

void VlasovBackwardEulerFormulation::finishTimeStep()
{
  const int rank = MPIWrapper::CommWorld()->MyPID();
  
  Teuchos::RCP<GMGSolver> gmgSolver;
  
  std::vector<int>    substepIterationCounts(_maxSubsteps);
  std::vector<double> substepStepSizesVlasov;
  
  if (_useGeometricMultigrid && (_gmgSolver == Teuchos::null))
  {
    reconstructGMGSolver();
  }
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, configurationViz;
  const bool exportViz = false; // for debugging of fixed-point iteration, set to true.  TODO: make this subject to external control.
  
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
  
  std::vector<FunctionPtr> density(_numSpecies);
  std::vector<std::string> densityLabels(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    density[i] = _solutionMoments[i]->solutionMomentFunction(0); // used in viz output
    densityLabels[i] = (_speciesLabels[i].size()) == 0 ? "n_e" : "n_" + _speciesLabels[i];
  }
  
  
  
  // get the PhysicalSolutionExtrusions ready to evaluate (in context of Vlasov BF)
  // updateConfigurationSpaceFunctions() will do MPI communication of configuration-space solution values so they're where they need to be for phase space mesh:
  _configurationSpaceFormulation->updateConfigurationSpaceFunctions();
  int numSubstepsTaken = _maxSubsteps;
  for (int substepOrdinal=0; substepOrdinal < _maxSubsteps; substepOrdinal++)
  {
    _vlasovSolnPrev->setSolution(_vlasovSoln);
    _configurationSpaceFormulation->copySolutionToPrevious();
    
    if ((rank == 0) && !_runQuietly)
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
    double fDiff_L2_squared = 0, f_L2_squared = 0;
    for (int i=0; i<_numSpecies; i++)
    {
      double fDiff_L2_i = _fDiff[i]->l2norm(vlasovMesh);
      double f_L2_i     = _fSoln[i]->l2norm(vlasovMesh);
      fDiff_L2_squared += fDiff_L2_i * fDiff_L2_i;
      f_L2_squared     += f_L2_i * f_L2_i;
    }
    const double fDiff_L2 = sqrt(fDiff_L2_squared);
    const double f_L2 = sqrt(f_L2_squared);
    const double fRelDiff = fDiff_L2 / f_L2;
    
    const double configurationSpaceRelativeDiff = _configurationSpaceFormulation->l2NormOfSolutionUpdate();
    
    substepStepSizesVlasov.push_back(fRelDiff);
    
    if ((rank == 0) && !_runQuietly)
    {
      std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in f: " << fRelDiff << std::endl;
      std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in E and/or B: " << configurationSpaceRelativeDiff << std::endl;
    }
    
    if ((fRelDiff < _fixedPointTolerance) && (substepOrdinal >= _minSubsteps-1))
    {
      // tolerance reached, min substeps taken
      numSubstepsTaken = substepOrdinal+1;
      break;
    }
    if (exportViz)
    {
      configurationViz->exportSolution(_configurationSpaceFormulation->solution(),density, densityLabels,substepOrdinal+1);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal+1);
    }
  } // substep loop (fixed point iteration)
  
  _substepCounts.push_back(numSubstepsTaken);
  _iterationCounts.push_back(substepIterationCounts);
  _fixedPointL2StepSizesVlasov.push_back(substepStepSizesVlasov);
  
  _t += _dt->getConstantValue();
}

Teuchos::RCP<GMGSolver> VlasovBackwardEulerFormulation::gmgSolver()
{
  return _gmgSolver;
}

bool VlasovBackwardEulerFormulation::hasL2Trace() const {
  return true;
}

const std::vector< std::vector<int> > & VlasovBackwardEulerFormulation::iterationCounts() const
{
  return _iterationCounts;
}

void VlasovBackwardEulerFormulation::loadFromCheckpoint(const std::string &checkpointDir)
{
  // for now, we require that the mesh agrees (i.e., refinements are replayed externally -- they aren't loaded from disk here; we haven't added ExtrudedMeshTopology to the Solution::load() infrastructure)
  // TODO: add some sort of sanity check on the mesh vs. loaded solution (cell numbers are present in mesh, say)
  
  std::string vlasovSolnPath = checkpointDir + "/vlasov.soln";
  _vlasovSoln->loadFromHDF5(vlasovSolnPath);
  
  std::string vlasovSolnPriorTimePath = checkpointDir + "/vlasovPriorTime.soln";
  _vlasovSolnPriorTime->loadFromHDF5(vlasovSolnPriorTimePath);
  
  std::string vlasovSolnPrevPath = checkpointDir + "/vlasovPrev.soln";
  _vlasovSolnPrev->loadFromHDF5(vlasovSolnPrevPath);
  
  {
    // load t:
    std::string timeFile = checkpointDir + "/time.txt";
    std::ifstream timeFileStream(timeFile, std::ios::in);
    
    timeFileStream >> _t;
    
    timeFileStream.close();
  }
}

int VlasovBackwardEulerFormulation::maxSubstepsTaken() const
{
  int maxSubstepsTaken = 0;
  for (auto numSubsteps : _substepCounts)
  {
    maxSubstepsTaken = std::max(numSubsteps, maxSubstepsTaken);
  }
  return maxSubstepsTaken;
}

int  VlasovBackwardEulerFormulation::minSubstepsTaken() const
{
  if (_substepCounts.size() == 0)
  {
    return 0;
  }
  int minSubstepsTaken = _maxSubsteps;
  for (auto numSubsteps : _substepCounts)
  {
    minSubstepsTaken = std::min(numSubsteps, minSubstepsTaken);
  }
  return minSubstepsTaken;
}

double VlasovBackwardEulerFormulation::averageSubstepsTaken() const
{
  double totalSubsteps = 0;
  for (auto numSubsteps : _substepCounts)
  {
    totalSubsteps += numSubsteps;
  }
  if (_substepCounts.size() > 0)
  {
    return totalSubsteps / _substepCounts.size();
  }
  else
  {
    return 0;
  }
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

void VlasovBackwardEulerFormulation::reconstructGMGSolver()
{
  if (_useGeometricMultigrid)
  {
    _gmgSolver = constructGMGSolver();
  }
}

void VlasovBackwardEulerFormulation::saveCheckpoint(const std::string &checkpointDir)
{
  const int rank = MPIWrapper::CommWorld()->MyPID();
  if (rank == 0)
  {
    // ensure directory exists
    bool dirExists = false;
    {
      struct stat info;
      bool statResult = stat( checkpointDir.c_str(), &info );
      if( statResult != 0)
        dirExists = false;
      else if(info.st_mode & S_IFDIR)
        dirExists = true;
      else
        dirExists = false;
    }
    if (!dirExists)
    {
      int success = mkdir(checkpointDir.c_str(), S_IRWXU | S_IRWXG);
      TEUCHOS_TEST_FOR_EXCEPTION(success != 0, std::invalid_argument, "Could not create checkpoint directory");
    }
  }
  
  std::string vlasovSolnPrefix = checkpointDir + "/vlasov";
  _vlasovSoln->save(vlasovSolnPrefix);
  
  std::string vlasovSolnPriorTimePrefix = checkpointDir + "/vlasovPriorTime";
  _vlasovSolnPriorTime->save(vlasovSolnPriorTimePrefix);
  
  std::string vlasovSolnPrevPrefix = checkpointDir + "/vlasovPrev";
  _vlasovSolnPrev->save(vlasovSolnPrevPrefix);
  
  {
    // save t:
    std::string timeFile = checkpointDir + "/time.txt";
    std::ofstream timeFileStream(timeFile, std::ios::out);
    
    timeFileStream << setprecision(15) << _t;
    
    timeFileStream.close();
  }
  
  _configurationSpaceFormulation->saveCheckpoint(checkpointDir);
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

void VlasovBackwardEulerFormulation::setRunQuietly(bool runQuietly)
{
  _runQuietly = runQuietly;
}

void VlasovBackwardEulerFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

void VlasovBackwardEulerFormulation::startTimeStep()
{
  _vlasovSolnPriorTime->setSolution(_vlasovSoln);
  _configurationSpaceFormulation->copySolutionToPriorTime();
}

void VlasovBackwardEulerFormulation::takeTimeStep()
{
  startTimeStep();
  finishTimeStep();
}

void VlasovBackwardEulerFormulation::setInitialConditions(FunctionPtr f0)
{
  setInitialConditions(std::vector<FunctionPtr>{f0});
}

void VlasovBackwardEulerFormulation::setInitialConditions(std::vector<FunctionPtr> f0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(int(f0.size()) != _numSpecies, std::invalid_argument, "Must define initial conditions for each species");
  std::map<int,FunctionPtr> projectionMap;
  for (int speciesOrdinal=0; speciesOrdinal<_numSpecies; speciesOrdinal++)
  {
    projectionMap[this->f(speciesOrdinal)->ID()] = f0[speciesOrdinal];
  }
  
  const int solutionOrdinal = 0;
  _vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  configurationSpaceSolve();
  
  if (_crankNicolson)
  {
    // Crank-Nicolson requires that the previous-solution trace be set as well.
    FunctionPtr n = Function::normal();
    FunctionPtr nx_dot_v = Function::zero();
    for (int vd=0; vd<_velocityDim; vd++)
    {
      auto n_d = n->spatialComponent(vd+1); // spatialComponent() is one-based
      FunctionPtr v_d = this->v_i(vd); // v_i() is zero-based
      nx_dot_v = nx_dot_v + n_d * v_d;
    }
    
    std::vector<FunctionPtr> grad_v_weight(_velocityDim); // the thing that gets dotted with grad_v of f (in the strong form)
    
    std::vector<FunctionPtr> E(3,Function::zero());
    std::vector<FunctionPtr> B(3,Function::zero());
    
    for (int vd=0; vd<_velocityDim; vd++)
    {
      E[vd] = _configurationSpaceFormulation->E_i(vd);
      B[vd] = _configurationSpaceFormulation->B_i(vd);
    }
    
    // we only include the v_cross_B term for 3V case
    // TODO: determine whether v_cross_B ever makes sense for 1V or 2V cases; if so, add something here.
    std::vector<FunctionPtr> v_cross_B(3,Function::zero());
    if (_velocityDim == 3)
    {
      FunctionPtr v_0 = this->v_i(0);
      FunctionPtr v_1 = this->v_i(1);
      FunctionPtr v_2 = this->v_i(2);
      v_cross_B[0] = v_1 * B[2] - v_2 * B[1];
      v_cross_B[1] = v_2 * B[0] - v_0 * B[2];
      v_cross_B[2] = v_0 * B[1] - v_1 * B[0];
    }
    
    for (int vd=0; vd<_velocityDim; vd++)
    {
      grad_v_weight[vd] = E[vd] + v_cross_B[vd];
    }
    
    for (int i=0; i<_numSpecies; i++)
    {
      FunctionPtr nv_dot_a = Function::zero();
      for (int vd=0; vd<_velocityDim; vd++)
      {
        auto a_vd  = (_q[i]/_m[i]) * grad_v_weight[vd];
        auto nv_vd = n->spatialComponent(_spaceDim + vd + 1); // spatialComponent() is one-based
        nv_dot_a = nv_dot_a + nv_vd * a_vd;
      }
      
      FunctionPtr tn0 = f0[i] * (nx_dot_v + nv_dot_a);
      FunctionPtr sgn_n = Function::sideParity();
      projectionMap[this->t_n_hat(i)->ID()] = tn0 * sgn_n;
    }
    _vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  }
  
  _vlasovSolnPrev->setSolution(_vlasovSoln);
  _vlasovSolnPriorTime->setSolution(_vlasovSoln);
  
  _t = 0;
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in)
{
  using std::vector;
  setInitialAndInflowConditions(vector<FunctionPtr>{f0}, vector<FunctionPtr>{f_in});
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(std::vector<FunctionPtr> f0, std::vector<FunctionPtr> f_in)
{
  TEUCHOS_TEST_FOR_EXCEPTION(int(f_in.size()) != _numSpecies, std::invalid_argument, "Must define inflow conditions for each species");
  auto vlasovBC = BC::bc();
  
  for (int speciesOrdinal=0; speciesOrdinal<_numSpecies; speciesOrdinal++)
  {
    VarPtr t_n = this->t_n_hat(speciesOrdinal);
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
    vlasovBC->addDirichlet(t_n, tagSpatialInflow, v_dot_nx * f_in[speciesOrdinal]);
  }

  _vlasovSoln->setBC(vlasovBC);
  
  setInitialConditions(f0);
}

Teuchos::RCP<SolutionMoments> VlasovBackwardEulerFormulation::solutionMoments(int i)
{
  return _solutionMoments[i];
}

Teuchos::RCP<SolutionMoments> VlasovBackwardEulerFormulation::solutionMomentsPriorTime(int i)
{
  return _solutionMomentsPriorTime[i];
}

// test variables:
VarPtr VlasovBackwardEulerFormulation::w(int i)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(_w_names[i], HGRAD);
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

double VlasovBackwardEulerFormulation::t() const
{
  return _t;
}

VarPtr VlasovBackwardEulerFormulation::t_n_hat(int i)
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(_tn_hat_names[i]);
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
  const bool useSerendipityBasis = true;
  VlasovBackwardEulerFormulation form(q, m, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
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

void testVelocityExtrusion(bool volumeVariable, bool doRefinement, bool useSerendipityBasis, Teuchos::FancyOStream &out, bool &success)
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
  
  auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
  
  auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
  std::vector<double> vxVertices;
  for (int i=0; i<=vElements; i++)
  {
    const double vx = v0 + (i * vWidth) / vElements;
    vxVertices.push_back(vx);
  }
  VlasovBackwardEulerFormulation form(q, m, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
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
  const bool useSerendipity = false;
  testVelocityExtrusion(volumeVariable, doRefinement, useSerendipity, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonBackwardEulerFormulation_1D1V_Serendipity, VolumeExtrusion )
{
  const bool volumeVariable = true;
  const bool doRefinement = false;
  const bool useSerendipity = true;
  testVelocityExtrusion(volumeVariable, doRefinement, useSerendipity, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonBackwardEulerFormulation_1D1V, VolumeExtrusionWithRefinement )
{
  const bool volumeVariable = true;
  const bool doRefinement = true;
  const bool useSerendipity = false;
  testVelocityExtrusion(volumeVariable, doRefinement, useSerendipity, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonBackwardEulerFormulation_1D1V_Serendipity, VolumeExtrusionWithRefinement )
{
  const bool volumeVariable = true;
  const bool doRefinement = true;
  const bool useSerendipity = true;
  testVelocityExtrusion(volumeVariable, doRefinement, useSerendipity, out, success);
}

} // end anonymous namespace
