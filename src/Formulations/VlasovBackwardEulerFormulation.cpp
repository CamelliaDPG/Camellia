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
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SharedScratchStorage.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TimeRampFunction.h"
#include "TypeDefs.h"

#include <sys/stat.h> // for mkdir, stat()

#include "CamelliaDataHelpers.h"

using namespace Camellia;

using std::string;
using std::vector;

const string VlasovBackwardEulerFormulation::S_F = "f";

const string VlasovBackwardEulerFormulation::S_F_HAT = "\\widehat{f}";
const string VlasovBackwardEulerFormulation::S_T_N_HAT = "\\widehat{t}_n";

const string VlasovBackwardEulerFormulation::S_PSI = "\\psi";
const string VlasovBackwardEulerFormulation::S_PHI = "\\phi";

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
                                                               const bool useCrankNicolson,
                                                               const bool enforcePositivity,
                                                               const bool useConformingTraces)
:
VlasovBackwardEulerFormulation({q},{m},{""},useSerendipityBasis,configurationSpaceFormulation,spatialMeshTopology,
                               dt, velocitySpaceVertices_x, velocitySpaceVertices_y, velocitySpaceVertices_z,
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson, enforcePositivity, useConformingTraces)
{}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               std::vector<double> velocitySpaceVertices_y,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson,
                                                               const bool enforcePositivity,
                                                               const bool useConformingTraces)
:
VlasovBackwardEulerFormulation(q, m, useSerendipityBasis, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, velocitySpaceVertices_y, std::vector<double>(),
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson, enforcePositivity, useConformingTraces) {}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson,
                                                               const bool enforcePositivity,
                                                               const bool useConformingTraces)
:
VlasovBackwardEulerFormulation(q, m, useSerendipityBasis, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, std::vector<double>(), std::vector<double>(),
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson, enforcePositivity, useConformingTraces) {}

VlasovBackwardEulerFormulation::VlasovBackwardEulerFormulation(vector<double> q, vector<double> m, vector<string> speciesLabels,
                                                               bool useSerendipityBasis,
                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                               MeshTopologyPtr spatialMeshTopology,
                                                               double dt,
                                                               std::vector<double> velocitySpaceVertices_x,
                                                               const int H1Order, const int delta_k,
                                                               const int maxMomentNumberForDriver,
                                                               const bool useCrankNicolson,
                                                               const bool enforcePositivity,
                                                               const bool useConformingTraces)
:
VlasovBackwardEulerFormulation(q, m, speciesLabels, useSerendipityBasis, configurationSpaceFormulation, spatialMeshTopology, dt,
                               velocitySpaceVertices_x, std::vector<double>(), std::vector<double>(),
                               H1Order, delta_k, maxMomentNumberForDriver, useCrankNicolson, enforcePositivity, useConformingTraces) {}

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
                                                               const bool useCrankNicolson,
                                                               const bool enforcePositivity,
                                                               const bool useConformingTraces)
:
_spaceDim(spatialMeshTopology->getDimension()),
_useSerendipityBasis(useSerendipityBasis),
_useConformingTraces(useConformingTraces),
_delta_k(delta_k),
_crankNicolson(useCrankNicolson),
_enforcePositivity(enforcePositivity),
_configurationSpaceFormulation(configurationSpaceFormulation),
_numSpecies(q.size()),
_q(q),
_m(m),
_speciesLabels(speciesLabels)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_q.size() != _m.size(), std::invalid_argument, "q and m arguments must agree in length");
  
  TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies == 0, std::invalid_argument, "Must have at least one species");
  
  _strongInflowBCFlag = ParameterFunction::parameterFunction(1); // default to BCs strongly imposed at inflow; can be overridden when setInitialAndInflowConditions() is called.
  
  const bool l2Trace = !_useConformingTraces;
  
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
  
  const int numLatentVariables = _enforcePositivity ? _numSpecies : 0;
  
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
  std::vector<VarPtr> phi(numLatentVariables); // test function for equation defining latent variable psi = ln f

  std::vector<VarPtr> psi(numLatentVariables);
  
  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();
  
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  map<int,int> trialOrderEnhancements;
  _f_names.resize(_numSpecies);
  _w_names.resize(_numSpecies);
  _psi_names.resize(numLatentVariables);
  _phi_names.resize(numLatentVariables);
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
    
    f[i]       = f_i;
    w[i]       = w_i;
    if (l2Trace)
    {
      VarPtr t_n_hat_i = vlasovVF->fluxVar(_tn_hat_names[i]);
      trialOrderEnhancements[t_n_hat_i->ID()] =  1; // +1 for non-conforming
      t_n_hat[i] = t_n_hat_i;
    }
    else
    {
      VarPtr f_hat_i = vlasovVF->traceVar(_f_hat_names[i]);
      f_hat[i] = f_hat_i;
    }
    
    if (_enforcePositivity)
    {
      _psi_names[i] = S_PSI + suffix;
      _phi_names[i] = S_PHI + suffix;
      psi[i] = vlasovVF->fieldVar(_psi_names[i], CONSTANT_SCALAR);
      // CONSTANT_SCALAR: setting the latent variables psi to have element-wise constant solutions (saves us from doing an inexact projection, e.g., of the exponential of a polynomial onto a polynomial during the flux-limiting -- that could be OK, of course, but it might not be uniformly positive post-projection.)
      
      phi[i] = vlasovVF->testVar(_phi_names[i], L2); // CONSTANT_SCALAR: Unclear on whether this is better than L2 for the test function.  We have some issues with Cholesky factorization failing; trying CONSTANT_SCALAR to ameliorate.
      _alphaPG = ParameterFunction::parameterFunction(1);
    }
  }
  
  FunctionPtr alphaPG = _alphaPG; // cast to FunctionPtr to enable operator overloads below
  
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
  Teuchos_CommPtr comm = MPIWrapper::TeuchosCommWorld();
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
    VarPtr traceVar = _useConformingTraces ? this->f_hat(i) : this->t_n_hat(i);
    for (int vd=0; vd<_velocityDim; vd++)
    {
      const int tagMin = this->tagVlasovMinVelocity(vd);
      const int tagMax = this->tagVlasovMaxVelocity(vd);
      vlasovBC->addDirichlet(traceVar, tagMin, Function::zero());
      vlasovBC->addDirichlet(traceVar, tagMax, Function::zero());
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
    
  FunctionPtr v_dot_nx = Function::zero();
  for (int i=0; i<_velocityDim; i++)
  {
    v_dot_nx = v_dot_nx + v[i] * n->spatialComponent(i+indexOffsetSpaceOneBased); // spatialComponent() is 1-based
  }
  
  // revision 6-8-23: move weights on f[i] onto w[i], so that we get something like b(f[i], g(w[i])), where g incorporates diff ops and weights
  std::vector<LinearTermPtr> w_against_f_i(_numSpecies);
  
  // theta for time-stepping rule: 1.0 for backward Euler; 0.5 for Crank-Nicolson
  const double theta = _crankNicolson ? 0.5 : 1.0;
  
  for (int i=0; i<_numSpecies; i++)
  {
    w_against_f_i[i] = w[i] / _dt;
    
    if (_enforcePositivity)
    {
      _vlasovBF->addTerm(psi[i] / alphaPG, w[i]);
    }
    
    for (int d=0; d<_spaceDim; d++)
    {
      w_against_f_i[i] = w_against_f_i[i] - theta * v[d] * w[i]->di(indexOffsetSpaceOneBased + d); // dx_d
    }
  }
  
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
  
  FunctionPtr a_dot_nv = Function::zero();
  // initialize acceleration term, _a
  {
    _a = std::vector<FunctionPtr>(_velocityDim); // this gets dotted with  grad_v (q[i]/m[i]) f_i (in the strong form)
    
    for (int d=0; d<_velocityDim; d++)
    {
      _a[d] = E[d] + v_cross_B[d] + extraAcceleration[d];
      auto nvd = n->spatialComponent(d+indexOffsetVelocityOneBased);
      a_dot_nv = a_dot_nv + _a[d] * nvd;
    }
  }
  
  auto vlasovSolnWeakPtr          = Teuchos::rcp(_vlasovSoln.get(),          false);
  auto vlasovSolnPriorTimeWeakPtr = Teuchos::rcp(_vlasovSolnPriorTime.get(), false);
  auto vlasovSolnPrevWeakPtr      = Teuchos::rcp(_vlasovSolnPrev.get(),      false);
  _traceSolnPriorTime.resize(_numSpecies);
  auto vlasovRHS = RHS::rhs();
  
  // add trace terms
  for (int i=0; i<_numSpecies; i++)
  {
    auto trace_i = l2Trace ? t_n_hat[i] : f_hat[i];
    const bool weightFluxesBySideParity = true;
    const SolutionComponent solnComponent = STANDARD_SOLUTION;
    const int stageOrdinal = 0; // would be -1 if we had an ERKTableau
    _traceSolnPriorTime[i] = Function::solution(trace_i, vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity,
                                                solnComponent, stageOrdinal, std::string("k"));
    
    FunctionPtr strongBCFilter = _strongInflowBCFlag; // will be 1 when strong BCs are imposed, 0 otherwise
    if (l2Trace)
    {
      _vlasovBF->addTerm(theta * t_n_hat[i], w[i]);
      
      vlasovRHS->addTerm(-(1-theta) * _traceSolnPriorTime[i] * strongBCFilter * w[i]);
    }
    else
    {
      // _boltzmannBF->addTerm((v * n->x() + (q/m) * _E_extruded * n->y()) * f_hat, w); // for explicit, f_hat should be understood as the trace of the solution at prior time…
      _vlasovBF->addTerm(f_hat[i], (v_dot_nx + (_q[i]/_m[i]) * a_dot_nv) * w[i]);
      vlasovRHS->addTerm(-(1-theta) * _traceSolnPriorTime[i] * strongBCFilter * (v_dot_nx + (_q[i]/_m[i]) * a_dot_nv) * w[i]); // TODO: ensure that Solution's trace projection is enabled; I'm pretty sure it's off right now.  We will get some wrong values if the inflow is refined and the trace projection is disabled.
    }
  }
  
  for (int i=0; i<_numSpecies; i++)
  {
    for (int d=0; d<_velocityDim; d++)
    {
      auto weight = _a[d];
      w_against_f_i[i] = w_against_f_i[i] - theta * (_q[i]/_m[i]) * weight * w[i]->di(indexOffsetVelocityOneBased + d); // d/dv_d
    }
  }
  // now add in the f terms:
  for (int i=0; i<_numSpecies; i++)
  {
    _vlasovBF->addTerm(f[i], w_against_f_i[i]);
  }
  
  _fSoln.resize(_numSpecies);
  _fDiff.resize(_numSpecies);
  _fSolnPriorTime.resize(_numSpecies);
  _expPsiSoln.resize(numLatentVariables);
  
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
      grad_v_weight_priorTime[vd] = E_priorTime[vd] + v_cross_B_priorTime[vd] + extraAcceleration[vd];
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
    auto f_priorTime = Function::solution(f[i], vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity,
                                         solnComponent, stageOrdinal, std::string("k"));
    
    _fSolnPriorTime[i] = f_priorTime;
    
    vlasovRHS->addTerm(f_priorTime / FunctionPtr(_dt) * w[i]);
  
    if (_enforcePositivity)
    {
      vlasovSolnWeakPtr->setPolyTypeForQuadrature(Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      _expPsiSoln[i] = Exp<double>::exp(Function::solution(psi[i], vlasovSolnWeakPtr));
      
      FunctionPtr psi_solnPrev = Function::solution(psi[i], vlasovSolnPrevWeakPtr);
      vlasovRHS->addTerm((psi_solnPrev / alphaPG) * w[i]);
      
      FunctionPtr exp_psi = Exp<double>::exp(psi_solnPrev);
      
      _vlasovBF->addTerm(f[i] - exp_psi * psi[i], phi[i]);
      vlasovRHS->addTerm((exp_psi - psi_solnPrev * exp_psi) * phi[i]);
    }
    
    if (theta != 1.0)
    {
      for (int d=0; d<_spaceDim; d++)
      {
        vlasovRHS->addTerm( (1-theta) * v[d] * f_priorTime * w[i]->di(indexOffsetSpaceOneBased + d));  // dx_d
      }
      
      for (int vd=0; vd<_velocityDim; vd++)
      {
        auto weight = grad_v_weight_priorTime[vd];
        vlasovRHS->addTerm( (1-theta) * (_q[i]/_m[i]) * weight * f_priorTime * w[i]->di(indexOffsetVelocityOneBased + vd)); // d/dv_vd
      }
    }
  }
  
  _vlasovSoln->setRHS(vlasovRHS);
    
  vlasovMesh->registerSolution(_vlasovSoln);
  vlasovMesh->registerSolution(_vlasovSolnPriorTime);
  vlasovMesh->registerSolution(_vlasovSolnPrev);
    
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

VlasovBackwardEulerFormulation::~VlasovBackwardEulerFormulation()
{
  if ((_vlasovSoln != Teuchos::null) && (_vlasovSoln->mesh() != Teuchos::null))
  {
    _vlasovSoln->mesh()->unregisterObserver(this);
  }
}

vector<FunctionPtr> VlasovBackwardEulerFormulation::a()
{
  return _a;
}

FunctionPtr VlasovBackwardEulerFormulation::alpha()
{
  return _alphaPG;
}

MeshPtr VlasovBackwardEulerFormulation::vlasovMesh()
{
  return _vlasovSoln->mesh();
}

SolutionPtr VlasovBackwardEulerFormulation::vlasovSolution()
{
  return _vlasovSoln;
}

SolutionPtr VlasovBackwardEulerFormulation::vlasovSolutionPrevious()
{
  return _vlasovSolnPrev;
}

SolutionPtr VlasovBackwardEulerFormulation::vlasovSolutionPriorTime()
{
  return _vlasovSolnPriorTime;
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

void VlasovBackwardEulerFormulation::didPRefine(const set<GlobalIndexType> &cellIDs, int pToAdd)
{
  _gmgSolver = Teuchos::null;
}

void VlasovBackwardEulerFormulation::didRepartition(MeshTopologyPtr meshTopo)
{
  _gmgSolver = Teuchos::null;
}

VarPtr VlasovBackwardEulerFormulation::E_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->E_hat(zeroBasedComp);
}

double VlasovBackwardEulerFormulation::expPsiForCell(GlobalIndexType cellID)
{
  return _expPsiForCell[cellID]; // TODO: support multiple species here
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
  
  _fluxLimitingWeightForCell.clear();
  
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
    
    if (rank == 0)
    {
      std::cout << "VlasovBackwardEulerFormulation: exported debugging visualizations for solveNumber " << solveNumber << std::endl;
    }
    
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
  double alpha = _alpha0;
  
  double fRelDiff = 1.0;
  
  int alphaUpdateFrequency = 4; // let alpha stay same for this many substeps
  std::vector<double> prior_fRelDiffs(alphaUpdateFrequency, std::numeric_limits<double>::max());
  
  for (int substepOrdinal=0; substepOrdinal < _maxSubsteps; substepOrdinal++)
  {
    if (_enforcePositivity)
    {
      _alphaPG->setValue(alpha);
    }
    
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
    bool fluxLimiting = true;
    if (_enforcePositivity && fluxLimiting)
    {
      // set f = theta_K f + (1-theta_K) * exp (psi)
      // want one value of theta per species per cell
      // we can do this on the local MPI rank (no communication required)
      
      const auto & myCellIDs = _vlasovSoln->mesh()->cellIDsInPartition();
      const auto numMyCells = myCellIDs.size();
      
      for (const auto & cellID : myCellIDs)
      {
        auto basisCache = BasisCache::basisCacheForCell(_vlasovSoln->mesh(), cellID, false, 0, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
        auto f_values   = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(_fSoln[0],      basisCache);
        auto psi_values = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(_expPsiSoln[0], basisCache);
        
        // we assume all the species use a common basis for f
        auto trialDofOrdering = _vlasovSoln->mesh()->getElementType(cellID)->trialOrderPtr;
        auto fBasis = trialDofOrdering->getBasis(this->f(0)->ID());
        int basisCardinality   = fBasis->getCardinality();
        FieldContainer<double> basisCoefficients(1, basisCardinality); // (C,F)
        
        auto psiBasis = trialDofOrdering->getBasis(this->psi(0)->ID());
        TEUCHOS_TEST_FOR_EXCEPTION(psiBasis->getCardinality() != 1, std::invalid_argument, "This implementation assumes constant basis for psi");
        FieldContainer<double> psiBasisCoefficients(1, psiBasis->getCardinality()); // (C,F)
        
        auto allCoefficientsForCell = _vlasovSoln->allCoefficientsForCellID(cellID);
        
        // first, impose a minimum value for psi (avoid FPEs, etc.)
        const double PSI_MIN = -40;
        const double PSI_MAX =  20;
        for (int i=0; i<_numSpecies; i++)
        {
          auto psi_soln = Function::solution(this->psi(i), _vlasovSoln);
          
          psi_soln->values(psi_values, ASSIGN, basisCache);
          
          double psi_value;
          Kokkos::Min<double> minPsi(psi_value);
          
          int numPoints  = psi_values.extent_int(1);
          Kokkos::parallel_reduce("VlasovBackwardEulerFormulation flux limiting, setting psi floor", numPoints,
          KOKKOS_LAMBDA (const int &pointOrdinal, double& intermediatePsi)
          {
            const auto & psi_value_point = psi_values(0,pointOrdinal);
            intermediatePsi = std::min(psi_value_point, intermediatePsi);
          }, minPsi);
          
          psi_value = std::max(PSI_MIN, psi_value);
          psi_value = std::min(PSI_MAX, psi_value);
          
          Projector<double>::projectFunctionOntoBasis(psiBasisCoefficients, Function::constant(psi_value), psiBasis, basisCache);
          const double exp_psi_value = exp(psi_value);
          
          const int psi_ID = this->psi(i)->ID();
          for (int dofOrdinal=0; dofOrdinal < psiBasis->getCardinality(); dofOrdinal++)
          {
            int localDofIndex = trialDofOrdering->getDofIndex(psi_ID, dofOrdinal);
            const auto & projected_psi_coefficient = psiBasisCoefficients(0, dofOrdinal);
            allCoefficientsForCell(localDofIndex) = projected_psi_coefficient;
          }
          
          double theta = 1;
          auto & f       = _fSoln[i];
          
          f->values      (f_values,       ASSIGN, basisCache);
          
          Kokkos::Min<double> minTheta(theta);
          
          numPoints  = f_values.extent_int(1);
          Kokkos::parallel_reduce("VlasovBackwardEulerFormulation flux limiting, theta determination", numPoints,
          KOKKOS_LAMBDA (const int &pointOrdinal, double& intermediateTheta)
          {
            double theta_pt = 1.0;
            const auto & f_value = f_values(0,pointOrdinal);
            
            if (f_value < 0)
            {
              theta_pt = 1 - f_value / (f_value - exp_psi_value);
            }
            intermediateTheta = std::min(theta_pt, intermediateTheta);
          }, minTheta);
          
//          if (cellID == 1024) // DEBUGGING output for Christenson-Lau with 32x64 coarse mesh, 0 UR
//          {
//            std::cout << "theta for cell " << cellID << ": " << theta << std::endl;
//          }
          
          _fluxLimitingWeightForCell[cellID] = theta;
          _expPsiForCell[cellID] = exp_psi_value;
          
          Projector<double>::projectFunctionOntoBasis(basisCoefficients, theta * f + (1-theta) * exp_psi_value, fBasis, basisCache);
          
          const int f_ID = this->f(i)->ID();
          for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++)
          {
            int localDofIndex = trialDofOrdering->getDofIndex(f_ID, dofOrdinal);
            allCoefficientsForCell(localDofIndex) = basisCoefficients(0, dofOrdinal);
          }
        }
        
        _vlasovSoln->setSolnCoeffsForCellID(allCoefficientsForCell, cellID);

        SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(f_values);
        SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(psi_values);
      }
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
    fRelDiff = fDiff_L2 / f_L2;
    
    if (_enforcePositivity)
    {
      const double alphaMin = 1e-8;
      const double alphaMax = 1e8;
      // adjust alpha for next substep:
      if (_alphaMultiplier > 0)
      {
        alpha *= _alphaMultiplier;
      }
      else if (prior_fRelDiffs[(substepOrdinal % alphaUpdateFrequency)] != std::numeric_limits<double>::max())
      {
        const double fRelDiff_previous = prior_fRelDiffs[(substepOrdinal % alphaUpdateFrequency)];
        double relativeChange = (fRelDiff_previous - fRelDiff) / std::max(fRelDiff, fRelDiff_previous);
        
        if (std::abs(relativeChange) < 1e-1)
        {
          // stagnated: increase alpha
          if (alpha != alphaMax)
          {
            alpha = std::min(2.0 * alpha, alphaMax);
            // ensure that we won't adjust alpha for at least alphaUpdateFrequency steps:
            prior_fRelDiffs = std::vector<double>(alphaUpdateFrequency, std::numeric_limits<double>::max());
          }
          if (rank == 0) std::cout << "stagnated: alpha: " << alpha << std::endl;
        }
        else if (relativeChange < 0)
        {
          // diverging: decrease alpha
          if (alpha != alphaMin)
          {
            alpha = std::max(0.25 * alpha, alphaMin);
            // ensure that we won't adjust alpha for at least alphaUpdateFrequency steps:
            prior_fRelDiffs = std::vector<double>(alphaUpdateFrequency, std::numeric_limits<double>::max());
          }
          if (rank == 0) std::cout << "diverging: alpha: " << alpha << std::endl;
        }
        else
        {
          // converging rapidly enough: leave alpha alone
          if (rank == 0) std::cout << "converging: alpha: " << alpha << std::endl;
        }
      }
    }
    prior_fRelDiffs[(substepOrdinal % alphaUpdateFrequency)] = fRelDiff;
    
    const double configurationSpaceRelativeDiff = _configurationSpaceFormulation->l2NormOfSolutionUpdate();
    
    substepStepSizesVlasov.push_back(fRelDiff);
    
    if ((rank == 0) && !_runQuietly)
    {
//      if (f_L2 < _fixedPointTolerance)
//      {
        std::cout << "Substep " << substepOrdinal+1 << ", absolute L^2 norm of difference in f: " << fDiff_L2 << std::endl;
//      }
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
  
  _t->setValue(_t->getConstantValue() + _dt->getConstantValue());
}

double VlasovBackwardEulerFormulation::fluxLimitingWeightForCell(GlobalIndexType cellID)
{
  return _fluxLimitingWeightForCell[cellID];
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
    
    double t;
    timeFileStream >> t;
    _t->setValue(t);
    
    timeFileStream.close();
  }
}

double VlasovBackwardEulerFormulation::m(const int speciesOrdinal) const
{
  return _m[speciesOrdinal];
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

VarPtr VlasovBackwardEulerFormulation::phi(const int speciesOrdinal)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(_phi_names[speciesOrdinal], L2);
}

VarPtr VlasovBackwardEulerFormulation::psi(const int speciesOrdinal)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(_psi_names[speciesOrdinal]);
}

double VlasovBackwardEulerFormulation::q(const int speciesOrdinal) const
{
  return _q[speciesOrdinal];
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

void VlasovBackwardEulerFormulation::setAlpha(const double alpha0, const double alphaMultiplier)
{
  _alpha0 = alpha0;
  _alphaMultiplier = alphaMultiplier;
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

void VlasovBackwardEulerFormulation::setTestIP(VlasovBackwardEulerFormulation::TestIPChoice ipChoice)
{
  IPPtr ip;
  switch (ipChoice) {
    case VLASOV_TIME_MARCHING_GRAPH_INNER_PRODUCT:
      ip = this->vlasovSolution()->bf()->graphNorm();
      break;
    
    case VLASOV_TIME_MARCHING_L2_INNER_PRODUCT:
      ip = this->vlasovSolution()->bf()->l2Norm();
      break;
    case VLASOV_TIME_MARCHING_CONSERVATIVE_INNER_PRODUCT:
    {
      double theta = _crankNicolson ? 0.5 : 1.0;
      ip = IP::ip();
      FunctionPtr dt = _dt;
      auto vx = this->v_i(0); // TODO: generalize this to higher dimensions
      auto ax = this->a()[0]; // TODO: generalize this to higher dimensions
      
      for (int i=0; i<_numSpecies; i++)
      {
        auto w  = this->w(i);
        auto w_dx = w->di(1);
        auto w_dv = w->di(2);
        
        ip->addTerm(w);
        
        if (!_enforcePositivity)
        {
          ip->addTerm(w - (2 * theta * dt * vx) * w_dx - (2*theta*dt*ax)* w_dv);
        }
        else
        {
          auto phi = this->phi(i);
          ip->addTerm(w - (2 * theta * dt * vx) * w_dx - (2*theta*dt*ax)* w_dv + 2 * dt * phi);
          
          auto psi_prev = Function::solution(this->psi(i), this->vlasovSolutionPrevious());
          auto exp_psi_prev = Exp<double>::exp(psi_prev);
          FunctionPtr alpha = this->alpha();
          auto sqrt_beta = Function::sqrtFunction(2 * alpha * dt * exp_psi_prev);
          ip->addTerm(sqrt_beta * phi);
        }
      }
    }
      break;
  }
  this->vlasovSolution()->setIP(ip);
}

void VlasovBackwardEulerFormulation::setTestIP(std::string ipChoiceString)
{
  VlasovBackwardEulerFormulation::TestIPChoice ipChoice;
  if (ipChoiceString == "graph")
  {
    ipChoice = Camellia::VlasovBackwardEulerFormulation::VLASOV_TIME_MARCHING_GRAPH_INNER_PRODUCT;
  }
  else if (ipChoiceString == "L2")
  {
    ipChoice = Camellia::VlasovBackwardEulerFormulation::VLASOV_TIME_MARCHING_L2_INNER_PRODUCT;
  }
  else if (ipChoiceString == "conservative")
  {
    ipChoice = Camellia::VlasovBackwardEulerFormulation::VLASOV_TIME_MARCHING_CONSERVATIVE_INNER_PRODUCT;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported ipChoice");
  }
  this->setTestIP(ipChoice);
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
//  {
//    // DEBUGGING
//    double E_extruded_norm = _configurationSpaceFormulation->E_i(0)->l2norm(_vlasovSoln->mesh());
//    const int rank = MPIWrapper::CommWorld()->MyPID();
//
//    if (rank == 0)
//    {
//      std::cout << "E_extruded_norm = " << E_extruded_norm << std::endl;
//    }
//  }
  
  _vlasovSolnPriorTime->setSolution(_vlasovSoln);
  _configurationSpaceFormulation->copySolutionToPriorTime();
}

void VlasovBackwardEulerFormulation::takeTimeStep()
{
  startTimeStep();
  finishTimeStep();
}

void VlasovBackwardEulerFormulation::setInitialConditions(FunctionPtr f0, bool enforceInflowBCsWeakly)
{
  setInitialConditions(std::vector<FunctionPtr>{f0}, enforceInflowBCsWeakly);
}

void VlasovBackwardEulerFormulation::setInitialConditions(std::vector<FunctionPtr> f0, bool enforceInflowBCsWeakly)
{
  TEUCHOS_TEST_FOR_EXCEPTION(int(f0.size()) != _numSpecies, std::invalid_argument, "Must define initial conditions for each species");
  std::map<int,FunctionPtr> projectionMap;
    
  FunctionPtr n = Function::normal();
  FunctionPtr v_dot_nx = Function::zero();
  for (int vd=0; vd<std::min(_velocityDim,_spaceDim); vd++)
  {
    FunctionPtr v_d = this->v_i(vd);
    FunctionPtr n_d = n->spatialComponent(vd+1); // 1-based index for Function::spatialComponent();
    v_dot_nx = v_dot_nx + v_d * n_d;
  }
  
  FunctionPtr a_dot_nv = Function::zero();
  {
    const int indexOffsetVelocityOneBased = _spaceDim + 1;
    for (int d=0; d<_velocityDim; d++)
    {
      auto nvd = n->spatialComponent(d+indexOffsetVelocityOneBased);
      a_dot_nv = a_dot_nv + _a[d] * nvd;
    }
  }
  
  for (int speciesOrdinal=0; speciesOrdinal<_numSpecies; speciesOrdinal++)
  {
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr n   = Function::normal();
    FunctionPtr f0_trace;
 
    if (_useConformingTraces)
      f0_trace = f0[speciesOrdinal];
    else
      f0_trace = f0[speciesOrdinal] * (v_dot_nx + (_q[speciesOrdinal]/_m[speciesOrdinal]) * a_dot_nv) * sgn;
    
    if (enforceInflowBCsWeakly)
    {
      // then we want to set homogeneous conditions at the boundaries
      const int tagSpatialInflow = this->tagVlasovSpatialInflow();
      FunctionPtr inflow_characteristic = Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, tagSpatialInflow);
      
      f0_trace = f0_trace - inflow_characteristic * f0_trace; // 0 at inflow boundary, original f0_trace everywhere else
    }
    
    VarPtr traceVar = _useConformingTraces ? this->f_hat(speciesOrdinal) : this->t_n_hat(speciesOrdinal);
    
    projectionMap[this->f(speciesOrdinal)->ID()] = f0[speciesOrdinal];
    projectionMap[traceVar->ID()] = f0_trace;
    
    if (_enforcePositivity)
    {
      FunctionPtr psi0;
      if (f0[speciesOrdinal]->isZero())
      {
        psi0 = Function::constant(-40);
      }
      else
      {
        psi0 = Teuchos::rcp( new Ln<double>(f0[speciesOrdinal]) );
      }
      projectionMap[this->psi(speciesOrdinal)->ID()] = psi0;
    }
  }
  
  const int solutionOrdinal = 0;
  _vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  configurationSpaceSolve();
  
  // project again to get an f0_trace that reflects configuration-space solution
  _vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  
  _vlasovSolnPrev->setSolution(_vlasovSoln);
  _vlasovSolnPriorTime->setSolution(_vlasovSoln);
  
  _t->setValue(0);
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in,
                                                                   double timeRampForInflowConditions)
{
  using std::vector;
  setInitialAndInflowConditions(vector<FunctionPtr>{f0}, vector<FunctionPtr>{f_in}, timeRampForInflowConditions, _enforcePositivity);
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in,
                                                                   double timeRampForInflowConditions, bool imposeInflowWeakly)
{
  using std::vector;
  setInitialAndInflowConditions(vector<FunctionPtr>{f0}, vector<FunctionPtr>{f_in}, timeRampForInflowConditions, imposeInflowWeakly);
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(std::vector<FunctionPtr> f0, std::vector<FunctionPtr> f_in,
                                                                   double timeRampForInflowConditions)
{
  setInitialAndInflowConditions(f0, f_in, timeRampForInflowConditions, _enforcePositivity);
}

void VlasovBackwardEulerFormulation::setInitialAndInflowConditions(std::vector<FunctionPtr> f0, std::vector<FunctionPtr> f_in,
                                                                   double timeRampForInflowConditions, bool imposeInflowWeakly)
{
  TEUCHOS_TEST_FOR_EXCEPTION(int(f_in.size()) != _numSpecies, std::invalid_argument, "Must define inflow conditions for each species");
  auto vlasovBC = BC::bc();
  
  for (int speciesOrdinal=0; speciesOrdinal<_numSpecies; speciesOrdinal++)
  {
    VarPtr traceVar = _useConformingTraces ? this->f_hat(speciesOrdinal) : this->t_n_hat(speciesOrdinal);
    
    for (int vd=0; vd<_velocityDim; vd++)
    {
      const int tagMin = this->tagVlasovMinVelocity(vd);
      const int tagMax = this->tagVlasovMaxVelocity(vd);
      vlasovBC->addDirichlet(traceVar, tagMin, Function::zero());
      vlasovBC->addDirichlet(traceVar, tagMax, Function::zero());
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
    
    FunctionPtr f_BC = f_in[speciesOrdinal];
    if (timeRampForInflowConditions != 0.0)
    {
      ParameterFunctionPtr t  = this->t();
      double dt = _dt->getConstantValue(); // want the time ramp at time t to be evaluated at t + dt
      FunctionPtr timeRamp = TimeRampFunction::timeRamp(this->t(), timeRampForInflowConditions, -dt); // timeRamp evaluated at current time
      f_BC = f_BC * timeRamp;
    }
    if (!imposeInflowWeakly)
    {
      FunctionPtr traceBC = _useConformingTraces ? f_BC : v_dot_nx * f_BC;
      vlasovBC->addDirichlet(traceVar, tagSpatialInflow, traceBC);
    }
    else
    {
      // if imposeInflowWeakly is set, don't project BC functions onto the trace space
      // instead, use homogeneous BCs, and add something to the RHS at inflow
      _strongInflowBCFlag->setValue(0); // turns off a term that gets added to the RHS otherwise
      vlasovBC->addDirichlet(traceVar, tagSpatialInflow, Function::zero());
      FunctionPtr inflow_characteristic = Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, tagSpatialInflow);
      f_BC = _crankNicolson ? 0.5 * (f_in[speciesOrdinal] + _fSolnPriorTime[speciesOrdinal]) : f_in[speciesOrdinal];
      FunctionPtr tn_BC = inflow_characteristic * v_dot_nx * f_BC;
      _vlasovSoln->rhs()->addTerm(-tn_BC * this->w(speciesOrdinal)); // minus sign comes from moving this term from the LHS to the RHS

      // note: the above is only correct for non-conforming traces
      TEUCHOS_TEST_FOR_EXCEPTION(_useConformingTraces, std::invalid_argument, "conforming traces not yet supported for weakly imposed BCs");
    }
  }

  _vlasovSoln->setBC(vlasovBC);
  
  setInitialConditions(f0, imposeInflowWeakly);
}

Teuchos::RCP<SolutionMoments> VlasovBackwardEulerFormulation::solutionMoments(int i)
{
  return _solutionMoments[i];
}

Teuchos::RCP<SolutionMoments> VlasovBackwardEulerFormulation::solutionMomentsPriorTime(int i)
{
  return _solutionMomentsPriorTime[i];
}

double VlasovBackwardEulerFormulation::timeStep() const
{
  return _dt->getConstantValue();
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

ParameterFunctionPtr VlasovBackwardEulerFormulation::t() const
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
