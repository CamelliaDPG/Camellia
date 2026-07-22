//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  RelativisticVlasovSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/15/25.
//
//

#include "RelativisticVlasovSpaceTimeFormulation.h"

#include "AbsFunction.h"
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "SolutionMomentFunction.h"

using namespace Camellia;
using std::vector;
using std::string;

RelativisticVlasovSpaceTimeFormulation::RelativisticVlasovSpaceTimeFormulation(const double c, const double q, const double m,
                                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                                               MeshTopologyPtr spatialMeshTopology,
                                                                               vector<double> temporalVertices,
                                                                               vector<double> velocityVertices_x,
                                                                               vector<double> velocityVertices_y,
                                                                               vector<double> velocityVertices_z,
                                                                               const int H1Order, const int delta_k,
                                                                               const int numUniformRefinements,
                                                                               const TestSpaceInnerProduct testSpaceIP,
                                                                               const double testSpaceL2Weight,
                                                                               const EBasisFamily &basisFamily,
                                                                               const int maxMomentNumberForDriver,
                                                                               const bool enforcePositivity)
:
_c(c),
_testSpaceInnerProduct(testSpaceIP),
_testSpaceL2Weight(testSpaceL2Weight),
VlasovSpaceTimeFormulation(q, m, configurationSpaceFormulation,
                           spatialMeshTopology, temporalVertices,
                           velocityVertices_x,
                           velocityVertices_y,
                           velocityVertices_z,
                           H1Order, delta_k,
                           numUniformRefinements,
                           basisFamily,
                           maxMomentNumberForDriver,
                           true, // true: defer initialization; we call initialize() below.
                           enforcePositivity)
{
  this->_useSumFactorizedAssembly = false; // set base class's flag: we don't (yet) support sum-factorized assembly here.
  this->initialize();
}

IPPtr RelativisticVlasovSpaceTimeFormulation::constructVlasovInnerProduct(const bool sumFactorizedAssembly)
{
  TEUCHOS_TEST_FOR_EXCEPTION(sumFactorizedAssembly, std::invalid_argument, "RelativisticVlasovSpaceTimeFormulation does not support sum-factorized Gram assembly.");
  
  const int spaceDim    = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  vector<FunctionPtr> a = this->a();
  
  IPPtr ip = IP::ip();
  
  const int numSpecies = this->numSpecies();
  
  for (int i=0; i<numSpecies; i++)
  {
    VarPtr w = this->w(i);
    
    if (_testSpaceInnerProduct == GraphTestSpaceInnerProduct)
    {
      LinearTermPtr f_term = - w->di(indexOffsetTimeOneBased); // -dw/dt
      for (int d=0; d<spaceDim; d++)
      {
        auto  v_d    = v_i(d);
        auto dw_dx_d = w->di(indexOffsetSpaceOneBased + d);
        f_term = f_term - v_d * dw_dx_d; // - v \cdot dw/dx
      }
      for (int vd=0; vd<velocityDim; vd++)
      {
        auto dw_dv_vd = w->di(indexOffsetVelocityOneBased + vd);
        f_term = f_term - a[vd] * dw_dv_vd;
      }
      ip->addTerm(f_term);
      
      LinearTermPtr stabilization_term = sqrt(_testSpaceL2Weight) * w;
      ip->addTerm(stabilization_term);
    }
    else if (_testSpaceInnerProduct == L2TestSpaceInnerProduct)
    {
      ip->addTerm(w);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported test space inner product selection");
    }
    
    if (this->enforcePositivity())
    {
      VarPtr phi = this->phi(i);
      ip->addTerm(phi);
    }
  }
  
  return ip;
}

VarFactoryPtr RelativisticVlasovSpaceTimeFormulation::constructVlasovVarFactory()
{
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  BasisFamilyPtr basisFamily = BasisFamily::basisFamily(this->basisFamily());
  
  if (this->enforcePositivity())
  {
    this->initializeProximalGalerkinAlpha();
  }
  
  const int numSpecies = this->numSpecies();
  std::vector<VarPtr> f(numSpecies);
  std::vector<VarPtr> w(numSpecies);
  std::vector<VarPtr> t_n_hat(numSpecies);
  
  for (int i=0; i<numSpecies; i++)
  {
    f[i] = vlasovVF->fieldVar(this->f_name(i));
    w[i] = vlasovVF->testVar(this->w_name(i), HGRAD);
    t_n_hat[i] = vlasovVF->fluxVar(this->t_n_hat_name(i));
    
    if (this->enforcePositivity())
    {
      VarPtr psi_i = vlasovVF->fieldVar(this->psi_name(i), CONSTANT_SCALAR); // element-wise constant latent positivity field on the full phase-space-time cell, including time
      VarPtr phi_i = vlasovVF->testVar(this->phi_name(i), L2);               // latent positivity test function
    }
  }
  
  vlasovVF->setTrialBasisFamily(basisFamily);
  vlasovVF->setTestBasisFamily(basisFamily);
  
  return vlasovVF;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::gamma() const
{
  int ux_component_zero_based = spaceDim() + 1; // plus 1 for temporal dimension
  
  const int uDim = VlasovSpaceTimeFormulation::velocityDim();
  
  FunctionPtr uSquared = Function::zero();
  for (int d=0; d<uDim; d++)
  {
    auto ud = MonomialFunction::xi_n(ux_component_zero_based + d, 1);
    uSquared = uSquared + ud * ud;
  }
  
  FunctionPtr gamma = Function::sqrtFunction(1 + uSquared / (_c * _c));
  
  return gamma;
}

void RelativisticVlasovSpaceTimeFormulation::initializeVlasovBilinearForm(BFPtr vlasovBF)
{
  const int spaceDim = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  const int numSpecies = this->numSpecies();
  std::vector<VarPtr> f(numSpecies);
  std::vector<VarPtr> w(numSpecies);
  std::vector<VarPtr> t_n_hat(numSpecies);
  
  Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation = this->configurationSpaceFormulation();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
    
  // The below assumes that spaceDim <= velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > velocityDim, std::invalid_argument, "velocityDim must be at least as large as spaceDim");
  
  
  std::vector<FunctionPtr> v;
  for (int vd=0; vd<velocityDim; vd++)
  {
    v.push_back(this->v_i(vd));
  }
  
  for (int i=0; i<numSpecies; i++)
  {
    // trace
    VarPtr t_n_hat = this->t_n_hat(i);
    
    // field
    VarPtr f = this->f(i);
    
    // test
    VarPtr w = this->w(i);
    
    if (this->enforcePositivity())
    {
      VarPtr psi = this->psi(i);
      VarPtr phi = this->phi(i);
      FunctionPtr alphaPG = this->proximalGalerkinAlpha();
      auto vlasovSolnPrevWeakPtr = Teuchos::rcp(this->vlasovSolutionPreviousSubstep().get(), false);
      FunctionPtr psi_solnPrev = Function::solution(psi, vlasovSolnPrevWeakPtr);
      FunctionPtr exp_psi = Exp<double>::exp(psi_solnPrev);
      
      vlasovBF->addTerm(psi / alphaPG, w);
      vlasovBF->addTerm(f - exp_psi * psi, phi);
    }
    
    for (int d=0; d<spaceDim; d++)
    {
      vlasovBF->addTerm( -v[d] * f, w->di(indexOffsetSpaceOneBased + d));  // dx_d
    }
    vlasovBF->addTerm( -f, w->di(indexOffsetTimeOneBased)); // dt
    
    const double q = this->q(i);
    const double m = this->m(i);
    auto a = this->a(-1); // acceleration term (unweighted by species q or m)
    for (int vd=0; vd<velocityDim; vd++)
    {
      // grad u of f
      vlasovBF->addTerm(-q/m * a[vd] * f, w->di(indexOffsetVelocityOneBased + vd));
    }
    
    vlasovBF->addTerm(t_n_hat, w);
  }
}

Teuchos::RCP<VlasovSpaceTimeFormulation> RelativisticVlasovSpaceTimeFormulation::nextTimeSlab(double tWidth, int numTemporalElements)
{
  auto temporalVertices = MeshTools::verticesFor1DIntervalMesh(t1(), tWidth, numTemporalElements);
  
  // create a new spatial MeshTopology based on the root of the current _spatialMeshTopology
  auto rootSpatialMeshTopology = MeshFactory::rootMeshTopologyCopy(spatialMeshTopology());
  
  auto nextSlabForm = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(_c, q(), m(), configurationSpaceFormulation(), rootSpatialMeshTopology,
                                                                              temporalVertices, velocitySpaceVertices_x(), velocitySpaceVertices_y(), velocitySpaceVertices_z(),
                                                                              H1Order(), delta_k(), numUniformRefinements(),
                                                                              _testSpaceInnerProduct, _testSpaceL2Weight,
                                                                              basisFamily(),
                                                                              0, this->enforcePositivity()));
  
  configureNextSlab(nextSlabForm);
  
  return nextSlabForm;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::v_i(const int zeroBasedComp) const
{
  int ux_component_zero_based = spaceDim() + 1; // plus 1 for temporal dimension
  auto ux = MonomialFunction::xi_n(ux_component_zero_based + zeroBasedComp, 1);
  
  FunctionPtr gammaFxn = gamma();
  return ux / gammaFxn;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::relativisticFirstMoment(const int zeroBasedVelocityComponent, const int speciesOrdinal)
{
  TEUCHOS_TEST_FOR_EXCEPTION((speciesOrdinal < 0) || (speciesOrdinal >= numSpecies()), std::invalid_argument,
                             "species ordinal out of range");
  std::pair<int,int> key{zeroBasedVelocityComponent, speciesOrdinal};
  auto cached = _relativisticFirstMomentFunctions.find(key);
  if (cached != _relativisticFirstMomentFunctions.end())
  {
    return cached->second;
  }

  auto drivingMoments = solutionMoments(speciesOrdinal);
  auto firstMoment = Teuchos::rcp(new RelativisticFirstMomentFunction(drivingMoments.get(),
                                                                      vlasovSolution(), f(speciesOrdinal),
                                                                      configurationSpaceMesh(), _c,
                                                                      zeroBasedVelocityComponent));
  _relativisticFirstMomentFunctions[key] = firstMoment;
  return firstMoment;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::currentX() const
{
  auto nonConstThis = const_cast<RelativisticVlasovSpaceTimeFormulation*>(this);
  FunctionPtr Jx = Function::zero();
  for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
  {
    Jx = Jx + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(0, speciesOrdinal);
  }
  return Jx;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::currentY() const
{
  if (velocityDim() < 2) return Function::zero();
  auto nonConstThis = const_cast<RelativisticVlasovSpaceTimeFormulation*>(this);
  FunctionPtr Jy = Function::zero();
  for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
  {
    Jy = Jy + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(1, speciesOrdinal);
  }
  return Jy;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::currentZ() const
{
  auto nonConstThis = const_cast<RelativisticVlasovSpaceTimeFormulation*>(this);
  FunctionPtr Jz = Function::zero();
  if (velocityDim() == 2)
  {
    for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
    {
      Jz = Jz + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(1, speciesOrdinal);
    }
    return Jz;
  }
  else if (velocityDim() >= 3)
  {
    for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
    {
      Jz = Jz + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(2, speciesOrdinal);
    }
    return Jz;
  }
  return Function::zero();
}
