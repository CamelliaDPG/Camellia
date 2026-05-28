//
//  ConfigurationSpaceFormulationMaxwell1D2V.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/7/26.
//

#include "ConfigurationSpaceFormulationMaxwell1D2V.h"

#include "BC.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "RHS.h"

using namespace Camellia;

// fields:
const string ConfigurationSpaceFormulationMaxwell1D2V::S_EX = "E_x";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_BY = "B_y";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_PHI = "\\phi";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_A = "A";

// traces:
const string ConfigurationSpaceFormulationMaxwell1D2V::S_BY_HAT = "\\widehat{B}_y";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_EX_HAT = "\\widehat{E}_x";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_PHI_HAT = "\\widehat{\\phi}";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_A_HAT = "\\widehat{A}";

// tests:
const string ConfigurationSpaceFormulationMaxwell1D2V::S_W_JX = "w_{J_x}";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_W_JZ = "w_{J_z}";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_W_RHO = "\\tau_{\\rho}";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_TAU_A   = "\\tau_{A}";
const string ConfigurationSpaceFormulationMaxwell1D2V::S_TAU_PHI = "\\tau_{\\phi}";

ConfigurationSpaceFormulationMaxwell1D2V::ConfigurationSpaceFormulationMaxwell1D2V(const std::vector<double> &q, const double eps0, const double mu0)
:
ConfigurationSpaceFormulation(),
_q(q),
_eps0(eps0),
_mu0(mu0)
{}

ConfigurationSpaceFormulationMaxwell1D2V::ConfigurationSpaceFormulationMaxwell1D2V(const double &q, const double eps0, const double mu0)
:
ConfigurationSpaceFormulationMaxwell1D2V(std::vector<double>{q},eps0,mu0)
{}

void ConfigurationSpaceFormulationMaxwell1D2V::initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k)
{
  this->_configurationMeshTopo = meshTopo;
  
  _spaceDim = meshTopo->getDimension() - 1; // assumption is meshTopo is space-time
  
  Space tauSpace    = HGRAD;
  Space E_hat_space = L2; // non-conforming
  Space B_hat_space = L2; // non-conforming
  Space ESpace      = L2;
  Space BSpace      = L2;

  // fields
  VarPtr Ex, Ey, Ez; // electric field
  VarPtr Bx, By, Bz; // magnetic field
  VarPtr phi; // potential
  VarPtr A;

  // traces
  VarPtr By_hat;
  VarPtr Ex_hat;
  VarPtr phi_hat;
  VarPtr A_hat;

  // tests
  VarPtr w_Jx, w_Jz;
  VarPtr w_rho, tau_A, tau_phi;
  
  // 1D-specific tau variables:
  VarPtr tau1y, tau1z;
  VarPtr tau2x, tau2y, tau2z;
  VarPtr tau3x;
  
  // 3D tau variables (2D, too?)
  VarPtr tau1, tau2; // spatially H(curl)
  VarPtr tau3;       // spatially H(div)
  
  VarFactoryPtr maxwellVF = VarFactory::varFactory();
  Ex = maxwellVF->fieldVarSpaceTime(S_EX, ESpace, L2); // L2 in time, too
  By = maxwellVF->fieldVarSpaceTime(S_BY, BSpace, L2);
  phi = maxwellVF->fieldVarSpaceTime(S_PHI, L2, L2);
  A  = maxwellVF->fieldVarSpaceTime(S_A, ESpace, L2);

  // we define these as fluxes rather than traces because fluxes include any explicit normals in the trace integrals, and Camellia will add a side parity for fluxes to account for an implicit normal in the variable definition.
  Ex_hat = maxwellVF->traceVarSpaceTime(S_EX_HAT, Ex, E_hat_space, L2); // L2 in space and time
  By_hat = maxwellVF->traceVarSpaceOnly(S_BY_HAT, By, B_hat_space, L2); // L2 in space
  phi_hat = maxwellVF->traceVarSpaceOnly(S_PHI_HAT, phi, L2);
  A_hat = maxwellVF->traceVarSpaceOnly(S_A_HAT, A, L2);
  
  const int X_INDEX = 1;
  const int T_INDEX = _spaceDim + 1;
  
  FunctionPtr n  = TFunction<double>::normal();
  FunctionPtr nx = n->spatialComponent(X_INDEX);
  FunctionPtr nt = n->spatialComponent(T_INDEX);

//  w_Jx    = maxwellVF->testVarSpaceTime(S_W_JX,       L2, HGRAD);   //    L2 in space, HGRAD in time
//  w_Jz    = maxwellVF->testVarSpaceTime(S_W_JZ,    HGRAD,    L2);   // HGRAD in space,    L2 in time
//  w_rho   = maxwellVF->testVarSpaceTime(S_W_RHO,   HGRAD,    L2);   // HGRAD in space,    L2 in time
//  tau_A   = maxwellVF->testVarSpaceTime(S_TAU_A,   HGRAD,    L2);   // HGRAD in space,    L2 in time
//  tau_phi = maxwellVF->testVarSpaceTime(S_TAU_PHI, HGRAD,    L2);   // HGRAD in space,    L2 in time
  
  // what's commented out above are the minimum requirements on the test functions.  But we don't handle
  // ref-to-physical transformations for those mixed spaces at the moment (which surprises me; I thought
  // that was there).  So I'm just making everything HGRAD everywhere: more than we need, but shouldn't hurt.
  w_Jx    = maxwellVF->testVarSpaceTime(S_W_JX,    HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
  w_Jz    = maxwellVF->testVarSpaceTime(S_W_JZ,    HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
  w_rho   = maxwellVF->testVarSpaceTime(S_W_RHO,   HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
  tau_A   = maxwellVF->testVarSpaceTime(S_TAU_A,   HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
  tau_phi = maxwellVF->testVarSpaceTime(S_TAU_PHI, HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
  
  _maxwellBF = BF::bf(maxwellVF);
  
  // space-time cell topology:
  CellTopoPtr spaceTopo = CellTopology::line();
  CellTopoPtr spaceTimeTopo = CellTopology::lineTensorTopology(spaceTopo); // space x time
  std::vector<int> spatialSides;
  std::vector<int> temporalSides;
  for (int spatialSide=0; spatialSide<spaceTopo->getSideCount(); spatialSide++)
  {
    int spaceTimeSideOrdinal = spaceTimeTopo->getExtrudedSideOrdinal(spatialSide);
    spatialSides.push_back(spaceTimeSideOrdinal);
  }
  for (int temporalSide=0; temporalSide<2; temporalSide++)
  {
    int spaceTimeSideOrdinal = spaceTimeTopo->getTemporalSideOrdinal(temporalSide);
    temporalSides.push_back(spaceTimeSideOrdinal);
  }
  
  // tau_phi
  _maxwellBF->addTerm(Ex, tau_phi);
  _maxwellBF->addTerm(phi_hat * nx, tau_phi); // integral over spatial sides (those with n_x != 0)
  _maxwellBF->addTerm(phi, - tau_phi->dx());
  
  // tau_A
  _maxwellBF->addTerm(By, tau_A);
  _maxwellBF->addTerm(A_hat * nx, tau_A); // integral over spatial sides (those with n_x != 0)
  _maxwellBF->addTerm(A, - tau_A->dx());
  
  // w_Jx
  _maxwellBF->addTerm(Ex_hat * nt, w_Jx); // integral over temporal sides (those with n_t != 0)
  _maxwellBF->addTerm(Ex, - w_Jx->dy()); // dy: this is really dt()
  
  // w_Jz
  _maxwellBF->addTerm(By_hat * nx, w_Jz); // integral over spatial sides (those with n_x != 0)
  _maxwellBF->addTerm(By, - w_Jz->dx());
  
  // w_rho
  _maxwellBF->addTerm(Ex_hat * nx, w_rho); // integral over spatial sides (those with n_x != 0)
  _maxwellBF->addTerm(Ex, - w_rho->dx());
  
  Teuchos_CommPtr comm = Teuchos::null;
  const bool allowPruning = false;
  map<int,int> trialOrderEnhancements;
  map<int,int> testOrderEnhancements;

  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  MeshPtr maxwellMesh = Teuchos::rcp( new Mesh(meshTopo, _maxwellBF, H1Order, delta_k,
                                               trialOrderEnhancements, testOrderEnhancements, partitionPolicy,
                                               comm, allowPruning) );
  
  auto maxwellSoln     = Solution::solution(_maxwellBF, maxwellMesh);
  auto maxwellSolnPrev = Solution::solution(_maxwellBF, maxwellMesh); // will copy out after fixed point solve
  
  maxwellSoln->setIP(_maxwellBF->graphNorm());
  
  setConfigurationSolution(maxwellSoln);
  setConfigurationSolutionPrevious(maxwellSolnPrev);
  
  auto maxwellSolnWeakPtr     = Teuchos::rcp(maxwellSoln.get(),     false);
  auto maxwellSolnPrevWeakPtr = Teuchos::rcp(maxwellSolnPrev.get(), false);
  
  std::vector<FunctionPtr> ESolnVector(_spaceDim);
  std::vector<FunctionPtr> ESolnPrevVector(_spaceDim);
  for (int i=0; i<_spaceDim; i++)
  {
    ESolnVector[i]     = Function::solution(Ex, maxwellSolnWeakPtr);
    ESolnPrevVector[i] = Function::solution(Ex, maxwellSolnPrevWeakPtr);
  }
  _ESoln     = Function::vectorize(ESolnVector);
  FunctionPtr E_solnPrev = Function::vectorize(ESolnPrevVector);
  
  _EDiff = _ESoln - E_solnPrev;
  
  {
    // to avoid an exception being thrown for an unset solution when _E_extruded->updateValues() is first called, we project zero onto the Maxwell solution first
    std::map<int,FunctionPtr> zeroMaxwell;
    FunctionPtr zero = Function::zero();
    for (int i=0; i<_spaceDim; i++)
    {
      zeroMaxwell[this->E_fieldVar(i)->ID()] = zero;
      zeroMaxwell[this->E_hat(i)->ID()] = zero;
      const bool Bx_in_1D = (_spaceDim == 1) && (i == 0);
      if (!Bx_in_1D)
      {
        zeroMaxwell[this->B_fieldVar(i)->ID()] = zero;
        zeroMaxwell[this->B_hat(i)->ID()] = zero;
      }
    }
    maxwellSoln->projectOntoMesh(zeroMaxwell);
  }
}

std::string ConfigurationSpaceFormulationMaxwell1D2V::name() const
{
  return "Maxwell1D2V";
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::A_fieldVar(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return Teuchos::null;
    case 1: return vf->trialVar(S_A);
    case 2: return Teuchos::null;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::A_hat(int zeroBasedComp)
{
  if (zeroBasedComp == 1) // A_y
  {
    VarFactoryPtr vf = _maxwellBF->varFactory();
    return vf->trialVar(S_A_HAT);
  }
  else
  {
    return Teuchos::null;
  }
}

FunctionPtr ConfigurationSpaceFormulationMaxwell1D2V::B_i(int zeroBasedComp)
{
  if (zeroBasedComp != 1)
  {
    return Function::zero();
  }
  else return _By_extruded;
}

FunctionPtr ConfigurationSpaceFormulationMaxwell1D2V::B_i_priorTime(int zeroBasedComp)
{
  if (zeroBasedComp == 1)
  {
    return _By_priorTime_extruded;
  }
  else
  {
    return Function::zero();
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::B_fieldVar(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return Teuchos::null;
    case 1: return vf->trialVar(S_BY);
    case 2: return Teuchos::null;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::B_hat(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return Teuchos::null;
    case 1: return vf->trialVar(S_BY_HAT);
    case 2: return Teuchos::null;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::E_fieldVar(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return vf->trialVar(S_EX);
    case 1: return Teuchos::null;
    case 2: return Teuchos::null;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::E_hat(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return vf->trialVar(S_EX_HAT);
    case 1: return Teuchos::null;
    case 2: return Teuchos::null;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

FunctionPtr ConfigurationSpaceFormulationMaxwell1D2V::E_i(int zeroBasedComp)
{
  if (zeroBasedComp == 0)
  {
    return _Ex_extruded;
  }
  else
  {
    return Function::zero();
  }
}

FunctionPtr ConfigurationSpaceFormulationMaxwell1D2V::E_i_priorTime(int zeroBasedComp)
{
  if (zeroBasedComp == 0)
  {
    return _Ex_priorTime_extruded;
  }
  else
  {
    return Function::zero();
  }
}

//! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
int ConfigurationSpaceFormulationMaxwell1D2V::maxMomentNumber()
{
  return 1;
}

void ConfigurationSpaceFormulationMaxwell1D2V::setCurrentSources(FunctionPtr rho, FunctionPtr Jx, FunctionPtr Jy, FunctionPtr Jz)
{
  auto rhs = RHS::rhs();
  rhs->addTerm(rho * w_rho());
  rhs->addTerm(-Jx / _eps0 * w_Jx());
  rhs->addTerm( Jz / _eps0 * w_Jz());
  _configurationSoln->setRHS(rhs);
}

void ConfigurationSpaceFormulationMaxwell1D2V::setRHSForTesting(FunctionPtr rho_over_eps0,
                                                                FunctionPtr Jx_over_eps0,
                                                                FunctionPtr Jz_over_eps0)
{
  auto rhs = RHS::rhs();
  rhs->addTerm(rho_over_eps0 * w_rho());
  rhs->addTerm(Jx_over_eps0  * w_Jx());
  rhs->addTerm(Jz_over_eps0  * w_Jz());
  _configurationSoln->setRHS(rhs);
}

void ConfigurationSpaceFormulationMaxwell1D2V::setMoments(std::vector< Teuchos::RCP<SolutionMoments> > fMoments, FunctionPtr n0)
{
  this->ConfigurationSpaceFormulation::setMoments(fMoments,n0);
  FunctionPtr velocityIntegral;
  FunctionPtr firstMoment_x, firstMoment_y, firstMoment_z;
  int numSpecies = _q.size();
  TEUCHOS_TEST_FOR_EXCEPTION(numSpecies != int(fMoments.size()), std::invalid_argument, "The number of moments objects provided to setMoments() should match the number of q values at construction of ConfigurationSpaceFormulationMaxwell1D2V");
  
  auto vlasovMesh = fMoments[0]->phaseSpaceMesh();
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  // initialize RHS
  auto rhs = RHS::rhs();
  
  // TODO: consider making rho, Jx, Jz ParameterFunctions (member variables) and then setting the RHS during construction; here, we would then update the ParameterFunctions.
  FunctionPtr rho = Function::zero();
  FunctionPtr Jx  = Function::zero();
  FunctionPtr Jy  = Function::zero();
  FunctionPtr Jz  = Function::zero();
  
  for (int i=0; i<numSpecies; i++)
  {
    auto moments_i = _fMoments[i];
    int numVelocityDims = moments_i->numVelocityDims();
    TEUCHOS_TEST_FOR_EXCEPTION(numVelocityDims != 2, std::invalid_argument, "This implementation assumes 1D2V");
    const double & q_i = _q[i];
    velocityIntegral = moments_i->solutionMomentFunction(0,0);
    firstMoment_x    = moments_i->solutionMomentFunction(1,0);
    firstMoment_z    = moments_i->solutionMomentFunction(0,1); // velocity dimensions are x,z
    
    rho = rho + q_i / _eps0 * (velocityIntegral - n0);
    Jx  = Jx  + q_i * firstMoment_x;
    Jz  = Jz  + q_i * firstMoment_z;
  }
  
  rhs->addTerm( rho * w_rho());
  rhs->addTerm( -Jx / _eps0 * w_Jx());
  rhs->addTerm( Jz / _eps0 * w_Jz());
  
  _configurationSoln->setRHS(rhs);
  
  // define extrusion of E, B to the phase-space mesh
  _Ex_extruded = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSoln, this->E_fieldVar(0), vlasovMeshWeakPtr) );
  _By_extruded = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSoln, this->B_fieldVar(1), vlasovMeshWeakPtr) );
  
  if (_configurationSolnPriorTime != Teuchos::null)
  {
    _Ex_priorTime_extruded = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSolnPriorTime, this->E_fieldVar(0), vlasovMeshWeakPtr) );
    _By_priorTime_extruded = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSolnPriorTime, this->B_fieldVar(1), vlasovMeshWeakPtr) );
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::tau_A()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU_A);
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::tau_phi()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU_PHI);
}

void ConfigurationSpaceFormulationMaxwell1D2V::updateConfigurationSpaceFunctions()
{
  static_cast<PhysicalSolutionExtrusion*>(_Ex_extruded.get())->updateValues();
  static_cast<PhysicalSolutionExtrusion*>(_By_extruded.get())->updateValues();
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::VE_fieldVar()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->trialVar(S_PHI);
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::VE_hat()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->trialVar(S_PHI_HAT);
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::w_Jx()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_W_JX);
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::w_Jz()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_W_JZ);
}

VarPtr ConfigurationSpaceFormulationMaxwell1D2V::w_rho()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_W_RHO);
}
