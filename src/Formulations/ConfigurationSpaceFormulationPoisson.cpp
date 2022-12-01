//
//  ConfigurationSpaceFormulationPoisson.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#include "ConfigurationSpaceFormulationPoisson.h"
#include "PhysicalSolutionExtrusion.hpp"

#include "Camellia_TimeDiscretizationType.h"

using namespace Camellia;

// fields:
const string ConfigurationSpaceFormulationPoisson::S_EX = "E_x";
const string ConfigurationSpaceFormulationPoisson::S_EY = "E_y";
const string ConfigurationSpaceFormulationPoisson::S_EZ = "E_z";
const string ConfigurationSpaceFormulationPoisson::S_VE = "V_E";

// traces:
const string ConfigurationSpaceFormulationPoisson::S_E_HAT = "\\widehat{E}";
const string ConfigurationSpaceFormulationPoisson::S_VE_HAT = "\\widehat{V}_E";

// tests:
const string ConfigurationSpaceFormulationPoisson::S_V_POISSON = "v_{\\rm Poisson}";
const string ConfigurationSpaceFormulationPoisson::S_TAU_POISSON = "\\tau_{\\rm Poisson}";

ConfigurationSpaceFormulationPoisson::ConfigurationSpaceFormulationPoisson(const double &q, const double &eps0, const TimeDiscretizationType &timeDiscretizationType)
:
ConfigurationSpaceFormulation(),
_q(q),
_eps0(eps0),
_timeDiscretizationType(timeDiscretizationType)
{}

void ConfigurationSpaceFormulationPoisson::initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k)
{
  this->_configurationMeshTopo = meshTopo;
  
  const bool spaceTime = (_timeDiscretizationType == SPACE_TIME) || (_timeDiscretizationType == EXPONENTIAL_INTEGRATOR);
  _spaceDim = spaceTime ? meshTopo->getDimension() - 1 : meshTopo->getDimension();
  
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim < 1, std::invalid_argument, "Invalid spaceDim");
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim > 3, std::invalid_argument, "Invalid spaceDim");
  
  Space tauSpace    = (_spaceDim > 1) ? HDIV : HGRAD;
  Space E_hat_space = L2; // non-conforming
  Space ESpace      = L2;

  // fields
  VarPtr Ex, Ey, Ez;
  VarPtr VE;

  // traces
  VarPtr E_hat, VE_hat; // t_n_hat: L2 trace, f_hat: H^1 trace

  // tests
  VarPtr w;   // Vlasov test function

  VarFactoryPtr poissonVF = VarFactory::varFactory();
  if (spaceTime)
  {
    VE = poissonVF->fieldVarSpaceTime(S_VE, L2,     L2); // L2 in time, too
    Ex = poissonVF->fieldVarSpaceTime(S_EX, ESpace, L2); // L2 in time, too
    if (_spaceDim > 1) Ey = poissonVF->fieldVarSpaceTime(S_EY, ESpace, L2);
    if (_spaceDim > 2) Ez = poissonVF->fieldVarSpaceTime(S_EZ, ESpace, L2);
  }
  else
  {
    VE = poissonVF->fieldVar(S_VE, L2);
    Ex = poissonVF->fieldVar(S_EX, ESpace); // L2 in time, too
    if (_spaceDim > 1) Ey = poissonVF->fieldVar(S_EY, ESpace);
    if (_spaceDim > 2) Ez = poissonVF->fieldVar(S_EZ, ESpace);
  }
      
  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();

  std::vector<VarPtr> E;
  if (_spaceDim == 1)      E = std::vector<VarPtr>{Ex};
  else if (_spaceDim == 2) E = std::vector<VarPtr>{Ex,Ey};
  else if (_spaceDim == 3) E = std::vector<VarPtr>{Ex,Ey,Ez};
  
  if (spaceTime)
  {
    if (_spaceDim > 1)
      VE_hat = poissonVF->traceVarSpaceOnly(S_VE_HAT, VE, L2);
    else
      VE_hat = poissonVF->fluxVarSpaceOnly(S_VE_HAT, VE * (Function::normal_1D() * parity), L2); // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
  }
  else
  {
    if (_spaceDim > 1)
      VE_hat = poissonVF->traceVar(S_VE_HAT, VE, L2);
    else
      VE_hat = poissonVF->fluxVar(S_VE_HAT, VE * (Function::normal_1D() * parity), L2); // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
  }
  
  if (_spaceDim > 1)
  {
    LinearTermPtr E_dot_n;
    if (_spaceDim == 2)
    {
      E_dot_n = Ex * n->x() + Ey * n->y();
    }
    else
    {
      E_dot_n = Ex * n->x() + Ey * n->y() + Ez * n->z();
    }
    
    if (spaceTime)
    {
      E_hat = poissonVF->fluxVarSpaceOnly(S_E_HAT, E_dot_n * parity);
    }
    else
    {
      E_hat = poissonVF->fluxVar(S_E_HAT, E_dot_n * parity);
    }
  }
  else
  {
    if (spaceTime)
    {
      E_hat = poissonVF->fluxVarSpaceOnly(S_E_HAT, Ex * (Function::normal_1D() * parity));
    }
    else
    {
      E_hat = poissonVF->fluxVar(S_E_HAT, Ex * (Function::normal_1D() * parity));
    }
  }
  
  VarPtr v_Poisson, tau_Poisson;
  
  if (spaceTime)
  {
    v_Poisson   = poissonVF->testVarSpaceTime(S_V_POISSON, HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
    tau_Poisson = poissonVF->testVarSpaceTime(S_TAU_POISSON, tauSpace, HGRAD); // HDIV or HGRAD in space, HGRAD in time
  }
  else
  {
    v_Poisson   = poissonVF->testVar(S_V_POISSON, HGRAD);   // HGRAD in space
    tau_Poisson = poissonVF->testVar(S_TAU_POISSON, tauSpace); // HDIV or HGRAD in space
  }
  _poissonBF = BF::bf(poissonVF);
  
  if (_spaceDim==1)
  {
    // for spaceDim==1, the "normal" component is in the flux-ness of VE_hat (it's a plus or minus 1)
    _poissonBF->addTerm(VE, tau_Poisson->di(1));
    _poissonBF->addTerm( -Ex, tau_Poisson);
    _poissonBF->addTerm(-VE_hat, tau_Poisson);

    _poissonBF->addTerm(-Ex, v_Poisson->di(1));
    _poissonBF->addTerm(E_hat, v_Poisson);
  }
  else
  {
    _poissonBF->addTerm(VE, tau_Poisson->div());
    for (int i=0; i<_spaceDim; i++)
    {
      _poissonBF->addTerm(-E[i], tau_Poisson->spatialComponent(i+1)); // E dot tau; spatialComponent() takes a 1-based argument
      _poissonBF->addTerm(-VE_hat, tau_Poisson->spatialComponent(i+1) * n->spatialComponent(i+1) ); // tau dot n
      _poissonBF->addTerm(-E[i], v_Poisson->di(i+1));
    }
    
    _poissonBF->addTerm(E_hat, v_Poisson);
  }
  
  Epetra_CommPtr comm = Teuchos::null;
  const bool allowPruning = false;
  map<int,int> trialOrderEnhancements;
  map<int,int> testOrderEnhancements;
  trialOrderEnhancements[ E_hat->ID()] = 1;
  trialOrderEnhancements[VE_hat->ID()] = 1;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  MeshPtr poissonMesh = Teuchos::rcp( new Mesh(meshTopo, _poissonBF, H1Order, delta_k,
                                               trialOrderEnhancements, testOrderEnhancements, partitionPolicy,
                                               comm, allowPruning) );
  
  auto poissonSoln     = Solution::solution(_poissonBF, poissonMesh);
  auto poissonSolnPrev = Solution::solution(_poissonBF, poissonMesh); // will copy out after fixed point solve
  
  poissonSoln->setIP(_poissonBF->graphNorm());
  
  // nullify _configurationSoln, _configurationSolnPrev: if these are initialized, they refer to a previous mesh, not the poissonMesh we just created
  _configurationSoln     = Teuchos::null;
  _configurationSolnPrev = Teuchos::null;
  setConfigurationSolution(poissonSoln);
  setConfigurationSolutionPrevious(poissonSolnPrev);
  
  if (spaceTime)
  {
    setConfigurationSolutionPriorTime(Teuchos::null);
  }
  else
  {
    // it's not clear to me that we actually need this for Poisson
    auto poissonSolnPriorTime = Solution::solution(_poissonBF, poissonMesh);
    setConfigurationSolutionPriorTime(poissonSolnPriorTime);
  }
  
  auto poissonSolnWeakPtr     = Teuchos::rcp(poissonSoln.get(),     false);
  auto poissonSolnPrevWeakPtr = Teuchos::rcp(poissonSolnPrev.get(), false);
  
  std::vector<FunctionPtr> ESolnVector(_spaceDim);
  std::vector<FunctionPtr> ESolnPrevVector(_spaceDim);
  for (int i=0; i<_spaceDim; i++)
  {
    ESolnVector[i]     = Function::solution(E[i], poissonSolnWeakPtr);
    ESolnPrevVector[i] = Function::solution(E[i], poissonSolnPrevWeakPtr);
  }
  _ESoln     = Function::vectorize(ESolnVector);
  FunctionPtr E_solnPrev = Function::vectorize(ESolnPrevVector);
  
  _EDiff = _ESoln - E_solnPrev;
  
  {
    // to avoid an exception being thrown for an unset solution when _E_extruded->updateValues() is first called, we project zero onto the Poisson solution first
    std::map<int,FunctionPtr> zeroPoisson;
    FunctionPtr zero = Function::zero();
    for (int i=0; i<_spaceDim; i++)
    {
      zeroPoisson[this->E_fieldVar(i)->ID()] = zero;
    }
    zeroPoisson[this->En_hat()->ID()] = zero;
    poissonSoln->projectOntoMesh(zeroPoisson);
  }
}

std::string ConfigurationSpaceFormulationPoisson::name() const
{
  return "Poisson";
}

FunctionPtr ConfigurationSpaceFormulationPoisson::B_i(int zeroBasedComp)
{
  return Function::zero();
}

VarPtr ConfigurationSpaceFormulationPoisson::E_fieldVar(int zeroBasedComp)
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  if (zeroBasedComp < _spaceDim)
  {
    switch (zeroBasedComp)
    {
      case 0: return vf->trialVar(S_EX);
      case 1: return vf->trialVar(S_EY);
      case 2: return vf->trialVar(S_EZ);
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
}

VarPtr ConfigurationSpaceFormulationPoisson::E_hat(int zeroBasedComp)
{
  if ((_spaceDim > 1) || (zeroBasedComp != 0))
  {
    // Poisson does not define individual components of E_hat; it defines the trace of E dot n.
    return Teuchos::null;
  }
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_E_HAT);
}

VarPtr ConfigurationSpaceFormulationPoisson::En_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_E_HAT);
}

FunctionPtr ConfigurationSpaceFormulationPoisson::E_i(int zeroBasedComp)
{
  if (zeroBasedComp < _spaceDim)
  {
    return _E_extruded[zeroBasedComp];
  }
  else
  {
    return Function::zero();
  }
}

double ConfigurationSpaceFormulationPoisson::l2NormOfSolutionUpdate()
{
  const double E_L2 = _ESoln->l2norm(_configurationSoln->mesh());
  return _EDiff->l2norm(_configurationSoln->mesh()) / E_L2;
}

//! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
int ConfigurationSpaceFormulationPoisson::maxMomentNumber()
{
  return 0;
}

void ConfigurationSpaceFormulationPoisson::setMoments(Teuchos::RCP<SolutionMoments> fMoments, FunctionPtr n0)
{
  this->ConfigurationSpaceFormulation::setMoments(fMoments, n0);
  FunctionPtr velocityIntegral = _fMoments->solutionMomentFunction(0);
  
  auto vlasovMesh = fMoments->phaseSpaceMesh();
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  // set RHS
  auto poissonRHS = RHS::rhs();
  auto v_Poisson  = this->v();
  poissonRHS->addTerm( _q / _eps0 * (velocityIntegral - n0) * v_Poisson);
  _configurationSoln->setRHS(poissonRHS);
  
  // define extrusion of E to the phase-space mesh
  _E_extruded = std::vector<FunctionPtr>(_spaceDim);
  for (int i=0; i<_spaceDim; i++)
  {
    _E_extruded[i] = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSoln, this->E_fieldVar(i), vlasovMeshWeakPtr) );
  }
}

void ConfigurationSpaceFormulationPoisson::updateConfigurationSpaceFunctions()
{
  for (int i=0; i<_spaceDim; i++)
  {
    static_cast<PhysicalSolutionExtrusion*>(_E_extruded[i].get())->updateValues();
  }
}

VarPtr ConfigurationSpaceFormulationPoisson::tau()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->testVar(S_TAU_POISSON);
}

VarPtr ConfigurationSpaceFormulationPoisson::v()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->testVar(S_V_POISSON);
}

VarPtr ConfigurationSpaceFormulationPoisson::VE_fieldVar()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_VE);
}

VarPtr ConfigurationSpaceFormulationPoisson::VE_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_VE_HAT);
}
