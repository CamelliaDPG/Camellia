//
//  ConfigurationSpaceFormulationMaxwell.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#include "ConfigurationSpaceFormulationMaxwell.h"
#include "PhysicalSolutionExtrusion.hpp"

using namespace Camellia;

// fields:
const string ConfigurationSpaceFormulationMaxwell::S_BX = "B_x";
const string ConfigurationSpaceFormulationMaxwell::S_BY = "B_y";
const string ConfigurationSpaceFormulationMaxwell::S_BZ = "B_z";
const string ConfigurationSpaceFormulationMaxwell::S_EX = "E_x";
const string ConfigurationSpaceFormulationMaxwell::S_EY = "E_y";
const string ConfigurationSpaceFormulationMaxwell::S_EZ = "E_z";
const string ConfigurationSpaceFormulationMaxwell::S_VE = "V_E";

// traces:
const string ConfigurationSpaceFormulationMaxwell::S_BX_HAT = "\\widehat{B}_x";
const string ConfigurationSpaceFormulationMaxwell::S_BY_HAT = "\\widehat{B}_y";
const string ConfigurationSpaceFormulationMaxwell::S_BZ_HAT = "\\widehat{B}_z";
const string ConfigurationSpaceFormulationMaxwell::S_EX_HAT = "\\widehat{E}_x";
const string ConfigurationSpaceFormulationMaxwell::S_EY_HAT = "\\widehat{E}_y";
const string ConfigurationSpaceFormulationMaxwell::S_EZ_HAT = "\\widehat{E}_z";
const string ConfigurationSpaceFormulationMaxwell::S_VE_HAT = "\\widehat{V}_E";

// tests:
const string ConfigurationSpaceFormulationMaxwell::S_W1 = "w1";
const string ConfigurationSpaceFormulationMaxwell::S_W2 = "w2";
const string ConfigurationSpaceFormulationMaxwell::S_TAU1 = "\\tau_{1}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU2 = "\\tau_{2}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU3 = "\\tau_{3}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU1Y = "\\tau_{1y}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU1Z = "\\tau_{1z}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU2X = "\\tau_{2x}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU2Y = "\\tau_{2y}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU2Z = "\\tau_{2z}";
const string ConfigurationSpaceFormulationMaxwell::S_TAU3X = "\\tau_{3x}";

ConfigurationSpaceFormulationMaxwell::ConfigurationSpaceFormulationMaxwell(const std::vector<double> &q, const double eps0, const double mu0)
:
ConfigurationSpaceFormulation(),
_q(q),
_eps0(eps0),
_mu0(mu0)
{}

ConfigurationSpaceFormulationMaxwell::ConfigurationSpaceFormulationMaxwell(const double &q, const double eps0, const double mu0)
:
ConfigurationSpaceFormulationMaxwell(std::vector<double>{q},eps0,mu0)
{}

void ConfigurationSpaceFormulationMaxwell::initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k)
{
  this->_configurationMeshTopo = meshTopo;
  
  _spaceDim = meshTopo->getDimension() - 1; // assumption is meshTopo is space-time
  
  Space tauSpace    = (_spaceDim > 1) ? HDIV : HGRAD;
  Space E_hat_space = L2; // non-conforming
  Space B_hat_space = L2; // non-conforming
  Space ESpace      = L2;
  Space BSpace      = L2;

  // fields
  VarPtr Ex, Ey, Ez; // electric field
  VarPtr Bx, By, Bz; // magnetic field
  VarPtr VE; // potential

  // traces
  VarPtr Bx_hat, By_hat, Bz_hat;
  VarPtr Ex_hat, Ey_hat, Ez_hat;
  VarPtr VE_hat;

  // tests
  VarPtr w1; // test var for Gauss' Law
  VarPtr w2; // test var for Gauss' Law for magnetism (unused in 1D)
  
  // 1D-specific tau variables:
  VarPtr tau1y, tau1z;
  VarPtr tau2x, tau2y, tau2z;
  VarPtr tau3x;
  
  // 3D tau variables (2D, too?)
  VarPtr tau1, tau2; // spatially H(curl)
  VarPtr tau3;       // spatially H(div)
  
  VarFactoryPtr maxwellVF = VarFactory::varFactory();
  Ex = maxwellVF->fieldVarSpaceTime(S_EX, ESpace, L2); // L2 in time, too
  Ey = maxwellVF->fieldVarSpaceTime(S_EY, ESpace, L2);
  Ez = maxwellVF->fieldVarSpaceTime(S_EZ, ESpace, L2);
  
  if (_spaceDim > 1) Bx = maxwellVF->fieldVarSpaceTime(S_BX, BSpace, L2); // L2 in time, too
  By = maxwellVF->fieldVarSpaceTime(S_BY, BSpace, L2);
  Bz = maxwellVF->fieldVarSpaceTime(S_BZ, BSpace, L2);
  
  VE = maxwellVF->fieldVarSpaceTime(S_VE, L2,     L2); // L2 in time, too
  
  Ex_hat = maxwellVF->traceVarSpaceTime(S_EX_HAT, Ex, E_hat_space, L2); // L2 in space and time
  Ey_hat = maxwellVF->traceVarSpaceTime(S_EY_HAT, Ey, E_hat_space, L2); // L2 in space and time
  Ez_hat = maxwellVF->traceVarSpaceTime(S_EZ_HAT, Ez, E_hat_space, L2); // L2 in space and time
  
  if (_spaceDim > 1) Bx_hat = maxwellVF->traceVarSpaceTime(S_BX_HAT, Bx, B_hat_space, L2); // L2 in space and time
  By_hat = maxwellVF->traceVarSpaceTime(S_BY_HAT, By, B_hat_space, L2); // L2 in space and time
  Bz_hat = maxwellVF->traceVarSpaceTime(S_BZ_HAT, Bz, B_hat_space, L2); // L2 in space and time
  
  VE_hat = maxwellVF->traceVarSpaceOnly(S_VE_HAT, VE, L2);
  
  TFunctionPtr<double> n  = TFunction<double>::normal();
  TFunctionPtr<double> nx = n->spatialComponent(1);

  w1      = maxwellVF->testVarSpaceTime(S_W1,    HGRAD,    L2);   // HGRAD in space, L2 in time
  tau1y   = maxwellVF->testVarSpaceTime(S_TAU1Y,    L2, HGRAD);   // L2 in space,    HGRAD in time
  tau1z   = maxwellVF->testVarSpaceTime(S_TAU1Z, HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
  tau2x   = maxwellVF->testVarSpaceTime(S_TAU2X,    L2, HGRAD);   // L2 in space,    HGRAD in time
  tau2y   = maxwellVF->testVarSpaceTime(S_TAU2Y, HGRAD, HGRAD);   // L2 in space,    HGRAD in time
  tau2z   = maxwellVF->testVarSpaceTime(S_TAU2Z, HGRAD, HGRAD);   // L2 in space,    HGRAD in time
  tau3x   = maxwellVF->testVarSpaceTime(S_TAU3X, HGRAD,    L2);   // HGRAD in space, L2 in time
  
  _maxwellBF = BF::bf(maxwellVF);
  
  std::vector<VarPtr> E {Ex,Ey,Ez};
  std::vector<VarPtr> E_hat {Ex_hat,Ey_hat,Ez_hat};
  std::vector<VarPtr> B_hat {Bx_hat,By_hat,Bz_hat}; // first entry is null if _spaceDim == 1
  
  const int X_INDEX = 1;
  const int T_INDEX = _spaceDim + 1;
  
  FunctionPtr nt = n->spatialComponent(T_INDEX);
  if (_spaceDim==1)
  {
    // w1 equation
    _maxwellBF->addTerm( Ex_hat * nx, w1);
    _maxwellBF->addTerm( -Ex * nx, w1->di(X_INDEX));
    // RHS: (rho / eps0, w1)
    
    // tau1y
    _maxwellBF->addTerm(-By,     tau1y->di(T_INDEX));
    _maxwellBF->addTerm( By_hat, tau1y * nt);
    
    // tau1z
    _maxwellBF->addTerm(-Ey,     tau1z->di(X_INDEX));
    _maxwellBF->addTerm( Ey_hat, tau1z * nx);
    _maxwellBF->addTerm(-Bz,     tau1z->di(T_INDEX));
    _maxwellBF->addTerm( Bz_hat, tau1z * nt);
    
    // tau2x
    _maxwellBF->addTerm(  _mu0 * _eps0 * Ex,     tau2x->di(T_INDEX));
    _maxwellBF->addTerm(- _mu0 * _eps0 * Ex_hat, tau2x * nt);
    // RHS: (Jx, tau2x)
    
    // tau2y
    _maxwellBF->addTerm(                Bz,     tau2y->di(X_INDEX));
    _maxwellBF->addTerm(                Bz_hat, tau2y * nx);
    _maxwellBF->addTerm( _mu0 * _eps0 * Ey,     tau2y->di(T_INDEX));
    _maxwellBF->addTerm(-_mu0 * _eps0 * Ey_hat, tau2y * nt);
    // RHS: (Jy, tau2y)
    
    // tau2z
    _maxwellBF->addTerm(               -By,     tau2z->di(X_INDEX));
    _maxwellBF->addTerm(                By_hat, tau2z * nx);
    _maxwellBF->addTerm( _mu0 * _eps0 * Ez,     tau2z->di(T_INDEX));
    _maxwellBF->addTerm(-_mu0 * _eps0 * Ez_hat, tau2z * nt);
    // RHS: (Jz, tau2z)
    
    // tau3x
    _maxwellBF->addTerm( Ex,     tau3x);
    _maxwellBF->addTerm( VE_hat, tau3x * nx);
    _maxwellBF->addTerm(-VE,     tau3x->di(X_INDEX));
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim > 1 cases not yet implemented");
  }
  
  Teuchos_CommPtr comm = Teuchos::null;
  const bool allowPruning = false;
  map<int,int> trialOrderEnhancements;
  map<int,int> testOrderEnhancements;
  for (int i=0; i<3; i++)
  {
    trialOrderEnhancements[ E_hat[i]->ID()] = 1;
    const bool Bx_in_1D = (_spaceDim == 1) && (i == 0);
    if (! Bx_in_1D) trialOrderEnhancements[ B_hat[i]->ID()] = 1;
  }
  
  trialOrderEnhancements[VE_hat->ID()] = 1;
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
    ESolnVector[i]     = Function::solution(E[i], maxwellSolnWeakPtr);
    ESolnPrevVector[i] = Function::solution(E[i], maxwellSolnPrevWeakPtr);
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

std::string ConfigurationSpaceFormulationMaxwell::name() const
{
  return "Maxwell";
}

Teuchos::RCP<ConfigurationSpaceFormulation> ConfigurationSpaceFormulationMaxwell::newInstance() const
{
  auto instance = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell(_q, _eps0, _mu0));
  instance->setExtraAcceleration(_extraAcceleration);
  instance->setExportViz(_exportViz, _vizOutputDir);
  return instance;
}

FunctionPtr ConfigurationSpaceFormulationMaxwell::B_i(int zeroBasedComp)
{
  return Function::zero();
}

VarPtr ConfigurationSpaceFormulationMaxwell::B_fieldVar(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  if (zeroBasedComp < _spaceDim)
  {
    switch (zeroBasedComp)
    {
      case 0: return vf->trialVar(S_BX);
      case 1: return vf->trialVar(S_BY);
      case 2: return vf->trialVar(S_BZ);
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
}

VarPtr ConfigurationSpaceFormulationMaxwell::B_hat(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return vf->trialVar(S_BX_HAT);
    case 1: return vf->trialVar(S_BY_HAT);
    case 2: return vf->trialVar(S_BZ_HAT);
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell::E_fieldVar(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
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

VarPtr ConfigurationSpaceFormulationMaxwell::E_hat(int zeroBasedComp)
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  switch (zeroBasedComp)
  {
    case 0: return vf->trialVar(S_EX_HAT);
    case 1: return vf->trialVar(S_EY_HAT);
    case 2: return vf->trialVar(S_EZ_HAT);
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid component index");
  }
}

FunctionPtr ConfigurationSpaceFormulationMaxwell::E_i(int zeroBasedComp)
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

FunctionPtr ConfigurationSpaceFormulationMaxwell::E_i_priorTime(int zeroBasedComp)
{
  if (zeroBasedComp < _spaceDim)
  {
    return _E_priorTime_extruded[zeroBasedComp];
  }
  else
  {
    return Function::zero();
  }
}

//! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
int ConfigurationSpaceFormulationMaxwell::maxMomentNumber()
{
  return 0;
}

void ConfigurationSpaceFormulationMaxwell::setMoments(std::vector< Teuchos::RCP<SolutionMoments> > fMoments, FunctionPtr n0)
{
  this->ConfigurationSpaceFormulation::setMoments(fMoments,n0);
  FunctionPtr velocityIntegral;
  FunctionPtr firstMoment_x, firstMoment_y, firstMoment_z;
  int numSpecies = _q.size();
  TEUCHOS_TEST_FOR_EXCEPTION(numSpecies != int(fMoments.size()), std::invalid_argument, "The number of moments objects provided to setMoments() should match the number of q values at construction of ConfigurationSpaceFormulationMaxwell");
  
  auto vlasovMesh = fMoments[0]->phaseSpaceMesh();
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  // initialize RHS
  auto rhs = RHS::rhs();
  auto w1  = this->w1();
  
  FunctionPtr rho = Function::zero();
  FunctionPtr Jx  = Function::zero();
  FunctionPtr Jy  = Function::zero();
  FunctionPtr Jz  = Function::zero();
  
  for (int i=0; i<numSpecies; i++)
  {
    auto moments_i = _fMoments[i];
    int numVelocityDims = moments_i->numVelocityDims();
    const double & q_i = _q[i];
    switch (numVelocityDims)
    {
      case 1:
      {
        velocityIntegral = moments_i->solutionMomentFunction(0);
        firstMoment_x    = moments_i->solutionMomentFunction(1);
        firstMoment_y    = Function::zero();
        firstMoment_z    = Function::zero();
      }
        break;
      case 2:
      {
        velocityIntegral = moments_i->solutionMomentFunction(0,0);
        firstMoment_x    = moments_i->solutionMomentFunction(1,0);
        firstMoment_y    = moments_i->solutionMomentFunction(0,1);
        firstMoment_z    = Function::zero();
      }
        break;
      case 3:
      {
        velocityIntegral = moments_i->solutionMomentFunction(0,0,0);
        firstMoment_x    = moments_i->solutionMomentFunction(1,0,0);
        firstMoment_y    = moments_i->solutionMomentFunction(0,1,0);
        firstMoment_z    = moments_i->solutionMomentFunction(0,0,1);
      }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported number of velocity dims");
    }
    rho = rho + q_i / _eps0 * (velocityIntegral - n0);
    Jx  = Jx  + q_i * firstMoment_x;
    Jy  = Jy  + q_i * firstMoment_y;
    Jz  = Jz  + q_i * firstMoment_z;
  }
  
  rhs->addTerm( rho * w1);
  
  // TODO: think about how we should handle the 2D case; probably there's an exception thrown due to tau2->z()…
  VarPtr tau2x = this->tau2x();
  VarPtr tau2y = this->tau2y();
  VarPtr tau2z = this->tau2z();
  
  rhs->addTerm(_mu0 * Jx * tau2x);
  rhs->addTerm(_mu0 * Jy * tau2y);
  rhs->addTerm(_mu0 * Jz * tau2z);
  
  _configurationSoln->setRHS(rhs);
  
  // define extrusion of E, B to the phase-space mesh
  _E_extruded = std::vector<FunctionPtr>(_spaceDim);
  _B_extruded = std::vector<FunctionPtr>(_spaceDim);
  for (int i=0; i<_spaceDim; i++)
  {
    _E_extruded[i] = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSoln, this->E_fieldVar(i), vlasovMeshWeakPtr) );
    const bool Bx_in_1D = (_spaceDim == 1) && (i == 0);
    if (!Bx_in_1D)
      _B_extruded[i] = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSoln, this->B_fieldVar(i), vlasovMeshWeakPtr) );
    else
      _B_extruded[0] = Function::zero();
  }
  
  if (_configurationSolnPriorTime != Teuchos::null)
  {
    // define extrusion of prior-time E, B to the phase-space mesh
    _E_priorTime_extruded = std::vector<FunctionPtr>(_spaceDim);
    _B_priorTime_extruded = std::vector<FunctionPtr>(_spaceDim);
    
    for (int i=0; i<_spaceDim; i++)
    {
      _E_priorTime_extruded[i] = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSolnPriorTime, this->E_fieldVar(i), vlasovMeshWeakPtr) );
      const bool Bx_in_1D = (_spaceDim == 1) && (i == 0);
      if (!Bx_in_1D)
        _B_priorTime_extruded[i] = Teuchos::rcp( new PhysicalSolutionExtrusion(_configurationSolnPriorTime, this->B_fieldVar(i), vlasovMeshWeakPtr) );
      else
        _B_priorTime_extruded[0] = Function::zero();
    }
  }
}

void ConfigurationSpaceFormulationMaxwell::updateConfigurationSpaceFunctions()
{
  for (int i=0; i<_spaceDim; i++)
  {
    static_cast<PhysicalSolutionExtrusion*>(_E_extruded[i].get())->updateValues();
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau1()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU1);
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau1y()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU1Y);
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau1z()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU1Z);
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau2()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU2);
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau2x()
{
  if (_spaceDim == 1)
  {
    VarFactoryPtr vf = _maxwellBF->varFactory();
    return vf->testVar(S_TAU2X);
  }
  else
  {
    return tau2()->x();
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau2y()
{
  if (_spaceDim == 1)
  {
    VarFactoryPtr vf = _maxwellBF->varFactory();
    return vf->testVar(S_TAU2Y);
  }
  else
  {
    return tau2()->y();
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau2z()
{
  if (_spaceDim == 1)
  {
    VarFactoryPtr vf = _maxwellBF->varFactory();
    return vf->testVar(S_TAU2Z);
  }
  else
  {
    return tau2()->z();
  }
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau3()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU3);
}

VarPtr ConfigurationSpaceFormulationMaxwell::tau3x()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_TAU3X);
}

VarPtr ConfigurationSpaceFormulationMaxwell::VE_fieldVar()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->trialVar(S_VE);
}

VarPtr ConfigurationSpaceFormulationMaxwell::VE_hat()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->trialVar(S_VE_HAT);
}

VarPtr ConfigurationSpaceFormulationMaxwell::w1()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_W1);
}

VarPtr ConfigurationSpaceFormulationMaxwell::w2()
{
  VarFactoryPtr vf = _maxwellBF->varFactory();
  return vf->testVar(S_W2);
}
