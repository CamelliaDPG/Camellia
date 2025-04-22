//
//  IdealMHDFormulation.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 5/1/18.
//

#include "IdealMHDFormulation.hpp"

#include "BC.h"
#include "BF.h"
#include "ExpFunction.h"
#include "Functions.hpp"
#include "LagrangeConstraints.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RHS.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"
#include "VarFunction.h"

using namespace Camellia;
using namespace std;

const string IdealMHDFormulation::S_rho = "rho";
const string IdealMHDFormulation::S_m1  = "m1";
const string IdealMHDFormulation::S_m2  = "m2";
const string IdealMHDFormulation::S_m3  = "m3";
const string IdealMHDFormulation::S_E   = "E";
const string IdealMHDFormulation::S_B1 = "B1";
const string IdealMHDFormulation::S_B2 = "B2";
const string IdealMHDFormulation::S_B3 = "B3";

// primitive variable strings
const string IdealMHDFormulation::S_u1  = "u1";
const string IdealMHDFormulation::S_u2  = "u2";
const string IdealMHDFormulation::S_u3  = "u3";
const string IdealMHDFormulation::S_T   = "T";

const string IdealMHDFormulation::S_tc = "tc";
const string IdealMHDFormulation::S_tm1 = "tm1";
const string IdealMHDFormulation::S_tm2 = "tm2";
const string IdealMHDFormulation::S_tm3 = "tm3";
const string IdealMHDFormulation::S_te = "te";
const string IdealMHDFormulation::S_tB1 = "tB1";
const string IdealMHDFormulation::S_tB2 = "tB2";
const string IdealMHDFormulation::S_tB3 = "tB3";
const string IdealMHDFormulation::S_tGauss = "tGauss";

const string IdealMHDFormulation::S_vc = "vc";
const string IdealMHDFormulation::S_vm1  = "vm1";
const string IdealMHDFormulation::S_vm2  = "vm2";
const string IdealMHDFormulation::S_vm3  = "vm3";
const string IdealMHDFormulation::S_ve   = "ve";
const string IdealMHDFormulation::S_vB1  = "vB1";
const string IdealMHDFormulation::S_vB2  = "vB2";
const string IdealMHDFormulation::S_vB3  = "vB3";
const string IdealMHDFormulation::S_vGauss = "vGauss";

const string IdealMHDFormulation::S_m[3]    = {S_m1, S_m2, S_m3};
const string IdealMHDFormulation::S_B[3]    = {S_B1, S_B2, S_B3};
const string IdealMHDFormulation::S_u[3]    = {S_u1, S_u2, S_u3};
const string IdealMHDFormulation::S_tm[3]   = {S_tm1, S_tm2, S_tm3};
const string IdealMHDFormulation::S_tB[3]   = {S_tB1, S_tB2, S_tB3};
const string IdealMHDFormulation::S_vm[3]   = {S_vm1, S_vm2, S_vm3};
const string IdealMHDFormulation::S_vB[3]   = {S_vB1, S_vB2, S_vB3};

void IdealMHDFormulation::CHECK_VALID_COMPONENT(int i) const // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  const int trueSpaceDim = 3;
  if ((i > trueSpaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::spaceTimeFormulation(int spaceDim, MeshTopologyPtr meshTopo,
                                                                            int spatialPolyOrder, int temporalPolyOrder, int delta_k, double gamma)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", true);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("temporalPolyOrder", temporalPolyOrder);
  parameters.set("delta_k", delta_k);
  parameters.set("gamma", gamma);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::timeSteppingFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k, double gamma)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  parameters.set("gamma", gamma);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::timeSteppingPicardFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k, double gamma)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  parameters.set("linearization", "Picard"); // Newton is default
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  parameters.set("gamma", gamma);
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

Teuchos::RCP<IdealMHDFormulation> IdealMHDFormulation::timeSteppingPrimitiveVariableFormulation(int spaceDim, MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k, double gamma)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  parameters.set("gamma", gamma);
  
  parameters.set("variableSet", "primitive"); // as opposed to conservation variables, the default
  
  return Teuchos::rcp(new IdealMHDFormulation(meshTopo, parameters));
}

IdealMHDFormulation::IdealMHDFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;
  
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  _spaceDim = spaceDim;
  const int trueSpaceDim = 3; // as opposed to _spaceDim, which indicates the number of dimensions that can have non-zero derivatives (also the dimension of the mesh)

  std::string linearizationType = parameters.get<std::string>("linearization", "Newton"); // Picard or Newton
  bool picardIteration = (linearizationType == "Picard");
  _usePicardIteration = picardIteration;
  if (_usePicardIteration)
  {
    std::cout << "WARNING: Picard iteration is still in development, and is known not to work right now.  It is unclear whether the issue is algorithmic or a bug in the implementation.\n";
  }
  
  std::string variableSet = parameters.get<std::string>("variableSet", "conservation"); // primitive or conservation
  _usePrimitiveVariables = (variableSet == "primitive");
  
  _fc = ParameterFunction::parameterFunction(0.0);
  _fe = ParameterFunction::parameterFunction(0.0);
  _fm = vector<Teuchos::RCP<ParameterFunction> >(trueSpaceDim);
  _fB = vector<Teuchos::RCP<ParameterFunction> >(trueSpaceDim);
  _fc->setName("fc");
  _fe->setName("fe");
  for (int d=0; d<trueSpaceDim; d++)
  {
    _fm[d] = ParameterFunction::parameterFunction(0.0);
    _fB[d] = ParameterFunction::parameterFunction(0.0);
    ostringstream name_fm;
    name_fm << "fm" << d+1;
    _fm[d]->setName(name_fm.str());
    if ((d > 0) || (spaceDim > 1))
    {
      ostringstream name_fB;
      name_fB << "fB" << d+1;
      _fB[d]->setName(name_fB.str());
//      cout << "set name of fB[" << d << "] to " << name_fB.str() << endl;
    }
  }
  _gamma = parameters.get<double>("gamma",2.0); // 2.0 is what is used for Brio-Wu
  _Cv = parameters.get<double>("Cv",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");
  string normName = parameters.get<string>("norm", "Graph");
  
  // nonlinear parameters
  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double initialDt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);
  
  string problemName = parameters.get<string>("problemName", "");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");
  
  _useConformingTraces = useConformingTraces;
  _spatialPolyOrder = spatialPolyOrder;
  _temporalPolyOrder =temporalPolyOrder;
  _dt = ParameterFunction::parameterFunction(initialDt);
  _dt->setName("dt");
  _t = ParameterFunction::parameterFunction(0);
  _t0 = parameters.get<double>("t0",0);
  _delta_k = delta_k;
  
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;
  
  // field variables -- conservation form
  VarPtr rhoVar;
  vector<VarPtr> mVar(trueSpaceDim);
  VarPtr EVar;
  vector<VarPtr> BVar(trueSpaceDim); // In 1D, first component will be null
  
  // field variables -- primitive form
  vector<VarPtr> uVar(trueSpaceDim);
  VarPtr TVar;
  
  // trace variables
  VarPtr tc;
  vector<VarPtr> tm(trueSpaceDim);
  VarPtr te;
  vector<VarPtr> tB(trueSpaceDim); // first component null in 1D
  VarPtr tGauss; // only defined for spaceDim > 1
  
  // test variables
  VarPtr vc;
  vector<VarPtr> vm(trueSpaceDim);
  VarPtr ve;
  vector<VarPtr> vB(trueSpaceDim);
  VarPtr vGauss; // only defined for spaceDim > 1
  
  _vf = VarFactory::varFactory();
  
  rhoVar = _vf->fieldVar(S_rho);
  
  if (_usePrimitiveVariables)
  {
    for (int d=0; d<trueSpaceDim; d++)
    {
      uVar[d] = _vf->fieldVar(S_u[d]);
    }
    TVar = _vf->fieldVar(S_T);
  }
  else
  {
    for (int d=0; d<trueSpaceDim; d++)
    {
      mVar[d] = _vf->fieldVar(S_m[d]);
    }
    EVar = _vf->fieldVar(S_E);
  }
  
  if (_spaceDim > 1)
  {
    BVar[0] = _vf->fieldVar(S_B[0]);
  }
  for (int d=1; d<trueSpaceDim; d++)
  {
    BVar[d] = _vf->fieldVar(S_B[d]);
  }
  
  FunctionPtr B; // 3-component vector Function.  First component is concrete in 1D; others are abstract.
  vector<FunctionPtr> B_comps(trueSpaceDim);
  if (_spaceDim == 1)
  {
    // then the vars that remain as unknowns are By, Bz
    _Bx = ParameterFunction::parameterFunction(0.75); // (This is the value we use for Brio-Wu; can be over-ridden via setBx())
    _Bx->setName("Bx"); // used in displayString
    B_comps[0] = _Bx;
    B_comps[1] = VarFunction<double>::abstractFunction(BVar[1]);
    B_comps[2] = VarFunction<double>::abstractFunction(BVar[2]);
  }
  else
  {
    B_comps[0] = VarFunction<double>::abstractFunction(BVar[0]);
    B_comps[1] = VarFunction<double>::abstractFunction(BVar[1]);
    B_comps[2] = VarFunction<double>::abstractFunction(BVar[2]);
  }
  B = Function::vectorize(B_comps);
  
  // FunctionPtr n = Function::normal();
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  
  // TODO: add in here the definitions of the LinearTerms that the fluxes below trace.
  //       (See the exactSolution_tc(), etc. methods for an indication of the right thing here.)
  tc = _vf->fluxVar(S_tc);
  for (int d=0; d<trueSpaceDim; d++)
  {
    tm[d] = _vf->fluxVar(S_tm[d]);
  }
  te = _vf->fluxVar(S_te);
  
  vc = _vf->testVar(S_vc, HGRAD);
  for (int d=0; d<trueSpaceDim; d++)
  {
    vm[d] = _vf->testVar(S_vm[d], HGRAD);
  }
  ve = _vf->testVar(S_ve, HGRAD);
  
  int dStart = (_spaceDim == 1) ? 1 : 0; // for 1D, skip the "x" equation in B
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    vB[d] = _vf->testVar(S_vB[d], HGRAD);
    tB[d] = _vf->fluxVar(S_tB[d]);
  }
  
  // Gauss' Law, test and trial:  (Only need this for _spaceDim > 1)
  if (_spaceDim > 1)
  {
    vGauss = _vf->testVar(S_vGauss, HGRAD);
    tGauss = _vf->fluxVar(S_tGauss);
  }
  
  // now that we have all our variables defined, process any adjustments
  map<int,VarPtr> trialVars = _vf->trialVars();
  for (auto entry : trialVars)
  {
    VarPtr var = entry.second;
    string lookupString = var->name() + "-polyOrderAdjustment";
    int adjustment = parameters.get<int>(lookupString,0);
    if (adjustment != 0)
    {
      _trialVariablePolyOrderAdjustments[var->ID()] = adjustment;
    }
  }
  vector<int> H1Order;
  if (_spaceTime)
  {
    H1Order = {spatialPolyOrder+1,temporalPolyOrder+1}; // not dead certain that temporalPolyOrder+1 is the best choice; it depends on whether the indicated poly order means L^2 as it does in space, or whether it means H^1...
  }
  else
  {
    H1Order = {spatialPolyOrder+1};
  }
  
  BCPtr bc = BC::bc();
  _bf = Teuchos::rcp( new BF(_vf) );
  _steadyBF = Teuchos::rcp( new BF(_vf) );
  _rhs = RHS::rhs();
  
  MeshPtr mesh;
  if (savedSolutionAndMeshPrefix == "")
  {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
    mesh->setTensorTopologyMeansSpaceTime(useSpaceTime);
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
  }
  else
  {
#ifdef HAVE_EPETRAEXT_HDF5
    mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    mesh->setTensorTopologyMeansSpaceTime(useSpaceTime);
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
    _backgroundFlow->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    _solnIncrement->loadFromHDF5(savedSolutionAndMeshPrefix+"_increment.soln");
    _solnPrevTime->loadFromHDF5(savedSolutionAndMeshPrefix+"_prevtime.soln");
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "File I/O is not supported when Trilinos was built without EpetraExt_ENABLE_HDF5=ON");
#endif
  }
  
  std::string backgroundFlowIdentifierExponent = "";
  std::string previousTimeIdentifierExponent = "";
  if (_timeStepping)
  {
    backgroundFlowIdentifierExponent = "k";
    previousTimeIdentifierExponent = "k-1";
  }
  // to avoid needing a bunch of casts below, do a cast once here:
  FunctionPtr dt = (FunctionPtr)_dt;
  
  // avoid circular references when setting up bf object
  SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);
  SolutionPtr solnPrevTimeWeakReference   = Teuchos::rcp(_solnPrevTime.get(), false);
  
  bool weightFluxesByParity = true; // TODO: decide whether true or false is more appropriate (so far, we haven't used the DPG fluxes that are stored here...)
  for (auto varEntry : trialVars)
  {
    VarPtr var = varEntry.second;
    _backgroundFlowMap  [var->ID()] = Function::solution(var, backgroundFlowWeakReference, weightFluxesByParity, backgroundFlowIdentifierExponent);
    _solnPreviousTimeMap[var->ID()] = Function::solution(var, solnPrevTimeWeakReference,   weightFluxesByParity, previousTimeIdentifierExponent);
    _solnIncrementMap   [var->ID()] = Function::solution(var, _solnIncrement,  weightFluxesByParity, "");
  }
  
  // Let's try writing the flux terms in terms of Functions
  // We write these in terms of u, m, P*, B, E
  FunctionPtr rho, m, E, I, u, T, p;
  {
    int trueSpaceDim = 3; // as opposed to our mesh space dim, _spaceDim.
    auto Cv = this->Cv();
    auto gamma = this->gamma();
    
    rho = VarFunction<double>::abstractFunction(rhoVar);
    
    if (_usePrimitiveVariables)
    {
      vector<FunctionPtr> u_comps;
      for (int d=0; d<trueSpaceDim; d++)
      {
        auto u_comp = VarFunction<double>::abstractFunction(uVar[d]);
        u_comps.push_back(u_comp);
      }
      u   = Function::vectorize(u_comps);
      m   = rho * u;
    }
    else
    {
      vector<FunctionPtr> m_comps;
      for (int d=0; d<trueSpaceDim; d++)
      {
        auto m_comp = VarFunction<double>::abstractFunction(mVar[d]);
        m_comps.push_back(m_comp);
      }
      m   = Function::vectorize(m_comps);
      u   = m / rho;
    }
    
    auto mDotu = dot(trueSpaceDim, m, u);
    auto BDotB = dot(trueSpaceDim, B, B);
    
    if (_usePrimitiveVariables)
    {
      T   = VarFunction<double>::abstractFunction(TVar);
      E   = Cv * rho * T + 0.5 * mDotu + 0.5 * BDotB;
      double R = (gamma - 1.) * Cv;
      p   = R * rho * T;
    }
    else
    {
      E   = VarFunction<double>::abstractFunction(EVar);
      T   = (1.0/Cv) / rho  * (E - 0.5 * mDotu - 0.5 * BDotB);
      p   = (gamma - 1.0)   * (E - 0.5 * mDotu - 0.5 * BDotB);
    }
    
    _abstractTemperature = T;
    _abstractPressure = p;
    _abstractVelocity = u;
    _abstractMomentum = m;
    _abstractEnergy = E;
    _abstractDensity = rho;
    _abstractMagnetism = B;

    I   = identityMatrix<double>(trueSpaceDim);
    
    _massFlux     = m;
    if (!picardIteration)
    {
      
      // P* is p + 0.5 * (B * B)
      //    cout << "B: " << B->displayString() << endl;
      FunctionPtr BdotB   = dot(trueSpaceDim, B, B);
      FunctionPtr BouterB = outerProduct(trueSpaceDim, B, B);
      FunctionPtr Bdotu   = dot(trueSpaceDim,B,u);
      FunctionPtr p_star = p + 0.5 * BdotB;
      
      _momentumFlux = outerProduct(trueSpaceDim, u, m) + p_star * I - BouterB;
      _energyFlux   = (E + p_star) * u - B * Bdotu;
      _magneticFlux = outerProduct(trueSpaceDim, u, B) - outerProduct(trueSpaceDim, B, u);
    }
    else
    {
      // For Picard iteration, we'll want to use u in terms of background flow
      FunctionPtr uBackgroundFlow, BBackgroundFlow, rhoBackgroundFlow;
      {
        uBackgroundFlow = u->evaluateAt(_backgroundFlowMap);
        BBackgroundFlow = B->evaluateAt(_backgroundFlowMap);
        rhoBackgroundFlow = rho->evaluateAt(_backgroundFlowMap);
      }
      FunctionPtr u   = uBackgroundFlow;
      
      //    cout << "B: " << B->displayString() << endl;
      FunctionPtr BdotB   = dot(trueSpaceDim, BBackgroundFlow, B);
      
      // (gas) pressure is (gamma - 1) * (E - 0.5 * m * u - 0.5 * B * B)
      FunctionPtr p   = (gamma - 1.0) * (E - 0.5 * dot(trueSpaceDim,m,u) - 0.5 * BdotB);
      
      FunctionPtr BouterB = 0.5 * (outerProduct(trueSpaceDim, BBackgroundFlow, B) + outerProduct(trueSpaceDim, B, BBackgroundFlow)); // make symmetric
      FunctionPtr Bdotu   = 0.5 * (dot(trueSpaceDim,BBackgroundFlow,u) + dot(trueSpaceDim,BBackgroundFlow, m / rhoBackgroundFlow)); // make sorta symmetric
      
      // P* is p + 0.5 * (B * B)
      FunctionPtr p_star = p + 0.5 * BdotB;
      
      FunctionPtr I   = identityMatrix<double>(trueSpaceDim);
      
      _momentumFlux = outerProduct(trueSpaceDim, u, m) + p_star * I - BouterB;
      _energyFlux   = (E + p_star) * u - B * Bdotu;
      _magneticFlux = outerProduct(trueSpaceDim, u, B) - outerProduct(trueSpaceDim, B, u);
    }
    if (spaceDim == 2)
    {
      _gaussFlux    = Function::vectorize(B->spatialComponent(1), B->spatialComponent(2)); // in 2D, the d/dz part falls away by definition -- but VectorizedFunction gets upset if we have the Bz component there anyway...
    }
    else if (spaceDim == 3)
    {
      _gaussFlux    = B; // only used for spaceDim > 1
    }
    
    if (_spaceDim == 1)
    {
      // take the x component of each flux term: this is the 1D flux (we drop y and z derivative terms)
      int x_comp = 1;
      _massFlux     = _massFlux->spatialComponent(x_comp);
      _momentumFlux = _momentumFlux->spatialComponent(x_comp);
      _energyFlux   = _energyFlux->spatialComponent(x_comp);
      _magneticFlux = _magneticFlux->spatialComponent(x_comp); // the first component of this is zero; we don't test against this in 1D
    }
    else if (_spaceDim == 2)
    {
      // take x,y components of each flux: this is 2D flux (we drop z derivative terms)
      int x_comp = 1, y_comp = 2;
      _massFlux     = Function::vectorize(_massFlux->spatialComponent    (x_comp),  _massFlux->spatialComponent    (y_comp));
      _momentumFlux = Function::vectorize(_momentumFlux->spatialComponent(x_comp),  _momentumFlux->spatialComponent(y_comp));
      _energyFlux   = Function::vectorize(_energyFlux->spatialComponent  (x_comp),  _energyFlux->spatialComponent  (y_comp));
      _magneticFlux = Function::vectorize(_magneticFlux->spatialComponent(x_comp),  _magneticFlux->spatialComponent(y_comp));
    }
    
//    cout << "massFlux: " << _massFlux->displayString() << endl;
//    cout << "energyFlux: " << _energyFlux->displayString() << endl;
//    cout << "momentumFlux: " << _momentumFlux->displayString() << endl;
//    cout << "magneticFlux: " << _magneticFlux->displayString() << endl;
  }
  
  _fluxEquations[vc->ID()] = {vc,_massFlux,tc,rho,_fc};
  for (int d=0; d<trueSpaceDim; d++)
  {
    auto momentumColumn = (_spaceDim > 1) ? column(_spaceDim,_momentumFlux,d+1) : _momentumFlux->spatialComponent(d+1);
//    {
//      // A small tweak for 1D: cancel out the (B_x, B_x) component in the m_1 equation -- this is a constant.
//      // It's likely just fine without this tweak: the placement of fluxPrevious on the RHS should take care of it.
//      // Still, it's a slightly unusual thing to have a constant in the flux definition, and the fact that we are taking
//      // the divergence means that mathematically getting rid of the constant is equivalent.
//      if ((_spaceDim == 1) && (d==0))
//      {
//        momentumColumn = momentumColumn + B->x() * B->x();
//      }
//    }
    _fluxEquations[vm[d]->ID()] = {vm[d],momentumColumn,tm[d],m->spatialComponent(d+1),_fm[d]};
    
//    cout << "for variable " << mVar[d]->name() << ", flux is " << momentumColumn->displayString() << endl;
    if ((d > 0) || (_spaceDim != 1))
    {
      auto magneticColumn = (_spaceDim > 1) ? column(_spaceDim,_magneticFlux,d+1) : _magneticFlux->spatialComponent(d+1);
      _fluxEquations[vB[d]->ID()] = {vB[d],magneticColumn,tB[d],B->spatialComponent(d+1),_fB[d]};
    }
  }
  _fluxEquations[ve->ID()] = {ve,_energyFlux,te,E,_fe};
  if (_spaceDim > 1)
  {
    // enforce Gauss' Law
    FunctionPtr timeTerm = Teuchos::null;
    _fluxEquations[vGauss->ID()] = {vGauss,_gaussFlux,tGauss,timeTerm,Function::zero()};
  }
  
  for (auto eqnEntry : _fluxEquations)
  {
    auto eqn = eqnEntry.second;
    auto testVar  = eqn.testVar;
    auto timeTerm = eqn.timeTerm;
    auto flux     = eqn.flux;
    auto traceVar = eqn.traceVar;
    auto f_rhs    = eqn.f_rhs;
//    {
//      // DEBUGGING
//      cout << "testVar:  " << testVar->name() << endl;
//      if (timeTerm != Teuchos::null) cout << "timeTerm: " << timeTerm->name() << endl;
//      cout << "flux:     " << flux->displayString() << endl;
//      cout << "traceVar: " << traceVar->name() << endl;
//      cout << "f_rhs:    " << f_rhs->displayString() << endl;
//    }
    if (timeTerm != Teuchos::null) // time term is null for Gauss' Law
    {
      auto timeTermBackground   = timeTerm->evaluateAt(_backgroundFlowMap);
      auto timeTermPrevTime     = timeTerm->evaluateAt(_solnPreviousTimeMap);
      auto timeTermIncrement    = timeTerm->jacobian(_backgroundFlowMap);
      if (_spaceTime)
      {
        _bf ->addTerm(-timeTermIncrement,  testVar->dt());
        _rhs->addTerm(timeTermBackground * testVar->dt());
      }
      else if (_timeStepping)
      {
        _bf ->addTerm(  timeTermIncrement / dt, testVar);
        if (!_usePicardIteration)
        {
          _rhs->addTerm(- (timeTermBackground - timeTermPrevTime) / dt  * testVar);
        }
        else
        {
          _rhs->addTerm( timeTermPrevTime / dt  * testVar);
        }
      }
    }
    auto fluxPrevious = flux->evaluateAt(_backgroundFlowMap); // unused for Picard
    auto fluxJacobian = flux->jacobian(_backgroundFlowMap);   // should be just the flux (dropping Bx * Bx bit, which has zero Jacobian, in 1D)
    Camellia::EOperator gradOp = (_spaceDim > 1) ? OP_GRAD : OP_DX;
    
    fluxJacobian->collectTerms();
    _bf->      addTerm(-fluxJacobian, testVar->applyOp(gradOp)); // negative from integration by parts
    _steadyBF->addTerm(-fluxJacobian, testVar->applyOp(gradOp));
    if (!picardIteration) // Picard doesn't evaluate the previous on RHS
    {
      _rhs     ->addTerm(fluxPrevious * testVar->applyOp(gradOp));
    }
    
    _bf->      addTerm(traceVar, testVar);
    _steadyBF->addTerm(traceVar, testVar);
    _rhs     ->addTerm(f_rhs * testVar);
  }
  
  if (picardIteration)
  {
    // DEBUGGING
    cout << "bf: " << _bf->displayString() << endl;
    cout << "rhs: " << _rhs->linearTerm()->displayString() << endl;
  }
  
  vector<VarPtr> missingTestVars = _bf->missingTestVars();
  vector<VarPtr> missingTrialVars = _bf->missingTrialVars();
  if (missingTestVars.size() > 0)
    cout << "WARNING: in IdealMHDFormulation, missing test vars:\n";
  for (auto var : missingTestVars)
  {
    cout << var->displayString() << endl;
  }
  if (missingTrialVars.size() > 0)
    cout << "WARNING: in IdealMHDFormulation, missing trial vars:\n";
  for (auto var : missingTrialVars)
  {
    cout << var->displayString() << endl;
  }

  IPPtr ip = _bf->graphNorm();
  
  _solnIncrement->setRHS(_rhs);
  _solnIncrement->setIP(ip);
  
  mesh->registerSolution(_backgroundFlow);
  mesh->registerSolution(_solnIncrement);
  mesh->registerSolution(_solnPrevTime);
  
  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );
  
  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );
  
  // Set up Functions for L^2 norm computations
  auto fieldVars = _vf->fieldVars();
  _L2IncrementFunction = Function::zero();
  _L2SolutionFunction = Function::zero();
  for (auto fieldVar : fieldVars)
  {
    auto fieldSolution   =  _backgroundFlowMap[fieldVar->ID()];
    _L2SolutionFunction  = _L2SolutionFunction + fieldSolution * fieldSolution;
    auto fieldIncrement  = _solnIncrementMap[fieldVar->ID()];
    // for Picard, the real increment is actually the difference between _solnIncrement and _backgroundFlow
    if (picardIteration)
    {
      fieldIncrement       = fieldIncrement - fieldSolution;
    }
    _L2IncrementFunction = _L2IncrementFunction + fieldIncrement * fieldIncrement;
  }
  
  _solver = Solver::getDirectSolver();
  
  _nonlinearIterationCount = 0;
}

FunctionPtr IdealMHDFormulation::abstractDensity() const
{
  return _abstractDensity;
}

FunctionPtr IdealMHDFormulation::abstractEnergy() const
{
  return _abstractEnergy;
}

FunctionPtr IdealMHDFormulation::abstractMagnetism() const
{
  return _abstractMagnetism;
}

FunctionPtr IdealMHDFormulation::abstractMomentum() const
{
  return _abstractMomentum;
}

FunctionPtr IdealMHDFormulation::abstractPressure() const
{
  return _abstractPressure;
}

FunctionPtr IdealMHDFormulation::abstractTemperature() const
{
  return _abstractTemperature;
}

FunctionPtr IdealMHDFormulation::abstractVelocity() const
{
  return _abstractVelocity;
}

void IdealMHDFormulation::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr tc_exact)
{
  VarPtr tc = this->tc();
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
}

void IdealMHDFormulation::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr te_exact)
{
  VarPtr te = this->te();
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
}

void IdealMHDFormulation::addMagneticFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tB_exact = exactSolution_tB(rho, u, E, B, includeParity);
  int trueSpaceDim = 3;
  int dStart = (_spaceDim > 1) ? 0 : 1; // B_x is not a solution variable in 1D
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    VarPtr tB_i = this->tB(d+1);
    _solnIncrement->bc()->addDirichlet(tB_i, region, tB_exact[d]);
//    cout << "adding boundary condition tB" << d+1 << " = " << tB_exact[d]->displayString() << endl;
  }
}

void IdealMHDFormulation::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  VarPtr tc = this->tc();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tc_exact = this->exactSolution_tc(rho, u, E, B, includeParity);
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
//  cout << "adding boundary condition tc = " << tc_exact->displayString() << endl;
}

void IdealMHDFormulation::addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tm_exact = exactSolution_tm(rho, u, E, B, includeParity);
  int trueSpaceDim = 3;
  for (int d=0; d<trueSpaceDim; d++)
  {
    VarPtr tm_i = this->tm(d+1);
    _solnIncrement->bc()->addDirichlet(tm_i, region, tm_exact[d]);
//    cout << "adding boundary condition tm" << d+1 << " = " << tm_exact[d]->displayString() << endl;
  }
}

void IdealMHDFormulation::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  VarPtr te = this->te();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto te_exact = exactSolution_te(rho, u, E, B, includeParity);
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
//  cout << "adding boundary condition te = " << te_exact->displayString() << endl;
}

VarPtr IdealMHDFormulation::B(int i) const
{
  return _vf->trialVar(S_B[i-1]);
}

BFPtr IdealMHDFormulation::bf()
{
  return _bf;
}

double IdealMHDFormulation::Cv()
{
  return _Cv;
}

double IdealMHDFormulation::Cp()
{
  return _gamma*_Cv;
}

VarPtr IdealMHDFormulation::E() const
{
  return _vf->trialVar(S_E);
}

// ! For an exact solution (rho, u, E, B), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_fB(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _magneticFlux;
  
  vector<FunctionPtr> f(3);
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto fluxes = abstractFlux->evaluateAt(exactMap);
  
  int dStart = (_spaceDim == 1) ? 1 : 0; // for 1D, skip the "x" equation in B
  for (int i=dStart; i<3; i++)
  {
    // get the column (row in 1D) of the magnetism tensor
    auto flux = (_spaceDim > 1) ? column(_spaceDim,fluxes,i+1) : fluxes->spatialComponent(i+1);
    
    FunctionPtr timeTerm = B->spatialComponent(i+1); // the thing that gets differentiated in time
    f[i] = Function::zero();
    if (timeTerm->dt() != Teuchos::null)
    {
      f[i] = f[i] + timeTerm->dt();
    }
    
    Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    f[i] = f[i] + Function::op(flux, divOp);
  }
  return f;
}

// ! For an exact solution (rho, u, E, B), returns the corresponding forcing in the continuity equation
FunctionPtr IdealMHDFormulation::exactSolution_fc(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _massFlux;
  FunctionPtr timeTerm = rho; // the thing that gets differentiated in time
  
  FunctionPtr f = Function::zero();
  if (timeTerm->dt() != Teuchos::null)
  {
    f = f + timeTerm->dt();
  }
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto flux = abstractFlux->evaluateAt(exactMap);

  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f = f + Function::op(flux, divOp);
  return f;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the energy equation
FunctionPtr IdealMHDFormulation::exactSolution_fe(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _energyFlux;
  FunctionPtr timeTerm = E; // the thing that gets differentiated in time
  
  FunctionPtr f = Function::zero();
  if (timeTerm->dt() != Teuchos::null)
  {
    f = f + timeTerm->dt();
  }
  
  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto flux = abstractFlux->evaluateAt(exactMap);
  
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f = f + Function::op(flux, divOp);
  return f;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_fm(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  auto abstractFlux = _momentumFlux;
  
  vector<FunctionPtr> f(3);

  bool includeFluxParity = false; // DPG fluxes won't be used here
  map<int,FunctionPtr> exactMap = this->exactSolutionMap(rho, u, E, B, includeFluxParity);
  auto fluxes = abstractFlux->evaluateAt(exactMap);
    
  for (int i=0; i<3; i++)
  {
    // get the column (row in 1D) of the momentum tensor
    auto flux = (_spaceDim > 1) ? column(_spaceDim,fluxes,i+1) : fluxes->spatialComponent(i+1);
    
    FunctionPtr timeTerm = (rho * u)->spatialComponent(i+1); // the thing that gets differentiated in time
    f[i] = Function::zero();
    if (timeTerm->dt() != Teuchos::null)
    {
      f[i] = f[i] + timeTerm->dt();
    }
    
    Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    f[i] = f[i] + Function::op(flux, divOp);
  }
  return f;
}

std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_tB(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  const int trueSpaceDim = 3;
  std::vector<FunctionPtr> tB(trueSpaceDim);
  int dStart = (_spaceDim == 1) ? 1 : 0; // for 1D, skip the "x" equation in B
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    auto testVar = this->vB(d+1);
    tB[d] = exactSolutionFlux(testVar, rho, u, E, B, includeParity);
  }
  return tB;
}

FunctionPtr IdealMHDFormulation::exactSolution_tc(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  auto testVar = this->vc();
  return exactSolutionFlux(testVar, rho, u, E, B, includeParity);
}

FunctionPtr IdealMHDFormulation::exactSolution_te(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  auto testVar = this->ve();
  return exactSolutionFlux(testVar, rho, u, E, B, includeParity);
}

std::vector<FunctionPtr> IdealMHDFormulation::exactSolution_tm(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeParity)
{
  const int trueSpaceDim = 3;
  std::vector<FunctionPtr> tm(trueSpaceDim);
  for (int d=0; d<trueSpaceDim; d++)
  {
    auto testVar = this->vm(d+1);
    tm[d] = exactSolutionFlux(testVar, rho, u, E, B, includeParity);
  }
  return tm;
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionFieldMap(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(u->rank() != 1, std::invalid_argument, "u must be a vector-valued function");
  TEUCHOS_TEST_FOR_EXCEPTION(B->rank() != 1, std::invalid_argument, "B must be a vector-valued function");
  
  return exactSolutionFieldMapFromConservationVariables(rho, u*rho, E, B);
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionFieldMapFromConservationVariables(FunctionPtr rho, FunctionPtr m, FunctionPtr E, FunctionPtr B)
{
  TEUCHOS_TEST_FOR_EXCEPTION(m->rank() != 1, std::invalid_argument, "m must be a vector-valued function");
  TEUCHOS_TEST_FOR_EXCEPTION(B->rank() != 1, std::invalid_argument, "B must be a vector-valued function");
  
  map<int, FunctionPtr> exactMap;
  const int trueSpaceDim = 3;
  
  exactMap[this->rho()->ID()] = rho;
  
  if (_usePrimitiveVariables)
  {
    auto u = m / rho;
    auto mDotu = dot(trueSpaceDim, m, u);
    
    exactMap[this->T()->ID()] = (1.0/_Cv) / rho  * (E  - 0.5 * mDotu);
    for (int d=1; d<=trueSpaceDim; d++)
    {
      exactMap[this->u(d)->ID()] = u->spatialComponent(d);
      if ((d > 1) || (_spaceDim > 1))
      {
        exactMap[this->B(d)->ID()] = B->spatialComponent(d);
      }
    }
  }
  else
  {
    exactMap[this->E()->ID()] = E;
    for (int d=1; d<=trueSpaceDim; d++)
    {
      exactMap[this->m(d)->ID()] = m->spatialComponent(d);
      if ((d > 1) || (_spaceDim > 1))
      {
        exactMap[this->B(d)->ID()] = B->spatialComponent(d);
      }
    }
  }
  return exactMap;
}

FunctionPtr IdealMHDFormulation::exactSolutionFlux(VarPtr testVar, FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B,
                                                   bool includeParity)
{
  ScalarFluxEquation equation = _fluxEquations.find(testVar->ID())->second;
  auto abstractFlux = equation.flux;
  auto timeTerm = equation.timeTerm;
  
//  cout << "u: " << u->displayString() << endl;
  map<int,FunctionPtr> exactMap = exactSolutionFieldMap(rho, u, E, B);
//  {
//    // DEBUGGING
//    cout << "exactMap:\n";
//    for (auto entry : exactMap)
//    {
//      VarPtr var = _vf->trial(entry.first);
//      cout << var->name() << " -> " << entry.second->displayString() << endl;
//    }
//    cout << "abstract flux for test variable " << testVar->name() << ": " << abstractFlux->displayString() << endl;
//  }
  
  auto flux = abstractFlux->evaluateAt(exactMap);
  
//  cout << "Concrete flux: " << flux->displayString() << endl;
  
  auto n = TFunction<double>::normal();
  FunctionPtr flux_dot_n;
  if (_spaceDim == 1) //flux is a scalar, then
  {
    auto n_x = n->spatialComponent(1);
    flux_dot_n = flux * n_x;
  }
  else
  {
    flux_dot_n = dot(_spaceDim,flux,n);
  }
  auto exactFlux = flux_dot_n;
  if (_spaceTime)
  {
    FunctionPtr n_t = TFunction<double>::normalSpaceTime()->t();
    auto timeTermExact = timeTerm->evaluateAt(exactMap);
    exactFlux = exactFlux + timeTermExact * n_t;
  }
  if (includeParity)
  {
    exactFlux = exactFlux * Function::sideParity();
  }
  return exactFlux;
}

std::map<int, FunctionPtr> IdealMHDFormulation::exactSolutionMap(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B, bool includeFluxParity)
{
  map<int, FunctionPtr> exactMap;
  
  auto varE   = this->E();
  auto varRho = this->rho();
  
  exactMap[varRho->ID()] = rho;
  exactMap[varE->ID()]   = E;
  exactMap[this->tc()->ID()] = this->exactSolution_tc(rho, u, E, B, includeFluxParity);
  exactMap[this->te()->ID()] = this->exactSolution_te(rho, u, E, B, includeFluxParity);
  
  int trueSpaceDim = 3;
  auto tmExact = this->exactSolution_tm(rho, u, E, B, includeFluxParity);
  for (int d=0; d<trueSpaceDim; d++)
  {
    exactMap[this->m(d+1)->ID()]  = u->spatialComponent(d+1) * rho;
    exactMap[this->tm(d+1)->ID()] = tmExact[d];
  }
  
  int dStart = (_spaceDim > 1) ? 0 : 1;
  auto tBExact = this->exactSolution_tB(rho, u, E, B, includeFluxParity);
  for (int d=dStart; d<trueSpaceDim; d++)
  {
    exactMap[this->B(d+1)->ID()]  = B->spatialComponent(d+1);
    exactMap[this->tB(d+1)->ID()] = tBExact[d];
  }
  
  return exactMap;
}

double IdealMHDFormulation::gamma()
{
  return _gamma;
}

std::vector<GlobalIndexType> IdealMHDFormulation::getLocalInadmissibleCells()
{
  auto rho = _backgroundFlowMap.find(this->rho()->ID())->second;
  auto T   = _abstractTemperature->evaluateAt(_backgroundFlowMap);
  
  vector<FunctionPtr> positiveFunctions = {rho, T};
  vector<string> positiveFunctionNames = {"rho", "T"};
  
  // use a set to ensure uniqueness
  std::set<GlobalIndexType> localCellsSet;
  auto mesh = _backgroundFlow->mesh();
  
  for (int i=0; i<positiveFunctions.size(); i++)
  {
    auto f = positiveFunctions[i];
    auto fName = positiveFunctionNames[i];
    
    std::vector<GlobalIndexType> negativeCells = f->localCellsWithNegativeValues(mesh);
    localCellsSet.insert(negativeCells.begin(), negativeCells.end());
  }
  
  std::vector<GlobalIndexType> localInadmissibleCells(localCellsSet.begin(),localCellsSet.end());
  return localInadmissibleCells;
}

int IdealMHDFormulation::getSolveCode()
{
  return _solveCode;
}

FunctionPtr IdealMHDFormulation::getTimeStep()
{
  return _dt;
}

double IdealMHDFormulation::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double IdealMHDFormulation::L2NormSolutionIncrement()
{
  return _l2NormOfIncrement;
}

FunctionPtr IdealMHDFormulation::getMomentumFluxComponent(FunctionPtr momentumFlux, int i)
{
  // get the column (row in 1D) of the momentum tensor
  auto flux = (_spaceDim > 1) ? column(_spaceDim,momentumFlux,i+1) : momentumFlux->spatialComponent(i+1);
  return flux;
}

VarPtr IdealMHDFormulation::m(int i) const
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->trialVar(S_m[i-1]);
}

bool IdealMHDFormulation::newtonSolve(double newtonThreshold, int maxSteps, bool print)
{
  int stepNumber=0;
  double L2NormOfIncrement = 1e5;
  
  set<int> nonlinearVariables, linearVariables;
  
  if (_usePicardIteration)
  {
    // then all variables are linear variables, in terms of accumulation
    auto trialIDs = _vf->trialIDs();
    linearVariables.insert(trialIDs.begin(),trialIDs.end());
  }
  else
  {
    if (_usePrimitiveVariables)
    {
      nonlinearVariables = {{rho()->ID(), T()->ID()}};
      linearVariables = {{tc()->ID(), te()->ID()}};
      
      const int trueSpaceDim = 3;
      for (int d1=0; d1<trueSpaceDim; d1++)
      {
        nonlinearVariables.insert(u(d1+1)->ID());
        linearVariables.insert(tm(d1+1)->ID());
        if ((d1>0) || (_spaceDim > 1))
        {
          nonlinearVariables.insert(B(d1+1)->ID());
          linearVariables.insert(tB(d1+1)->ID());
        }
      }
    }
    else
    {
      nonlinearVariables = {{rho()->ID(), E()->ID()}};
      linearVariables = {{tc()->ID(), te()->ID()}};
      
      const int trueSpaceDim = 3;
      for (int d1=0; d1<trueSpaceDim; d1++)
      {
        nonlinearVariables.insert(m(d1+1)->ID());
        linearVariables.insert(tm(d1+1)->ID());
        if ((d1>0) || (_spaceDim > 1))
        {
          nonlinearVariables.insert(B(d1+1)->ID());
          linearVariables.insert(tB(d1+1)->ID());
        }
      }
    }
  }
  
  double alpha = 1.0;
  
  while ((L2NormOfIncrement >= newtonThreshold) && (stepNumber < maxSteps))
  {
    _solveCode = _solnIncrement->solve(_solver);
    double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
    L2NormOfIncrement = sqrt(l2_squared);
    
    if (!_usePicardIteration)
    {
      _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, nonlinearVariables, linearVariables);
    }
    else
    {
      // this will do a convex sum of background flow and soln increment in linearVariables.  nonlinearVariables is empty.
      _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, 1.0 - alpha, linearVariables, nonlinearVariables);
    }
    if (print)
    {
      cout << "Newton step " << stepNumber << ", L^2(increment) = " << L2NormOfIncrement << endl;
    }
    stepNumber++;
  }
  
  _l2NormOfIncrement = L2NormOfIncrement;
  if (_l2NormOfIncrement > newtonThreshold)
  {
    return false;
  }
  
  _nonlinearIterationCount += stepNumber;
  
  return true; // for now, we pretend that we always succeed....
}

double IdealMHDFormulation::R()
{
  return Cp()-Cv();
}

VarPtr IdealMHDFormulation::rho() const
{
  return _vf->trialVar(S_rho);
}

RHSPtr IdealMHDFormulation::rhs()
{
  return _rhs;
}

void IdealMHDFormulation::setBx(FunctionPtr Bx)
{
  if (_spaceDim != 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setBx() is only supported for 1D");
  }
  _Bx->setValue(Bx);
}

void IdealMHDFormulation::setForcing(FunctionPtr f_continuity, std::vector<FunctionPtr> f_momentum, FunctionPtr f_energy,
                                     std::vector<FunctionPtr> f_magnetic)
{
  const int trueSpaceDim = 3;
  TEUCHOS_TEST_FOR_EXCEPTION(f_momentum.size() != trueSpaceDim, std::invalid_argument, "f_momentum should have size equal to 3, the true spatial dimension");
  TEUCHOS_TEST_FOR_EXCEPTION(f_magnetic.size() != trueSpaceDim, std::invalid_argument, "f_magnetic should have size equal to 3, the true spatial dimension");
  _fc->setValue(f_continuity);
  _fe->setValue(f_energy);
  for (int d=0; d<trueSpaceDim; d++)
  {
    _fm[d]->setValue(f_momentum[d]);
    _fB[d]->setValue(f_magnetic[d]);
  }
}

void IdealMHDFormulation::setInitialCondition(FunctionPtr rho, FunctionPtr u, FunctionPtr E, FunctionPtr B)
{
  if (_spaceDim == 1)
  {
    // then Bx needs to be set (it's not a variable; just material data)
    this->setBx(B->spatialComponent(1));
  }
  
  auto initialState = this->exactSolutionFieldMap(rho, u, E, B);
  setInitialState(initialState);
    
  if (_spaceTime)
  {
    // for space-time, we set initial conditions as BCs for the mesh at time = 0
    auto initialTime = SpatialFilter::matchingT(0.0);
    addMassFluxCondition    (initialTime, rho, u, E, B);
    addMomentumFluxCondition(initialTime, rho, u, E, B);
    addEnergyFluxCondition  (initialTime, rho, u, E, B);
    addMagneticFluxCondition(initialTime, rho, u, E, B);
  }
}

void IdealMHDFormulation::setInitialState(const std::map<int, FunctionPtr> &initialState)
{
  _backgroundFlow->projectOntoMesh(initialState);
  if (!_spaceTime)
  {
    _solnPrevTime->projectOntoMesh(initialState);
  }
}

void IdealMHDFormulation::setMaxLineSearchSteps(int maxLineSearchSteps)
{
  _maxLineSearchSteps = maxLineSearchSteps;
}

// ! set current time step used for transient solve
void IdealMHDFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

double IdealMHDFormulation::solveAndAccumulate()
{
//  cout << "_solnIncrement->bf(): " << _solnIncrement->bf()->displayString() << endl;
  
  _solveCode = _solnIncrement->solve(_solver);
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  _l2NormOfIncrement = sqrt(l2_squared);
  
  int rank = _solnIncrement->mesh()->Comm()->getRank();
  
  set<int> nonlinearVariables, linearVariables;
  
  if (_usePicardIteration)
  {
    // then all variables are linear variables, in terms of accumulation
    auto trialIDs = _vf->trialIDs();
    linearVariables.insert(trialIDs.begin(),trialIDs.end());
  }
  else
  {
    if (_usePrimitiveVariables)
    {
      nonlinearVariables = {{rho()->ID(), T()->ID()}};
      linearVariables = {{tc()->ID(), te()->ID()}};
      
      const int trueSpaceDim = 3;
      for (int d1=0; d1<trueSpaceDim; d1++)
      {
        nonlinearVariables.insert(u(d1+1)->ID());
        linearVariables.insert(tm(d1+1)->ID());
        if ((d1>0) || (_spaceDim > 1))
        {
          nonlinearVariables.insert(B(d1+1)->ID());
          linearVariables.insert(tB(d1+1)->ID());
        }
      }
    }
    else
    {
      nonlinearVariables = {{rho()->ID(), E()->ID()}};
      linearVariables = {{tc()->ID(), te()->ID()}};
      
      const int trueSpaceDim = 3;
      for (int d1=0; d1<trueSpaceDim; d1++)
      {
        nonlinearVariables.insert(m(d1+1)->ID());
        linearVariables.insert(tm(d1+1)->ID());
        if ((d1>0) || (_spaceDim > 1))
        {
          nonlinearVariables.insert(B(d1+1)->ID());
          linearVariables.insert(tB(d1+1)->ID());
        }
      }
    }
  }
  
  double alpha = 1.0;
  bool admissibleSolutionFound = false;
  
  if (_usePicardIteration)
  {
    // for Picard, line search becomes a search for an alpha value such that
    //     alpha * _solnIncrement + (1-alpha) * _backgroundFlow
    // is admissible
    
    ParameterFunctionPtr alphaParameter = ParameterFunction::parameterFunction(alpha);
    FunctionPtr alphaFxn = alphaParameter;
    FunctionPtr oneMinusAlpha = 1.0 - alphaFxn;
    
    auto newSolnMap = _solnIncrementMap;
    for (auto entry : _solnIncrementMap)
    {
      newSolnMap[entry.first] = entry.second * alphaFxn + _backgroundFlowMap[entry.first] * oneMinusAlpha;
    }
    
    FunctionPtr rhoNew = _abstractDensity->evaluateAt(newSolnMap);
    FunctionPtr ENew   = _abstractEnergy->evaluateAt(newSolnMap);
    FunctionPtr TNew   = _abstractTemperature->evaluateAt(newSolnMap);
    
    vector<FunctionPtr> positiveFunctions = {rhoNew, TNew};
    vector<string> positiveFunctionNames = {"rho", "T"};
    double minDistanceFromZero = 0.0; // "positive" values should not get *too* small...
    int posEnrich = 5;
    
    // lambda for positivity checking
    auto isPositive = [&] () -> bool
    {
      for (int i=0; i<positiveFunctions.size(); i++)
      {
        auto f = positiveFunctions[i];
        auto fName = positiveFunctionNames[i];
        FunctionPtr f_smaller = f - minDistanceFromZero;
        //        bool isPositive = f_smaller->isPositive(_solnIncrement->mesh(),posEnrich); // does MPI communication
        auto mesh = _solnIncrement->mesh();
        double minValue = f->minimumValue(mesh, posEnrich);
        bool isPositive = (minValue - minDistanceFromZero) >= 0.0;
        if (!isPositive)
        {
          double minValue = f->minimumValue(mesh, posEnrich);
//          if (rank == 0)
//          {
//            cout << "During positivity check, " << fName << " check failed with minimum value of " << minValue << endl;
//          }
          return false;
        }
      }
      return true;
    };
    
    double lineSearchFactor = .5;
    int iter = 0;
    while (!isPositive() && iter < _maxLineSearchSteps)
    {
      alpha = alpha*lineSearchFactor;
      alphaParameter->setValue(alpha);
      iter++;
    }
    
    admissibleSolutionFound = isPositive();
  }
  else // standard Newton iteration (not Picard)
  {
    bool useFancyNewPositivityCheck = true; // it's pretty nice-looking code, but I'm concerned there might be a bug...
    if (useFancyNewPositivityCheck)
    {
      ParameterFunctionPtr alphaParameter = ParameterFunction::parameterFunction(alpha);
      alphaParameter->setName("alpha");
      
      map<int,FunctionPtr> updatedFieldMap;
      for (auto entry : _backgroundFlowMap)
      {
        auto uID = entry.first;
        auto uBackground = entry.second;
        FunctionPtr uIncr = _solnIncrementMap.find(uID)->second;
        FunctionPtr uUpdated = uBackground + FunctionPtr(alphaParameter) * uIncr;
        updatedFieldMap[uID] = uUpdated;
      }

      auto rhoUpdated = updatedFieldMap.find(this->rho()->ID())->second;
      auto TUpdated   = _abstractTemperature->evaluateAt(updatedFieldMap);
      
      auto PUpdated  = _abstractPressure->evaluateAt(updatedFieldMap);
      
      auto EUpdated  = _abstractEnergy->evaluateAt(updatedFieldMap);
      auto BUpdated  = _abstractMagnetism->evaluateAt(updatedFieldMap);
      
    //  cout << "TUpdated: " << TUpdated->displayString() << endl;
    //  cout << "rhoUpdated: " << rhoUpdated->displayString() << endl;
      
      // we may need to do something else to ensure positive changes in entropy;
      // if we just add ds to the list of positive functions, we stall on the first Newton step...
      vector<FunctionPtr> positiveFunctions = {rhoUpdated, TUpdated};
      vector<string> positiveFunctionNames = {"rho", "T"};
      double minDistanceFromZero = 0.0; // "positive" values should not get *too* small...
      int posEnrich = 5;
      
      // lambda for positivity checking
      auto isPositive = [&] () -> bool
      {
        for (int i=0; i<positiveFunctions.size(); i++)
        {
          auto f = positiveFunctions[i];
          auto fName = positiveFunctionNames[i];
          FunctionPtr f_smaller = f - minDistanceFromZero;
          //        bool isPositive = f_smaller->isPositive(_solnIncrement->mesh(),posEnrich); // does MPI communication
          auto mesh = _solnIncrement->mesh();
          double minValue = f->minimumValue(mesh, posEnrich);
          bool isPositive = (minValue - minDistanceFromZero) >= 0.0;
          if (!isPositive)
          {
            double minValue = f->minimumValue(mesh, posEnrich);
            double minPressureValue = PUpdated->minimumValue(mesh, posEnrich);
            double maxPressureValue = PUpdated->maximumValue(mesh, posEnrich);
            double minTValue = TUpdated->minimumValue(mesh, posEnrich);
            double maxTValue = TUpdated->maximumValue(mesh, posEnrich);
            double minRhoValue = rhoUpdated->minimumValue(mesh, posEnrich);
            double maxRhoValue = rhoUpdated->maximumValue(mesh, posEnrich);
            double minEValue   = EUpdated->minimumValue(mesh, posEnrich);
            double maxEValue   = EUpdated->maximumValue(mesh, posEnrich);
            double minBxValue  = BUpdated->spatialComponent(1)->minimumValue(mesh, posEnrich);
            double maxBxValue  = BUpdated->spatialComponent(1)->maximumValue(mesh, posEnrich);
            double minByValue  = BUpdated->spatialComponent(2)->minimumValue(mesh, posEnrich);
            double maxByValue  = BUpdated->spatialComponent(2)->maximumValue(mesh, posEnrich);
            double minBzValue  = BUpdated->spatialComponent(3)->minimumValue(mesh, posEnrich);
            double maxBzValue  = BUpdated->spatialComponent(3)->maximumValue(mesh, posEnrich);
            if (rank == 0)
            {
              cout << "During positivity check, " << fName << " check failed with minimum value of " << minValue << endl;
              cout << "Pressure value range: (" << minPressureValue << ", " << maxPressureValue << ")\n";
              cout << "T value range:        (" << minTValue << ", " << maxTValue << ")\n";
              cout << "rho value range:      (" << minRhoValue << ", " << maxRhoValue << ")\n";
              cout << "E  value range:       (" << minEValue   << ", " << maxEValue   << ")\n";
              cout << "Bx value range:       (" << minBxValue  << ", " << maxBxValue  << ")\n";
              cout << "By value range:       (" << minByValue  << ", " << maxByValue  << ")\n";
              cout << "Bz value range:       (" << minBzValue  << ", " << maxBzValue  << ")\n";
            }
            return false;
          }
        }
        return true;
      };
      
      bool useLineSearch = true;
      
      if (useLineSearch)
      {
        double lineSearchFactor = .5;
        int iter = 0;
        while (!isPositive() && iter < _maxLineSearchSteps)
        {
          alpha = alpha*lineSearchFactor;
          alphaParameter->setValue(alpha);
          iter++;
        }
      }
      if (isPositive())
      {
        admissibleSolutionFound = true;
      }
    }
    else // not the fancy new, but the well-worn old, positivity check
    {
      ParameterFunctionPtr alphaParameter = ParameterFunction::parameterFunction(alpha);
      
      FunctionPtr rhoPrevious  = Function::solution(rho(),_backgroundFlow);
      FunctionPtr rhoIncrement = Function::solution(rho(),_solnIncrement);
      FunctionPtr rhoUpdated   = rhoPrevious + FunctionPtr(alphaParameter) * rhoIncrement;
      
      FunctionPtr EPrevious    = Function::solution(E(),_backgroundFlow);
      FunctionPtr EIncrement   = Function::solution(E(),_solnIncrement);
      FunctionPtr EUpdated     = EPrevious + FunctionPtr(alphaParameter) * EIncrement;
      
      vector<FunctionPtr> mPrevious(_spaceDim), mIncrement(_spaceDim), mUpdated(_spaceDim);
      FunctionPtr mDotmPrevious = Function::zero();
      FunctionPtr mDotmUpdated = Function::zero();
      for (int d=0; d<_spaceDim; d++)
      {
        mPrevious[d] = Function::solution(m(d+1),_backgroundFlow);
        mIncrement[d] = Function::solution(m(d+1),_solnIncrement);
        mUpdated[d] = mPrevious[d] + FunctionPtr(alphaParameter) * mIncrement[d];
        mDotmPrevious = mDotmPrevious + mPrevious[d] * mPrevious[d];
        mDotmUpdated  = mDotmUpdated  + mUpdated [d] * mUpdated [d];
      }
      
      double Cv = this->Cv();
      FunctionPtr TUpdated  = (1.0/Cv) / rhoUpdated  * (EUpdated  - 0.5 * mDotmUpdated  / rhoUpdated);
      FunctionPtr TPrevious = (1.0/Cv) / rhoPrevious * (EPrevious - 0.5 * mDotmPrevious / rhoPrevious);
      
      // pointwise change in Entropy should also be positive
      // s2 - s1 = c_p ln (T2/T1) - R ln (p2/p1)
      
      FunctionPtr ds;
      {
        // pressure = rho * R * T
        double R  = this->R();
        double Cp = this->Cp();
        FunctionPtr pPrevious = R * rhoPrevious * TPrevious;
        FunctionPtr pUpdated  = R * rhoUpdated  * TUpdated;
        
        auto ln = [&] (FunctionPtr arg) -> FunctionPtr
        {
          return Teuchos::rcp(new Ln<double>(arg));
        };
        
        ds = Cp * ln(TUpdated / TPrevious) - R * ln(pUpdated / pPrevious);
      }
      
      // we may need to do something else to ensure positive changes in entropy;
      // if we just add ds to the list of positive functions, we stall on the first Newton step...
      vector<FunctionPtr> positiveFunctions = {rhoUpdated, TUpdated};
      vector<string> positiveFunctionNames = {"rho", "T"};
      double minDistanceFromZero = 0.0; // "positive" values should not get *too* small...
      int posEnrich = 5;
      
      // lambda for positivity checking
      auto isPositive = [&] () -> bool
      {
        for (int i=0; i<positiveFunctions.size(); i++)
        {
          auto f = positiveFunctions[i];
          auto fName = positiveFunctionNames[i];
          FunctionPtr f_smaller = f - minDistanceFromZero;
          //        bool isPositive = f_smaller->isPositive(_solnIncrement->mesh(),posEnrich); // does MPI communication
          auto mesh = _solnIncrement->mesh();
          double minValue = f->minimumValue(mesh, posEnrich);
          bool isPositive = (minValue - minDistanceFromZero) >= 0.0;
          if (!isPositive)
          {
            double minValue = f->minimumValue(mesh, posEnrich);
//            if (rank == 0)
//            {
//              cout << "During positivity check, " << fName << " check failed with minimum value of " << minValue << endl;
//            }
            return false;
          }
        }
        return true;
      };
      
  //    {
  //      // DEBUGGING: check that the positive functions are positive with a zero alpha weight
  //      double savedAlpha = alpha;
  //      alphaParameter->setValue(0.0);
  //      for (auto f : positiveFunctions)
  //      {
  //        auto mesh = _solnIncrement->mesh();
  //        double minValue = f->minimumValue(mesh, posEnrich);
  //        if (minValue < 0.0)
  //        {
  //          if (rank == 0)
  //          {
  //            cout << "ERROR: Prior to adding solution increment, 'positive' " << f->displayString() << " has minimum value of " << minValue << endl;
  //          }
  //        }
  //      }
  //      alphaParameter->setValue(savedAlpha);
  //    }
    
      bool useLineSearch = true;
      
      if (useLineSearch)
      {
        double lineSearchFactor = .5;
        int iter = 0;
        while (!isPositive() && iter < _maxLineSearchSteps)
        {
          alpha = alpha*lineSearchFactor;
          alphaParameter->setValue(alpha);
          iter++;
        }
      }
      if (isPositive())
      {
        admissibleSolutionFound = true;
      }
    }
  }
  
  if (admissibleSolutionFound)
  {
    if (!_usePicardIteration)
    {
      _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, nonlinearVariables, linearVariables);
    }
    else
    {
      // this will do a convex sum of background flow and soln increment in linearVariables.  nonlinearVariables is empty.
      _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, 1.0 - alpha, linearVariables, nonlinearVariables);
    }
    _nonlinearIterationCount++;
    return alpha;
  }
  else
  {
    return -1;
  }
}

// ! Returns the solution (at current time)
SolutionPtr IdealMHDFormulation::solution()
{
  return _backgroundFlow;
}

// ! Returns a map containing the current time step's solution data for each trial variable
const std::map<int, FunctionPtr> & IdealMHDFormulation::solutionFieldMap() const
{
  return _backgroundFlowMap;
}

bool IdealMHDFormulation::solutionIsAdmissible(double minDistanceFromZero, bool printOnFailure) const
{
  auto rho = _backgroundFlowMap.find(this->rho()->ID())->second;
  auto T   = _abstractTemperature->evaluateAt(_backgroundFlowMap);
  
  auto P  = _abstractPressure->evaluateAt(_backgroundFlowMap);
  
  auto E  = _abstractEnergy->evaluateAt(_backgroundFlowMap);
  auto B  = _abstractMagnetism->evaluateAt(_backgroundFlowMap);
  
  //  cout << "TUpdated: " << TUpdated->displayString() << endl;
  //  cout << "rhoUpdated: " << rhoUpdated->displayString() << endl;
  
  // we may need to do something else to ensure positive changes in entropy;
  // if we just add ds to the list of positive functions, we stall on the first Newton step...
  vector<FunctionPtr> positiveFunctions = {rho, T};
  vector<string> positiveFunctionNames = {"rho", "T"};
  int posEnrich = 0;
  
  for (int i=0; i<positiveFunctions.size(); i++)
  {
    auto f = positiveFunctions[i];
    auto fName = positiveFunctionNames[i];
    auto mesh = _solnIncrement->mesh();
    double minValue = f->minimumValue(mesh, posEnrich);
    bool isPositive = (minValue - minDistanceFromZero) >= 0.0;
    if (!isPositive)
    {
      double minValue = f->minimumValue(mesh, posEnrich);
      double minPressureValue = P->minimumValue(mesh, posEnrich);
      double maxPressureValue = P->maximumValue(mesh, posEnrich);
      double minTValue = T->minimumValue(mesh, posEnrich);
      double maxTValue = T->maximumValue(mesh, posEnrich);
      double minRhoValue = rho->minimumValue(mesh, posEnrich);
      double maxRhoValue = rho->maximumValue(mesh, posEnrich);
      double minEValue   = E->minimumValue(mesh, posEnrich);
      double maxEValue   = E->maximumValue(mesh, posEnrich);
      double minBxValue  = B->spatialComponent(1)->minimumValue(mesh, posEnrich);
      double maxBxValue  = B->spatialComponent(1)->maximumValue(mesh, posEnrich);
      double minByValue  = B->spatialComponent(2)->minimumValue(mesh, posEnrich);
      double maxByValue  = B->spatialComponent(2)->maximumValue(mesh, posEnrich);
      double minBzValue  = B->spatialComponent(3)->minimumValue(mesh, posEnrich);
      double maxBzValue  = B->spatialComponent(3)->maximumValue(mesh, posEnrich);
      if (printOnFailure)
      {
        cout << "During positivity check, " << fName << " check failed with minimum value of " << minValue << endl;
        cout << "Pressure value range: (" << minPressureValue << ", " << maxPressureValue << ")\n";
        cout << "T value range:        (" << minTValue << ", " << maxTValue << ")\n";
        cout << "rho value range:      (" << minRhoValue << ", " << maxRhoValue << ")\n";
        cout << "E  value range:       (" << minEValue   << ", " << maxEValue   << ")\n";
        cout << "Bx value range:       (" << minBxValue  << ", " << maxBxValue  << ")\n";
        cout << "By value range:       (" << minByValue  << ", " << maxByValue  << ")\n";
        cout << "Bz value range:       (" << minBzValue  << ", " << maxBzValue  << ")\n";
      }
      
      return false;
    }
  }
  return true;
}

SolutionPtr IdealMHDFormulation::solutionIncrement()
{
  return _solnIncrement;
}

const std::map<int, FunctionPtr> & IdealMHDFormulation::solutionIncrementFieldMap() const
{
  return _solnIncrementMap;
}


// ! Returns the solution (at previous time)
SolutionPtr IdealMHDFormulation::solutionPreviousTimeStep()
{
  return _solnPrevTime;
}

const std::map<int, FunctionPtr> & IdealMHDFormulation::solutionPreviousTimeStepFieldMap() const
{
  return _solnPreviousTimeMap;
}

BFPtr IdealMHDFormulation::steadyBF()
{
  return _steadyBF;
}

VarPtr IdealMHDFormulation::u(int i) const
{
  return _vf->trialVar(S_u[i-1]);
}

VarPtr IdealMHDFormulation::T() const
{
  return _vf->trialVar(S_T);
}

VarPtr IdealMHDFormulation::tB(int i) const
{
  CHECK_VALID_COMPONENT(i);
  return _vf->trialVar(S_tB[i-1]);
}

VarPtr IdealMHDFormulation::tc() const
{
  return _vf->trialVar(S_tc);
}

VarPtr IdealMHDFormulation::te() const
{
  return _vf->trialVar(S_te);
}

VarPtr IdealMHDFormulation::tm(int i) const
{
  CHECK_VALID_COMPONENT(i);
  return _vf->trialVar(S_tm[i-1]);
}

VarPtr IdealMHDFormulation::vB(int i) const
{
  CHECK_VALID_COMPONENT(i);
  return _vf->testVar(S_vB[i-1]);
}

VarPtr IdealMHDFormulation::vc() const
{
  return _vf->testVar(S_vc);
}

VarPtr IdealMHDFormulation::vm(int i) const
{
  CHECK_VALID_COMPONENT(i);
  return _vf->testVar(S_vm[i-1]);
}

VarPtr IdealMHDFormulation::ve() const
{
  return _vf->testVar(S_ve);
}
