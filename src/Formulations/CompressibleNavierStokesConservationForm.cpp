//
//  CompressibleNavierStokesConservationForm.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/2/18.
//

#include "CompressibleNavierStokesConservationForm.hpp"

#include "BC.h"
#include "BF.h"
#include "CompressibleNavierStokesProblem.hpp"
#include "ExpFunction.h"
#include "LagrangeConstraints.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RHS.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"

using namespace Camellia;
using namespace std;

const string CompressibleNavierStokesConservationForm::S_rho = "rho";
const string CompressibleNavierStokesConservationForm::S_m1  = "m1";
const string CompressibleNavierStokesConservationForm::S_m2  = "m2";
const string CompressibleNavierStokesConservationForm::S_m3  = "m3";
const string CompressibleNavierStokesConservationForm::S_E   = "E";
const string CompressibleNavierStokesConservationForm::S_D11 = "D11";
const string CompressibleNavierStokesConservationForm::S_D12 = "D12";
const string CompressibleNavierStokesConservationForm::S_D13 = "D13";
const string CompressibleNavierStokesConservationForm::S_D21 = "D21";
const string CompressibleNavierStokesConservationForm::S_D22 = "D22";
const string CompressibleNavierStokesConservationForm::S_D23 = "D23";
const string CompressibleNavierStokesConservationForm::S_D31 = "D31";
const string CompressibleNavierStokesConservationForm::S_D32 = "D32";
const string CompressibleNavierStokesConservationForm::S_D33 = "D33";
const string CompressibleNavierStokesConservationForm::S_q1 = "q1";
const string CompressibleNavierStokesConservationForm::S_q2 = "q2";
const string CompressibleNavierStokesConservationForm::S_q3 = "q3";

const string CompressibleNavierStokesConservationForm::S_tc = "tc";
const string CompressibleNavierStokesConservationForm::S_tm1 = "tm1";
const string CompressibleNavierStokesConservationForm::S_tm2 = "tm2";
const string CompressibleNavierStokesConservationForm::S_tm3 = "tm3";
const string CompressibleNavierStokesConservationForm::S_te = "te";
const string CompressibleNavierStokesConservationForm::S_u1_hat = "u1_hat";
const string CompressibleNavierStokesConservationForm::S_u2_hat = "u2_hat";
const string CompressibleNavierStokesConservationForm::S_u3_hat = "u3_hat";
const string CompressibleNavierStokesConservationForm::S_T_hat = "T_hat";

const string CompressibleNavierStokesConservationForm::S_vc = "vc";
const string CompressibleNavierStokesConservationForm::S_vm1  = "vm1";
const string CompressibleNavierStokesConservationForm::S_vm2  = "vm2";
const string CompressibleNavierStokesConservationForm::S_vm3  = "vm3";
const string CompressibleNavierStokesConservationForm::S_ve   = "ve";
const string CompressibleNavierStokesConservationForm::S_S1 = "S1";
const string CompressibleNavierStokesConservationForm::S_S2 = "S2";
const string CompressibleNavierStokesConservationForm::S_S3 = "S3";
const string CompressibleNavierStokesConservationForm::S_tau = "tau";


const string CompressibleNavierStokesConservationForm::S_m[3]    = {S_m1, S_m2, S_m3};
const string CompressibleNavierStokesConservationForm::S_q[3]    = {S_q1, S_q2, S_q3};
const string CompressibleNavierStokesConservationForm::S_D[3][3] = {{S_D11, S_D12, S_D13},
                                                                    {S_D21, S_D22, S_D23},
                                                                    {S_D31, S_D32, S_D33}};
const string CompressibleNavierStokesConservationForm::S_S[3]    = {S_S1, S_S2, S_S3};
const string CompressibleNavierStokesConservationForm::S_tm[3]   = {S_tm1, S_tm2, S_tm3};
const string CompressibleNavierStokesConservationForm::S_u_hat[3]= {S_u1_hat, S_u2_hat, S_u3_hat};
const string CompressibleNavierStokesConservationForm::S_vm[3]   = {S_vm1, S_vm2, S_vm3};

void CompressibleNavierStokesConservationForm::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

Teuchos::RCP<CompressibleNavierStokesConservationForm> CompressibleNavierStokesConservationForm::steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                                                           MeshTopologyPtr meshTopo, int polyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",1.0 / Re);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("spatialPolyOrder", polyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new CompressibleNavierStokesConservationForm(meshTopo, parameters));
}

Teuchos::RCP<CompressibleNavierStokesConservationForm> CompressibleNavierStokesConservationForm::timeSteppingFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                                                                 MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",1.0 / Re);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new CompressibleNavierStokesConservationForm(meshTopo, parameters));
}

Teuchos::RCP<CompressibleNavierStokesConservationForm> CompressibleNavierStokesConservationForm::steadyEulerFormulation(int spaceDim, bool useConformingTraces,
                                                                                                                        MeshTopologyPtr meshTopo,
                                                                                                                        int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu", 0.0);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new CompressibleNavierStokesConservationForm(meshTopo, parameters));
}

Teuchos::RCP<CompressibleNavierStokesConservationForm> CompressibleNavierStokesConservationForm::timeSteppingEulerFormulation(int spaceDim, bool useConformingTraces,
                                                                                    MeshTopologyPtr meshTopo,
                                                                                    int spatialPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu", 0.0);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  
  parameters.set("t0",0.0);
  
  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("delta_k", delta_k);
  
  return Teuchos::rcp(new CompressibleNavierStokesConservationForm(meshTopo, parameters));
}

CompressibleNavierStokesConservationForm::CompressibleNavierStokesConservationForm(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;
  
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  _spaceDim = spaceDim;
  _fc = ParameterFunction::parameterFunction(0.0);
  _fe = ParameterFunction::parameterFunction(0.0);
  _fm = vector<Teuchos::RCP<ParameterFunction> >(_spaceDim, ParameterFunction::parameterFunction(0.0));
  _fc->setName("fc");
  _fe->setName("fe");
  for (int d=0; d<spaceDim; d++)
  {
    ostringstream name;
    name << "fm" << d+1;
    _fm[d]->setName(name.str());
  }
  _mu = parameters.get<double>("mu",1.0);
  _gamma = parameters.get<double>("gamma",1.4);
  _Pr = parameters.get<double>("Pr",0.713);
  _Cv = parameters.get<double>("Cv",1.0);
  _pureEulerMode = (_mu == 0.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");
  string normName = parameters.get<string>("norm", "Graph");
  
  // nonlinear parameters
  bool neglectFluxesOnRHS = true; // if ever we want to support a false value here, we will need to add terms corresponding to traces/fluxes to the RHS.
  
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
  _neglectFluxesOnRHS = neglectFluxesOnRHS;
  _delta_k = delta_k;
  
  _muParamFunc = ParameterFunction::parameterFunction(_mu);
  _muSqrtParamFunc = ParameterFunction::parameterFunction(sqrt(_mu));
  _muFunc = _muParamFunc;
  _muSqrtFunc = _muSqrtParamFunc;
  
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;
  
  // TEUCHOS_TEST_FOR_EXCEPTION(_timeStepping, std::invalid_argument, "Time stepping not supported");
  
  // field variables
  VarPtr rho;
  vector<VarPtr> m(_spaceDim);
  VarPtr E;
  vector<vector<VarPtr>> D(_spaceDim,vector<VarPtr>(_spaceDim));
  vector<VarPtr> q(_spaceDim);
  
  // trace variables
  VarPtr tc;
  vector<VarPtr> tm(_spaceDim);
  VarPtr te;
  vector<VarPtr> u_hat(_spaceDim);
  VarPtr T_hat;
  
  // test variables
  VarPtr vc;
  vector<VarPtr> vm(_spaceDim);
  VarPtr ve;
  vector<VarPtr> S(_spaceDim);
  VarPtr tau;
  
  _vf = VarFactory::varFactory();
  
  if (!_spaceTime)
  {
    rho = _vf->fieldVar(S_rho);
    
    for (int d=0; d<spaceDim; d++)
    {
      m[d] = _vf->fieldVar(S_m[d]);
    }
    
    if (!_pureEulerMode)
    {
      for (int d=0; d<spaceDim; d++)
      {
        q[d] = _vf->fieldVar(S_q[d]);
      }
      for (int d1=0; d1<spaceDim; d1++)
      {
        for (int d2=0; d2<spaceDim; d2++)
        {
          D[d1][d2] = _vf->fieldVar(S_D[d1][d2]);
        }
      }
    }
    
    E = _vf->fieldVar(S_E);
  }
  else
  {
    rho = _vf->fieldVarSpaceTime(S_rho);
    
    for (int d=0; d<spaceDim; d++)
    {
      m[d] = _vf->fieldVarSpaceTime(S_m[d]);
    }
    
    if (!_pureEulerMode)
    {
      for (int d=0; d<spaceDim; d++)
      {
        q[d] = _vf->fieldVarSpaceTime(S_q[d]);
      }
      for (int d1=0; d1<spaceDim; d1++)
      {
        for (int d2=0; d2<spaceDim; d2++)
        {
          D[d1][d2] = _vf->fieldVarSpaceTime(S_D[d1][d2]);
        }
      }
    }
    
    E = _vf->fieldVarSpaceTime(S_E);
  }
  
  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  
  // u_hat, T_hat equations are not defined in Euler formulation...
  if (!_pureEulerMode)
  {
    // since u_hat, T_hat are only defined on spatial interfaces (if we're doing space-time), we only need to define a function space in space
    Space uHatSpace = useConformingTraces ? HGRAD : L2;
    Space THatSpace = useConformingTraces ? HGRAD : L2;
    if (! _spaceTime)
    {
      for (int d=0; d<spaceDim; d++)
      {
        // unfortunately, we don't have a way right now to say what u_hat is the trace of,
        // because it is not linear in the field variables: it is m[d] / rho...
        // We could resolve this by using a Function that depends on ParameterFunctions,
        // and a map in place of the LinearTerm below, but there's some more work to be done to support this
        // in the places that make use of the "termTraced" argument below, as well as adding support to Var
        // TODO: add this support.  (Main use case right now is geometric multi-grid preconditioning.)
        u_hat[d] = _vf->traceVar(S_u_hat[d], uHatSpace);
      }
      // same story with the T hat trace as with u_hat:
      // T is a nonlinear function of the field variables, T = trace((1/rho) * (E - m * m / rho)).
      T_hat = _vf->traceVar(S_T_hat, THatSpace);
    }
    else // _spaceTime == TRUE
    {
      // since T_hat is only defined on spatial interfaces, we only need to define a function space in space
      Space uHatSpace = useConformingTraces ? HGRAD : L2;
      for (int d=0; d<spaceDim; d++)
      {
        // see note above about the term traced...
        u_hat[d] = _vf->traceVarSpaceOnly(S_u_hat[d], uHatSpace);
      }
      // same story with the T hat trace as with u_hat:
      // T is a nonlinear function of the field variables, T = trace((1/rho) * (E - m * m / rho)).
      T_hat = _vf->traceVarSpaceOnly(S_T_hat, THatSpace);
    }
  }
  
  // FunctionPtr n = Function::normal();
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  
  // TODO: add in here the definitions of the LinearTerms that the fluxes below trace.
  //       (See the exactSolution_tc(), etc. methods for an indication of the right thing here.)
  tc = _vf->fluxVar(S_tc);
  for (int d=0; d<spaceDim; d++)
  {
    tm[d] = _vf->fluxVar(S_tm[d]);
  }
  te = _vf->fluxVar(S_te);
  
  vc = _vf->testVar(S_vc, HGRAD);
  for (int d=0; d<spaceDim; d++)
  {
    vm[d] = _vf->testVar(S_vm[d], HGRAD);
  }
  ve = _vf->testVar(S_ve, HGRAD);
  
  if (! _pureEulerMode)
  {
    for (int d=0; d<spaceDim; d++)
    {
      Space S_space = (spaceDim == 1) ? HGRAD : HDIV;
      S[d] = _vf->testVar(S_S[d], S_space);
    }
    
    if (spaceDim == 1)  tau = _vf->testVar(S_tau, HGRAD);
    else                tau = _vf->testVar(S_tau, HDIV);
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
  
  _bf = Teuchos::rcp( new BF(_vf) );
  _steadyBF = Teuchos::rcp( new BF(_vf) );
  BFPtr transientBF = Teuchos::rcp( new BF(_vf) ); // we'll end up writing _bf = _steadyBF + transientBF, basically
  _rhs = RHS::rhs();
  
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
  
  MeshPtr mesh;
  Teuchos::RCP<CompressibleNavierStokesProblem> problem;
  if (problemName != "")
  {
    problem = CompressibleNavierStokesProblem::namedProblem(problemName);
  }
  if (savedSolutionAndMeshPrefix == "")
  {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
  }
  else
  {
    mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    _backgroundFlow = Solution::solution(_bf, mesh, bc);
    _solnIncrement = Solution::solution(_bf, mesh, bc);
    _solnPrevTime = Solution::solution(_bf, mesh, bc);
    _backgroundFlow->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    _solnIncrement->loadFromHDF5(savedSolutionAndMeshPrefix+"_increment.soln");
    _solnPrevTime->loadFromHDF5(savedSolutionAndMeshPrefix+"_prevtime.soln");
  }
  
  // avoid circular references when setting up bf object
  SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);
  SolutionPtr solnPrevTimeWeakReference   = Teuchos::rcp(_solnPrevTime.get(), false);
  
  // Previous solution values
  std::string backgroundFlowIdentifierExponent = "";
  std::string previousTimeIdentifierExponent = "";
  if (_timeStepping)
  {
    backgroundFlowIdentifierExponent = "k";
    previousTimeIdentifierExponent = "k-1";
  }
  FunctionPtr rho_prev = Function::solution(rho, backgroundFlowWeakReference, backgroundFlowIdentifierExponent);
  FunctionPtr rho_prev_time = Function::solution(rho, solnPrevTimeWeakReference, previousTimeIdentifierExponent);
  FunctionPtr E_prev       = Function::solution(E, backgroundFlowWeakReference, backgroundFlowIdentifierExponent);
  FunctionPtr E_prev_time  = Function::solution(E, solnPrevTimeWeakReference, previousTimeIdentifierExponent);
  
  vector<FunctionPtr> q_prev(spaceDim);
  vector<FunctionPtr> m_prev(spaceDim);
  vector<FunctionPtr> m_prev_time(spaceDim);
  vector<vector<FunctionPtr>> D_prev(spaceDim,vector<FunctionPtr>(spaceDim));
  
  FunctionPtr rho_prev_squared = rho_prev * rho_prev;
  FunctionPtr m_prev_dot_m_prev = Function::zero();
  
  for (int d1=0; d1<spaceDim; d1++)
  {
    m_prev[d1]      = Function::solution(m[d1], backgroundFlowWeakReference, backgroundFlowIdentifierExponent);
    m_prev_time[d1] = Function::solution(m[d1], solnPrevTimeWeakReference, previousTimeIdentifierExponent);
    if (!_pureEulerMode)
    {
      q_prev[d1]      = Function::solution(q[d1], backgroundFlowWeakReference, backgroundFlowIdentifierExponent);
      for (int d2=0; d2<spaceDim; d2++)
      {
        D_prev[d1][d2] = Function::solution(D[d1][d2], backgroundFlowWeakReference, backgroundFlowIdentifierExponent);
      }
    }
    m_prev_dot_m_prev = m_prev_dot_m_prev + m_prev[d1] * m_prev[d1];
  }
  
  double Cp    = this->Cp();
  double Pr    = this->Pr();
  double Cv    = this->Cv();
  double gamma = this->gamma();
  
  if (!_pureEulerMode) // S and tau equations (heat flux, scaled velocity gradient) vanish for Euler
  {
    // S terms:
    Camellia::EOperator S_divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    auto n_S = (_spaceDim > 1) ? n_x : n_x->x();
    for (int d=0; d<spaceDim; d++)
    {
      _steadyBF->addTerm(m[d]/rho_prev - m_prev[d]/rho_prev_squared * rho, S[d]->applyOp(S_divOp)); // D_i = mu() * grad u_i
      _rhs->addTerm(-m_prev[d] / rho_prev * S[d]->applyOp(S_divOp)); // D_i = mu() * grad u_i
      for (int d2=0; d2<spaceDim; d2++)
      {
        VarPtr S_d2 = (_spaceDim > 1) ? S[d]->spatialComponent(d2+1) : S[d];
        _steadyBF->addTerm (1./_muFunc * D[d][d2],        S_d2);
        _rhs->addTerm(-1./_muFunc * D_prev[d][d2] * S_d2);
      }
      _steadyBF->addTerm(-u_hat[d], S[d] * n_S);
    }
    
    // tau terms:
    auto n_tau = n_S;
    FunctionPtr qWeight = Pr / (Cp * _muFunc); // q is defined such that q * qWeight = - grad T
    Camellia::EOperator tauDivOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    
    _steadyBF->addTerm (-1.0 / rho_prev * E + E_prev / rho_prev_squared * rho, (1.0 / Cv) * tau->applyOp(tauDivOp));
    _rhs->addTerm(E_prev / rho_prev                                    * (1.0 / Cv) * tau->applyOp(tauDivOp));
    
    _steadyBF->addTerm(T_hat,     tau * n_tau);
    
    _steadyBF->addTerm( - m_prev_dot_m_prev / (rho_prev_squared * rho_prev) * rho, (1.0 / Cv) * tau->applyOp(tauDivOp));
    _rhs->addTerm( - m_prev_dot_m_prev / (2. * rho_prev_squared)           * (1.0 / Cv) * tau->applyOp(tauDivOp));
    for (int d=0; d<spaceDim; d++)
    {
      VarPtr tau_d = (_spaceDim > 1) ? tau->spatialComponent(d+1) : tau;
      _steadyBF->addTerm ( qWeight * q[d],       tau_d);
      _rhs->addTerm(-qWeight * q_prev[d] * tau_d);
      
      _steadyBF->addTerm(   m_prev[d] / rho_prev_squared * m[d], (1.0 / Cv) * tau->applyOp(tauDivOp));
    }
  }

  // to avoid needing a bunch of casts below, do a cast once here:
  FunctionPtr dt = (FunctionPtr)_dt;
  
  // vc terms:
  if (_spaceTime)
  {
    transientBF->addTerm (-rho, vc->dt());
    _rhs->addTerm( rho_prev   * vc->dt());
  }
  else if (_timeStepping)
  {
    transientBF->addTerm (rho/dt,                     vc);
    _rhs->addTerm(- (rho_prev - rho_prev_time) / dt * vc);
  }
  
  for (int d=0; d<spaceDim; d++)
  {
    _steadyBF->addTerm( -m[d], vc->di(d+1));
    _rhs->addTerm( m_prev[d] * vc->di(d+1));
  }
  _steadyBF->addTerm(tc, vc);
  _rhs->addTerm(FunctionPtr(_fc) * vc);
  
  // D is the mu-weighted gradient of u
  double D_traceWeight = -2./3.; // In Truman's code, this is hard-coded to -2/3 for 1D, 3D, and -2/2 for 2D.  This value arises from Stokes' hypothesis, and I think he probably was implementing a variant of this for 2D.  I'm going with what I think is the more standard choice of using the same value regardless of spatial dimension.
  LinearTermPtr D_trace;
  FunctionPtr D_trace_prev = Function::zero();
  if (!_pureEulerMode) // D is not defined in Euler formulation
  {
    for (int d=0; d<spaceDim; d++)
    {
      D_trace = D_trace + D[d][d];
      D_trace_prev = D_trace_prev +  D_prev[d][d];
    }
  }
  
  LinearTermPtr m_prev_dot_m = Teuchos::rcp(new LinearTerm);
  for (int d=0; d<spaceDim; d++)
  {
    m_prev_dot_m = m_prev_dot_m + m_prev[d] * m[d];
  }
  
  // vm
  for (int d1=0; d1<spaceDim; d1++)
  {
    if (_spaceTime)
    {
      transientBF->addTerm( -m[d1], vm[d1]->dt() );
      _rhs->addTerm( m_prev[d1]   * vm[d1]->dt() );
    }
    if (_timeStepping)
    {
      transientBF->addTerm( m[d1],                         vm[d1] / dt);
      _rhs->addTerm(-(m_prev[d1] - m_prev_time[d1]) / dt * vm[d1] );
    }
    
    { // terms of the form (Scalar * I, grad vm) -- simplifies to (Scalar, trace(grad vm))
      _steadyBF->addTerm(-(gamma - 1.) * E,       vm[d1]->di(d1+1));
      _rhs->addTerm((gamma - 1.) * E_prev * vm[d1]->di(d1+1));
      
      _steadyBF->addTerm( -(gamma - 1.) * m_prev_dot_m_prev / (2 * rho_prev_squared) * rho, vm[d1]->di(d1+1));
      _rhs->addTerm(-(gamma - 1.) * m_prev_dot_m_prev / (2 * rho_prev)              * vm[d1]->di(d1+1));
      
      _steadyBF->addTerm( (gamma - 1.) * m_prev_dot_m / rho_prev, vm[d1]->di(d1+1));
      
      if (!_pureEulerMode) // no D for Euler
      {
        _steadyBF->addTerm(  D_traceWeight * D_trace,       vm[d1]->di(d1+1));
        _rhs->addTerm(-D_traceWeight * D_trace_prev * vm[d1]->di(d1+1));
      }
    }
    for (int d2=0; d2<spaceDim; d2++)
    {
      _steadyBF->addTerm(-(m_prev[d2]/rho_prev * m[d1] + m_prev[d1]/rho_prev * m[d2]), vm[d1]->di(d2+1));
      _steadyBF->addTerm( (m_prev[d1] * m_prev[d2] / rho_prev_squared) * rho,          vm[d1]->di(d2+1));
      _rhs->addTerm( m_prev[d1] * m_prev[d2] / rho_prev                        * vm[d1]->di(d2+1));
      
      if (!_pureEulerMode) // no D for Euler
      {
        _steadyBF->addTerm(   D[d1][d2]      + D[d2][d1],        vm[d1]->di(d2+1));
        _rhs->addTerm(-(D_prev[d1][d2] + D_prev[d2][d1]) * vm[d1]->di(d2+1));
      }
    }
    _steadyBF->addTerm(tm[d1], vm[d1]);
    _rhs->addTerm(FunctionPtr(_fm[d1]) * vm[d1]);
  }
  
  // ve:
  if (_spaceTime)
  {
    transientBF->addTerm(-E, ve->dt());
    _rhs->addTerm(E_prev   * ve->dt());
  }
  if (_timeStepping)
  {
    transientBF->addTerm(E,                      ve / dt);
    _rhs->addTerm(-(E_prev - E_prev_time) / dt * ve);
  }
  for (int d1=0; d1<spaceDim; d1++)
  {
    _steadyBF->addTerm(- gamma * (E_prev / rho_prev) * m[d1],                    ve->di(d1+1));
    _steadyBF->addTerm(  gamma * (E_prev * m_prev[d1] / rho_prev_squared) * rho, ve->di(d1+1));
    _steadyBF->addTerm(- gamma * (m_prev[d1] / rho_prev) * E,                    ve->di(d1+1));
    _rhs->addTerm( gamma * E_prev / rho_prev * m_prev[d1]                * ve->di(d1+1));
    
    _steadyBF->addTerm( -(gamma - 1.)     * m_prev_dot_m_prev * m_prev[d1] / (rho_prev_squared * rho_prev) * rho, ve->di(d1+1));
    _steadyBF->addTerm( ((gamma - 1.)/2.) * m_prev_dot_m_prev / rho_prev_squared * m[d1],                         ve->di(d1+1));
    _rhs->addTerm(-((gamma - 1.)/2.) * m_prev_dot_m_prev * m_prev[d1]/ rho_prev_squared                   * ve->di(d1+1));
    
    _steadyBF->addTerm( (gamma-1.) * m_prev_dot_m * m_prev[d1] / rho_prev_squared,                                ve->di(d1+1));
    
    if (!_pureEulerMode) // no viscous terms for Euler
    {
      _steadyBF->addTerm(-q[d1], ve->di(d1+1));
      _rhs->addTerm(q_prev[d1] * ve->di(d1+1));
      
      _steadyBF->addTerm((D_traceWeight * m_prev[d1] / rho_prev) * D_trace, ve->di(d1+1));
      _steadyBF->addTerm((D_traceWeight / rho_prev) * D_trace_prev * m[d1], ve->di(d1+1));
      _steadyBF->addTerm(-(D_traceWeight * m_prev[d1] / rho_prev_squared) * D_trace_prev * rho, ve->di(d1+1));
      _rhs->addTerm(-(D_traceWeight * m_prev[d1] / rho_prev) * D_trace_prev * ve->di(d1+1));
      for (int d2=0; d2<spaceDim; d2++)
      {
        _steadyBF->addTerm(m_prev[d2]/rho_prev * (D[d1][d2] + D[d2][d1]),                        ve->di(d1+1));
        _steadyBF->addTerm((1.0/rho_prev) * (D_prev[d1][d2] + D_prev[d2][d1])*m[d2],             ve->di(d1+1));
        _steadyBF->addTerm(-(D_prev[d1][d2] + D_prev[d2][d1])*m_prev[d2]/rho_prev_squared * rho, ve->di(d1+1));
        _rhs->addTerm(-(D_prev[d1][d2] + D_prev[d2][d1])*m_prev[d2]/rho_prev             * ve->di(d1+1));
      }
    }
  }
  _steadyBF->addTerm(te, ve);
  _rhs->addTerm(FunctionPtr(_fe) * ve);
  
  // now, combine transient and steady into _bf:
  auto steadyTerms = _steadyBF->getTerms();
  auto transientTerms = transientBF->getTerms();
  for (auto steadyTerm : steadyTerms)
  {
    _bf->addTerm(steadyTerm.first, steadyTerm.second);
  }
  for (auto transientTerm : transientTerms)
  {
    _bf->addTerm(transientTerm.first, transientTerm.second);
  }
  
  vector<VarPtr> missingTestVars = _bf->missingTestVars();
  vector<VarPtr> missingTrialVars = _bf->missingTrialVars();
  for (int i=0; i < missingTestVars.size(); i++)
  {
    VarPtr var = missingTestVars[i];
    cout << var->displayString() << endl;
  }
  for (int i=0; i < missingTrialVars.size(); i++)
  {
    VarPtr var = missingTrialVars[i];
    cout << var->displayString() << endl;
  }
  
  // TODO: consider adding support for Truman's various IP definitions
  // (For now, we just support the graph norm.)
  
  TEUCHOS_TEST_FOR_EXCEPTION(normName != "Graph", std::invalid_argument, "non-graph norms not yet supported in the refactor; use legacy instead");
  _ips["Graph"] = _bf->graphNorm();
  IPPtr ip = _ips.at(normName);
  
  // _solnIncrement->setBC(bc);
  _solnIncrement->setRHS(_rhs);
  _solnIncrement->setIP(ip);
  // _solnIncrement->setRHS(rhs);
  
  mesh->registerSolution(_backgroundFlow);
  mesh->registerSolution(_solnIncrement);
  mesh->registerSolution(_solnPrevTime);
  
  // LinearTermPtr residual = rhs->linearTerm() - _bf->testFunctional(_solnIncrement,true); // false: don't exclude boundary terms
  // LinearTermPtr residual = _rhsForResidual->linearTerm() - _bf->testFunctional(_solnIncrement,false); // false: don't exclude boundary terms
  // LinearTermPtr residual = _rhsForSolve->linearTerm() - _bf->testFunctional(_solnIncrement,true); // false: don't exclude boundary terms
  
  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );
  // _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, _ips[normName], energyThreshold ) );
  
  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );
  
  // Set up Functions for L^2 norm computations
  
  FunctionPtr rho_incr = Function::solution(rho, _solnIncrement);
  FunctionPtr E_incr = Function::solution(E, _solnIncrement);
  
  _L2IncrementFunction = rho_incr * rho_incr + E_incr * E_incr;
  _L2SolutionFunction = rho_prev * rho_prev + E_prev * E_prev;
  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    FunctionPtr m_i_incr = Function::solution(this->m(comp_i), _solnIncrement);
    FunctionPtr m_i_prev = Function::solution(this->m(comp_i), _backgroundFlow);
    
    _L2IncrementFunction = _L2IncrementFunction + m_i_incr * m_i_incr;
    _L2SolutionFunction = _L2SolutionFunction + m_i_prev * m_i_prev;
    if (!_pureEulerMode)
    {
      FunctionPtr q_i_incr = Function::solution(this->q(comp_i), _solnIncrement);
      FunctionPtr q_i_prev = Function::solution(this->q(comp_i), _backgroundFlow);
      _L2IncrementFunction = _L2IncrementFunction + q_i_incr * q_i_incr;
      _L2SolutionFunction = _L2SolutionFunction + q_i_prev * q_i_prev;
      
      for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
      {
        FunctionPtr D_ij_incr = Function::solution(this->D(comp_i,comp_j), _solnIncrement);
        FunctionPtr D_ij_prev = Function::solution(this->D(comp_i,comp_j), _backgroundFlow);
        _L2IncrementFunction = _L2IncrementFunction + D_ij_incr * D_ij_incr;
        _L2SolutionFunction = _L2SolutionFunction + D_ij_prev * D_ij_prev;
      }
    }
  }
  
  _solver = Solver::getDirectSolver();
  
  _nonlinearIterationCount = 0;

  // the following does not appear to work.  Unclear on what's wrong; we may need an entirely different strategy
  // IT MAY BE THAT LAGRANGE CONSTRAINTS ARE BROKEN.  (Constraint_StokesLocalConservation_UnitTest has been failing...)
//  if (_timeStepping)
//  {
//    cout << "TRYING CONSERVATION ENFORCEMENT.\n";
//    Teuchos::RCP<LagrangeConstraints> constraints = Teuchos::rcp(new LagrangeConstraints);
//    // vc constraint:
//    VarPtr vc = this->vc();
//    map<int, FunctionPtr> vcEqualsOne = {{vc->ID(), Function::constant(1.0)}};
//    LinearTermPtr vcTrialFunctional = _bf->trialFunctional(vcEqualsOne) * dt;
//    FunctionPtr   vcRHSFunction     = _rhs->linearTerm()->evaluate(vcEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
//    constraints->addConstraint(vcTrialFunctional == vcRHSFunction);
//    cout << "Added element constraint " << vcTrialFunctional->displayString() << " == " << vcRHSFunction->displayString() << endl;
//
//    // vm constraint(s):
//    for (int d=0; d<spaceDim; d++)
//    {
//      // test with 1
//      VarPtr vm = this->vm(d+1);
//      map<int, FunctionPtr> vmEqualsOne = {{vm->ID(), Function::constant(1.0)}};
//      LinearTermPtr trialFunctional = _bf->trialFunctional(vmEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
//      FunctionPtr rhsFxn = _rhs->linearTerm()->evaluate(vmEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
//      constraints->addConstraint(trialFunctional == rhsFxn);
//
//      cout << "Added element constraint " << trialFunctional->displayString() << " == " << rhsFxn->displayString() << endl;
//    }
//    // ve constraint:
//    VarPtr ve = this->ve();
//    map<int, FunctionPtr> veEqualsOne = {{ve->ID(), Function::constant(1.0)}};
//    LinearTermPtr veTrialFunctional = _bf->trialFunctional(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
//    FunctionPtr   veRHSFunction     = _rhs->linearTerm()->evaluate(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
//    constraints->addConstraint(veTrialFunctional == veRHSFunction);
//    cout << "Added element constraint " << veTrialFunctional->displayString() << " == " << veRHSFunction->displayString() << endl;
//
//    // although enforcement only happens in solnIncrement, the constraints change numbering of dofs, so we need to set constraints in each Solution object
//    _solnIncrement->setLagrangeConstraints(constraints);
//    _backgroundFlow->setLagrangeConstraints(constraints);
//    _solnPrevTime->setLagrangeConstraints(constraints);
//  }
}

void CompressibleNavierStokesConservationForm::addVelocityTraceComponentCondition(SpatialFilterPtr region, FunctionPtr ui_exact, int i)
{
  VarPtr ui_hat = this->u_hat(i);
  _solnIncrement->bc()->addDirichlet(ui_hat, region, ui_exact);
}


void CompressibleNavierStokesConservationForm::addVelocityTraceCondition(SpatialFilterPtr region, FunctionPtr u_exact)
{
  if (_spaceDim==1)
    addVelocityTraceComponentCondition(region, u_exact, 1);
  else
  {
    for (int d=0; d<_spaceDim; d++)
    {
      addVelocityTraceComponentCondition(region, u_exact->spatialComponent(d+1), d+1);
    }
  }
}

void CompressibleNavierStokesConservationForm::addTemperatureTraceCondition(SpatialFilterPtr region, FunctionPtr T_exact)
{
  VarPtr T_hat = this->T_hat();
  _solnIncrement->bc()->addDirichlet(T_hat, region, T_exact);
}

void CompressibleNavierStokesConservationForm::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr tc_exact)
{
  VarPtr tc = this->tc();
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
//  cout << "adding boundary condition tc = " << tc_exact->displayString() << endl;
}

void CompressibleNavierStokesConservationForm::addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr tm_i_exact, int i)
{
  VarPtr tm_i = this->tm(i);
  _solnIncrement->bc()->addDirichlet(tm_i, region, tm_i_exact);
//  cout << "adding boundary condition tm" << i << " = " << tm_i_exact->displayString() << endl;
}

void CompressibleNavierStokesConservationForm::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr te_exact)
{
  VarPtr te = this->te();
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
//  cout << "adding boundary condition te = " << te_exact->displayString() << endl;
}

void CompressibleNavierStokesConservationForm::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr tc = this->tc();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tc_exact = this->exactSolution_tc(u_exact, rho_exact, T_exact, includeParity);
  _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
//  cout << "adding boundary condition tc = " << tc_exact->displayString() << endl;
}

void CompressibleNavierStokesConservationForm::addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact, int i)
{
  VarPtr tm_i = this->tm(i);
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  FunctionPtr tm_i_exact = exactSolution_tm(u_exact, rho_exact, T_exact, includeParity)[i-1];
  _solnIncrement->bc()->addDirichlet(tm_i, region, tm_i_exact);
}

void CompressibleNavierStokesConservationForm::addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto tm_exact = exactSolution_tm(u_exact, rho_exact, T_exact, includeParity);
  for (int d=0; d<_spaceDim; d++)
  {
    VarPtr tm_i = this->tm(d+1);
    _solnIncrement->bc()->addDirichlet(tm_i, region, tm_exact[d]);
//    cout << "adding boundary condition tm" << d+1 << " = " << tm_exact[d]->displayString() << endl;
  }
}

void CompressibleNavierStokesConservationForm::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr te = this->te();
  bool includeParity = true; // in the usual course of things, this should not matter for BCs, because the parity is always 1 on boundary.  But conceptually, the more correct thing is to include, because here we are imposing what ought to be a unique value, and if ever we have an internal boundary which also has non-positive parity on one of its sides, we'd want to include...
  auto te_exact = exactSolution_te(u_exact, rho_exact, T_exact, includeParity);
  _solnIncrement->bc()->addDirichlet(te, region, te_exact);
//  cout << "adding boundary condition te = " << te_exact->displayString() << endl;
}

BFPtr CompressibleNavierStokesConservationForm::bf()
{
  return _bf;
}

double CompressibleNavierStokesConservationForm::Cv()
{
  return _Cv;
}

double CompressibleNavierStokesConservationForm::Cp()
{
  return _gamma*_Cv;
}

VarPtr CompressibleNavierStokesConservationForm::D(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  
  return _vf->fieldVar(S_D[i-1][j-1]);
}

VarPtr CompressibleNavierStokesConservationForm::E()
{
  return _vf->fieldVar(S_E);
}


// ! For an exact solution (u, rho, T), returns the corresponding forcing in the continuity equation
FunctionPtr CompressibleNavierStokesConservationForm::exactSolution_fc(FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  // strong form of the equation has
  // d/dt rho + div ( rho u ) = f_c
  FunctionPtr f_c = Function::zero();
  if (rho->dt() != Teuchos::null)
  {
    f_c = f_c + rho->dt();
  }
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f_c = f_c + Function::op(rho * u, divOp);
  return f_c;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the energy equation
FunctionPtr CompressibleNavierStokesConservationForm::exactSolution_fe(FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  bool includeParity = false; // we don't use any traces or fluxes below, so this does not matter
  auto exactMap = this->exactSolutionMap(u, rho, T, includeParity);
  // strong form of the equation has
  // f_e =   d/dt ( rho * ( c_v T + 0.5 * u * u) )
  //       + div  ( rho * u * ( c_v T + 0.5 * u * u) + rho * u * R * T + q - u \dot (D + D^T - 2/3 tr(D) I) )
  // define g_e = rho * ( c_v T + 0.5 * u * u)
  FunctionPtr f_e = Function::zero();
  double c_v = this->Cv();
  double R   = this->R();
  FunctionPtr g_e = rho * ( c_v * T + 0.5 * u * u);
  if (g_e->dt() != Teuchos::null)
  {
    f_e = f_e + g_e->dt();
  }
  FunctionPtr q;
  if (_spaceDim == 1)
    q = exactMap[this->q(1)->ID()];
  else
  {
    vector<FunctionPtr> q_vector(_spaceDim);
    for (int d=0; d<_spaceDim; d++)
    {
      q_vector[d] = exactMap[this->q(d+1)->ID()];
    }
    q = (_spaceDim > 1) ? Function::vectorize(q_vector) : q_vector[0];
  }
  FunctionPtr D_trace = Function::zero();
  for (int d=0; d<_spaceDim; d++)
  {
    VarPtr D_dd = this->D(d+1,d+1);
    D_trace = D_trace + exactMap[D_dd->ID()];
  }
  FunctionPtr u_dot_sigma = Function::zero();
  for (int d1=0; d1<_spaceDim; d1++)
  {
    int i = d1+1;
    FunctionPtr u_i = exactMap[this->m(i)->ID()] / rho;
    vector<FunctionPtr> sigmaRow_vector(_spaceDim); // sigma_i
    for (int d2=0; d2<_spaceDim; d2++)
    {
      int j = d2 + 1;
      FunctionPtr D_ij = exactMap[this->D(i,j)->ID()];
      FunctionPtr D_ji = exactMap[this->D(j,i)->ID()];
      FunctionPtr D_trace_contribution = (i==j)? -2./3. * D_trace : Function::zero();
      sigmaRow_vector[d2] = D_ij + D_ji + D_trace_contribution;
    }
    FunctionPtr sigmaRow =  (_spaceDim > 1) ? Function::vectorize(sigmaRow_vector) : sigmaRow_vector[0];
    u_dot_sigma = u_dot_sigma + u_i * sigmaRow;
  }
  FunctionPtr flux_term = u * g_e + rho * R * T * u + q - u_dot_sigma;
  Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
  f_e = f_e + Function::op(flux_term, divOp);
  return f_e;
}

// ! For an exact solution (u, rho, T), returns the corresponding forcing in the momentum equation
std::vector<FunctionPtr> CompressibleNavierStokesConservationForm::exactSolution_fm(FunctionPtr u, FunctionPtr rho, FunctionPtr T)
{
  bool includeParity = false; // we don't use any traces or fluxes below, so this does not matter
  auto exactMap = this->exactSolutionMap(u, rho, T, includeParity);
  double R   = this->R();
  vector<FunctionPtr> f_m(_spaceDim, Function::zero());
  FunctionPtr rho_u = rho * u;
  if (rho_u->dt() != Teuchos::null)
  {
    if (_spaceDim == 1)
    {
      f_m[0] = f_m[0] + rho_u->dt();
    }
    else
    {
      FunctionPtr dt_part = rho_u->dt();
      for (int d=0; d<_spaceDim; d++)
      {
        f_m[d] = f_m[d] + dt_part->spatialComponent(d+1);
      }
    }
  }
  vector<FunctionPtr> u_vector(_spaceDim);
  for (int d=0; d<_spaceDim; d++)
  {
    u_vector[d] = exactMap[this->m(d+1)->ID()] / rho;
  }

  for (int d2=0; d2<_spaceDim; d2++)
  {
    int j = d2+1;
    vector<FunctionPtr> column_vector(_spaceDim, Function::zero());
    for (int d1=0; d1<_spaceDim; d1++)
    {
      int i = d1+1;
      column_vector[d1] = rho * u_vector[d1] * u_vector[d2];
      if (i==j)
      {
        column_vector[d1] = column_vector[d1] + R * (rho * T);
      }
    }
    FunctionPtr column = (_spaceDim > 1) ? Function::vectorize(column_vector) : column_vector[0];
    Camellia::EOperator divOp = (_spaceDim > 1) ? OP_DIV : OP_DX;
    f_m[d2] = f_m[d2] + Function::op(column, divOp);
  }
  return f_m;
}

FunctionPtr CompressibleNavierStokesConservationForm::exactSolution_tc(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeParity)
{
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  FunctionPtr tc_exact = Function::zero();
  for (int d=0; d<_spaceDim; d++)
  {
    auto u_i = (_spaceDim > 1) ? velocity->spatialComponent(d+1) : velocity;
    auto n_i = n->spatialComponent(d+1);
    tc_exact = tc_exact + rho * u_i * n_i;
  }
  if (_spaceTime)
  {
    FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
    FunctionPtr n_t = n_xt->t();
    tc_exact = tc_exact + rho * n_t;
  }
  if (includeParity)
  {
    tc_exact = tc_exact * Function::sideParity();
  }
  return tc_exact;
}

FunctionPtr CompressibleNavierStokesConservationForm::exactSolution_te(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeParity)
{
  // t_e is the trace of:
  // ((c_v + R) T rho u + 0.5 * (u dot u) rho u + q - u dot (D + D^T - 2/3 tr(D) I)) dot n
  
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  
  FunctionPtr D_trace = Function::zero();
  std::vector<FunctionPtr> Di_exact(_spaceDim);
  std::vector<std::vector<FunctionPtr>> Dij_exact(_spaceDim, std::vector<FunctionPtr>(_spaceDim));
  std::vector<FunctionPtr> u_vector(_spaceDim);
  FunctionPtr q_exact;
  
  FunctionPtr qWeight = (-Cp()/Pr())*_muFunc;
  if (_spaceDim == 1)
  {
    q_exact = qWeight * T->dx();
  }
  else
  {
    q_exact = qWeight * T->grad();
  }
  
  for (int d=0; d<_spaceDim; d++)
  {
    if (_spaceDim == 1)
    {
      u_vector[d] = velocity;
      Di_exact[d] = _mu * u_vector[d]->dx();
      D_trace = Di_exact[d];
      Dij_exact[0][0] = Di_exact[0];
    }
    else
    {
      u_vector[d] = velocity->spatialComponent(d+1);
      Di_exact[d] = _mu * u_vector[d]->grad();
      D_trace = D_trace + Di_exact[d]->spatialComponent(d+1);
      for (int d2=0; d2<_spaceDim; d2++)
      {
        Dij_exact[d][d2] = Di_exact[d]->spatialComponent(d2+1);
      }
    }
  }
  
  double R = this->R();
  double Cv = this->Cv();
  double D_traceWeight = -2./3.; // Stokes' hypothesis
  
  // defer the dotting with the normal until we've accumulated the other terms (other than the D + D^T terms, which we treat separately)
  FunctionPtr te_exact = ((Cv + R) * T + (0.5 * velocity * velocity)) * rho * velocity + q_exact;
  te_exact = te_exact - D_traceWeight * D_trace * velocity; // simplification of u dot (2/3 tr(D) I)
  
  // now, dot with normal
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  if (_spaceDim == 1)
  {
    // for 1D, the product with normal should yield a scalar result
    te_exact = te_exact * n->x();
  }
  else
  {
    te_exact = te_exact * n; // dot product
  }
  // u dot (D + D^T) dot n = ((D + D^T) u) dot n
  for (int d1=0; d1<_spaceDim; d1++)
  {
    for (int d2=0; d2<_spaceDim; d2++)
    {
      te_exact = te_exact - (Dij_exact[d1][d2] + Dij_exact[d2][d1]) * u_vector[d2] * n->spatialComponent(d1+1);
    }
  }
  if (includeParity)
  {
    te_exact = te_exact * Function::sideParity();
  }
  return te_exact;
}

std::vector<FunctionPtr> CompressibleNavierStokesConservationForm::exactSolution_tm(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeParity)
{
  vector<FunctionPtr> tm_exact(_spaceDim);
  for (int i=1; i<= _spaceDim; i++)
  {
    VarPtr tm_i = this->tm(i);
    
    // tm: trace of rho (u xx u) n + rho R T I n - (D + D^T - 2./3. * tr(D)I) n
    //     (where xx is the outer product operator)
    
    FunctionPtr n = TFunction<double>::normal(); // spatial normal
    
    FunctionPtr D_trace = Function::zero();
    std::vector<FunctionPtr> Di_exact(_spaceDim);
    std::vector<std::vector<FunctionPtr>> Dij_exact(_spaceDim, std::vector<FunctionPtr>(_spaceDim));
    std::vector<FunctionPtr> u_vector(_spaceDim);
    for (int d=0; d<_spaceDim; d++)
    {
      if (_spaceDim == 1)
      {
        u_vector[d] = velocity;
        Di_exact[d] = _mu * u_vector[d]->dx();
        D_trace = Di_exact[d];
        Dij_exact[0][0] = Di_exact[0];
      }
      else
      {
        u_vector[d] = velocity->spatialComponent(d+1);
        Di_exact[d] = _mu * u_vector[d]->grad();
        D_trace = D_trace + Di_exact[d]->spatialComponent(d+1);
        for (int d2=0; d2<_spaceDim; d2++)
        {
          Dij_exact[d][d2] = Di_exact[d]->spatialComponent(d2+1);
        }
      }
    }
    
    double R = this->R();
    double D_traceWeight = -2./3.; // Stokes' hypothesis
    
    FunctionPtr tm_i_exact = Function::zero();
    for (int d=0; d<_spaceDim; d++)
    {
      // rho (u xx u) n
      tm_i_exact = tm_i_exact + rho * u_vector[d] * u_vector[i-1] * n->spatialComponent(d+1);
      // - (D + D^T) n
      tm_i_exact = tm_i_exact - (Dij_exact[d][i-1] + Dij_exact[i-1][d]) * n->spatialComponent(d+1);
    }
    // rho R T I n
    tm_i_exact = tm_i_exact + rho * R * T * n->spatialComponent(i);
    // - D_traceWeight * tr(D) I n
    tm_i_exact = tm_i_exact - D_traceWeight * D_trace * n->spatialComponent(i);
    
    if (_spaceTime)
    {
      // TODO: confirm that this is correct (I'm not really focused on the space-time case in this refactor...)
      FunctionPtr n_t = Function::normalSpaceTime()->t();
      FunctionPtr u_i = u_vector[i-1];
      tm_i_exact = tm_i_exact + rho * u_i * n_t;
    }
    
    if (includeParity)
    {
      tm_i_exact = tm_i_exact * Function::sideParity();
    }
    tm_exact[i-1] = tm_i_exact;
  }
  return tm_exact;
}

std::map<int, FunctionPtr> CompressibleNavierStokesConservationForm::exactSolutionMap(FunctionPtr velocity, FunctionPtr rho, FunctionPtr T, bool includeFluxParity)
{
  using namespace std;
  vector<FunctionPtr> q(_spaceDim);
  vector<vector<FunctionPtr>> D(_spaceDim,vector<FunctionPtr>(_spaceDim));
  vector<FunctionPtr> u(_spaceDim);
  FunctionPtr qWeight = (-Cp()/Pr())*_muFunc;
  FunctionPtr E = rho * Cv() * T; // we add u*u/2 to this below
  if (_spaceDim == 1)
  {
    D[0][0] = _muFunc * velocity->dx();
    q[0] = qWeight * T->dx();
    u[0] = velocity;
    E = E + 0.5 * rho * u[0]*u[0];
  }
  else
  {
    for (int d1=0; d1<_spaceDim; d1++)
    {
      q[d1] = qWeight * T->di(d1+1);
      u[d1] = velocity->spatialComponent(d1+1);
      for (int d2=0; d2<_spaceDim; d2++)
      {
        D[d1][d2] = _muFunc * u[d1]->di(d2+1);
      }
      E = E + 0.5 * rho * u[d1] * u[d1];
    }
  }
  vector<FunctionPtr> tm = exactSolution_tm(velocity, rho, T, includeFluxParity);
  FunctionPtr         te = exactSolution_te(velocity, rho, T, includeFluxParity);
  FunctionPtr         tc = exactSolution_tc(velocity, rho, T, includeFluxParity);
  
  FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
  map<int, FunctionPtr> solnMap;
  solnMap[this->E()->ID()]     = E;
  solnMap[this->T_hat()->ID()] = T * traceFxn;
  solnMap[this->rho()->ID()]   = rho;
  solnMap[this->tc()->ID()]    = tc;
  solnMap[this->te()->ID()]    = te;
  for (int d1=0; d1<_spaceDim; d1++)
  {
    solnMap[this->m(d1+1)->ID()]     = u[d1] * rho;
    solnMap[this->u_hat(d1+1)->ID()] = u[d1] * traceFxn;
    
    solnMap[this->q(d1+1)->ID()]     = q[d1];
    
    solnMap[this->tm(d1+1)->ID()]    = tm[d1];
    
    for (int d2=0; d2<_spaceDim; d2++)
    {
      solnMap[this->D(d1+1,d2+1)->ID()] = D[d1][d2];
    }
  }
  return solnMap;
}

double CompressibleNavierStokesConservationForm::gamma()
{
  return _gamma;
}

int CompressibleNavierStokesConservationForm::getSolveCode()
{
  return _solveCode;
}

FunctionPtr CompressibleNavierStokesConservationForm::getTimeStep()
{
  return _dt;
}

double CompressibleNavierStokesConservationForm::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double CompressibleNavierStokesConservationForm::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

VarPtr CompressibleNavierStokesConservationForm::m(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->fieldVar(S_m[i-1]);
}

double CompressibleNavierStokesConservationForm::mu()
{
  return _mu;
}

double CompressibleNavierStokesConservationForm::Pr()
{
  return _Pr;
}

VarPtr CompressibleNavierStokesConservationForm::q(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  return _vf->fieldVar(S_q[i-1]);
}

double CompressibleNavierStokesConservationForm::R()
{
  return Cp()-Cv();
}

VarPtr CompressibleNavierStokesConservationForm::rho()
{
  return _vf->fieldVar(S_rho);
}

RHSPtr CompressibleNavierStokesConservationForm::rhs()
{
  return _rhs;
}

VarPtr CompressibleNavierStokesConservationForm::S(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_pureEulerMode, std::invalid_argument, "S test function is not defined in Euler formulation");
  CHECK_VALID_COMPONENT(i);
  Space SSpace = (_spaceDim == 1) ? HGRAD : HDIV;
  return _vf->testVar(S_S[i-1], SSpace);
}

void CompressibleNavierStokesConservationForm::setForcing(FunctionPtr f_continuity, vector<FunctionPtr> f_momentum, FunctionPtr f_energy)
{
  TEUCHOS_TEST_FOR_EXCEPTION(f_momentum.size() != _spaceDim, std::invalid_argument, "f_momentum should have size equal to the spatial dimension");
  _fc->setValue(f_continuity);
  _fe->setValue(f_energy);
  for (int d=0; d<_spaceDim; d++)
  {
    _fm[d]->setValue(f_momentum[d]);
  }
}

void CompressibleNavierStokesConservationForm::setMu(double value)
{
  _mu = value;
  _muParamFunc->setValue(_mu);
  _muSqrtParamFunc->setValue(sqrt(_mu));
}

// ! set current time step used for transient solve
void CompressibleNavierStokesConservationForm::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

double CompressibleNavierStokesConservationForm::solveAndAccumulate()
{
  _solveCode = _solnIncrement->solve(_solver);
  
  set<int> nonlinearVariables = {{rho()->ID(), E()->ID()}};
  set<int> linearVariables = {{tc()->ID(), te()->ID()}};
  
  if (!_pureEulerMode) // T_hat not defined for Euler
  {
    linearVariables.insert(T_hat()->ID());
  }
  for (int d1=0; d1<_spaceDim; d1++)
  {
    nonlinearVariables.insert(m(d1+1)->ID());
    linearVariables.insert(tm(d1+1)->ID());
    if (!_pureEulerMode) // u_hat, q, D not defined in Euler
    {
      linearVariables.insert(u_hat(d1+1)->ID());
      nonlinearVariables.insert(q(d1+1)->ID());
      for (int d2=0; d2<_spaceDim; d2++)
      {
        nonlinearVariables.insert(D(d1+1,d2+1)->ID());
      }
    }
  }
  
  double alpha = 1.0;
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
  double minDistanceFromZero = 0.001; // "positive" values should not get *too* small...
  int posEnrich = 5;
  
  // lambda for positivity checking
  auto isPositive = [&] () -> bool
  {
    for (auto f : positiveFunctions)
    {
      FunctionPtr f_smaller = f - minDistanceFromZero;
      bool isPositive = f_smaller->isPositive(_solnIncrement->mesh(),posEnrich); // does MPI communication
      if (!isPositive) return false;
    }
    return true;
  };
  
  bool useLineSearch = true;

  if (useLineSearch)
  {
    double lineSearchFactor = .5;
    int iter = 0; int maxIter = 20;
    while (!isPositive() && iter < maxIter)
    {
      alpha = alpha*lineSearchFactor;
      alphaParameter->setValue(alpha);
      iter++;
    }
  }
  
  _backgroundFlow->addReplaceSolution(_solnIncrement, alpha, nonlinearVariables, linearVariables);
  _nonlinearIterationCount++;
  
  return alpha;
}

// ! Returns the solution (at current time)
SolutionPtr CompressibleNavierStokesConservationForm::solution()
{
  return _backgroundFlow;
}

SolutionPtr CompressibleNavierStokesConservationForm::solutionIncrement()
{
  return _solnIncrement;
}

// ! Returns the solution (at previous time)
SolutionPtr CompressibleNavierStokesConservationForm::solutionPreviousTimeStep()
{
  return _solnPrevTime;
}

BFPtr CompressibleNavierStokesConservationForm::steadyBF()
{
  return _steadyBF;
}

VarPtr CompressibleNavierStokesConservationForm::T_hat()
{
  TEUCHOS_TEST_FOR_EXCEPTION(_pureEulerMode, std::invalid_argument, "T_hat is not defined in Euler formulation");
  if (! _spaceTime)
    return _vf->traceVar(S_T_hat);
  else
    return _vf->traceVarSpaceOnly(S_T_hat);
}

VarPtr CompressibleNavierStokesConservationForm::tau()
{
  TEUCHOS_TEST_FOR_EXCEPTION(_pureEulerMode, std::invalid_argument, "tau test function is not defined in Euler formulation");
  Space tauSpace = (_spaceDim == 1) ? HGRAD : HDIV;
  return _vf->testVar(S_tau, tauSpace);
}

VarPtr CompressibleNavierStokesConservationForm::tc()
{
  return _vf->fluxVar(S_tc);
}

VarPtr CompressibleNavierStokesConservationForm::te()
{
  return _vf->fluxVar(S_te);
}

VarPtr CompressibleNavierStokesConservationForm::tm(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->fluxVar(S_tm[i-1]);
}

// traces:
VarPtr CompressibleNavierStokesConservationForm::u_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  TEUCHOS_TEST_FOR_EXCEPTION(_pureEulerMode, std::invalid_argument, "u_hat is not defined in Euler formulation");
  if (! _spaceTime)
    return _vf->traceVar(S_u_hat[i-1]);
  else
    return _vf->traceVarSpaceOnly(S_u_hat[i-1]);
}

VarPtr CompressibleNavierStokesConservationForm::vc()
{
  return _vf->testVar(S_vc, HGRAD);
}

VarPtr CompressibleNavierStokesConservationForm::vm(int i)
{
  CHECK_VALID_COMPONENT(i);
  return _vf->testVar(S_vm[i-1], HGRAD);
}

VarPtr CompressibleNavierStokesConservationForm::ve()
{
  return _vf->testVar(S_ve, HGRAD);
}
