
//  CompressibleNavierStokesFormulation.cpp
//  Camellia
//
//  Created by Truman Ellis on 12/4/15.
//
//

#include "CompressibleNavierStokesFormulation.h"

#include "ConstantScalarFunction.h"
#include "Constraint.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

using namespace Camellia;

const string CompressibleNavierStokesFormulation::S_rho = "rho";
const string CompressibleNavierStokesFormulation::S_u1  = "u1";
const string CompressibleNavierStokesFormulation::S_u2  = "u2";
const string CompressibleNavierStokesFormulation::S_u3  = "u3";
const string CompressibleNavierStokesFormulation::S_T   = "T";
const string CompressibleNavierStokesFormulation::S_D11 = "D11";
const string CompressibleNavierStokesFormulation::S_D12 = "D12";
const string CompressibleNavierStokesFormulation::S_D13 = "D13";
const string CompressibleNavierStokesFormulation::S_D21 = "D21";
const string CompressibleNavierStokesFormulation::S_D22 = "D22";
const string CompressibleNavierStokesFormulation::S_D23 = "D23";
const string CompressibleNavierStokesFormulation::S_D31 = "D31";
const string CompressibleNavierStokesFormulation::S_D32 = "D32";
const string CompressibleNavierStokesFormulation::S_D33 = "D33";
const string CompressibleNavierStokesFormulation::S_q1 = "q1";
const string CompressibleNavierStokesFormulation::S_q2 = "q2";
const string CompressibleNavierStokesFormulation::S_q3 = "q3";

const string CompressibleNavierStokesFormulation::S_tc = "tc";
const string CompressibleNavierStokesFormulation::S_tm1 = "tm1";
const string CompressibleNavierStokesFormulation::S_tm2 = "tm2";
const string CompressibleNavierStokesFormulation::S_tm3 = "tm3";
const string CompressibleNavierStokesFormulation::S_te = "te";
const string CompressibleNavierStokesFormulation::S_u1_hat = "u1_hat";
const string CompressibleNavierStokesFormulation::S_u2_hat = "u2_hat";
const string CompressibleNavierStokesFormulation::S_u3_hat = "u3_hat";
const string CompressibleNavierStokesFormulation::S_T_hat = "T_hat";

const string CompressibleNavierStokesFormulation::S_vc = "vc";
const string CompressibleNavierStokesFormulation::S_vm1  = "vm1";
const string CompressibleNavierStokesFormulation::S_vm2  = "vm2";
const string CompressibleNavierStokesFormulation::S_vm3  = "vm3";
const string CompressibleNavierStokesFormulation::S_ve   = "ve";
const string CompressibleNavierStokesFormulation::S_S1 = "S1";
const string CompressibleNavierStokesFormulation::S_S2 = "S2";
const string CompressibleNavierStokesFormulation::S_S3 = "S3";
const string CompressibleNavierStokesFormulation::S_tau = "tau";

static const int INITIAD_CONDITION_TAG = 1;

CompressibleNavierStokesFormulation CompressibleNavierStokesFormulation::steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
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

  return CompressibleNavierStokesFormulation(meshTopo, parameters);
}

CompressibleNavierStokesFormulation CompressibleNavierStokesFormulation::spaceTimeFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                            MeshTopologyPtr meshTopo, int spatialPolyOrder, int temporalPolyOrder, int delta_k)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",1.0 / Re);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", true);

  parameters.set("t0",0.0);

  parameters.set("spatialPolyOrder", spatialPolyOrder);
  parameters.set("temporalPolyOrder", temporalPolyOrder);
  parameters.set("delta_k", delta_k);

  return CompressibleNavierStokesFormulation(meshTopo, parameters);
}

CompressibleNavierStokesFormulation::CompressibleNavierStokesFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  _ctorParameters = parameters;

  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  _mu = parameters.get<double>("mu",1.0);
  _gamma = parameters.get<double>("gamma",1.4);
  _Pr = parameters.get<double>("Pr",0.713);
  _Cv = parameters.get<double>("Cv",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");
  string normName = parameters.get<string>("norm", "Graph");

  // nonlinear parameters
  bool neglectFluxesOnRHS = true;

  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double dt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);
  TimeStepType timeStepType = parameters.get<TimeStepType>("timeStepType", BACKWARD_EULER); // Backward Euler is immune to oscillations (which Crank-Nicolson can/does exhibit)

  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _spatialPolyOrder = spatialPolyOrder;
  _temporalPolyOrder =temporalPolyOrder;
  _dt = ParameterFunction::parameterFunction(dt);
  _t = ParameterFunction::parameterFunction(0);
  _t0 = parameters.get<double>("t0",0);
  _neglectFluxesOnRHS = neglectFluxesOnRHS;
  _delta_k = delta_k;

  double thetaValue;
  switch (timeStepType) {
    case FORWARD_EULER:
      thetaValue = 0.0;
      break;
    case CRANK_NICOLSON:
      thetaValue = 0.5;
      break;
    case BACKWARD_EULER:
      thetaValue = 1.0;
      break;
  }

  _theta = ParameterFunction::parameterFunction(thetaValue);
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;

  TEUCHOS_TEST_FOR_EXCEPTION(_timeStepping, std::invalid_argument, "Time stepping not supported");

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr rho;
  VarPtr u1, u2, u3;
  VarPtr T;
  VarPtr D11, D12, D13, D21, D22, D23, D31, D32, D33;
  VarPtr q1, q2, q3;

  // traces
  VarPtr tc;
  VarPtr tm1, tm2, tm3;
  VarPtr te;
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr T_hat;

  // tests
  VarPtr vc;
  VarPtr vm1, vm2, vm3;
  VarPtr ve;
  VarPtr S1, S2, S3;
  VarPtr tau;

  _vf = VarFactory::varFactory();

  rho = _vf->fieldVar(S_rho);

  u1 = _vf->fieldVar(S_u1);
  if (spaceDim>=2) u2 = _vf->fieldVar(S_u2);
  if (spaceDim==3) u3 = _vf->fieldVar(S_u3);
  vector<VarPtr> u(spaceDim);
  u[0] = u1;
  if (spaceDim>=2) u[1] = u2;
  if (spaceDim==3) u[2] = u3;

  T = _vf->fieldVar(S_T);

  vector<vector<VarPtr>> D(spaceDim,vector<VarPtr>(spaceDim));
  D11 = _vf->fieldVar(S_D11);
  D[0][0] = D11;
  if (spaceDim>=2)
  {
    D12 = _vf->fieldVar(S_D12);
    D21 = _vf->fieldVar(S_D21);
    D22 = _vf->fieldVar(S_D22);
    D[0][1] = D12;
    D[1][0] = D21;
    D[1][1] = D22;
  }
  if (spaceDim==3)
  {
    D13 = _vf->fieldVar(S_D13);
    D23 = _vf->fieldVar(S_D23);
    D31 = _vf->fieldVar(S_D31);
    D32 = _vf->fieldVar(S_D32);
    D33 = _vf->fieldVar(S_D33);
    D[0][2] = D13;
    D[1][2] = D23;
    D[2][0] = D31;
    D[2][1] = D32;
    D[2][2] = D33;
  }

  q1 = _vf->fieldVar(S_q1);
  if (spaceDim>=2) q2 = _vf->fieldVar(S_q2);
  if (spaceDim==3) q3 = _vf->fieldVar(S_q3);
  // vector<VarPtr> q(spaceDim);
  // q[0] = q1;
  // if (spaceDim>=2) q[1] = q2;
  // if (spaceDim==3) q[2] = q3;

  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  if (! _spaceTime)
  {
    Space uHatSpace = useConformingTraces ? HGRAD : L2;
    if (spaceDim > 0) u1_hat = _vf->traceVar(S_u1_hat, one * u1, uHatSpace);
    if (spaceDim > 1) u2_hat = _vf->traceVar(S_u2_hat, one * u2, uHatSpace);
    if (spaceDim > 2) u3_hat = _vf->traceVar(S_u3_hat, one * u3, uHatSpace);
  }
  else
  {
    Space uHatSpace = useConformingTraces ? HGRAD_SPACE_L2_TIME : L2;
    if (spaceDim > 0) u1_hat = _vf->traceVarSpaceOnly(S_u1_hat, one * u1, uHatSpace);
    if (spaceDim > 1) u2_hat = _vf->traceVarSpaceOnly(S_u2_hat, one * u2, uHatSpace);
    if (spaceDim > 2) u3_hat = _vf->traceVarSpaceOnly(S_u3_hat, one * u3, uHatSpace);
  }

  Space THatSpace = useConformingTraces ? HGRAD_SPACE_L2_TIME : L2;
  T_hat = _vf->traceVarSpaceOnly(S_T_hat, one * T, THatSpace);

  // FunctionPtr n = Function::normal();
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();

  // Too complicated at the moment to define where these other trace variables comes from
  tc = _vf->fluxVar(S_tc);
  tm1 = _vf->fluxVar(S_tm1);
  if (spaceDim >= 2) tm2 = _vf->fluxVar(S_tm2);
  if (spaceDim == 3) tm3 = _vf->fluxVar(S_tm3);
  te = _vf->fluxVar(S_te);

  vc = _vf->testVar(S_vc, HGRAD);
  vm1 = _vf->testVar(S_vm1, HGRAD);
  if (spaceDim >= 2) vm2 = _vf->testVar(S_vm2, HGRAD);
  if (spaceDim == 3) vm3 = _vf->testVar(S_vm3, HGRAD);
  ve = _vf->testVar(S_ve, HGRAD);

  if (spaceDim == 1)
    S1 = _vf->testVar(S_S1, HGRAD);
  else
    S1 = _vf->testVar(S_S1, HDIV);
    // S1 = _vf->testVar(S_S1, HGRAD);
  if (spaceDim >= 2) S2 = _vf->testVar(S_S2, HDIV);
  // if (spaceDim >= 2) S2 = _vf->testVar(S_S2, HGRAD);
  if (spaceDim == 3) S3 = _vf->testVar(S_S3, HDIV);

  // vector<VarPtr> S(spaceDim,VarPtr);
  // S[0] = S1;
  // if (spaceDim >= 2) S[1] = S2;
  // if (spaceDim == 3) S[2] = S3;

  if (spaceDim == 1)
    tau = _vf->testVar(S_tau, HGRAD);
  else
    tau = _vf->testVar(S_tau, HDIV);
    // tau = _vf->testVar(S_tau, HGRAD);

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

  // FunctionPtr beta;
  FunctionPtr beta_x = Function::constant(1);
  FunctionPtr beta_y = Function::zero();
  FunctionPtr beta_z = Function::zero();
  if (spaceDim == 1)
    _beta = beta_x;
  else if (spaceDim == 2)
    _beta = Function::vectorize(beta_x, beta_y);
  else if (spaceDim == 3)
    _beta = Function::vectorize(beta_x, beta_y, beta_z);

  _bf = Teuchos::rcp( new BF(_vf) );
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

  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
  BCPtr bc = BC::bc();

  _backgroundFlow = Solution::solution(_bf, mesh, bc);
  _solnIncrement = Solution::solution(_bf, mesh, bc);

  // Project ones as initial guess
  map<int, FunctionPtr> initialGuess;
  // FunctionPtr one = Function::constant(1);
  initialGuess[this->rho()->ID()] = one;
  initialGuess[this->u(1)->ID()] = one;
  if (_spaceDim > 1)
    initialGuess[this->u(2)->ID()] = one;
  if (_spaceDim > 2)
    initialGuess[this->u(3)->ID()] = one;
  initialGuess[this->T()->ID()] = one;
  _backgroundFlow->projectOntoMesh(initialGuess);

  // // Project ones as initial guess
  // map<int, FunctionPtr> initialGuess;
  // initialGuess[this->rho()->ID()] = 2*one;
  // initialGuess[this->u(1)->ID()] = 2*one;
  // if (_spaceDim > 1)
  //   initialGuess[this->u(2)->ID()] = 2*one;
  // if (_spaceDim > 2)
  //   initialGuess[this->u(3)->ID()] = 2*one;
  // initialGuess[this->T()->ID()] = 2*one;
  // _backgroundFlow->projectOntoMesh(initialGuess);

  // Previous solution values
  FunctionPtr rho_prev;
  FunctionPtr u1_prev, u2_prev, u3_prev;
  FunctionPtr T_prev;
  FunctionPtr D11_prev, D12_prev, D13_prev, D21_prev, D22_prev, D23_prev, D31_prev, D32_prev, D33_prev;
  FunctionPtr q1_prev, q2_prev, q3_prev;
  switch (_spaceDim)
  {
    case 1:
      rho_prev = Function::solution(this->rho(), _backgroundFlow);
      u1_prev = Function::solution(this->u(1), _backgroundFlow);
      T_prev  = Function::solution(this->T(), _backgroundFlow);
      D11_prev = Function::solution(this->D(1,1), _backgroundFlow);
      q1_prev = Function::solution(this->q(1), _backgroundFlow);
      break;
    case 2:
      rho_prev = Function::solution(this->rho(), _backgroundFlow);
      u1_prev = Function::solution(this->u(1), _backgroundFlow);
      u2_prev = Function::solution(this->u(2), _backgroundFlow);
      T_prev  = Function::solution(this->T(), _backgroundFlow);
      D11_prev = Function::solution(this->D(1,1), _backgroundFlow);
      D12_prev = Function::solution(this->D(1,2), _backgroundFlow);
      D21_prev = Function::solution(this->D(2,1), _backgroundFlow);
      D22_prev = Function::solution(this->D(2,2), _backgroundFlow);
      q1_prev = Function::solution(this->q(1), _backgroundFlow);
      q2_prev = Function::solution(this->q(2), _backgroundFlow);
      break;
    case 3:
      rho_prev = Function::solution(this->rho(), _backgroundFlow);
      u1_prev = Function::solution(this->u(1), _backgroundFlow);
      u2_prev = Function::solution(this->u(2), _backgroundFlow);
      u3_prev = Function::solution(this->u(3), _backgroundFlow);
      T_prev  = Function::solution(this->T(), _backgroundFlow);
      D11_prev = Function::solution(this->D(1,1), _backgroundFlow);
      D12_prev = Function::solution(this->D(1,2), _backgroundFlow);
      D13_prev = Function::solution(this->D(1,3), _backgroundFlow);
      D21_prev = Function::solution(this->D(2,1), _backgroundFlow);
      D22_prev = Function::solution(this->D(2,2), _backgroundFlow);
      D23_prev = Function::solution(this->D(2,3), _backgroundFlow);
      D31_prev = Function::solution(this->D(3,1), _backgroundFlow);
      D32_prev = Function::solution(this->D(3,2), _backgroundFlow);
      D33_prev = Function::solution(this->D(3,3), _backgroundFlow);
      q1_prev = Function::solution(this->q(1), _backgroundFlow);
      q2_prev = Function::solution(this->q(2), _backgroundFlow);
      q3_prev = Function::solution(this->q(3), _backgroundFlow);
      break;

    default:
      break;
  }

  // S terms:
  switch (_spaceDim)
  {
    case 1:
      _bf->addTerm(u1, S1->dx()); // D1 = _mu * grad u1
      _bf->addTerm(1./_mu * D11, S1); // (D1, S1)
      _bf->addTerm(-u1_hat, S1*n_x->x());

      _rhs->addTerm(-u1_prev * S1->dx()); // D1 = _mu * grad u1
      _rhs->addTerm(-1./_mu * D11_prev * S1); // (D1, S1)
      break;
    case 2:
      _bf->addTerm(u1, S1->div()); // D1 = _mu * grad u1
      _bf->addTerm(u2, S2->div()); // D2 = _mu * grad u2
      // _bf->addTerm(u1, S1->x()->dx() + S1->y()->dy()); // D1 = _mu * grad u1
      // _bf->addTerm(u2, S2->x()->dx() + S2->y()->dy()); // D2 = _mu * grad u2
      _bf->addTerm(1./_mu * D11, S1->x()); // (D1, S1)
      _bf->addTerm(1./_mu * D12, S1->y());
      _bf->addTerm(1./_mu * D21, S2->x()); // (D2, S2)
      _bf->addTerm(1./_mu * D22, S2->y());
      _bf->addTerm(-u1_hat, S1*n_x);
      _bf->addTerm(-u2_hat, S2*n_x);

      _rhs->addTerm(-u1_prev * S1->div()); // D1 = _mu * grad u1
      _rhs->addTerm(-u2_prev * S2->div()); // D2 = _mu * grad u2
      _rhs->addTerm(-1./_mu * D11_prev * S1->x()); // (D1, S1)
      _rhs->addTerm(-1./_mu * D12_prev * S1->y());
      _rhs->addTerm(-1./_mu * D21_prev * S2->x()); // (D2, S2)
      _rhs->addTerm(-1./_mu * D22_prev * S2->y());
      break;
    case 3:
      _bf->addTerm(u1, S1->div()); // D1 = _mu * grad u1
      _bf->addTerm(u2, S2->div()); // D2 = _mu * grad u2
      _bf->addTerm(u3, S3->div()); // D3 = _mu * grad u3
      _bf->addTerm(1./_mu * D11, S1->x()); // (D1, S1)
      _bf->addTerm(1./_mu * D12, S1->y());
      _bf->addTerm(1./_mu * D13, S1->z());
      _bf->addTerm(1./_mu * D21, S2->x()); // (D2, S2)
      _bf->addTerm(1./_mu * D22, S2->y());
      _bf->addTerm(1./_mu * D23, S2->z());
      _bf->addTerm(1./_mu * D31, S3->x()); // (D3, S3)
      _bf->addTerm(1./_mu * D32, S3->y());
      _bf->addTerm(1./_mu * D33, S3->z());
      _bf->addTerm(-u1_hat, S1*n_x);
      _bf->addTerm(-u2_hat, S2*n_x);
      _bf->addTerm(-u3_hat, S3*n_x);

      _rhs->addTerm(-u1_prev * S1->div()); // D1 = _mu * grad u1
      _rhs->addTerm(-u2_prev * S2->div()); // D2 = _mu * grad u2
      _rhs->addTerm(-u3_prev * S3->div()); // D3 = _mu * grad u3
      _rhs->addTerm(-1./_mu * D11_prev * S1->x()); // (D1, S1)
      _rhs->addTerm(-1./_mu * D12_prev * S1->y());
      _rhs->addTerm(-1./_mu * D13_prev * S1->z());
      _rhs->addTerm(-1./_mu * D21_prev * S2->x()); // (D2, S2)
      _rhs->addTerm(-1./_mu * D22_prev * S2->y());
      _rhs->addTerm(-1./_mu * D23_prev * S2->z());
      _rhs->addTerm(-1./_mu * D31_prev * S3->x()); // (D3, S3)
      _rhs->addTerm(-1./_mu * D32_prev * S3->y());
      _rhs->addTerm(-1./_mu * D33_prev * S3->z());
      break;
    default:
      break;
  }

  // tau terms:
  switch (_spaceDim)
  {
    case 1:
      _bf->addTerm(-T, tau->dx()); // tau = Cp*mu/Pr * grad T
      _bf->addTerm(Pr()/(Cp()*mu()) * q1, tau); // (D1, S1)
      _bf->addTerm(T_hat, tau*n_x->x());

      _rhs->addTerm( T_prev * tau->dx()); // tau = Cp*_mu/Pr * grad T
      _rhs->addTerm(-Pr()/(Cp()*mu()) * q1_prev * tau); // (D1, S1)
      break;
    case 2:
      _bf->addTerm(-T, tau->div()); // tau = Cp*mu/Pr * grad T
      _bf->addTerm(Pr()/(Cp()*mu()) * q1, tau->x()); // (D1, S1)
      _bf->addTerm(Pr()/(Cp()*mu()) * q2, tau->y()); // (D1, S1)
      _bf->addTerm(T_hat, tau*n_x);

      _rhs->addTerm( T_prev * tau->div()); // tau = Cp*_mu/Pr * grad T
      _rhs->addTerm(-Pr()/(Cp()*mu()) * q1_prev * tau->x()); // (D1, S1)
      _rhs->addTerm(-Pr()/(Cp()*mu()) * q2_prev * tau->y());
      break;
    case 3:
      _bf->addTerm(-T, tau->div()); // tau = Cp*mu/Pr * grad T
      _bf->addTerm(Pr()/(Cp()*mu()) * q1, tau->x()); // (D1, S1)
      _bf->addTerm(Pr()/(Cp()*mu()) * q2, tau->y());
      _bf->addTerm(Pr()/(Cp()*mu()) * q3, tau->z());
      _bf->addTerm(T_hat, tau*n_x);

      _rhs->addTerm( T_prev * tau->div()); // tau = Cp*_mu/Pr * grad T
      _rhs->addTerm(-Pr()/(Cp()*mu()) * q1_prev * tau->x()); // (D1, S1)
      _rhs->addTerm(-Pr()/(Cp()*mu()) * q2_prev * tau->y());
      _rhs->addTerm(-Pr()/(Cp()*mu()) * q3_prev * tau->z());
      break;
    default:
      break;
  }

  // if (_spaceTime)
  //   _bf->addTerm(-rho, vc->dt());
  // _bf->addTerm(-beta_x*rho, vc->dx());
  // if (_spaceDim >= 2) _bf->addTerm(-beta_y*rho, vc->dy());
  // if (_spaceDim == 3) _bf->addTerm(-beta_z*rho, vc->dz());
  // _bf->addTerm(tc, vc);
  // vc:
  switch (_spaceDim)
  {
    case 1:
      if (_spaceTime)
        _bf->addTerm(-rho, vc->dt());
      _bf->addTerm(-(u1_prev*rho+rho_prev*u1), vc->dx());
      _bf->addTerm(tc, vc);

      if (_spaceTime)
        _rhs->addTerm( rho_prev * vc->dt());
      _rhs->addTerm( rho_prev*u1_prev * vc->dx());
      break;
    case 2:
      if (_spaceTime)
        _bf->addTerm(-rho, vc->dt());
      _bf->addTerm(-(u1_prev*rho+rho_prev*u1), vc->dx());
      _bf->addTerm(-(u2_prev*rho+rho_prev*u2), vc->dy());
      _bf->addTerm(tc, vc);

      if (_spaceTime)
        _rhs->addTerm( rho_prev * vc->dt());
      _rhs->addTerm( rho_prev*u1_prev * vc->dx());
      _rhs->addTerm( rho_prev*u2_prev * vc->dy());
      break;
    case 3:
      if (_spaceTime)
        _bf->addTerm(-rho, vc->dt());
      _bf->addTerm(-(u1_prev*rho+rho_prev*u1), vc->dx());
      _bf->addTerm(-(u2_prev*rho+rho_prev*u2), vc->dy());
      _bf->addTerm(-(u3_prev*rho+rho_prev*u3), vc->dz());
      _bf->addTerm(tc, vc);

      if (_spaceTime)
        _rhs->addTerm( rho_prev * vc->dt());
      _rhs->addTerm( rho_prev*u1_prev * vc->dx());
      _rhs->addTerm( rho_prev*u2_prev * vc->dy());
      _rhs->addTerm( rho_prev*u3_prev * vc->dz());
      break;
    default:
      break;
  }

  // vm
  switch (_spaceDim)
  {
    case 1:
      if (_spaceTime)
      {
        _bf->addTerm(-(u1_prev*rho+rho_prev*u1), vm1->dt());
      }
      _bf->addTerm(-u1_prev*u1_prev*rho, vm1->dx());
      _bf->addTerm(-rho_prev*u1_prev*u1, vm1->dx());
      _bf->addTerm(-rho_prev*u1_prev*u1, vm1->dx());
      _bf->addTerm(-R()*T_prev*rho, vm1->dx());
      _bf->addTerm(-R()*rho_prev*T, vm1->dx());
      _bf->addTerm( D11+D11-2./3*D11, vm1->dx());
      _bf->addTerm(tm1, vm1);

      if (_spaceTime)
      {
        _rhs->addTerm(rho_prev*u1_prev * vm1->dt());
      }
      _rhs->addTerm( rho_prev*u1_prev*u1_prev * vm1->dx() );
      _rhs->addTerm( R()*rho_prev*T_prev * vm1->dx() );
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev) * vm1->dx() );
      break;
    case 2:
      if (_spaceTime)
      {
        _bf->addTerm(-(u1_prev*rho+rho_prev*u1), vm1->dt());
        _bf->addTerm(-(u2_prev*rho+rho_prev*u2), vm2->dt());
      }
      _bf->addTerm(-u1_prev*u1_prev*rho, vm1->dx());
      _bf->addTerm(-u1_prev*u2_prev*rho, vm1->dy());
      _bf->addTerm(-u2_prev*u1_prev*rho, vm2->dx());
      _bf->addTerm(-u2_prev*u2_prev*rho, vm2->dy());
      _bf->addTerm(-rho_prev*u1_prev*u1, vm1->dx());
      _bf->addTerm(-rho_prev*u1_prev*u2, vm1->dy());
      _bf->addTerm(-rho_prev*u2_prev*u1, vm2->dx());
      _bf->addTerm(-rho_prev*u2_prev*u2, vm2->dy());
      _bf->addTerm(-rho_prev*u1_prev*u1, vm1->dx());
      _bf->addTerm(-rho_prev*u2_prev*u1, vm1->dy());
      _bf->addTerm(-rho_prev*u1_prev*u2, vm2->dx());
      _bf->addTerm(-rho_prev*u2_prev*u2, vm2->dy());
      _bf->addTerm(-R()*T_prev*rho, vm1->dx());
      _bf->addTerm(-R()*rho_prev*T, vm1->dx());
      _bf->addTerm(-R()*T_prev*rho, vm2->dy());
      _bf->addTerm(-R()*rho_prev*T, vm2->dy());
      _bf->addTerm( D11+D11-2./3*D11-2./3*D22, vm1->dx());
      _bf->addTerm( D12+D21, vm1->dy());
      _bf->addTerm( D21+D12, vm2->dx());
      _bf->addTerm( D22+D22-2./3*D11-2./3*D22, vm2->dy());
      _bf->addTerm(tm1, vm1);
      _bf->addTerm(tm2, vm2);

      if (_spaceTime)
      {
        _rhs->addTerm( rho_prev*u1_prev * vm1->dt() );
        _rhs->addTerm( rho_prev*u2_prev * vm2->dt() );
      }
      _rhs->addTerm( rho_prev*u1_prev*u1_prev * vm1->dx());
      _rhs->addTerm( rho_prev*u1_prev*u2_prev * vm1->dy());
      _rhs->addTerm( rho_prev*u2_prev*u1_prev * vm2->dx());
      _rhs->addTerm( rho_prev*u2_prev*u2_prev * vm2->dy());
      _rhs->addTerm( R()*rho_prev*T_prev * vm1->dx());
      _rhs->addTerm( R()*rho_prev*T_prev * vm2->dy());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev-2./3*D22_prev) * vm1->dx());
      _rhs->addTerm(-(D12_prev+D21_prev) * vm1->dy());
      _rhs->addTerm(-(D21_prev+D12_prev) * vm2->dx());
      _rhs->addTerm(-(D22_prev+D22_prev-2./3*D11_prev-2./3*D22_prev) * vm2->dy());
      break;
    case 3:
      if (_spaceTime)
      {
        _bf->addTerm(-(u1_prev*rho+rho_prev*u1), vm1->dt());
        _bf->addTerm(-(u2_prev*rho+rho_prev*u2), vm2->dt());
        _bf->addTerm(-(u3_prev*rho+rho_prev*u3), vm3->dt());
      }
      _bf->addTerm(-u1_prev*u1_prev*rho, vm1->dx());
      _bf->addTerm(-u1_prev*u2_prev*rho, vm1->dy());
      _bf->addTerm(-u1_prev*u3_prev*rho, vm1->dz());
      _bf->addTerm(-u2_prev*u1_prev*rho, vm2->dx());
      _bf->addTerm(-u2_prev*u2_prev*rho, vm2->dy());
      _bf->addTerm(-u2_prev*u3_prev*rho, vm2->dz());
      _bf->addTerm(-u3_prev*u1_prev*rho, vm3->dx());
      _bf->addTerm(-u3_prev*u2_prev*rho, vm3->dy());
      _bf->addTerm(-u3_prev*u3_prev*rho, vm3->dz());
      _bf->addTerm(-rho_prev*u1_prev*u1, vm1->dx());
      _bf->addTerm(-rho_prev*u1_prev*u2, vm1->dy());
      _bf->addTerm(-rho_prev*u1_prev*u3, vm1->dz());
      _bf->addTerm(-rho_prev*u2_prev*u1, vm2->dx());
      _bf->addTerm(-rho_prev*u2_prev*u2, vm2->dy());
      _bf->addTerm(-rho_prev*u2_prev*u3, vm2->dz());
      _bf->addTerm(-rho_prev*u3_prev*u1, vm3->dx());
      _bf->addTerm(-rho_prev*u3_prev*u2, vm3->dy());
      _bf->addTerm(-rho_prev*u3_prev*u3, vm3->dz());
      _bf->addTerm(-rho_prev*u1_prev*u1, vm1->dx());
      _bf->addTerm(-rho_prev*u2_prev*u1, vm1->dy());
      _bf->addTerm(-rho_prev*u3_prev*u1, vm1->dz());
      _bf->addTerm(-rho_prev*u1_prev*u2, vm2->dx());
      _bf->addTerm(-rho_prev*u2_prev*u2, vm2->dy());
      _bf->addTerm(-rho_prev*u3_prev*u2, vm2->dz());
      _bf->addTerm(-rho_prev*u1_prev*u3, vm3->dx());
      _bf->addTerm(-rho_prev*u2_prev*u3, vm3->dy());
      _bf->addTerm(-rho_prev*u3_prev*u3, vm3->dz());
      _bf->addTerm(-R()*T_prev*rho, vm1->dx());
      _bf->addTerm(-R()*rho_prev*T, vm1->dx());
      _bf->addTerm(-R()*T_prev*rho, vm2->dy());
      _bf->addTerm(-R()*rho_prev*T, vm2->dy());
      _bf->addTerm(-R()*T_prev*rho, vm3->dz());
      _bf->addTerm(-R()*rho_prev*T, vm3->dz());
      _bf->addTerm( D11+D11-2./3*D11-2./3*D22-2./3*D33, vm1->dx());
      _bf->addTerm( D12+D21, vm1->dy());
      _bf->addTerm( D13+D31, vm1->dz());
      _bf->addTerm( D21+D12, vm2->dx());
      _bf->addTerm( D22+D22-2./3*D11-2./3*D22-2./3*D33, vm2->dy());
      _bf->addTerm( D23+D32, vm2->dz());
      _bf->addTerm( D31+D13, vm3->dx());
      _bf->addTerm( D32+D23, vm3->dy());
      _bf->addTerm( D33+D33-2./3*D11-2./3*D22-2./3*D33, vm3->dz());
      _bf->addTerm(tm1, vm1);
      _bf->addTerm(tm2, vm2);
      _bf->addTerm(tm3, vm3);

      if (_spaceTime)
      {
        _rhs->addTerm( rho_prev*u1_prev * vm1->dt() );
        _rhs->addTerm( rho_prev*u2_prev * vm2->dt() );
        _rhs->addTerm( rho_prev*u3_prev * vm3->dt() );
      }
      _rhs->addTerm( rho_prev*u1_prev*u1_prev * vm1->dx() );
      _rhs->addTerm( rho_prev*u1_prev*u2_prev * vm1->dy() );
      _rhs->addTerm( rho_prev*u1_prev*u3_prev * vm1->dz() );
      _rhs->addTerm( rho_prev*u2_prev*u1_prev * vm2->dx() );
      _rhs->addTerm( rho_prev*u2_prev*u2_prev * vm2->dy() );
      _rhs->addTerm( rho_prev*u2_prev*u3_prev * vm2->dz() );
      _rhs->addTerm( rho_prev*u3_prev*u1_prev * vm3->dx() );
      _rhs->addTerm( rho_prev*u3_prev*u2_prev * vm3->dy() );
      _rhs->addTerm( rho_prev*u3_prev*u3_prev * vm3->dz() );
      _rhs->addTerm( R()*rho_prev*T_prev * vm1->dx());
      _rhs->addTerm( R()*rho_prev*T_prev * vm2->dy());
      _rhs->addTerm( R()*rho_prev*T_prev * vm3->dz());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev-2./3*D22_prev-2./3*D33_prev) * vm1->dx());
      _rhs->addTerm(-(D12_prev+D21_prev) * vm1->dy());
      _rhs->addTerm(-(D13_prev+D31_prev) * vm1->dz());
      _rhs->addTerm(-(D21_prev+D12_prev) * vm2->dx());
      _rhs->addTerm(-(D22_prev+D22_prev-2./3*D11_prev-2./3*D22_prev-2./3*D33_prev) * vm2->dy());
      _rhs->addTerm(-(D23_prev+D32_prev) * vm2->dz());
      _rhs->addTerm(-(D31_prev+D13_prev) * vm3->dx());
      _rhs->addTerm(-(D32_prev+D23_prev) * vm3->dy());
      _rhs->addTerm(-(D33_prev+D33_prev-2./3*D11_prev-2./3*D22_prev-2./3*D33_prev) * vm3->dz());
    default:
      break;
  }
  // // vm1:
  // if (_spaceTime)
  //   _bf->addTerm(-u1, vm1->dt());
  // _bf->addTerm(-beta_x*u1 + D11, vm1->dx());
  // if (_spaceDim >= 2) _bf->addTerm(-beta_y*u1 + D12, vm1->dy());
  // if (_spaceDim == 3) _bf->addTerm(-beta_z*u1 + D13, vm1->dz());
  // _bf->addTerm(tm1, vm1);

  // // vm2:
  // if (_spaceDim >= 2)
  // {
  //   if (_spaceTime)
  //     _bf->addTerm(-u2, vm2->dt());
  //   _bf->addTerm(-beta_x*u2 + D21, vm2->dx());
  //   _bf->addTerm(-beta_y*u2 + D22, vm2->dy());
  //   if (_spaceDim == 3) _bf->addTerm(-beta_z*u2 + D23, vm2->dz());
  //   _bf->addTerm(tm2, vm2);
  // }

  // // vm3:
  // if (_spaceDim == 3)
  // {
  //   if (_spaceTime)
  //     _bf->addTerm(-u3, vm3->dt());
  //   _bf->addTerm(-beta_x*u3 + D31, vm3->dx());
  //   _bf->addTerm(-beta_y*u3 + D32, vm3->dy());
  //   _bf->addTerm(-beta_z*u3 + D33, vm3->dz());
  //   _bf->addTerm(tm3, vm3);
  // }

  // ve:
  // if (_spaceTime)
  //   _bf->addTerm(-T, ve->dt());
  // _bf->addTerm(-beta_x*T + q1, ve->dx());
  // if (_spaceDim >= 2) _bf->addTerm(-beta_y*T + q2, ve->dy());
  // if (_spaceDim == 3) _bf->addTerm(-beta_z*T + q3, ve->dz());
  // _bf->addTerm(te, ve);
  switch (_spaceDim)
  {
    case 1:
      if (_spaceTime)
      {
        _bf->addTerm(-(Cv()*T_prev*rho+Cv()*rho_prev*T), ve->dt());
        _bf->addTerm(-0.5*u1_prev*u1_prev*rho, ve->dt());
        _bf->addTerm(-rho_prev*u1_prev*u1, ve->dt());
      }
      _bf->addTerm(-(Cv()*u1_prev*T_prev*rho+Cv()*rho_prev*T_prev*u1+Cv()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(0.5*u1_prev*u1_prev*u1_prev*rho), ve->dx());
      _bf->addTerm(-(0.5*rho_prev*u1_prev*u1_prev*u1), ve->dx());
      _bf->addTerm(-(rho_prev*u1_prev*u1_prev*u1), ve->dx());
      _bf->addTerm(-(R()*rho_prev*T_prev*u1), ve->dx());
      _bf->addTerm(-(R()*u1_prev*T_prev*rho), ve->dx());
      _bf->addTerm(-(R()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-q1, ve->dx());
      _bf->addTerm((D11_prev+D11_prev-2./3*D11_prev)*u1, ve->dx());
      _bf->addTerm(u1_prev*(D11+D11-2./3*D11), ve->dx());
      _bf->addTerm(te, ve);

      if (_spaceTime)
      {
        _rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
        _rhs->addTerm(0.5*rho_prev*u1_prev*u1_prev * ve->dt());
      }
      _rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(0.5*rho_prev*u1_prev*u1_prev*u1_prev * ve->dx());
      _rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(q1_prev * ve->dx());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev)*u1_prev * ve->dx());
      _rhs->addTerm(-u1_prev*(D11_prev+D11_prev-2./3*D11_prev) * ve->dx());
      break;
    case 2:
      if (_spaceTime)
      {
        _bf->addTerm(-(Cv()*T_prev*rho+Cv()*rho_prev*T), ve->dt());
        _bf->addTerm(-0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*rho, ve->dt());
        _bf->addTerm(-rho_prev*(u1_prev*u1+u2_prev*u2), ve->dt());
      }
      _bf->addTerm(-(Cv()*u1_prev*T_prev*rho+Cv()*rho_prev*T_prev*u1+Cv()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*u1_prev*rho), ve->dx());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u1), ve->dx());
      _bf->addTerm(-(rho_prev*u1_prev*(u1_prev*u1+u2_prev*u2)), ve->dx());
      _bf->addTerm(-(R()*rho_prev*T_prev*u1), ve->dx());
      _bf->addTerm(-(R()*u1_prev*T_prev*rho), ve->dx());
      _bf->addTerm(-(R()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(Cv()*u2_prev*T_prev*rho+Cv()*rho_prev*T_prev*u2+Cv()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev)*u2_prev*rho), ve->dy());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u2), ve->dy());
      _bf->addTerm(-(rho_prev*u2_prev*(u1_prev*u1+u2_prev*u2)), ve->dy());
      _bf->addTerm(-(R()*rho_prev*T_prev*u2), ve->dy());
      _bf->addTerm(-(R()*u2_prev*T_prev*rho), ve->dy());
      _bf->addTerm(-(R()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-q1, ve->dx());
      _bf->addTerm(-q2, ve->dy());
      _bf->addTerm((D11_prev+D11_prev-2./3*(D11_prev+D22_prev))*u1, ve->dx());
      _bf->addTerm((D12_prev+D21_prev)*u2, ve->dx());
      _bf->addTerm((D21_prev+D12_prev)*u1, ve->dy());
      _bf->addTerm((D22_prev+D22_prev-2./3*(D11_prev+D22_prev))*u2, ve->dy());
      _bf->addTerm(u1_prev*(1*D11+1*D11-2./3*D11-2./3*D22), ve->dx());
      _bf->addTerm(u2_prev*(1*D12+1*D21), ve->dx());
      _bf->addTerm(u1_prev*(1*D21+1*D12), ve->dy());
      _bf->addTerm(u2_prev*(1*D22+1*D22-2./3*D11-2./3*D22), ve->dy());
      _bf->addTerm(te, ve);

      if (_spaceTime)
      {
        _rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
        _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev) * ve->dt());
      }
      _rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u1_prev * ve->dx());
      _rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(Cv()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u2_prev * ve->dy());
      _rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(q1_prev * ve->dx());
      _rhs->addTerm(q2_prev * ve->dy());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*(D11_prev+D22_prev))*u1_prev * ve->dx());
      _rhs->addTerm(-(D12_prev+D21_prev)*u2_prev * ve->dx());
      _rhs->addTerm(-(D21_prev+D12_prev)*u1_prev * ve->dy());
      _rhs->addTerm(-(D22_prev+D22_prev-2./3*(D11_prev+D22_prev))*u2_prev * ve->dy());
      break;
    case 3:
      if (_spaceTime)
      {
        _bf->addTerm(-(Cv()*T_prev*rho+Cv()*rho_prev*T), ve->dt());
        _bf->addTerm(-0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*rho, ve->dt());
        _bf->addTerm(-rho_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3), ve->dt());
      }
      _bf->addTerm(-(Cv()*u1_prev*T_prev*rho+Cv()*rho_prev*T_prev*u1+Cv()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(Cv()*u2_prev*T_prev*rho+Cv()*rho_prev*T_prev*u2+Cv()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-(Cv()*u3_prev*T_prev*rho+Cv()*rho_prev*T_prev*u3+Cv()*rho_prev*u3_prev*T), ve->dz());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1_prev*rho), ve->dx());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2_prev*rho), ve->dy());
      _bf->addTerm(-(0.5*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3_prev*rho), ve->dz());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1), ve->dx());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2), ve->dy());
      _bf->addTerm(-(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3), ve->dz());
      _bf->addTerm(-(rho_prev*u1_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3)), ve->dx());
      _bf->addTerm(-(rho_prev*u2_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3)), ve->dy());
      _bf->addTerm(-(rho_prev*u3_prev*(u1_prev*u1+u2_prev*u2+u3_prev*u3)), ve->dz());
      _bf->addTerm(-(R()*rho_prev*T_prev*u1), ve->dx());
      _bf->addTerm(-(R()*rho_prev*T_prev*u2), ve->dy());
      _bf->addTerm(-(R()*rho_prev*T_prev*u3), ve->dz());
      _bf->addTerm(-(R()*u1_prev*T_prev*rho), ve->dx());
      _bf->addTerm(-(R()*u2_prev*T_prev*rho), ve->dy());
      _bf->addTerm(-(R()*u3_prev*T_prev*rho), ve->dz());
      _bf->addTerm(-(R()*rho_prev*u1_prev*T), ve->dx());
      _bf->addTerm(-(R()*rho_prev*u2_prev*T), ve->dy());
      _bf->addTerm(-(R()*rho_prev*u3_prev*T), ve->dz());
      _bf->addTerm(-q1, ve->dx());
      _bf->addTerm(-q2, ve->dy());
      _bf->addTerm(-q3, ve->dz());
      _bf->addTerm((D11_prev+D11_prev-2./3*(D11_prev+D22_prev+D33_prev))*u1, ve->dx());
      _bf->addTerm((D12_prev+D21_prev)*u2, ve->dx());
      _bf->addTerm((D13_prev+D31_prev)*u3, ve->dx());
      _bf->addTerm((D21_prev+D12_prev)*u1, ve->dy());
      _bf->addTerm((D22_prev+D22_prev-2./3*(D11_prev+D22_prev+D33_prev))*u2, ve->dy());
      _bf->addTerm((D31_prev+D13_prev)*u3, ve->dy());
      _bf->addTerm((D31_prev+D13_prev)*u1, ve->dz());
      _bf->addTerm((D32_prev+D23_prev)*u2, ve->dz());
      _bf->addTerm((D33_prev+D33_prev-2./3*(D11_prev+D22_prev+D33_prev))*u3, ve->dz());
      _bf->addTerm(u1_prev*(D11+D11-2./3*D11-2./3*D22-2./3*D33), ve->dx());
      _bf->addTerm(u2_prev*(D12+D21), ve->dx());
      _bf->addTerm(u3_prev*(D13+D31), ve->dx());
      _bf->addTerm(u1_prev*(D21+D12), ve->dy());
      _bf->addTerm(u2_prev*(D22+D22-2./3*D11-2./3*D22-2./3*D33), ve->dy());
      _bf->addTerm(u3_prev*(D31+D13), ve->dy());
      _bf->addTerm(u1_prev*(D31+D13), ve->dz());
      _bf->addTerm(u2_prev*(D32+D23), ve->dz());
      _bf->addTerm(u3_prev*(D33+D33-2./3*D11-2./3*D22-2./3*D33), ve->dz());
      _bf->addTerm(te, ve);

      if (_spaceTime)
      {
        _rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
        _rhs->addTerm(-0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev) * ve->dt());
      }
      _rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(Cv()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(Cv()*rho_prev*u3_prev*T_prev * ve->dz());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1_prev * ve->dx());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2_prev * ve->dy());
      _rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3_prev * ve->dz());
      _rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
      _rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
      _rhs->addTerm(R()*rho_prev*u3_prev*T_prev * ve->dz());
      _rhs->addTerm(q1_prev * ve->dx());
      _rhs->addTerm(q2_prev * ve->dy());
      _rhs->addTerm(q3_prev * ve->dz());
      _rhs->addTerm(-(D11_prev+D11_prev-2./3*(D11_prev+D22_prev+D33_prev))*u1_prev * ve->dx());
      _rhs->addTerm(-(D12_prev+D21_prev)*u2_prev * ve->dx());
      _rhs->addTerm(-(D13_prev+D31_prev)*u3_prev * ve->dx());
      _rhs->addTerm(-(D21_prev+D12_prev)*u1_prev * ve->dy());
      _rhs->addTerm(-(D22_prev+D22_prev-2./3*(D11_prev+D22_prev+D33_prev))*u2_prev * ve->dy());
      _rhs->addTerm(-(D31_prev+D13_prev)*u3_prev * ve->dy());
      _rhs->addTerm(-(D31_prev+D13_prev)*u1_prev * ve->dz());
      _rhs->addTerm(-(D32_prev+D23_prev)*u2_prev * ve->dz());
      _rhs->addTerm(-(D33_prev+D33_prev-2./3*(D11_prev+D22_prev+D33_prev))*u3_prev * ve->dz());
      break;
    default:
      break;
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

  LinearTermPtr adj_Cc = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Cm1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Cm2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Cm3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Ce = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fc = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fm1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fm2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fm3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Fe = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD11 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD12 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD13 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD21 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD22 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD23 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD31 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD32 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_KD33 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Kq1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Kq2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Kq3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD11 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD12 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD13 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD21 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD22 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD23 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD31 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD32 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_MD33 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Mq1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Mq2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Mq3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gc = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gm1 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gm2 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Gm3 = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_Ge = Teuchos::rcp( new LinearTerm );
  LinearTermPtr adj_vm = Teuchos::rcp( new LinearTerm );

  _ips["Graph"] = _bf->graphNorm();
  // cout << "Graph" << endl;
  // _ips["Graph"]->printInteractions();

  switch (_spaceDim)
  {
    case 1:
      adj_Cc->addTerm( vc->dt() + u1_prev*vm1->dt() + Cv()*T_prev*ve->dt() + 0.5*u1_prev*u1_prev*ve->dt() );
      adj_Cm1->addTerm( rho_prev*vm1->dt() + rho_prev*u1_prev*ve->dt() );
      adj_Ce->addTerm( Cv()*rho_prev*ve->dt() );
      adj_Fc->addTerm( u1_prev*vc->dx() + u1_prev*u1_prev*vm1->dx() + R()*T_prev*vm1->dx() + Cv()*T_prev*u1_prev*ve->dx()
          + 0.5*u1_prev*u1_prev*u1_prev*ve->dx() + R()*T_prev*u1_prev*ve->dx() );
      adj_Fm1->addTerm( rho_prev*vc->dx() + 2*rho_prev*u1_prev*vm1->dx() + Cv()*T_prev*rho_prev*ve->dx()
          + 0.5*rho_prev*u1_prev*u1_prev*ve->dx() + rho_prev*u1_prev*u1_prev*ve->dx() + R()*T_prev*rho_prev*ve->dx()
          - D11_prev*ve->dx() - D11_prev*ve->dx() + 2./3*D11_prev*ve->dx() );
      adj_Fe->addTerm( R()*rho_prev*vm1->dx() + Cv()*rho_prev*u1_prev*ve->dx() + R()*rho_prev*u1_prev*ve->dx() );
      adj_KD11->addTerm( vm1->dx() + vm1->dx() - 2./3*vm1->dx() + u1_prev*ve->dx() + u1_prev*ve->dx() - 2./3*u1_prev*ve->dx() );
      adj_Kq1->addTerm( -ve->dx() );
      adj_MD11->addTerm( 1./mu()*S1 );
      adj_Mq1->addTerm( Pr()/(Cp()*mu())*tau );
      adj_Gm1->addTerm( one*S1->dx() );
      adj_Ge->addTerm( -tau->dx() );

      _ips["ManualGraph"] = Teuchos::rcp(new IP);
      _ips["ManualGraph"]->addTerm( adj_MD11 + adj_KD11 );
      _ips["ManualGraph"]->addTerm( adj_Mq1 + adj_Kq1 );
      if (_spaceTime)
      {
        _ips["ManualGraph"]->addTerm( adj_Gc - adj_Fc - adj_Cc );
        _ips["ManualGraph"]->addTerm( adj_Gm1 - adj_Fm1 - adj_Cm1 );
        _ips["ManualGraph"]->addTerm( adj_Ge - adj_Fe - adj_Ce );
      }
      else
      {
        _ips["ManualGraph"]->addTerm( adj_Gc - adj_Fc );
        _ips["ManualGraph"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["ManualGraph"]->addTerm( adj_Ge - adj_Fe );
      }
      _ips["ManualGraph"]->addTerm( vc );
      _ips["ManualGraph"]->addTerm( vm1 );
      _ips["ManualGraph"]->addTerm( ve );
      _ips["ManualGraph"]->addTerm( S1 );
      _ips["ManualGraph"]->addTerm( tau );

      // cout << endl << "ManualGraph" << endl;
      // _ips["ManualGraph"]->printInteractions();

      _ips["Robust"] = Teuchos::rcp(new IP);
      // _ips["Robust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),Function::constant(1./sqrt(mu())))*mu()*adj_MD11);
      _ips["Robust"]->addTerm( Function::min(one/Function::h(),Function::constant(1./sqrt(mu())))*Cp()*mu()/Pr()*adj_Mq1);
      // _ips["Robust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["Robust"]->addTerm( sqrt(mu())*one*adj_KD11 );
      _ips["Robust"]->addTerm( sqrt(Cp()*mu()/Pr())*one*adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["Robust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["Robust"]->addTerm( adj_Fc + adj_Cc );
        _ips["Robust"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["Robust"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["Robust"]->addTerm(_beta*v->grad());
        _ips["Robust"]->addTerm( adj_Fc );
        _ips["Robust"]->addTerm( adj_Fm1 );
        _ips["Robust"]->addTerm( adj_Fe );
      }
      // _ips["Robust"]->addTerm(tau->div());
      _ips["Robust"]->addTerm( adj_Gc );
      _ips["Robust"]->addTerm( adj_Gm1 );
      _ips["Robust"]->addTerm( adj_Ge );
      // _ips["Robust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      _ips["Robust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );

      _ips["CoupledRobust"] = Teuchos::rcp(new IP);
      // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),Function::constant(1./sqrt(mu())))*mu()*adj_MD11);
      _ips["CoupledRobust"]->addTerm( Function::min(one/Function::h(),Function::constant(1./sqrt(mu())))*Cp()*mu()/Pr()*adj_Mq1);
      // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v->grad());
      _ips["CoupledRobust"]->addTerm( sqrt(mu())*one*adj_KD11 );
      _ips["CoupledRobust"]->addTerm( sqrt(Cp()*mu()/Pr())*one*adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc - adj_Cc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 - adj_Cm1 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe - adj_Ce );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad() + v->dt());
        _ips["CoupledRobust"]->addTerm( adj_Fc + adj_Cc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["CoupledRobust"]->addTerm(tau->div() - beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Gc - adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Gm1 - adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Ge - adj_Fe );
        // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
        _ips["CoupledRobust"]->addTerm( adj_Fc );
        _ips["CoupledRobust"]->addTerm( adj_Fm1 );
        _ips["CoupledRobust"]->addTerm( adj_Fe );
      }
      // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
      _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vc );
      _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*vm1 );
      _ips["CoupledRobust"]->addTerm( Function::min(sqrt(mu())*one/Function::h(),one)*ve );

      _ips["NSDecoupled"] = Teuchos::rcp(new IP);
      // _ips["NSDecoupled"]->addTerm(one/Function::h()*tau);
      _ips["NSDecoupled"]->addTerm( mu()/Function::h()*adj_MD11 );
      _ips["NSDecoupled"]->addTerm( Cp()*mu()/Pr()/Function::h()*adj_Mq1 );
      // _ips["NSDecoupled"]->addTerm(tau->div());
      _ips["NSDecoupled"]->addTerm( adj_KD11 );
      _ips["NSDecoupled"]->addTerm( adj_Kq1 );
      if (_spaceTime)
      {
        // _ips["NSDecoupled"]->addTerm(_beta*v->grad() + v->dt());
        _ips["NSDecoupled"]->addTerm( adj_Fc + adj_Cc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 + adj_Cm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fe + adj_Ce );
      }
      else
      {
        // _ips["NSDecoupled"]->addTerm(_beta*v->grad());
        _ips["NSDecoupled"]->addTerm( adj_Fc );
        _ips["NSDecoupled"]->addTerm( adj_Fm1 );
        _ips["NSDecoupled"]->addTerm( adj_Fe );
      }
      // _ips["NSDecoupled"]->addTerm(v->grad());
      _ips["NSDecoupled"]->addTerm( adj_Gc );
      _ips["NSDecoupled"]->addTerm( adj_Gm1 );
      _ips["NSDecoupled"]->addTerm( adj_Ge );
      // _ips["NSDecoupled"]->addTerm(v);
      _ips["NSDecoupled"]->addTerm( vc );
      _ips["NSDecoupled"]->addTerm( vm1 );
      _ips["NSDecoupled"]->addTerm( ve );
      break;
    case 2:
    _ips["Robust"] = Teuchos::rcp(new IP);
      break;
    case 3:
      break;
  }

  IPPtr ip = _ips.at(normName);

  // set the inner product to the graph norm:
  // setIP( _ips[normName] );

  // this->setForcingFunction(Teuchos::null); // will default to zero
  // _rhsForSolve = this->rhs(_neglectFluxesOnRHS);
  // _rhsForResidual = this->rhs(false);
  // _solnIncrement->setRHS(_rhsForSolve);

  // _solnIncrement->setBC(bc);
  _solnIncrement->setRHS(_rhs);
  _solnIncrement->setIP(ip);
  // _solnIncrement->setRHS(rhs);

  mesh->registerSolution(_backgroundFlow);
  mesh->registerSolution(_solnIncrement);

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
  FunctionPtr T_incr = Function::solution(T, _solnIncrement);

  _L2IncrementFunction = rho_incr * rho_incr + T_incr * T_incr;
  _L2SolutionFunction = rho_prev * rho_prev + T_prev * T_prev;
  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    FunctionPtr u_i_incr = Function::solution(this->u(comp_i), _solnIncrement);
    FunctionPtr u_i_prev = Function::solution(this->u(comp_i), _backgroundFlow);
    FunctionPtr q_i_incr = Function::solution(this->q(comp_i), _solnIncrement);
    FunctionPtr q_i_prev = Function::solution(this->q(comp_i), _backgroundFlow);

    _L2IncrementFunction = _L2IncrementFunction + u_i_incr * u_i_incr;
    _L2SolutionFunction = _L2SolutionFunction + u_i_prev * u_i_prev;
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

  _solver = Solver::getDirectSolver();

  _nonlinearIterationCount = 0;

}

void CompressibleNavierStokesFormulation::addVelocityTraceCondition(SpatialFilterPtr region, FunctionPtr u_exact)
{
  VarPtr u1_hat, u2_hat, u3_hat;
  u1_hat = this->u_hat(1);
  if (_spaceDim>=2) u2_hat = this->u_hat(2);
  if (_spaceDim==3) u3_hat = this->u_hat(3);

  if (_neglectFluxesOnRHS)
  {
    if (_spaceDim==1) _solnIncrement->bc()->addDirichlet(u1_hat, region, u_exact);
    else _solnIncrement->bc()->addDirichlet(u1_hat, region, u_exact->x());
    if (_spaceDim>=2) _solnIncrement->bc()->addDirichlet(u2_hat, region, u_exact->y());
    if (_spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, region, u_exact->z());
  }
  else
  {
    // SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );
    SolutionPtr backgroundFlowWeakReference = _backgroundFlow;

    FunctionPtr u1_hat_prev, u2_hat_prev, u3_hat_prev;
    u1_hat_prev = Function::solution(u1_hat,backgroundFlowWeakReference);
    if (_spaceDim >= 2) u2_hat_prev = Function::solution(u2_hat,backgroundFlowWeakReference);
    if (_spaceDim == 3) u3_hat_prev = Function::solution(u3_hat,backgroundFlowWeakReference);

    if (_spaceDim==1) _solnIncrement->bc()->addDirichlet(u1_hat, region, u_exact - u1_hat_prev);
    else _solnIncrement->bc()->addDirichlet(u1_hat, region, u_exact->x() - u1_hat_prev);
    if (_spaceDim>=2) _solnIncrement->bc()->addDirichlet(u2_hat, region, u_exact->y() - u2_hat_prev);
    if (_spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, region, u_exact->z() - u3_hat_prev);
  }
}

void CompressibleNavierStokesFormulation::addTemperatureTraceCondition(SpatialFilterPtr region, FunctionPtr T_exact)
{
  VarPtr T_hat = this->T_hat();

  if (_neglectFluxesOnRHS)
  {
    _solnIncrement->bc()->addDirichlet(T_hat, region, T_exact);
  }
  else
  {
    // SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );
    SolutionPtr backgroundFlowWeakReference = _backgroundFlow;
    FunctionPtr T_hat_prev = Function::solution(T_hat,backgroundFlowWeakReference);
    _solnIncrement->bc()->addDirichlet(T_hat, region, T_exact - T_hat_prev);
  }
}

void CompressibleNavierStokesFormulation::addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr tc = this->tc();

  // if (_spaceTime)
  //   n = Function::normalSpaceTime();
  // else
  //   n = Function::normal();
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  FunctionPtr n_x, n_y, n_z, n_t;
  n_x = n->x();
  if (_spaceDim>=2) n = n_x->y();
  if (_spaceDim==3) n = n_x->z();
  if (_spaceTime) n_t = n_xt->t();

  // FunctionPtr beta_x, beta_y, beta_z;
  // if (_spaceDim == 1)
  //   beta_x = _beta;
  // else
  //   beta_x = _beta->x();
  // if (_spaceDim >= 2) beta_y = _beta->y();
  // if (_spaceDim == 3) beta_z = _beta->z();

  FunctionPtr tc_exact;
  switch (_spaceDim)
  {
    case 1:
      tc_exact = rho_exact*u_exact*n_x;
      break;
    case 2:
      tc_exact = rho_exact*u_exact->x()*n_x + rho_exact*u_exact->y()*n_y;
      break;
    case 3:
      tc_exact = rho_exact*u_exact->x()*n_x + rho_exact*u_exact->y()*n_y + rho_exact*u_exact->z()*n_z;
      break;

    default:
      break;
  }
  if (_spaceTime) tc_exact = tc_exact + rho_exact*n_t;
  // FunctionPtr tc_exact = rho_exact*beta_x*n_x;
  // if (_spaceDim>=2) tc_exact = tc_exact + rho_exact*beta_y*n_y;
  // if (_spaceDim==3) tc_exact = tc_exact + rho_exact*beta_z*n_z;
  // if (_spaceTime) tc_exact = tc_exact + rho_exact*n_t;

  if (_neglectFluxesOnRHS)
  {
    _solnIncrement->bc()->addDirichlet(tc, region, tc_exact);
  }
  else
  {
    // SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );
    SolutionPtr backgroundFlowWeakReference = _backgroundFlow;
    FunctionPtr tc_prev = Function::solution(tc,backgroundFlowWeakReference);
    _solnIncrement->bc()->addDirichlet(tc, region, tc_exact - tc_prev);
  }
}

void CompressibleNavierStokesFormulation::addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr tm1, tm2, tm3;
  tm1 = this->tm(1);
  if (_spaceDim>=2) tm2 = this->tm(2);
  if (_spaceDim==3) tm3 = this->tm(3);

  FunctionPtr n_x, n_y, n_z, n_t;
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  // if (_spaceTime)
  //   n = Function::normalSpaceTime();
  // else
  //   n = Function::normal();
  n_x = n->x();
  if (_spaceDim>=2) n_y = n->y();
  if (_spaceDim==3) n_z = n->z();
  if (_spaceTime) n_t = n_xt->t();

  FunctionPtr D1_exact, D2_exact, D3_exact;
  if (_spaceDim==1) D1_exact = u_exact->dx();
  else D1_exact = u_exact->x()->grad();
  if (_spaceDim>=2) D2_exact = u_exact->y()->grad();
  if (_spaceDim==3) D3_exact = u_exact->z()->grad();

  // FunctionPtr beta_x, beta_y, beta_z;
  // if (_spaceDim == 1)
  //   beta_x = _beta;
  // else
  //   beta_x = _beta->x();
  // if (_spaceDim >= 2) beta_y = _beta->y();
  // if (_spaceDim == 3) beta_z = _beta->z();

  // FunctionPtr tm1_exact, tm2_exact, tm3_exact;
  // if (_spaceDim==1) tm1_exact = u_exact*beta_x*n_x;
  // else tm1_exact = u_exact->x()*beta_x*n_x;
  // if (_spaceDim>=2) tm1_exact = tm1_exact + u_exact->x()*beta_y*n_y;
  // if (_spaceDim==3) tm1_exact = tm1_exact + u_exact->x()*beta_z*n_z;
  // if (_spaceTime && _spaceDim == 1)
  //   tm1_exact = tm1_exact + u_exact*n_t;
  // else if (_spaceTime)
  //   tm1_exact = tm1_exact + u_exact->x()*n_t;

  // if (_spaceDim >= 2)
  // {
  //   tm2_exact = u_exact->y()*beta_x*n_x;
  //   tm2_exact = tm2_exact + u_exact->y()*beta_y*n_y;
  //   if (_spaceDim==3) tm2_exact = tm2_exact + u_exact->y()*beta_z*n_z;
  //   if (_spaceTime) tm2_exact = tm2_exact + u_exact->y()*n_t;
  // }

  // if (_spaceDim == 3)
  // {
  //   tm3_exact = u_exact->z()*beta_x*n_x;
  //   tm3_exact = tm3_exact + u_exact->z()*beta_y*n_y;
  //   tm3_exact = tm3_exact + u_exact->z()*beta_z*n_z;
  //   if (_spaceTime) tm3_exact = tm3_exact + u_exact->z()*n_t;
  // }
  FunctionPtr tm1_exact, tm2_exact, tm3_exact;
  switch (_spaceDim)
  {
    case 1:
      // tm1_exact = (rho_exact*u_exact*u_exact + R()*rho_exact*T_exact
      //     - (D1_exact+D1_exact-2./3*D1_exact))*n_x;
      tm1_exact = (rho_exact*u_exact*u_exact + R()*rho_exact*T_exact)*n_x;
      if (_spaceTime)
        tm1_exact = tm1_exact + rho_exact*u_exact*n_t;
      break;
    case 2:
      // tm1_exact = (rho_exact*u_exact->x()*u_exact->x() + R()*rho_exact*T_exact
      //     - (D1_exact->x()+D1_exact->x()-2./3*D1_exact->x()-2./3*D2_exact->y()))*n_x
      //   + (rho_exact*u_exact->x()*u_exact->y()
      //     - (D1_exact->y()+D2_exact->x()))*n_y;
      // tm2_exact = (rho_exact*u_exact->y()*u_exact->y() + R()*rho_exact*T_exact
      //     - (D2_exact->y()+D2_exact->y()-2./3*D1_exact->x()-2./3*D2_exact->y()))*n_y
      //   + (rho_exact*u_exact->x()*u_exact->y()
      //     - (D1_exact->y()+D2_exact->x()))*n_x;
      tm1_exact = (rho_exact*u_exact->x()*u_exact->x() + R()*rho_exact*T_exact)*n_x
        + (rho_exact*u_exact->x()*u_exact->y())*n_y;
      tm2_exact = (rho_exact*u_exact->x()*u_exact->y())*n_x
        + (rho_exact*u_exact->y()*u_exact->y() + R()*rho_exact*T_exact)*n_y;
      if (_spaceTime)
      {
        tm1_exact = tm1_exact + rho_exact*u_exact->x()*n_t;
        tm2_exact = tm2_exact + rho_exact*u_exact->y()*n_t;
      }
      break;
    case 3:
      tm1_exact = (rho_exact*u_exact->x()*u_exact->x() + R()*rho_exact*T_exact
          - (D1_exact->x()+D1_exact->x()-2./3*D1_exact->x()-2./3*D2_exact->y()-2./3*D3_exact->z()))*n_x
        + (rho_exact*u_exact->x()*u_exact->y()
          - (D1_exact->y()+D2_exact->x()))*n_y
        + (rho_exact*u_exact->x()*u_exact->z()
          - (D1_exact->z()+D3_exact->x()))*n_z;
      tm2_exact = (rho_exact*u_exact->y()*u_exact->y() + R()*rho_exact*T_exact
          - (D2_exact->y()+D2_exact->y()-2./3*D1_exact->x()-2./3*D2_exact->y()-2./3*D3_exact->z()))*n_y
        + (rho_exact*u_exact->x()*u_exact->y()
          - (D1_exact->y()+D2_exact->x()))*n_x
        + (rho_exact*u_exact->z()*u_exact->y()
          - (D3_exact->y()+D2_exact->z()))*n_z;
      tm3_exact = (rho_exact*u_exact->z()*u_exact->z() + R()*rho_exact*T_exact
          - (D3_exact->z()+D3_exact->z()-2./3*D1_exact->x()-2./3*D2_exact->y()-2./3*D3_exact->z()))*n_z
        + (rho_exact*u_exact->x()*u_exact->z()
          - (D1_exact->z()+D3_exact->x()))*n_x
        + (rho_exact*u_exact->y()*u_exact->z()
          - (D2_exact->z()+D3_exact->y()))*n_y;
      if (_spaceTime)
      {
        tm1_exact = tm1_exact + rho_exact*u_exact->x()*n_t;
        tm2_exact = tm2_exact + rho_exact*u_exact->y()*n_t;
        tm3_exact = tm3_exact + rho_exact*u_exact->z()*n_t;
      }
      break;

    default:
      break;
  }

  if (_neglectFluxesOnRHS)
  {
    _solnIncrement->bc()->addDirichlet(tm1, region, tm1_exact);
    if (_spaceDim >= 2) _solnIncrement->bc()->addDirichlet(tm2, region, tm2_exact);
    if (_spaceDim == 3) _solnIncrement->bc()->addDirichlet(tm3, region, tm3_exact);
  }
  else
  {
    // SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );
    SolutionPtr backgroundFlowWeakReference = _backgroundFlow;
    FunctionPtr tm1_prev, tm2_prev, tm3_prev;
    tm1_prev = Function::solution(tm1, backgroundFlowWeakReference);
    if (_spaceDim >= 2) tm2_prev = Function::solution(tm2, backgroundFlowWeakReference);
    if (_spaceDim == 3) tm3_prev = Function::solution(tm3, backgroundFlowWeakReference);
    _solnIncrement->bc()->addDirichlet(tm1, region, tm1_exact - tm1_prev);
    if (_spaceDim >= 2) _solnIncrement->bc()->addDirichlet(tm2, region, tm2_exact - tm2_prev);
    if (_spaceDim == 3) _solnIncrement->bc()->addDirichlet(tm3, region, tm3_exact - tm3_prev);
  }
}

void CompressibleNavierStokesFormulation::addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
{
  VarPtr te = this->te();

  FunctionPtr n_x, n_y, n_z, n_t;
  FunctionPtr n = TFunction<double>::normal(); // spatial normal
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  // if (_spaceTime)
  //   n = Function::normalSpaceTime();
  // else
  //   n = Function::normal();
  n_x = n->x();
  if (_spaceDim>=2) n_y = n->y();
  if (_spaceDim==3) n_z = n->z();
  if (_spaceTime) n_t = n_xt->t();

  // FunctionPtr beta_x, beta_y, beta_z;
  // if (_spaceDim == 1)
  //   beta_x = _beta;
  // else
  //   beta_x = _beta->x();
  // if (_spaceDim >= 2) beta_y = _beta->y();
  // if (_spaceDim == 3) beta_z = _beta->z();

  // FunctionPtr te_exact = T_exact*beta_x*n_x;
  // if (_spaceDim>=2) te_exact = te_exact + T_exact*beta_y*n_y;
  // if (_spaceDim==3) te_exact = te_exact + T_exact*beta_z*n_z;
  // if (_spaceTime) te_exact = te_exact + T_exact*n_t;

  FunctionPtr D1_exact, D2_exact, D3_exact;
  if (_spaceDim==1) D1_exact = u_exact->dx();
  else D1_exact = u_exact->x()->grad();
  if (_spaceDim>=2) D2_exact = u_exact->y()->grad();
  if (_spaceDim==3) D3_exact = u_exact->z()->grad();
  FunctionPtr q_exact;
  if (_spaceDim==1) q_exact = T_exact->dx();
  else q_exact = T_exact->grad();

  FunctionPtr te_exact;
  switch (_spaceDim)
  {
    case 1:
      // te_exact = (Cv()*rho_exact*u_exact*T_exact + 0.5*rho_exact*u_exact*u_exact*u_exact + R()*rho_exact*u_exact*T_exact
      //     + q_exact - u_exact*(D1_exact+D1_exact-2./3*D1_exact))*n_x;
      te_exact = (Cv()*rho_exact*u_exact*T_exact + 0.5*rho_exact*u_exact*u_exact*u_exact + R()*rho_exact*u_exact*T_exact)*n_x;
      if (_spaceTime) te_exact = te_exact + (Cv()*rho_exact*T_exact+0.5*u_exact*u_exact)*n_t;
      break;
    case 2:
      te_exact = ( Cv()*rho_exact*u_exact->x()*T_exact
          + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y())*u_exact->x()
          + R()*rho_exact*u_exact->x()*T_exact + q_exact->x()
          - u_exact->x()*(D1_exact->x()+D1_exact->x()-2./3*D1_exact->x()-2./3*D2_exact->y())
          - u_exact->y()*(D1_exact->y()+D2_exact->x()) )*n_x
        + ( Cv()*rho_exact*u_exact->y()*T_exact
            + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y())*u_exact->y()
            + R()*rho_exact*u_exact->y()*T_exact + q_exact->y()
            - u_exact->x()*(D1_exact->y()+D2_exact->x())
            - u_exact->y()*(D2_exact->y()+D2_exact->y()-2./3*D1_exact->x()-2./3*D2_exact->y()) )*n_y;
      // te_exact = ( Cv()*rho_exact*u_exact->x()*T_exact
      //     + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y())*u_exact->x()
      //     + R()*rho_exact*u_exact->x()*T_exact + q_exact->x() )*n_x
      //   + ( Cv()*rho_exact*u_exact->y()*T_exact
      //       + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y())*u_exact->y()
      //       + R()*rho_exact*u_exact->y()*T_exact + q_exact->y() )*n_y;
      if (_spaceTime) te_exact = te_exact + (Cv()*rho_exact*T_exact+0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y()))*n_t;
      break;
    case 3:
      te_exact = ( Cv()*rho_exact*u_exact->x()*T_exact
          + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y()+u_exact->z()*u_exact->z())*u_exact->x()
          + R()*rho_exact*u_exact->x()*T_exact + q_exact->x()
          - u_exact->x()*(D1_exact->x()+D1_exact->x()-2./3*D1_exact->x()-2./3*D2_exact->y()-2./3*D3_exact->z())
          - u_exact->y()*(D1_exact->y()+D2_exact->x())
          - u_exact->z()*(D1_exact->z()+D3_exact->x()) )*n_x
        + ( Cv()*rho_exact*u_exact->y()*T_exact
            + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y()+u_exact->z()*u_exact->z())*u_exact->y()
            + R()*rho_exact*u_exact->y()*T_exact + q_exact->y()
            - u_exact->x()*(D1_exact->y()+D2_exact->x())
            - u_exact->y()*(D2_exact->y()+D2_exact->y()-2./3*D1_exact->x()-2./3*D2_exact->y()-2./3*D3_exact->z())
            - u_exact->z()*(D1_exact->z()+D3_exact->x()) )*n_y
        + ( Cv()*rho_exact*u_exact->z()*T_exact
            + 0.5*rho_exact*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y()+u_exact->z()*u_exact->z())*u_exact->z()
            + R()*rho_exact*u_exact->z()*T_exact + q_exact->z()
            - u_exact->x()*(D1_exact->y()+D2_exact->x())
            - u_exact->y()*(D2_exact->y()+D2_exact->y())
            - u_exact->z()*(D1_exact->z()+D3_exact->x()-2./3*D1_exact->x()-2./3*D2_exact->y()-2./3*D3_exact->z()) )*n_z;
      if (_spaceTime) te_exact = te_exact + (Cv()*rho_exact*T_exact
          + 0.5*(u_exact->x()*u_exact->x()+u_exact->y()*u_exact->y()+u_exact->z()*u_exact->z()))*n_t;
      break;

    default:
      break;
  }

  if (_neglectFluxesOnRHS)
  {
    _solnIncrement->bc()->addDirichlet(te, region, te_exact);
  }
  else
  {
    // SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );
    SolutionPtr backgroundFlowWeakReference = _backgroundFlow;
    FunctionPtr te_prev = Function::solution(te,backgroundFlowWeakReference);
    _solnIncrement->bc()->addDirichlet(te, region, te_exact - te_prev);
  }
}


BFPtr CompressibleNavierStokesFormulation::bf()
{
  return _bf;
}

void CompressibleNavierStokesFormulation::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}


// FunctionPtr CompressibleNavierStokesFormulation::forcingFunction(FunctionPtr u_exact, FunctionPtr p_exact)
// {
//   // f1 and f2 are those for Navier-Stokes, but without the u \cdot \grad u term
//   FunctionPtr u1_exact = u_exact->x();
//   FunctionPtr u2_exact = u_exact->y();
//   FunctionPtr u3_exact = u_exact->z();

//   FunctionPtr f_stokes;

//   if (_spaceDim == 2)
//   {
//     FunctionPtr f1, f2;
//     f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
//     f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
//     if (_spaceTime)
//     {
//       f1 = f1 + u1_exact->dt();
//       f2 = f2 + u2_exact->dt();
//     }

//     f_stokes = Function::vectorize(f1, f2);
//   }
//   else
//   {
//     FunctionPtr f1, f2, f3;
//     f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz());
//     f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz());
//     f3 = p_exact->dz() - _mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz());
//     if (_spaceTime)
//     {
//       f1 = f1 + u1_exact->dt();
//       f2 = f2 + u2_exact->dt();
//       f3 = f3 + u3_exact->dt();
//     }

//     f_stokes = Function::vectorize(f1, f2, f3);
//   }


//   FunctionPtr convectiveTerm = OldroydBFormulation::convectiveTerm(spaceDim, u_exact);
//   return f_stokes + convectiveTerm;
// }

// void CompressibleNavierStokesFormulation::setForcingFunction(FunctionPtr forcingFunction)
// {
//   // set the RHS:
//   if (forcingFunction == Teuchos::null)
//   {
//     FunctionPtr scalarZero = Function::zero();
//     if (_spaceDim == 1)
//       forcingFunction = scalarZero;
//     else if (_spaceDim == 2)
//       forcingFunction = Function::vectorize(scalarZero, scalarZero);
//     else if (_spaceDim == 3)
//       forcingFunction = Function::vectorize(scalarZero, scalarZero, scalarZero);
//     else
//       TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported space dimension");
//   }
//
//   _rhsForSolve = this->rhs(forcingFunction, _neglectFluxesOnRHS);
//   _rhsForResidual = this->rhs(forcingFunction, false);
//   _solnIncrement->setRHS(_rhsForSolve);
// }

bool CompressibleNavierStokesFormulation::isSpaceTime() const
{
  return _spaceTime;
}

bool CompressibleNavierStokesFormulation::isSteady() const
{
  return !_timeStepping && !_spaceTime;
}


bool CompressibleNavierStokesFormulation::isTimeStepping() const
{
  return _timeStepping;
}

void CompressibleNavierStokesFormulation::setIP(IPPtr ip)
{
  _solnIncrement->setIP(ip);
}

void CompressibleNavierStokesFormulation::setIP(string normName)
{
  setIP( _ips[normName] );
}

// double CompressibleNavierStokesFormulation::relativeL2NormOfTimeStep()
// {
//   FunctionPtr rho_current = Function::solution(rho(), _solution);
//   FunctionPtr  u1_current = Function::solution( u(1), _solution);
//   FunctionPtr  u2_current = Function::solution( u(2), _solution);
//   FunctionPtr   T_current = Function::solution(  T(), _solution);
//   FunctionPtr rho_prev = Function::solution(rho(), _previousSolution);
//   FunctionPtr  u1_prev = Function::solution( u(1), _previousSolution);
//   FunctionPtr  u2_prev = Function::solution( u(2), _previousSolution);
//   FunctionPtr   T_prev = Function::solution(  T(), _previousSolution);
//
//   FunctionPtr squaredSum = (p_current+p_prev) * (p_current+p_prev) + (u1_current+u1_prev) * (u1_current+u1_prev) + (u2_current + u2_prev) * (u2_current + u2_prev);
//   // average would be each summand divided by 4
//   double L2OfAverage = sqrt( 0.25 * squaredSum->integrate(_solution->mesh()));
//
//   FunctionPtr squaredDiff = (p_current-p_prev) * (p_current-p_prev) + (u1_current-u1_prev) * (u1_current-u1_prev) + (u2_current - u2_prev) * (u2_current - u2_prev);
//
//   double valSquared = squaredDiff->integrate(_solution->mesh());
//   if (L2OfAverage < 1e-15) return sqrt(valSquared);
//
//   return sqrt(valSquared) / L2OfAverage;
// }

double CompressibleNavierStokesFormulation::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double CompressibleNavierStokesFormulation::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

int CompressibleNavierStokesFormulation::nonlinearIterationCount()
{
  return _nonlinearIterationCount;
}

double CompressibleNavierStokesFormulation::mu()
{
  return _mu;
}

double CompressibleNavierStokesFormulation::gamma()
{
  return _gamma;
}

double CompressibleNavierStokesFormulation::Pr()
{
  return _Pr;
}

double CompressibleNavierStokesFormulation::Cv()
{
  return _Cv;
}

double CompressibleNavierStokesFormulation::Cp()
{
  return _gamma*_Cv;
}

double CompressibleNavierStokesFormulation::R()
{
  return Cp()-Cv();
}

RefinementStrategyPtr CompressibleNavierStokesFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void CompressibleNavierStokesFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void CompressibleNavierStokesFormulation::refine()
{
  _refinementStrategy->refine();
  // _nonlinearIterationCount = 0;
  // this->clearSolutionIncrement();
  // _solnIncrement->setRHS(_rhsForResidual);
  // _refinementStrategy->refine();
  // _solnIncrement->setRHS(_rhsForSolve);
}

void CompressibleNavierStokesFormulation::hRefine()
{
  _hRefinementStrategy->refine();
}

void CompressibleNavierStokesFormulation::pRefine()
{
  _pRefinementStrategy->refine();
}

SolverPtr CompressibleNavierStokesFormulation::getSolver()
{
  return _solver;
}

void CompressibleNavierStokesFormulation::setSolver(SolverPtr solver)
{
  _solver = solver;
}

// RHSPtr CompressibleNavierStokesFormulation::rhs(FunctionPtr forcingFunction, bool excludeFluxesAndTraces)
// RHSPtr CompressibleNavierStokesFormulation::rhs(bool excludeFluxesAndTraces)
// {
//
//   // TO DO : UPDATE THIS!
//   RHSPtr rhs = RHS::rhs();
//
//   // SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);
//   SolutionPtr backgroundFlowWeakReference = _backgroundFlow;
//
//   FunctionPtr rho_prev;
//   FunctionPtr u1_prev, u2_prev, u3_prev;
//   FunctionPtr T_prev;
//   FunctionPtr D11_prev, D12_prev, D13_prev, D21_prev, D22_prev, D23_prev, D31_prev, D32_prev, D33_prev;
//   FunctionPtr q1_prev, q2_prev, q3_prev;
//   FunctionPtr tc_prev;
//   FunctionPtr tm1_prev, tm2_prev, tm3_prev;
//   FunctionPtr te_prev;
//   FunctionPtr u1_hat_prev, u2_hat_prev, u3_hat_prev;
//   FunctionPtr T_hat_prev;
//
//   VarPtr vc;
//   VarPtr vm1, vm2, vm3;
//   VarPtr ve;
//   VarPtr S1, S2, S3;
//   VarPtr tau;
//
//   FunctionPtr n;
//   if (_spaceTime)
//     n = Function::normalSpaceTime();
//   else
//     n = Function::normal();
//
//   switch (_spaceDim)
//   {
//     case 1:
//       vc = this->vc();
//       vm1 = this->vm(1);
//       ve = this->ve();
//       S1 = this->S(1);
//       tau = this->tau();
//       rho_prev = Function::solution(this->rho(),backgroundFlowWeakReference);
//       u1_prev = Function::solution(this->u(1),backgroundFlowWeakReference);
//       T_prev  = Function::solution(this->T(),backgroundFlowWeakReference);
//       D11_prev = Function::solution(this->D(1,1),backgroundFlowWeakReference);
//       q1_prev = Function::solution(this->q(1),backgroundFlowWeakReference);
//       tc_prev = Function::solution(this->tc(),backgroundFlowWeakReference,true);
//       tm1_prev = Function::solution(this->tm(1),backgroundFlowWeakReference,true);
//       te_prev = Function::solution(this->te(),backgroundFlowWeakReference,true);
//       u1_hat_prev = Function::solution(this->u_hat(1),backgroundFlowWeakReference);
//       T_hat_prev = Function::solution(this->T_hat(),backgroundFlowWeakReference);
//       // if (!excludeFluxesAndTraces)
//       // {
//       //   rhs->addTerm(-tc_prev * vc);
//       //   rhs->addTerm(-tm1_prev * vm1);
//       //   rhs->addTerm(-te_prev * ve);
//       //   rhs->addTerm(u1_hat_prev * S1*n->x());
//       //   rhs->addTerm(T_hat_prev * tau*n->x());
//       // }
//       break;
//     case 2:
//       vc = this->vc();
//       vm1 = this->vm(1);
//       vm2 = this->vm(2);
//       ve = this->ve();
//       S1 = this->S(1);
//       S2 = this->S(2);
//       tau = this->tau();
//       rho_prev = Function::solution(this->rho(),backgroundFlowWeakReference);
//       u1_prev = Function::solution(this->u(1),backgroundFlowWeakReference);
//       u2_prev = Function::solution(this->u(2),backgroundFlowWeakReference);
//       T_prev  = Function::solution(this->T(),backgroundFlowWeakReference);
//       D11_prev = Function::solution(this->D(1,1),backgroundFlowWeakReference);
//       D12_prev = Function::solution(this->D(1,2),backgroundFlowWeakReference);
//       D21_prev = Function::solution(this->D(2,1),backgroundFlowWeakReference);
//       D22_prev = Function::solution(this->D(2,2),backgroundFlowWeakReference);
//       q1_prev = Function::solution(this->q(1),backgroundFlowWeakReference);
//       q2_prev = Function::solution(this->q(2),backgroundFlowWeakReference);
//       tc_prev = Function::solution(this->tc(),backgroundFlowWeakReference,true);
//       tm1_prev = Function::solution(this->tm(1),backgroundFlowWeakReference,true);
//       tm2_prev = Function::solution(this->tm(2),backgroundFlowWeakReference,true);
//       te_prev = Function::solution(this->te(),backgroundFlowWeakReference,true);
//       u1_hat_prev = Function::solution(this->u_hat(1),backgroundFlowWeakReference);
//       u2_hat_prev = Function::solution(this->u_hat(2),backgroundFlowWeakReference);
//       T_hat_prev = Function::solution(this->T_hat(),backgroundFlowWeakReference);
//       // if (!excludeFluxesAndTraces)
//       // {
//       //   rhs->addTerm(-tc_prev * vc);
//       //   rhs->addTerm(-tm1_prev * vm1);
//       //   rhs->addTerm(-tm2_prev * vm2);
//       //   rhs->addTerm(-te_prev * ve);
//       //   rhs->addTerm(u1_hat_prev * S1*n->x());
//       //   rhs->addTerm(u2_hat_prev * S2*n->x());
//       //   rhs->addTerm(T_hat_prev * tau*n->x());
//       // }
//       break;
//     case 3:
//       vc = this->vc();
//       vm1 = this->vm(1);
//       vm2 = this->vm(2);
//       vm3 = this->vm(3);
//       ve = this->ve();
//       S1 = this->S(1);
//       S2 = this->S(2);
//       S3 = this->S(3);
//       tau = this->tau();
//       rho_prev = Function::solution(this->rho(),backgroundFlowWeakReference);
//       u1_prev = Function::solution(this->u(1),backgroundFlowWeakReference);
//       u2_prev = Function::solution(this->u(2),backgroundFlowWeakReference);
//       u3_prev = Function::solution(this->u(3),backgroundFlowWeakReference);
//       T_prev  = Function::solution(this->T(),backgroundFlowWeakReference);
//       D11_prev = Function::solution(this->D(1,1),backgroundFlowWeakReference);
//       D12_prev = Function::solution(this->D(1,2),backgroundFlowWeakReference);
//       D13_prev = Function::solution(this->D(1,3),backgroundFlowWeakReference);
//       D21_prev = Function::solution(this->D(2,1),backgroundFlowWeakReference);
//       D22_prev = Function::solution(this->D(2,2),backgroundFlowWeakReference);
//       D23_prev = Function::solution(this->D(2,3),backgroundFlowWeakReference);
//       D31_prev = Function::solution(this->D(3,1),backgroundFlowWeakReference);
//       D32_prev = Function::solution(this->D(3,2),backgroundFlowWeakReference);
//       D33_prev = Function::solution(this->D(3,3),backgroundFlowWeakReference);
//       q1_prev = Function::solution(this->q(1),backgroundFlowWeakReference);
//       q2_prev = Function::solution(this->q(2),backgroundFlowWeakReference);
//       q3_prev = Function::solution(this->q(3),backgroundFlowWeakReference);
//       tc_prev = Function::solution(this->tc(),backgroundFlowWeakReference,true);
//       tm1_prev = Function::solution(this->tm(1),backgroundFlowWeakReference,true);
//       tm2_prev = Function::solution(this->tm(2),backgroundFlowWeakReference,true);
//       tm3_prev = Function::solution(this->tm(3),backgroundFlowWeakReference,true);
//       te_prev = Function::solution(this->te(),backgroundFlowWeakReference,true);
//       u1_hat_prev = Function::solution(this->u_hat(1),backgroundFlowWeakReference);
//       u2_hat_prev = Function::solution(this->u_hat(2),backgroundFlowWeakReference);
//       u3_hat_prev = Function::solution(this->u_hat(3),backgroundFlowWeakReference);
//       T_hat_prev = Function::solution(this->T_hat(),backgroundFlowWeakReference);
//       // if (!excludeFluxesAndTraces)
//       // {
//       //   rhs->addTerm(-tc_prev * vc);
//       //   rhs->addTerm(-tm1_prev * vm1);
//       //   rhs->addTerm(-tm2_prev * vm2);
//       //   rhs->addTerm(-tm3_prev * vm3);
//       //   rhs->addTerm(-te_prev * ve);
//       //   rhs->addTerm(u1_hat_prev * S1*n->x());
//       //   rhs->addTerm(u2_hat_prev * S2*n->x());
//       //   rhs->addTerm(u3_hat_prev * S3*n->x());
//       //   rhs->addTerm(T_hat_prev * tau*n->x());
//       // }
//       break;
//
//     default:
//       break;
//   }
//
//   // if (f != Teuchos::null)
//   // {
//   //   rhs->addTerm( f->x() * v1 );
//   //   rhs->addTerm( f->y() * v2 );
//   //   if (_spaceDim == 3) rhs->addTerm( f->z() * v3 );
//   // }
//
//   // FunctionPtr beta_x, beta_y, beta_z;
//   // if (_spaceDim == 1)
//   //   beta_x = _beta;
//   // else
//   //   beta_x = _beta->x();
//   // if (_spaceDim >= 2) beta_y = _beta->y();
//   // if (_spaceDim == 3) beta_z = _beta->z();
//
//   // S terms:
//   switch (_spaceDim)
//   {
//     case 1:
//       rhs->addTerm(-u1_prev * S1->dx()); // D1 = _mu * grad u1
//       rhs->addTerm(-1./_mu * D11_prev * S1); // (D1, S1)
//       break;
//     case 2:
//       rhs->addTerm(-u1_prev * S1->div()); // D1 = _mu * grad u1
//       rhs->addTerm(-u2_prev * S2->div()); // D2 = _mu * grad u2
//       // rhs->addTerm(-u1_prev * (S1->x()->dx() + S1->y()->dy())); // D1 = _mu * grad u1
//       // rhs->addTerm(-u2_prev * (S2->x()->dx() + S2->y()->dy())); // D2 = _mu * grad u2
//       rhs->addTerm(-1./_mu * D11_prev * S1->x()); // (D1, S1)
//       rhs->addTerm(-1./_mu * D12_prev * S1->y());
//       rhs->addTerm(-1./_mu * D21_prev * S2->x()); // (D2, S2)
//       rhs->addTerm(-1./_mu * D22_prev * S2->y());
//       break;
//     case 3:
//       rhs->addTerm(-u1_prev * S1->div()); // D1 = _mu * grad u1
//       rhs->addTerm(-u2_prev * S2->div()); // D2 = _mu * grad u2
//       rhs->addTerm(-u3_prev * S3->div()); // D3 = _mu * grad u3
//       rhs->addTerm(-1./_mu * D11_prev * S1->x()); // (D1, S1)
//       rhs->addTerm(-1./_mu * D12_prev * S1->y());
//       rhs->addTerm(-1./_mu * D13_prev * S1->z());
//       rhs->addTerm(-1./_mu * D21_prev * S2->x()); // (D2, S2)
//       rhs->addTerm(-1./_mu * D22_prev * S2->y());
//       rhs->addTerm(-1./_mu * D23_prev * S2->z());
//       rhs->addTerm(-1./_mu * D31_prev * S3->x()); // (D3, S3)
//       rhs->addTerm(-1./_mu * D32_prev * S3->y());
//       rhs->addTerm(-1./_mu * D33_prev * S3->z());
//       break;
//     default:
//       break;
//   }
//
//   // tau terms:
//   switch (_spaceDim)
//   {
//     case 1:
//       rhs->addTerm( T_prev * tau->dx()); // tau = Cp*_mu/Pr * grad T
//       rhs->addTerm(-Pr()/(Cp()*mu()) * q1_prev * tau); // (D1, S1)
//       break;
//     case 2:
//       rhs->addTerm( T_prev * tau->div()); // tau = Cp*_mu/Pr * grad T
//       // rhs->addTerm(-T_prev * (tau->x()->dx() + tau->y()->dy())); // tau = Cp*_mu/Pr * grad T
//       rhs->addTerm(-Pr()/(Cp()*mu()) * q1_prev * tau->x()); // (D1, S1)
//       rhs->addTerm(-Pr()/(Cp()*mu()) * q2_prev * tau->y());
//       break;
//     case 3:
//       rhs->addTerm( T_prev * tau->div()); // tau = Cp*_mu/Pr * grad T
//       rhs->addTerm(-Pr()/(Cp()*mu()) * q1_prev * tau->x()); // (D1, S1)
//       rhs->addTerm(-Pr()/(Cp()*mu()) * q2_prev * tau->y());
//       rhs->addTerm(-Pr()/(Cp()*mu()) * q3_prev * tau->z());
//       break;
//     default:
//       break;
//   }
//
//
//   // vc:
//   // if (_spaceTime)
//   //   rhs->addTerm(rho_prev * vc->dt());
//   // rhs->addTerm(beta_x*rho_prev * vc->dx());
//   // if (_spaceDim >= 2) rhs->addTerm(beta_y*rho_prev * vc->dy());
//   // if (_spaceDim == 3) rhs->addTerm(beta_z*rho_prev * vc->dz());
//   // // rhs->addTerm(-tc_prev * vc);
//   switch (_spaceDim)
//   {
//     case 1:
//       if (_spaceTime)
//         rhs->addTerm( rho_prev * vc->dt());
//       rhs->addTerm( rho_prev*u1_prev * vc->dx());
//       break;
//     case 2:
//       if (_spaceTime)
//         rhs->addTerm( rho_prev * vc->dt());
//       rhs->addTerm( rho_prev*u1_prev * vc->dx());
//       rhs->addTerm( rho_prev*u2_prev * vc->dy());
//       break;
//     case 3:
//       if (_spaceTime)
//         rhs->addTerm( rho_prev * vc->dt());
//       rhs->addTerm( rho_prev*u1_prev * vc->dx());
//       rhs->addTerm( rho_prev*u2_prev * vc->dy());
//       rhs->addTerm( rho_prev*u3_prev * vc->dz());
//       break;
//     default:
//       break;
//   }
//
//   // vm
//   switch (_spaceDim)
//   {
//     case 1:
//       if (_spaceTime)
//       {
//         rhs->addTerm(rho_prev*u1_prev * vm1->dt());
//       }
//       rhs->addTerm( rho_prev*u1_prev*u1_prev * vm1->dx() );
//       rhs->addTerm( R()*rho_prev*T_prev * vm1->dx() );
//       rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev) * vm1->dx() );
//       break;
//     case 2:
//       if (_spaceTime)
//       {
//         rhs->addTerm( rho_prev*u1_prev * vm1->dt() );
//         rhs->addTerm( rho_prev*u2_prev * vm2->dt() );
//       }
//       rhs->addTerm( rho_prev*u1_prev*u1_prev * vm1->dx());
//       rhs->addTerm( rho_prev*u1_prev*u2_prev * vm1->dy());
//       rhs->addTerm( rho_prev*u2_prev*u1_prev * vm2->dx());
//       rhs->addTerm( rho_prev*u2_prev*u2_prev * vm2->dy());
//       rhs->addTerm( R()*rho_prev*T_prev * vm1->dx());
//       rhs->addTerm( R()*rho_prev*T_prev * vm2->dy());
//       rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev-2./3*D22_prev) * vm1->dx());
//       rhs->addTerm(-(D12_prev+D21_prev) * vm1->dy());
//       rhs->addTerm(-(D21_prev+D12_prev) * vm2->dx());
//       rhs->addTerm(-(D22_prev+D22_prev-2./3*D11_prev-2./3*D22_prev) * vm2->dy());
//       break;
//     case 3:
//       if (_spaceTime)
//       {
//         rhs->addTerm( rho_prev*u1_prev * vm1->dt() );
//         rhs->addTerm( rho_prev*u2_prev * vm2->dt() );
//         rhs->addTerm( rho_prev*u3_prev * vm3->dt() );
//       }
//       rhs->addTerm( rho_prev*u1_prev*u1_prev * vm1->dx() );
//       rhs->addTerm( rho_prev*u1_prev*u2_prev * vm1->dy() );
//       rhs->addTerm( rho_prev*u1_prev*u3_prev * vm1->dz() );
//       rhs->addTerm( rho_prev*u2_prev*u1_prev * vm2->dx() );
//       rhs->addTerm( rho_prev*u2_prev*u2_prev * vm2->dy() );
//       rhs->addTerm( rho_prev*u2_prev*u3_prev * vm2->dz() );
//       rhs->addTerm( rho_prev*u3_prev*u1_prev * vm3->dx() );
//       rhs->addTerm( rho_prev*u3_prev*u2_prev * vm3->dy() );
//       rhs->addTerm( rho_prev*u3_prev*u3_prev * vm3->dz() );
//       rhs->addTerm( R()*rho_prev*T_prev * vm1->dx());
//       rhs->addTerm( R()*rho_prev*T_prev * vm2->dy());
//       rhs->addTerm( R()*rho_prev*T_prev * vm3->dz());
//       rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev-2./3*D22_prev-2./3*D33_prev) * vm1->dx());
//       rhs->addTerm(-(D12_prev+D21_prev) * vm1->dy());
//       rhs->addTerm(-(D13_prev+D31_prev) * vm1->dz());
//       rhs->addTerm(-(D21_prev+D12_prev) * vm2->dx());
//       rhs->addTerm(-(D22_prev+D22_prev-2./3*D11_prev-2./3*D22_prev-2./3*D33_prev) * vm2->dy());
//       rhs->addTerm(-(D23_prev+D32_prev) * vm2->dz());
//       rhs->addTerm(-(D31_prev+D13_prev) * vm3->dx());
//       rhs->addTerm(-(D32_prev+D23_prev) * vm3->dy());
//       rhs->addTerm(-(D33_prev+D33_prev-2./3*D11_prev-2./3*D22_prev-2./3*D33_prev) * vm3->dz());
//     default:
//       break;
//   }
//   // // vm1:
//   // if (_spaceTime)
//   //   rhs->addTerm(u1_prev * vm1->dt());
//   // rhs->addTerm((beta_x*u1_prev - D11_prev) * vm1->dx());
//   // if (_spaceDim >= 2) rhs->addTerm((beta_y*u1_prev -D12_prev) * vm1->dy());
//   // if (_spaceDim == 3) rhs->addTerm((beta_z*u1_prev -D13_prev) * vm1->dz());
//   // // rhs->addTerm(-tm1_prev * vm1);
//
//   // // vm2:
//   // if (_spaceDim >= 2)
//   // {
//   //   if (_spaceTime)
//   //     rhs->addTerm(u2_prev * vm2->dt());
//   //   rhs->addTerm((beta_x*u2_prev - D21_prev) * vm2->dx());
//   //   rhs->addTerm((beta_y*u2_prev - D22_prev) * vm2->dy());
//   //   if (_spaceDim == 3) rhs->addTerm((beta_z*u2_prev - D23_prev) * vm2->dz());
//   //   // rhs->addTerm(-tm2_prev * vm2);
//   // }
//
//   // // vm3:
//   // if (_spaceDim == 3)
//   // {
//   //   if (_spaceTime)
//   //     rhs->addTerm(u3_prev * vm3->dt());
//   //   rhs->addTerm((beta_x*u3_prev -D31_prev) * vm3->dx());
//   //   rhs->addTerm((beta_y*u3_prev -D32_prev) * vm3->dy());
//   //   rhs->addTerm((beta_z*u3_prev -D33_prev) * vm3->dz());
//   //   // rhs->addTerm(-tm3_prev * vm3);
//   // }
//
//   // ve:
//   // if (_spaceTime)
//   //   rhs->addTerm(T_prev * ve->dt());
//   // rhs->addTerm((beta_x*T_prev - q1_prev) * ve->dx());
//   // if (_spaceDim >= 2) rhs->addTerm((beta_y*T_prev - q2_prev) * ve->dy());
//   // if (_spaceDim == 3) rhs->addTerm((beta_z*T_prev - q3_prev) * ve->dz());
//   // // rhs->addTerm(-te_prev * ve);
//   switch (_spaceDim)
//   {
//     case 1:
//       if (_spaceTime)
//       {
//         rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
//         rhs->addTerm(0.5*rho_prev*u1_prev*u1_prev * ve->dt());
//       }
//       rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
//       rhs->addTerm(0.5*rho_prev*u1_prev*u1_prev*u1_prev * ve->dx());
//       rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
//       rhs->addTerm(q1_prev * ve->dx());
//       rhs->addTerm(-(D11_prev+D11_prev-2./3*D11_prev)*u1_prev * ve->dx());
//       rhs->addTerm(-u1_prev*(D11_prev+D11_prev-2./3*D11_prev) * ve->dx());
//       break;
//     case 2:
//       if (_spaceTime)
//       {
//         rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
//         rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev) * ve->dt());
//       }
//       rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
//       rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u1_prev * ve->dx());
//       rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
//       rhs->addTerm(Cv()*rho_prev*u2_prev*T_prev * ve->dy());
//       rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u2_prev * ve->dy());
//       rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
//       // if (_spaceTime)
//       // {
//       //   rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
//       //   rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev) * ve->dt());
//       // }
//       // rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
//       // rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u1_prev * ve->dx());
//       // rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev)*u2_prev * ve->dy());
//       // rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
//       // rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
//       rhs->addTerm(q1_prev * ve->dx());
//       rhs->addTerm(q2_prev * ve->dy());
//       rhs->addTerm(-(D11_prev+D11_prev-2./3*(D11_prev+D22_prev))*u1_prev * ve->dx());
//       rhs->addTerm(-(D12_prev+D21_prev)*u2_prev * ve->dx());
//       rhs->addTerm(-(D21_prev+D12_prev)*u1_prev * ve->dy());
//       rhs->addTerm(-(D22_prev+D22_prev-2./3*(D11_prev+D22_prev))*u2_prev * ve->dy());
//       break;
//     case 3:
//       if (_spaceTime)
//       {
//         rhs->addTerm(Cv()*rho_prev*T_prev * ve->dt());
//         rhs->addTerm(-0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev) * ve->dt());
//       }
//       rhs->addTerm(Cv()*rho_prev*u1_prev*T_prev * ve->dx());
//       rhs->addTerm(Cv()*rho_prev*u2_prev*T_prev * ve->dy());
//       rhs->addTerm(Cv()*rho_prev*u3_prev*T_prev * ve->dz());
//       rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u1_prev * ve->dx());
//       rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u2_prev * ve->dy());
//       rhs->addTerm(0.5*rho_prev*(u1_prev*u1_prev+u2_prev*u2_prev+u3_prev*u3_prev)*u3_prev * ve->dz());
//       rhs->addTerm(R()*rho_prev*u1_prev*T_prev * ve->dx());
//       rhs->addTerm(R()*rho_prev*u2_prev*T_prev * ve->dy());
//       rhs->addTerm(R()*rho_prev*u3_prev*T_prev * ve->dz());
//       rhs->addTerm(q1_prev * ve->dx());
//       rhs->addTerm(q2_prev * ve->dy());
//       rhs->addTerm(q3_prev * ve->dz());
//       rhs->addTerm(-(D11_prev+D11_prev-2./3*(D11_prev+D22_prev+D33_prev))*u1_prev * ve->dx());
//       rhs->addTerm(-(D12_prev+D21_prev)*u2_prev * ve->dx());
//       rhs->addTerm(-(D13_prev+D31_prev)*u3_prev * ve->dx());
//       rhs->addTerm(-(D21_prev+D12_prev)*u1_prev * ve->dy());
//       rhs->addTerm(-(D22_prev+D22_prev-2./3*(D11_prev+D22_prev+D33_prev))*u2_prev * ve->dy());
//       rhs->addTerm(-(D31_prev+D13_prev)*u3_prev * ve->dy());
//       rhs->addTerm(-(D31_prev+D13_prev)*u1_prev * ve->dz());
//       rhs->addTerm(-(D32_prev+D23_prev)*u2_prev * ve->dz());
//       rhs->addTerm(-(D33_prev+D33_prev-2./3*(D11_prev+D22_prev+D33_prev))*u3_prev * ve->dz());
//       break;
//     default:
//       break;
//   }
//
//
//   return rhs;
// }

VarPtr CompressibleNavierStokesFormulation::rho()
{
  return _vf->fieldVar(S_rho);
}

VarPtr CompressibleNavierStokesFormulation::u(int i)
{
  CHECK_VALID_COMPONENT(i);

  static const vector<string> uStrings = {S_u1,S_u2,S_u3};
  return _vf->fieldVar(uStrings[i-1]);
}

VarPtr CompressibleNavierStokesFormulation::T()
{
  return _vf->fieldVar(S_T);
}

VarPtr CompressibleNavierStokesFormulation::D(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> DStrings = {{S_D11, S_D12, S_D13},{S_D21, S_D22, S_D23},{S_D31, S_D32, S_D33}};

  return _vf->fieldVar(DStrings[i-1][j-1]);
}

VarPtr CompressibleNavierStokesFormulation::q(int i)
{
  CHECK_VALID_COMPONENT(i);

  static const vector<string> qStrings = {S_q1,S_q2,S_q3};
  return _vf->fieldVar(qStrings[i-1]);
}

// traces:
VarPtr CompressibleNavierStokesFormulation::u_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> uHatStrings = {S_u1_hat,S_u2_hat,S_u3_hat};
  if (! _spaceTime)
    return _vf->traceVar(uHatStrings[i-1]);
  else
    return _vf->traceVarSpaceOnly(uHatStrings[i-1]);
}

VarPtr CompressibleNavierStokesFormulation::T_hat()
{
  if (! _spaceTime)
    return _vf->traceVar(S_T_hat);
  else
    return _vf->traceVarSpaceOnly(S_T_hat);
}

VarPtr CompressibleNavierStokesFormulation::tc()
{
  return _vf->fluxVar(S_tc);
}

VarPtr CompressibleNavierStokesFormulation::tm(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> tmHatStrings = {S_tm1,S_tm2,S_tm3};
  return _vf->fluxVar(tmHatStrings[i-1]);
}

VarPtr CompressibleNavierStokesFormulation::te()
{
  return _vf->fluxVar(S_te);
}

// test variables:
VarPtr CompressibleNavierStokesFormulation::vc()
{
  return _vf->testVar(S_vc, HGRAD);
}

VarPtr CompressibleNavierStokesFormulation::vm(int i)
{
  CHECK_VALID_COMPONENT(i);
  const static vector<string> vmStrings = {S_vm1,S_vm2,S_vm3};
  return _vf->testVar(vmStrings[i-1], HGRAD);
}

VarPtr CompressibleNavierStokesFormulation::ve()
{
  return _vf->testVar(S_ve, HGRAD);
}

VarPtr CompressibleNavierStokesFormulation::S(int i)
{
  CHECK_VALID_COMPONENT(i);
  const static vector<string> SStrings = {S_S1,S_S2,S_S3};
  if (_spaceDim == 1)
    return _vf->testVar(SStrings[i-1], HGRAD);
  else
    return _vf->testVar(SStrings[i-1], HDIV);
}

VarPtr CompressibleNavierStokesFormulation::tau()
{
  if (_spaceDim == 1)
    return _vf->testVar(S_tau, HGRAD);
  else
    return _vf->testVar(S_tau, HDIV);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void CompressibleNavierStokesFormulation::save(std::string prefixString)
{
  _backgroundFlow->mesh()->saveToHDF5(prefixString+".mesh");
  _backgroundFlow->saveToHDF5(prefixString+".soln");
}

// ! set current time step used for transient solve
void CompressibleNavierStokesFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

// ! Returns the solution (at current time)
SolutionPtr CompressibleNavierStokesFormulation::solution()
{
  return _backgroundFlow;
}

SolutionPtr CompressibleNavierStokesFormulation::solutionIncrement()
{
  return _solnIncrement;
}

void CompressibleNavierStokesFormulation::solveAndAccumulate(double weight)
{
  // RHSPtr savedRHS = _solnIncrement->rhs();
  // _solnIncrement->setRHS(_rhsForSolve);
  _solnIncrement->solve(_solver);
  // _solnIncrement->setRHS(savedRHS);

  bool allowEmptyCells = false;
  set<int> nlVars;
  set<int> lVars;
  nlVars.insert(rho()->ID());
  nlVars.insert(T()->ID());
  lVars.insert(tc()->ID());
  lVars.insert(te()->ID());
  lVars.insert(T_hat()->ID());
  if (_spaceDim == 1)
  {
    nlVars.insert(u(1)->ID());
    nlVars.insert(D(1,1)->ID());
    nlVars.insert(q(1)->ID());
    lVars.insert(tm(1)->ID());
    lVars.insert(u_hat(1)->ID());
  }
  else if (_spaceDim == 2)
  {
    nlVars.insert(u(1)->ID());
    nlVars.insert(u(2)->ID());
    nlVars.insert(D(1,1)->ID());
    nlVars.insert(D(1,2)->ID());
    nlVars.insert(D(2,1)->ID());
    nlVars.insert(D(2,2)->ID());
    nlVars.insert(q(1)->ID());
    nlVars.insert(q(2)->ID());
    lVars.insert(tm(1)->ID());
    lVars.insert(tm(2)->ID());
    lVars.insert(u_hat(1)->ID());
    lVars.insert(u_hat(2)->ID());
  }
  else if (_spaceDim == 3)
  {
    nlVars.insert(u(1)->ID());
    nlVars.insert(u(2)->ID());
    nlVars.insert(u(3)->ID());
    nlVars.insert(D(1,1)->ID());
    nlVars.insert(D(1,2)->ID());
    nlVars.insert(D(1,3)->ID());
    nlVars.insert(D(2,1)->ID());
    nlVars.insert(D(2,2)->ID());
    nlVars.insert(D(2,3)->ID());
    nlVars.insert(D(3,1)->ID());
    nlVars.insert(D(3,2)->ID());
    nlVars.insert(D(3,3)->ID());
    nlVars.insert(q(1)->ID());
    nlVars.insert(q(2)->ID());
    nlVars.insert(q(3)->ID());
    lVars.insert(tm(1)->ID());
    lVars.insert(tm(2)->ID());
    lVars.insert(tm(3)->ID());
    lVars.insert(u_hat(1)->ID());
    lVars.insert(u_hat(2)->ID());
    lVars.insert(u_hat(3)->ID());
  }
  // _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  // _backgroundFlow->addSolution(_solnIncrement, weight, nlVars, allowEmptyCells);
  _backgroundFlow->addReplaceSolution(_solnIncrement, weight, nlVars, lVars);
  _nonlinearIterationCount++;
}
// ! Returns the solution (at previous time)
// SolutionPtr CompressibleNavierStokesFormulation::solutionPreviousTimeStep()
// {
//   return _previousSolution;
// }

// ! Solves iteratively
void CompressibleNavierStokesFormulation::solveIteratively(int maxIters, double cgTol, int azOutputLevel, bool suppressSuperLUOutput)
{
  int kCoarse = 0;

  bool useCondensedSolve = _solnIncrement->usesCondensedSolve();

  vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(_solnIncrement->mesh(), kCoarse, 1);
  vector<MeshPtr> prunedMeshes;
  int minDofCount = 2000; // skip any coarse meshes that have fewer dofs than this
  for (int i=0; i<meshes.size()-2; i++) // leave the last two meshes, so we can guarantee there are at least two
  {
    MeshPtr mesh = meshes[i];
    GlobalIndexType numGlobalDofs;
    if (useCondensedSolve)
      numGlobalDofs = mesh->numFluxDofs(); // this might under-count, in case e.g. of pressure constraints.  But it's meant as a heuristic anyway.
    else
      numGlobalDofs = mesh->numGlobalDofs();

    if (numGlobalDofs > minDofCount)
    {
      prunedMeshes.push_back(mesh);
    }
  }
  prunedMeshes.push_back(meshes[meshes.size()-2]);
  prunedMeshes.push_back(meshes[meshes.size()-1]);

//  prunedMeshes = meshes;

  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(_solnIncrement, prunedMeshes, maxIters, cgTol, GMGOperator::V_CYCLE,
                                                                  Solver::getDirectSolver(true), useCondensedSolve) );
  if (suppressSuperLUOutput)
    turnOffSuperLUDistOutput(gmgSolver);

  gmgSolver->setAztecOutput(azOutputLevel);

  _solnIncrement->solve(gmgSolver);
}

int CompressibleNavierStokesFormulation::spaceDim()
{
  return _spaceDim;
}

// ! Returns the sum of the time steps taken thus far.
double CompressibleNavierStokesFormulation::getTime()
{
  return _time;
}

FunctionPtr CompressibleNavierStokesFormulation::getTimeFunction()
{
  return _t;
}

void CompressibleNavierStokesFormulation::turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver){
  Teuchos::RCP<GMGOperator> gmgOperator = gmgSolver->gmgOperator();
  while (gmgOperator->getCoarseOperator() != Teuchos::null)
  {
    gmgOperator = gmgOperator->getCoarseOperator();
  }
  SolverPtr coarseSolver = gmgOperator->getCoarseSolver();
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
  if (superLUSolver)
  {
    superLUSolver->setRunSilent(true);
  }
#endif
}

const std::map<int,int> & CompressibleNavierStokesFormulation::getTrialVariablePolyOrderAdjustments()
{
  return _trialVariablePolyOrderAdjustments;
}

void CompressibleNavierStokesFormulation::clearSolutionIncrement()
{
  _solnIncrement->clear(); // only clears the local cell coefficients, not the global solution vector
  if (_solnIncrement->getLHSVector().get() != NULL)
    _solnIncrement->getLHSVector()->PutScalar(0); // this clears global solution vector
  _solnIncrement->clearComputedResiduals();
}

Teuchos::ParameterList CompressibleNavierStokesFormulation::getConstructorParameters() const
{
  return _ctorParameters;
}
