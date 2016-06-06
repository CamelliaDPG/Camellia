
//  OldroydBFormulationUW.cpp
//  Camellia
//
//  Created by Brendan Keith, October 2015
//
//

#include "OldroydBFormulationUW.h"

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
#include "LagrangeConstraints.h"


using namespace Camellia;

const string OldroydBFormulationUW::S_U1 = "u_1";
const string OldroydBFormulationUW::S_U2 = "u_2";
const string OldroydBFormulationUW::S_U3 = "u_3";
const string OldroydBFormulationUW::S_L11 = "L_{11}";
const string OldroydBFormulationUW::S_L12 = "L_{12}";
const string OldroydBFormulationUW::S_L13 = "L_{13}";
const string OldroydBFormulationUW::S_L21 = "L_{21}";
const string OldroydBFormulationUW::S_L22 = "L_{22}";
const string OldroydBFormulationUW::S_L23 = "L_{23}";
const string OldroydBFormulationUW::S_L31 = "L_{31}";
const string OldroydBFormulationUW::S_L32 = "L_{32}";
const string OldroydBFormulationUW::S_L33 = "L_{33}";
const string OldroydBFormulationUW::S_T11 = "T_{11}";
const string OldroydBFormulationUW::S_T12 = "T_{12}";
const string OldroydBFormulationUW::S_T13 = "T_{13}";
const string OldroydBFormulationUW::S_T22 = "T_{22}";
const string OldroydBFormulationUW::S_T23 = "T_{23}";
const string OldroydBFormulationUW::S_T33 = "T_{33}";
const string OldroydBFormulationUW::S_P = "p";

const string OldroydBFormulationUW::S_U1_HAT = "\\widehat{u}_1";
const string OldroydBFormulationUW::S_U2_HAT = "\\widehat{u}_2";
const string OldroydBFormulationUW::S_U3_HAT = "\\widehat{u}_3";
const string OldroydBFormulationUW::S_SIGMAN1_HAT = "\\widehat{\\sigma}_{1n}";
const string OldroydBFormulationUW::S_SIGMAN2_HAT = "\\widehat{\\sigma}_{2n}";
const string OldroydBFormulationUW::S_SIGMAN3_HAT = "\\widehat{\\sigma}_{3n}";
const string OldroydBFormulationUW::S_TUN11_HAT = "\\hat{(T\\otimes u)_{n_{11}}}";
const string OldroydBFormulationUW::S_TUN12_HAT = "\\hat{(T\\otimes u)_{n_{12}}}";
const string OldroydBFormulationUW::S_TUN13_HAT = "\\hat{(T\\otimes u)_{n_{13}}}";
const string OldroydBFormulationUW::S_TUN22_HAT = "\\hat{(T\\otimes u)_{n_{22}}}";
const string OldroydBFormulationUW::S_TUN23_HAT = "\\hat{(T\\otimes u)_{n_{23}}}";
const string OldroydBFormulationUW::S_TUN33_HAT = "\\hat{(T\\otimes u)_{n_{33}}}";

const string OldroydBFormulationUW::S_V1 = "v_1";
const string OldroydBFormulationUW::S_V2 = "v_2";
const string OldroydBFormulationUW::S_V3 = "v_3";
const string OldroydBFormulationUW::S_M1 = "M_{1}";
const string OldroydBFormulationUW::S_M2 = "M_{2}";
const string OldroydBFormulationUW::S_M3 = "M_{3}";
const string OldroydBFormulationUW::S_S11 = "S_{11}";
const string OldroydBFormulationUW::S_S12 = "S_{12}";
const string OldroydBFormulationUW::S_S13 = "S_{13}";
const string OldroydBFormulationUW::S_S22 = "S_{22}";
const string OldroydBFormulationUW::S_S23 = "S_{23}";
const string OldroydBFormulationUW::S_S33 = "S_{33}";
const string OldroydBFormulationUW::S_Q = "q";

static const int INITIAL_CONDITION_TAG = 1;

// OldroydBFormulationUW OldroydBFormulationUW::steadyFormulation(int spaceDim, double mu, bool useConformingTraces)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",mu);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", false);
//   parameters.set("useSpaceTime", false);

//   return OldroydBFormulationUW(parameters);
// }

// OldroydBFormulationUW OldroydBFormulationUW::spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces,
//                                                                 bool includeVelocityTracesInFluxTerm)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",mu);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", false);
//   parameters.set("useSpaceTime", true);

//   parameters.set("includeVelocityTracesInFluxTerm",includeVelocityTracesInFluxTerm); // a bit easier to visualize traces when false (when true, tn in space and uhat in time get lumped together, and these can have fairly different scales)
//   parameters.set("t0",0.0);

//   return OldroydBFormulationUW(parameters);
// }

// OldroydBFormulationUW OldroydBFormulationUW::timeSteppingFormulation(int spaceDim, double mu, double dt,
//                                                                    bool useConformingTraces, TimeStepType timeStepType)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",mu);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", true);
//   parameters.set("useSpaceTime", false);
//   parameters.set("dt", dt);
//   parameters.set("timeStepType", timeStepType);

//   return OldroydBFormulationUW(parameters);
// }

OldroydBFormulationUW::OldroydBFormulationUW(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  double muS = parameters.get<double>("muS",1.0); // solvent viscosity
  double muP = parameters.get<double>("muP",1.0); // polymeric viscosity
  double alpha = parameters.get<double>("alpha",0);
  double lambda = parameters.get<double>("lambda",1.0);
  bool enforceLocalConservation = parameters.get<bool>("enforceLocalConservation");
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  int spatialPolyOrder = parameters.get<int>("spatialPolyOrder");
  int temporalPolyOrder = parameters.get<int>("temporalPolyOrder", 1);
  int delta_k = parameters.get<int>("delta_k");

  // nonlinear parameters
  bool stokesOnly = parameters.get<bool>("stokesOnly");
  bool conservationFormulation = parameters.get<bool>("useConservationFormulation");
  // bool neglectFluxesOnRHS = false; // DOES NOT WORK!!!!!
  bool neglectFluxesOnRHS = true;

  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double dt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);
  TimeStepType timeStepType = parameters.get<TimeStepType>("timeStepType", BACKWARD_EULER); // Backward Euler is immune to oscillations (which Crank-Nicolson can/does exhibit)

  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _enforceLocalConservation = enforceLocalConservation;
  _spatialPolyOrder = spatialPolyOrder;
  _temporalPolyOrder =temporalPolyOrder;
  _muS = muS;
  _muP = muP;
  _alpha = alpha;
  _lambda = ParameterFunction::parameterFunction(lambda);
  _dt = ParameterFunction::parameterFunction(dt);
  _t = ParameterFunction::parameterFunction(0);
  _includeVelocityTracesInFluxTerm = parameters.get<bool>("includeVelocityTracesInFluxTerm",true);
  _t0 = parameters.get<double>("t0",0);
  _stokesOnly = stokesOnly;
  _conservationFormulation = conservationFormulation;
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

  TEUCHOS_TEST_FOR_EXCEPTION((spaceDim != 2) && (spaceDim != 3), std::invalid_argument, "spaceDim must be 2 or 3");
  TEUCHOS_TEST_FOR_EXCEPTION(_timeStepping, std::invalid_argument, "Time stepping not supported");

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr p;
  VarPtr L11, L12, L13, L21, L22, L23, L31, L32, L33;
  VarPtr T11, T12, T13, T22, T23, T33;

  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr sigma1n_hat, sigma2n_hat, sigma3n_hat;
  VarPtr Tu11n_hat, Tu12n_hat, Tu22n_hat, Tu13n_hat, Tu23n_hat, Tu33n_hat;

  // tests
  VarPtr v1, v2, v3;
  VarPtr M1, M2, M3;
  VarPtr S11, S12, S13, S22, S23, S33;
  VarPtr q;

  _vf = VarFactory::varFactory();
  u1 = _vf->fieldVar(S_U1);
  u2 = _vf->fieldVar(S_U2);
  if (spaceDim==3) u3 = _vf->fieldVar(S_U3);

  vector<VarPtr> u(spaceDim);
  u[0] = u1;
  u[1] = u2;
  if (spaceDim==3) u[2] = u3;

  p = _vf->fieldVar(S_P);

  vector<vector<VarPtr>> L(spaceDim,vector<VarPtr>(spaceDim));
  L11 = _vf->fieldVar(S_L11);
  L12 = _vf->fieldVar(S_L12);
  L21 = _vf->fieldVar(S_L21);
  L22 = _vf->fieldVar(S_L22);
  L[0][0] = L11;
  L[0][1] = L12;
  L[1][0] = L21;
  L[1][1] = L22;
  if (spaceDim==3)
  {
    L13 = _vf->fieldVar(S_L13);
    L23 = _vf->fieldVar(S_L23);
    L31 = _vf->fieldVar(S_L31);
    L32 = _vf->fieldVar(S_L32);
    L33 = _vf->fieldVar(S_L33);
    L[0][2] = L13;
    L[1][2] = L23;
    L[2][0] = L31;
    L[2][1] = L32;
    L[2][2] = L33;
  }

  vector<vector<VarPtr>> T(spaceDim,vector<VarPtr>(spaceDim));
  T11 = _vf->fieldVar(S_T11);
  T12 = _vf->fieldVar(S_T12);
  T22 = _vf->fieldVar(S_T22);
  T[0][0] = T11;
  T[0][1] = T12;
  T[1][0] = T12;
  T[1][1] = T22;
  if (spaceDim==3)
  {
    T13 = _vf->fieldVar(S_T13);
    T23 = _vf->fieldVar(S_T23);
    T33 = _vf->fieldVar(S_T33);
    T[0][2] = T13;
    T[1][2] = T23;
    T[2][0] = T13;
    T[2][1] = T23;
    T[2][2] = T33;
  }

  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  if (! _spaceTime)
  {
    Space uHatSpace = useConformingTraces ? HGRAD : L2;
    if (spaceDim > 0) u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
    if (spaceDim > 1) u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
    if (spaceDim > 2) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);
  }
  else
  {
    if (_includeVelocityTracesInFluxTerm)
    {
      Space uHatSpace = useConformingTraces ? HGRAD_SPACE_L2_TIME : L2;
      if (spaceDim > 0) u1_hat = _vf->traceVarSpaceOnly(S_U1_HAT, one * u1, uHatSpace);
      if (spaceDim > 1) u2_hat = _vf->traceVarSpaceOnly(S_U2_HAT, one * u2, uHatSpace);
      if (spaceDim > 2) u3_hat = _vf->traceVarSpaceOnly(S_U3_HAT, one * u3, uHatSpace);
    }
    else
    {
      Space uHatSpace = useConformingTraces ? HGRAD : L2;
      if (spaceDim > 0) u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
      if (spaceDim > 1) u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
      if (spaceDim > 2) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);
    }
  }

  TFunctionPtr<double> n = TFunction<double>::normal();

  // Too complicated at the moment to define where these other trace variables comes from
  if (_spaceTime)
  {
    if (_includeVelocityTracesInFluxTerm)
    {
      sigma1n_hat = _vf->fluxVar(S_SIGMAN1_HAT);
      sigma2n_hat = _vf->fluxVar(S_SIGMAN2_HAT);
      if (spaceDim == 3) sigma3n_hat = _vf->fluxVar(S_SIGMAN3_HAT);
    }
    else
    {
      sigma1n_hat = _vf->fluxVarSpaceOnly(S_SIGMAN1_HAT);
      sigma2n_hat = _vf->fluxVarSpaceOnly(S_SIGMAN2_HAT);
      if (spaceDim == 3) sigma3n_hat = _vf->fluxVarSpaceOnly(S_SIGMAN3_HAT);
    }
  }
  else
  {
    sigma1n_hat = _vf->fluxVar(S_SIGMAN1_HAT);
    sigma2n_hat = _vf->fluxVar(S_SIGMAN2_HAT);
    if (spaceDim == 3) sigma3n_hat = _vf->fluxVar(S_SIGMAN3_HAT);
  }

  if (_spaceTime)
  {
    if (_includeVelocityTracesInFluxTerm)
    {
      Tu11n_hat = _vf->fluxVar(S_TUN11_HAT);
      Tu12n_hat = _vf->fluxVar(S_TUN12_HAT);
      Tu22n_hat = _vf->fluxVar(S_TUN22_HAT);
      if (spaceDim == 3)
      {
        Tu13n_hat = _vf->fluxVar(S_TUN13_HAT);
        Tu23n_hat = _vf->fluxVar(S_TUN23_HAT);
        Tu33n_hat = _vf->fluxVar(S_TUN33_HAT);
      }
    }
    else
    {
      Tu11n_hat = _vf->fluxVarSpaceOnly(S_TUN11_HAT);
      Tu12n_hat = _vf->fluxVarSpaceOnly(S_TUN12_HAT);
      Tu22n_hat = _vf->fluxVarSpaceOnly(S_TUN22_HAT);
      if (spaceDim == 3)
      {
        Tu13n_hat = _vf->fluxVarSpaceOnly(S_TUN13_HAT);
        Tu23n_hat = _vf->fluxVarSpaceOnly(S_TUN23_HAT);
        Tu33n_hat = _vf->fluxVarSpaceOnly(S_TUN33_HAT);
      }
    }
  }
  else
  {
    Tu11n_hat = _vf->fluxVar(S_TUN11_HAT);
    Tu12n_hat = _vf->fluxVar(S_TUN12_HAT);
    Tu22n_hat = _vf->fluxVar(S_TUN22_HAT);
    if (spaceDim == 3)
    {
      Tu13n_hat = _vf->fluxVar(S_TUN13_HAT);
      Tu23n_hat = _vf->fluxVar(S_TUN23_HAT);
      Tu33n_hat = _vf->fluxVar(S_TUN33_HAT);
    }
  }

  v1 = _vf->testVar(S_V1, HGRAD);
  v2 = _vf->testVar(S_V2, HGRAD);
  if (spaceDim == 3) v3 = _vf->testVar(S_V3, HGRAD);

  M1 = _vf->testVar(S_M1, HDIV);
  M2 = _vf->testVar(S_M2, HDIV);
  if (spaceDim == 3) M3 = _vf->testVar(S_M3, HDIV);

  vector<vector<VarPtr>> S(spaceDim,vector<VarPtr>(spaceDim));
  S11 = _vf->testVar(S_S11, HGRAD);
  S12 = _vf->testVar(S_S12, HGRAD);
  S22 = _vf->testVar(S_S22, HGRAD);
  S[0][0] = S11;
  S[0][1] = S12;
  S[1][0] = S12;
  S[1][1] = S22;
  if (spaceDim==3)
  {
    S13 = _vf->testVar(S_S13, HGRAD);
    S23 = _vf->testVar(S_S23, HGRAD);
    S33 = _vf->testVar(S_S33, HGRAD);
    S[0][2] = S13;
    S[1][2] = S23;
    S[2][0] = S13;
    S[2][1] = S23;
    S[2][2] = S33;
  }

  q = _vf->testVar(S_Q, HGRAD);

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

  _steadyStokesBF = Teuchos::rcp( new BF(_vf) );
  // M1 terms:
  _steadyStokesBF->addTerm(_muS * u1, M1->div()); // L1 = _muS * du1/dx
  _steadyStokesBF->addTerm(L11, M1->x()); // (L1, M1)
  _steadyStokesBF->addTerm(L12, M1->y());
  if (spaceDim == 3) _steadyStokesBF->addTerm(L13, M1->z());
  _steadyStokesBF->addTerm(-_muS * u1_hat, M1->dot_normal());

  // M2 terms:
  _steadyStokesBF->addTerm(_muS * u2, M2->div());
  _steadyStokesBF->addTerm(L21, M2->x());
  _steadyStokesBF->addTerm(L22, M2->y());
  if (spaceDim == 3) _steadyStokesBF->addTerm(L23, M2->z());
  _steadyStokesBF->addTerm(-_muS * u2_hat, M2->dot_normal());

  // M3:
  if (spaceDim == 3)
  {
    _steadyStokesBF->addTerm(_muS * u3, M3->div());
    _steadyStokesBF->addTerm(L31, M3->x());
    _steadyStokesBF->addTerm(L32, M3->y());
    _steadyStokesBF->addTerm(L33, M3->z());
    _steadyStokesBF->addTerm(-_muS * u3_hat, M3->dot_normal());
  }

  // v1:
  _steadyStokesBF->addTerm(L11, v1->dx()); // (L1, grad v1)
  _steadyStokesBF->addTerm(L12, v1->dy());
  if (spaceDim==3) _steadyStokesBF->addTerm(L13, v1->dz());
  _steadyStokesBF->addTerm( - p, v1->dx() );
  _steadyStokesBF->addTerm( sigma1n_hat, v1);

  // v2:
  _steadyStokesBF->addTerm(L21, v2->dx()); // (L2, grad v2)
  _steadyStokesBF->addTerm(L22, v2->dy());
  if (spaceDim==3) _steadyStokesBF->addTerm(L23, v2->dz());
  _steadyStokesBF->addTerm( - p, v2->dy());
  _steadyStokesBF->addTerm( sigma2n_hat, v2);

  // v3:
  if (spaceDim > 2)
  {
    _steadyStokesBF->addTerm(L31, v3->dx()); // (L3, grad v3)
    _steadyStokesBF->addTerm(L32, v3->dy());
    _steadyStokesBF->addTerm(L33, v3->dz());
    _steadyStokesBF->addTerm( - p, v3->dz());
    _steadyStokesBF->addTerm( sigma3n_hat, v3);
  }

  // q:
  if (spaceDim > 0) _steadyStokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  if (spaceDim > 1) _steadyStokesBF->addTerm(-u2,q->dy());
  if (spaceDim > 2) _steadyStokesBF->addTerm(-u3, q->dz());

  if (spaceDim==2)
  {
    _steadyStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
  }
  else if (spaceDim==3)
  {
    _steadyStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
  }

  if (!_spaceTime)
  {
    _oldroydBBF = _steadyStokesBF;
  }
  else
  {
    _oldroydBBF = Teuchos::rcp( new BF(*_steadyStokesBF) );

    TFunctionPtr<double> n_spaceTime = TFunction<double>::normalSpaceTime();

    // v1:
    _oldroydBBF->addTerm(-u1, v1->dt());

    // v2:
    _oldroydBBF->addTerm(-u2, v2->dt());

    // v3:
    if (_spaceDim == 3)
    {
      _oldroydBBF->addTerm(-u3, v3->dt());
    }

    if (!_includeVelocityTracesInFluxTerm)
    {
      _oldroydBBF->addTerm(u1_hat * n_spaceTime->t(), v1);
      _oldroydBBF->addTerm(u2_hat * n_spaceTime->t(), v2);
      if (_spaceDim == 3) _oldroydBBF->addTerm(u3_hat * n_spaceTime->t(), v3);
    }
  }


  // NONLINEAR TERMS //

  vector<int> H1Order;
  if (_spaceTime)
  {
    H1Order = {spatialPolyOrder+1,temporalPolyOrder+1}; // not dead certain that temporalPolyOrder+1 is the best choice; it depends on whether the indicated poly order means L^2 as it does in space, or whether it means H^1...
  }
  else
  {
    H1Order = {spatialPolyOrder+1};
  }

  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, _oldroydBBF, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;

  _backgroundFlow = TSolution<double>::solution(mesh);
  _solnIncrement = TSolution<double>::solution(mesh);


  // CONSERVATION  OF MOMENTUM

  // convective terms:
  // vector<FunctionPtr> L_prev, u_prev;

  double Re = 1.0 / _muS;
  TFunctionPtr<double> p_prev = TFunction<double>::solution(this->p(), _backgroundFlow);

  if (!_stokesOnly)
  {
    if (!_conservationFormulation)
    {
      for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
      {
        VarPtr u_i = this->u(comp_i);
        VarPtr v_i = this->v(comp_i);

        for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
        {
          VarPtr u_j = this->u(comp_j);
          VarPtr L_ij = this->L(comp_i, comp_j);

          FunctionPtr u_prev_j = TFunction<double>::solution(u_j, _backgroundFlow);
          FunctionPtr L_prev_ij = TFunction<double>::solution(L_ij, _backgroundFlow);

          _oldroydBBF->addTerm( Re * L_prev_ij * u_j, v_i);
          _oldroydBBF->addTerm( Re * u_prev_j * L_ij, v_i);
        }
      }
    }
    else
    {
      if (_spaceDim == 2)
      {
        FunctionPtr u_prev_1 = TFunction<double>::solution(u1, _backgroundFlow);
        FunctionPtr u_prev_2 = TFunction<double>::solution(u2, _backgroundFlow);

        _oldroydBBF->addTerm(-u_prev_1*u1, v1->dx());
        _oldroydBBF->addTerm(-u_prev_1*u1, v1->dx());
        _oldroydBBF->addTerm(-u_prev_2*u1, v1->dy());
        _oldroydBBF->addTerm(-u_prev_1*u2, v1->dy());

        _oldroydBBF->addTerm(-u_prev_2*u1, v2->dx());
        _oldroydBBF->addTerm(-u_prev_1*u2, v2->dx());
        _oldroydBBF->addTerm(-u_prev_2*u2, v2->dy());
        _oldroydBBF->addTerm(-u_prev_2*u2, v2->dy());
      }
      else if (_spaceDim == 3)
      {
        FunctionPtr u_prev_1 = TFunction<double>::solution(u1, _backgroundFlow);
        FunctionPtr u_prev_2 = TFunction<double>::solution(u2, _backgroundFlow);
        FunctionPtr u_prev_3 = TFunction<double>::solution(u3, _backgroundFlow);

        _oldroydBBF->addTerm(u_prev_1*u1, v1->dx());
        _oldroydBBF->addTerm(u_prev_1*u1, v1->dx());
        _oldroydBBF->addTerm(u_prev_2*u1, v1->dy());
        _oldroydBBF->addTerm(u_prev_1*u2, v1->dy());
        _oldroydBBF->addTerm(u_prev_3*u1, v1->dz());
        _oldroydBBF->addTerm(u_prev_1*u3, v1->dz());

        _oldroydBBF->addTerm(u_prev_1*u2, v2->dx());
        _oldroydBBF->addTerm(u_prev_2*u1, v2->dx());
        _oldroydBBF->addTerm(u_prev_2*u2, v2->dy());
        _oldroydBBF->addTerm(u_prev_2*u2, v2->dy());
        _oldroydBBF->addTerm(u_prev_3*u2, v2->dz());
        _oldroydBBF->addTerm(u_prev_2*u3, v2->dz());

        _oldroydBBF->addTerm(u_prev_1*u3, v3->dx());
        _oldroydBBF->addTerm(u_prev_3*u1, v3->dx());
        _oldroydBBF->addTerm(u_prev_2*u3, v3->dy());
        _oldroydBBF->addTerm(u_prev_3*u2, v3->dy());
        _oldroydBBF->addTerm(u_prev_3*u3, v3->dz());
        _oldroydBBF->addTerm(u_prev_3*u3, v3->dz());
      }
    }
  }

  // new constitutive terms:
  switch (_spaceDim) {
          case 3:
            _oldroydBBF->addTerm(T13, v1->dz());
            _oldroydBBF->addTerm(T23, v2->dz());
            _oldroydBBF->addTerm(T13, v3->dx());
            _oldroydBBF->addTerm(T23, v3->dy());
            _oldroydBBF->addTerm(T33, v3->dz());
          case 2:
            _oldroydBBF->addTerm(T11, v1->dx());
            _oldroydBBF->addTerm(T12, v1->dy());
            _oldroydBBF->addTerm(T12, v2->dx());
            _oldroydBBF->addTerm(T22, v2->dy());
            break;

          default:
            break;
        }


  // UPPER-CONVECTED MAXWELL EQUATION FOR T
  TFunctionPtr<double> lambdaFxn = _lambda; // cast to allow use of TFunctionPtr<double> operator overloads

  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
    {
      VarPtr T_ij = this->T(comp_i, comp_j);
      VarPtr Tu_ijn_hat = this->Tun_hat(comp_i, comp_j);
      VarPtr L_ij = this->L(comp_i, comp_j);
      VarPtr S_ij = this->S(comp_i, comp_j);

      FunctionPtr T_prev_ij = TFunction<double>::solution(T_ij, _backgroundFlow);

      _oldroydBBF->addTerm( T_ij, S_ij);
      //
      _oldroydBBF->addTerm( lambdaFxn * Tu_ijn_hat, S_ij);
      //
      _oldroydBBF->addTerm( -2 * Re * _muP * L_ij, S_ij);

      for (int comp_k=1; comp_k <= _spaceDim; comp_k++)
      {
        VarPtr u_k = this->u(comp_k);
        VarPtr L_ik = this->L(comp_i, comp_k);
        VarPtr T_kj = this->T(comp_k, comp_j);

        FunctionPtr u_prev_k = TFunction<double>::solution(u_k, _backgroundFlow);
        FunctionPtr L_prev_ik = TFunction<double>::solution(L_ik, _backgroundFlow);
        FunctionPtr T_prev_kj = TFunction<double>::solution(T_kj, _backgroundFlow);

        switch (comp_k) {
          case 1:
            _oldroydBBF->addTerm( -lambdaFxn * u_prev_k * T_ij, S_ij->dx());
            _oldroydBBF->addTerm( -lambdaFxn * T_prev_ij * u_k, S_ij->dx());
            break;
          case 2:
            _oldroydBBF->addTerm( -lambdaFxn * u_prev_k * T_ij, S_ij->dy());
            _oldroydBBF->addTerm( -lambdaFxn * T_prev_ij * u_k, S_ij->dy());
            break;
          case 3:
            _oldroydBBF->addTerm( -lambdaFxn * u_prev_k * T_ij, S_ij->dz());
            _oldroydBBF->addTerm( -lambdaFxn * T_prev_ij * u_k, S_ij->dz());
            break;

          default:
            break;
        }
        //
        _oldroydBBF->addTerm( -2 * lambdaFxn * Re * L_prev_ik * T_kj, S_ij);
        _oldroydBBF->addTerm( -2 * lambdaFxn * Re * T_prev_kj * L_ik, S_ij);

        // Giesekus model
        if (alpha > 0)
        {
          VarPtr T_ik = this->T(comp_i, comp_k);
          FunctionPtr T_prev_ik = TFunction<double>::solution(T_ik, _backgroundFlow);

          _oldroydBBF->addTerm( alpha * lambdaFxn / _muP * T_prev_ik * T_kj, S_ij);
          _oldroydBBF->addTerm( alpha * lambdaFxn / _muP * T_ik * T_prev_kj, S_ij);
        }

      }
    }
  }


  // TO DO:: Refine this

  // define tractions (used in outflow conditions)
  // definition of traction: _mu * ( (\nabla u) + (\nabla u)^T ) n - p n
  //                      = (L + L^T) n - p n
  if (spaceDim == 2)
  {
    _t1 = n->x() * (2 * L11 - p)       + n->y() * (L11 + L21);
    _t2 = n->x() * (L12 + L21) + n->y() * (2 * L22 - p);
  }
  else
  {
    _t1 = n->x() * (2 * L11 - p)       + n->y() * (L11 + L21) + n->z() * (L13 + L31);
    _t2 = n->x() * (L12 + L21) + n->y() * (2 * L22 - p)       + n->z() * (L23 + L32);
    _t3 = n->x() * (L13 + L31) + n->y() * (L23 + L32) + n->z() * (2 * L33 - p);
  }

  // cout << endl << _oldroydBBF->displayString() << endl;

  // set the inner product to the graph norm:
  setIP( _oldroydBBF->graphNorm() );

  this->setForcingFunction(Teuchos::null); // will default to zero

  _bc = BC::bc();

  mesh->registerSolution(_backgroundFlow);

  _solnIncrement->setBC(_bc);

  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );

  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );

  // Set up Functions for L^2 norm computations

  TFunctionPtr<double> p_incr = TFunction<double>::solution(this->p(), _solnIncrement);
  p_prev = TFunction<double>::solution(this->p(), _backgroundFlow);

  _L2IncrementFunction = p_incr * p_incr;
  _L2SolutionFunction = p_prev * p_prev;
  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    TFunctionPtr<double> u_i_incr = TFunction<double>::solution(this->u(comp_i), _solnIncrement);
    TFunctionPtr<double> u_i_prev = TFunction<double>::solution(this->u(comp_i), _backgroundFlow);

    _L2IncrementFunction = _L2IncrementFunction + u_i_incr * u_i_incr;
    _L2SolutionFunction = _L2SolutionFunction + u_i_prev * u_i_prev;

    for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
    {
      TFunctionPtr<double> L_ij_incr = TFunction<double>::solution(this->L(comp_i,comp_j), _solnIncrement);
      TFunctionPtr<double> L_ij_prev = TFunction<double>::solution(this->L(comp_i,comp_j), _backgroundFlow);
      _L2IncrementFunction = _L2IncrementFunction + L_ij_incr * L_ij_incr;
      _L2SolutionFunction = _L2SolutionFunction + L_ij_prev * L_ij_prev;
    }

    for (int comp_j=comp_i; comp_j <= _spaceDim; comp_j++)
    {
      TFunctionPtr<double> T_ij_incr = TFunction<double>::solution(this->T(comp_i,comp_j), _solnIncrement);
      TFunctionPtr<double> T_ij_prev = TFunction<double>::solution(this->T(comp_i,comp_j), _backgroundFlow);
      _L2IncrementFunction = _L2IncrementFunction + T_ij_incr * T_ij_incr;
      _L2SolutionFunction = _L2SolutionFunction + T_ij_prev * T_ij_prev;
    }
  }

  _solver = Solver::getDirectSolver();

  _nonlinearIterationCount = 0;

  // Enforce local conservation
  if (_enforceLocalConservation)
  {
    TFunctionPtr<double> zero = TFunction<double>::zero();
    if (_spaceDim == 2)
    {
      // CONSERVATION OF VOLUME
      _solnIncrement->lagrangeConstraints()->addConstraint(u1_hat->times_normal_x() + u2_hat->times_normal_y() == zero);
      // // CONSERVATION OF MOMENTUM (if Stokes)
      // if (_stokesOnly)
      // {
      //   // we are assuming that there is no body forcing in the problem.
      //   FunctionPtr x    = Function::xn(1);
      //   FunctionPtr y    = Function::yn(1);
      //   _solnIncrement->lagrangeConstraints()->addConstraint(sigma1n_hat == zero);
      //   _solnIncrement->lagrangeConstraints()->addConstraint(sigma2n_hat == zero);
      //   _solnIncrement->lagrangeConstraints()->addConstraint(x*sigma1n_hat - y*sigma2n_hat == zero); // seems to upset convergence 
      // }
      // _solnIncrement->lagrangeConstraints()->addConstraint(_muS*u1_hat->times_normal_x() - L11 == zero);
    }
    else if (_spaceDim == 3)
    {
      _solnIncrement->lagrangeConstraints()->addConstraint(u1_hat->times_normal_x() + u2_hat->times_normal_y() + u3_hat->times_normal_z() == zero);
    }
  }


  // TO DO: Set up stream function

}

void OldroydBFormulationUW::addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u)
{
  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (_spaceDim==3) u3_hat = this->u_hat(3);

  if (_neglectFluxesOnRHS)
  {
    // this also governs how we accumulate in the fluxes and traces, and hence whether we should use zero BCs or the true BCs for solution increment
    _solnIncrement->bc()->addDirichlet(u1_hat, inflowRegion, u->x());
    _solnIncrement->bc()->addDirichlet(u2_hat, inflowRegion, u->y());
    if (_spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, inflowRegion, u->z());
  }
  else
  {
    // we assume that _neglectFluxesOnRHS = true, in that we always use the full BCs, not their zero-imposing counterparts, when solving for solution increment
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_neglectFluxesOnRHS = true assumed various places");

    TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );

    TFunctionPtr<double> u1_hat_prev = TFunction<double>::solution(u1_hat,backgroundFlowWeakReference);
    TFunctionPtr<double> u2_hat_prev = TFunction<double>::solution(u2_hat,backgroundFlowWeakReference);
    TFunctionPtr<double> u3_hat_prev;
    if (_spaceDim == 3) u3_hat_prev = TFunction<double>::solution(u3_hat,backgroundFlowWeakReference);

    _solnIncrement->bc()->addDirichlet(u1_hat, inflowRegion, u->x() - u1_hat_prev);
    _solnIncrement->bc()->addDirichlet(u2_hat, inflowRegion, u->y() - u2_hat_prev);
    if (_spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, inflowRegion, u->z() - u3_hat_prev);
  }
}

void OldroydBFormulationUW::addInflowViscoelasticStress(SpatialFilterPtr inflowRegion, TFunctionPtr<double> T11un, TFunctionPtr<double> T12un, TFunctionPtr<double> T22un)
{
  if (_neglectFluxesOnRHS)
  {
    // this also governs how we accumulate in the fluxes and traces, and hence whether we should use zero BCs or the true BCs for solution increment

    _solnIncrement->bc()->addDirichlet(this->Tun_hat(1, 1), inflowRegion, T11un);
    _solnIncrement->bc()->addDirichlet(this->Tun_hat(1, 2), inflowRegion, T12un);
    _solnIncrement->bc()->addDirichlet(this->Tun_hat(2, 2), inflowRegion, T22un);
  }
  else
  {
    TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );

    TFunctionPtr<double> T11un_hat_prev = TFunction<double>::solution(this->Tun_hat(1, 1),backgroundFlowWeakReference);
    TFunctionPtr<double> T12un_hat_prev = TFunction<double>::solution(this->Tun_hat(1, 2),backgroundFlowWeakReference);
    TFunctionPtr<double> T22un_hat_prev = TFunction<double>::solution(this->Tun_hat(2, 2),backgroundFlowWeakReference);

    _solnIncrement->bc()->addDirichlet(this->Tun_hat(1, 1), inflowRegion, T11un - T11un_hat_prev);
    _solnIncrement->bc()->addDirichlet(this->Tun_hat(1, 2), inflowRegion, T12un - T12un_hat_prev);
    _solnIncrement->bc()->addDirichlet(this->Tun_hat(2, 2), inflowRegion, T22un - T22un_hat_prev);
  }

  // // zero inflow viscoelastic stress
  // TFunctionPtr<double> zero = TFunction<double>::zero();
  // for (int i=1; i<=_spaceDim; i++)
  // {
  //   for (int j=i; j<=_spaceDim; j++)
  //   {
  //     _solnIncrement->bc()->addDirichlet(Tun_hat(i, j), inflowRegion, zero);
  //   }
  // }
}

void OldroydBFormulationUW::addOutflowCondition(SpatialFilterPtr outflowRegion, double yMax, double muP, double lambda, bool usePhysicalTractions)
{
  _haveOutflowConditionsImposed = true;

  // point pressure and zero-mean pressures are not compatible with outflow conditions:
  VarPtr p = this->p();
  if (_solnIncrement->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    cout << "Removing zero-mean constraint on pressure by virtue of outflow condition.\n";
    _solnIncrement->bc()->removeZeroMeanConstraint(p->ID());
  }

  if (_solnIncrement->bc()->singlePointBC(p->ID()))
  {
    cout << "Removing zero-point condition on pressure by virtue of outflow condition.\n";
    _solnIncrement->bc()->removeSinglePointBC(p->ID());
  }

  if (usePhysicalTractions)
  {
    // my favorite way to do outflow conditions is via penalty constraints imposing a zero traction
    Teuchos::RCP<LocalStiffnessMatrixFilter> filter_incr = _solnIncrement->filter();

    Teuchos::RCP< PenaltyConstraints > pcRCP;
    PenaltyConstraints* pc;

    if (filter_incr.get() != NULL)
    {
      pc = dynamic_cast<PenaltyConstraints*>(filter_incr.get());
      if (pc == NULL)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't add PenaltyConstraints when a non-PenaltyConstraints LocalStiffnessMatrixFilter already in place");
      }
    }
    else
    {
      pcRCP = Teuchos::rcp( new PenaltyConstraints );
      pc = pcRCP.get();
    }
    TFunctionPtr<double> zero = TFunction<double>::zero();
    pc->addConstraint(_t1==zero, outflowRegion);
    pc->addConstraint(_t2==zero, outflowRegion);
    if (_spaceDim==3) pc->addConstraint(_t3==zero, outflowRegion);

    if (pcRCP != Teuchos::null)   // i.e., we're not just adding to a prior PenaltyConstraints object
    {
      _solnIncrement->setFilter(pcRCP);
    }
  }
  else
  {
    TFunctionPtr<double> zero = TFunction<double>::zero();
    // for (int d=1; d<=_spaceDim; d++)
    // {
    //   _solnIncrement->bc()->addDirichlet(sigman_hat(d), outflowRegion, zero);
    // }
    TFunctionPtr<double> y    = TFunction<double>::yn(1);
    // TFunctionPtr<double> T11 = Teuchos::rcp( new 18.0*muP*lambda*pow(y/(_height*_height),2));
    // _solnIncrement->bc()->addDirichlet(this->sigman_hat(1), outflowRegion, -18.0*muP*lambda*y*y/(yMax*yMax*yMax*yMax));
    _solnIncrement->bc()->addDirichlet(this->sigman_hat(1), outflowRegion, zero);
    _solnIncrement->bc()->addDirichlet(this->u_hat(2), outflowRegion, zero);
  }
}

void OldroydBFormulationUW::addPointPressureCondition(vector<double> vertex)
{
  if (_haveOutflowConditionsImposed)
  {
    cout << "ERROR: can't add pressure point condition if there are outflow conditions imposed.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
  }

  VarPtr p = this->p();

  if (vertex.size() == 0)
  {
    vertex = _solnIncrement->mesh()->getTopology()->getVertex(0);
    if (_spaceTime) // then the last coordinate is time; drop it
    {
      vertex.pop_back();
    }
  }
  _solnIncrement->bc()->addSpatialPointBC(p->ID(), 0.0, vertex);

//  cout << "setting point pressure condition at point (";
//  for (int d=0; d<vertex.size(); d++)
//  {
//    cout << vertex[d];
//    if (d < vertex.size() - 1) cout << ", ";
//  }
//  cout << ")\n";

  if (_solnIncrement->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    _solnIncrement->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void OldroydBFormulationUW::addWallCondition(SpatialFilterPtr wall)
{
  vector<double> zero(_spaceDim, 0.0);
  addInflowCondition(wall, TFunction<double>::constant(zero));
}

void OldroydBFormulationUW::addSymmetryCondition(SpatialFilterPtr symmetryRegion)
{
  TFunctionPtr<double> zero = TFunction<double>::zero();
  _solnIncrement->bc()->addDirichlet(this->sigman_hat(1), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->Tun_hat(1,2), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->u_hat(2), symmetryRegion, zero);
}

void OldroydBFormulationUW::addInitialCondition(double t0, vector<FunctionPtr> u0, FunctionPtr p0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_spaceTime, std::invalid_argument, "This method only supported for space-time formulations");
  TEUCHOS_TEST_FOR_EXCEPTION(u0.size() != _spaceDim, std::invalid_argument, "u0 should have length equal to the number of spatial dimensions");

  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_solution->mesh()->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "For the present, StokesFormulation only supports true MeshTopologies for its Solution object");
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, INITIAL_CONDITION_TAG, meshTopo->getEntitySetInitialTime());

  for (int d=1; d<=_spaceDim; d++)
  {
    VarPtr var;
    FunctionPtr functionToImpose;
    if (!_includeVelocityTracesInFluxTerm)
    {
      var = this->u_hat(d);
      functionToImpose = u0[d-1];
    }
    else
    {
      var = this->sigman_hat(d);
      FunctionPtr n_t = Function::normalSpaceTime()->t();  // under usual circumstances, n_t = -1
      functionToImpose = u0[d-1] * n_t;
    }
    _solution->bc()->addDirichlet(var, INITIAL_CONDITION_TAG, functionToImpose);
  }

  if (p0 != Teuchos::null)
  {
    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_solution->mesh()->getTopology().get());
    TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "For the present, StokesFormulation only supports true MeshTopologies for its Solution object");
    meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, INITIAL_CONDITION_TAG, meshTopo->getEntitySetInitialTime());
    _solution->bc()->addDirichlet(p(), INITIAL_CONDITION_TAG, p0);
  }
}

void OldroydBFormulationUW::addZeroInitialCondition(double t0)
{
  vector<FunctionPtr> zero(_spaceDim, Function::zero());
  addInitialCondition(t0, zero, Teuchos::null); // null: don't impose an initial condition on the pressure.
}

void OldroydBFormulationUW::addZeroMeanPressureCondition()
{
  if (_spaceTime)
  {
    cout << "zero-mean constraints for pressure not yet supported for space-time.  Use point constraints instead.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zero-mean constraints for pressure not yet supported for space-time.  Use point constraints instead.");
  }
  if (_haveOutflowConditionsImposed)
  {
    cout << "ERROR: can't add zero mean pressure condition if there are outflow conditions imposed.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
  }

  VarPtr p = this->p();

  _solnIncrement->bc()->addZeroMeanConstraint(p);

  if (_solnIncrement->bc()->singlePointBC(p->ID()))
  {
    _solnIncrement->bc()->removeSinglePointBC(p->ID());
  }
}

BFPtr OldroydBFormulationUW::bf()
{
  return _oldroydBBF;
}

void OldroydBFormulationUW::clearSolutionIncrement()
{
  _solnIncrement->clear(); // only clears the local cell coefficients, not the global solution vector
  if (_solnIncrement->getLHSVector().get() != NULL)
    _solnIncrement->getLHSVector()->PutScalar(0); // this clears global solution vector
  _solnIncrement->clearComputedResiduals();
}

void OldroydBFormulationUW::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

FunctionPtr OldroydBFormulationUW::convectiveTerm(int spaceDim, FunctionPtr u_exact)
{
  TEUCHOS_TEST_FOR_EXCEPTION((spaceDim != 2) && (spaceDim != 3), std::invalid_argument, "spaceDim must be 2 or 3");

  TFunctionPtr<double> f;

  vector<FunctionPtr> convectiveTermVector(spaceDim, Function::zero());
  for (int i=1; i<=spaceDim; i++)
  {
    FunctionPtr ui_exact;
    switch (i) {
      case 1:
        ui_exact = u_exact->x();
        break;
      case 2:
        ui_exact = u_exact->y();
        break;
      case 3:
        ui_exact = u_exact->z();
        break;

      default:
        break;
    }
    for (int j=1; j<=spaceDim; j++)
    {
      FunctionPtr ui_dj_exact;
      switch (j) {
        case 1:
          ui_dj_exact = ui_exact->dx();
          break;
        case 2:
          ui_dj_exact = ui_exact->dy();
          break;
        case 3:
          ui_dj_exact = ui_exact->dz();
          break;

        default:
          break;
      }
      FunctionPtr uj_exact;
      switch (j) {
        case 1:
          uj_exact = u_exact->x();
          break;
        case 2:
          uj_exact = u_exact->y();
          break;
        case 3:
          uj_exact = u_exact->z();
          break;

        default:
          break;
      }

      convectiveTermVector[i-1] = convectiveTermVector[i-1] + uj_exact * ui_dj_exact;
    }
  }
  if (spaceDim == 2)
  {
    return Function::vectorize(convectiveTermVector[0],convectiveTermVector[1]);
  }
  else
  {
    return Function::vectorize(convectiveTermVector[0],convectiveTermVector[1],convectiveTermVector[2]);
  }
}

// TFunctionPtr<double> OldroydBFormulationUW::forcingFunction(TFunctionPtr<double> u_exact, TFunctionPtr<double> p_exact)
// {
//   // f1 and f2 are those for Navier-Stokes, but without the u \cdot \grad u term
//   TFunctionPtr<double> u1_exact = u_exact->x();
//   TFunctionPtr<double> u2_exact = u_exact->y();
//   TFunctionPtr<double> u3_exact = u_exact->z();

//   TFunctionPtr<double> f_stokes;

//   if (_spaceDim == 2)
//   {
//     TFunctionPtr<double> f1, f2;
//     f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
//     f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
//     if (_spaceTime)
//     {
//       f1 = f1 + u1_exact->dt();
//       f2 = f2 + u2_exact->dt();
//     }

//     f_stokes = TFunction<double>::vectorize(f1, f2);
//   }
//   else
//   {
//     TFunctionPtr<double> f1, f2, f3;
//     f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz());
//     f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz());
//     f3 = p_exact->dz() - _mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz());
//     if (_spaceTime)
//     {
//       f1 = f1 + u1_exact->dt();
//       f2 = f2 + u2_exact->dt();
//       f3 = f3 + u3_exact->dt();
//     }

//     f_stokes = TFunction<double>::vectorize(f1, f2, f3);
//   }


//   FunctionPtr convectiveTerm = OldroydBFormulation::convectiveTerm(spaceDim, u_exact);
//   return f_stokes + convectiveTerm;
// }

void OldroydBFormulationUW::setForcingFunction(FunctionPtr forcingFunction)
{
  // set the RHS:
  if (forcingFunction == Teuchos::null)
  {
    FunctionPtr scalarZero = Function::zero();
    if (_spaceDim == 1)
      forcingFunction = scalarZero;
    else if (_spaceDim == 2)
      forcingFunction = Function::vectorize(scalarZero, scalarZero);
    else if (_spaceDim == 3)
      forcingFunction = Function::vectorize(scalarZero, scalarZero, scalarZero);
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported space dimension");
  }

  _rhsForSolve = this->rhs(forcingFunction, _neglectFluxesOnRHS);
  _rhsForResidual = this->rhs(forcingFunction, false);
  _solnIncrement->setRHS(_rhsForSolve);
}

void OldroydBFormulationUW::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, int temporalPolyOrder)
{
  this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction, "", temporalPolyOrder);
}

void OldroydBFormulationUW::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, int temporalPolyOrder)
{
  this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, forcingFunction, filePrefix, temporalPolyOrder);
}

void OldroydBFormulationUW::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, string savedSolutionAndMeshPrefix, int temporalPolyOrder)
{
  _haveOutflowConditionsImposed = false;
  BCPtr bc = BC::bc();

  vector<int> H1Order {fieldPolyOrder + 1};
  MeshPtr mesh;
  if (savedSolutionAndMeshPrefix == "")
  {
    if (_spaceTime) H1Order.push_back(temporalPolyOrder); // "H1Order" is a bit misleading for space-time; in fact in BasisFactory we ensure that the polynomial order in time is whatever goes in this slot, regardless of function space.  This is disanalogous to what happens in space, so we might want to revisit that at some point.
    mesh = Teuchos::rcp( new Mesh(meshTopo, _oldroydBBF, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
    _solution = TSolution<double>::solution(mesh,bc);
  }
  else
  {
    mesh = MeshFactory::loadFromHDF5(_oldroydBBF, savedSolutionAndMeshPrefix+".mesh");
    _solution = TSolution<double>::solution(mesh, bc);
    _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
  }

  RHSPtr rhs = this->rhs(forcingFunction, _neglectFluxesOnRHS); // in transient case, this will refer to _previousSolution
  IPPtr ip = _oldroydBBF->graphNorm();

//  cout << "graph norm for Stokes BF:\n";
//  ip->printInteractions();

  _solution->setRHS(rhs);
  _solution->setIP(ip);

  mesh->registerSolution(_solution); // will project both time steps during refinements...

  LinearTermPtr residual = rhs->linearTerm() - _oldroydBBF->testFunctional(_solution,false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );

  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold, maxDouble, maxP, true ) );

  _time = 0;
  _t->setTime(_time);

  if (_spaceDim==2)
  {
    // finally, set up a stream function solve for 2D
    _streamFormulation = Teuchos::rcp( new PoissonFormulation(_spaceDim,_useConformingTraces) );

    MeshPtr streamMesh;
    if (savedSolutionAndMeshPrefix == "")
    {
      MeshTopologyPtr streamMeshTopo = meshTopo->deepCopy();
      streamMesh = Teuchos::rcp( new Mesh(streamMeshTopo, _streamFormulation->bf(), H1Order, delta_k) ) ;
    }
    else
    {
      streamMesh = MeshFactory::loadFromHDF5(_streamFormulation->bf(), savedSolutionAndMeshPrefix+"_stream.mesh");
    }

    mesh->registerObserver(streamMesh); // refine streamMesh whenever mesh is refined

    LinearTermPtr u1_dy = (1.0 / _muS) * this->L(1,2);
    LinearTermPtr u2_dx = (1.0 / _muS) * this->L(2,1);

    TFunctionPtr<double> vorticity = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, u2_dx - u1_dy) );
    RHSPtr streamRHS = RHS::rhs();
    VarPtr q_stream = _streamFormulation->q();
    streamRHS->addTerm( -vorticity * q_stream );
    bool dontWarnAboutOverriding = true;
    ((PreviousSolutionFunction<double>*) vorticity.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

    /* Stream function phi is such that
     *    d/dx phi = -u2
     *    d/dy phi =  u1
     * Therefore, psi = grad phi = (-u2, u1), and psi * n = u1 n2 - u2 n1
     */

    TFunctionPtr<double> u1_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, this->u(1) ) );
    TFunctionPtr<double> u2_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, this->u(2) ) );
    ((PreviousSolutionFunction<double>*) u1_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
    ((PreviousSolutionFunction<double>*) u2_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

    TFunctionPtr<double> n = TFunction<double>::normal();

    BCPtr streamBC = BC::bc();
    VarPtr phi = _streamFormulation->phi();
    streamBC->addZeroMeanConstraint(phi);

    VarPtr psi_n = _streamFormulation->psi_n_hat();
    streamBC->addDirichlet(psi_n, SpatialFilter::allSpace(), u1_soln * n->y() - u2_soln * n->x());

    IPPtr streamIP = _streamFormulation->bf()->graphNorm();
    _streamSolution = TSolution<double>::solution(streamMesh,streamBC,streamRHS,streamIP);

    if (savedSolutionAndMeshPrefix != "")
    {
      _streamSolution->loadFromHDF5(savedSolutionAndMeshPrefix + "_stream.soln");
    }
  }
}

bool OldroydBFormulationUW::isSpaceTime() const
{
  return _spaceTime;
}

bool OldroydBFormulationUW::isSteady() const
{
  return !_timeStepping && !_spaceTime;
}


bool OldroydBFormulationUW::isTimeStepping() const
{
  return _timeStepping;
}

void OldroydBFormulationUW::setIP(IPPtr ip)
{
  _solnIncrement->setIP(ip);
}

double OldroydBFormulationUW::relativeL2NormOfTimeStep()
{
  TFunctionPtr<double>  p_current = TFunction<double>::solution( p(), _solution);
  TFunctionPtr<double> u1_current = TFunction<double>::solution(u(1), _solution);
  TFunctionPtr<double> u2_current = TFunction<double>::solution(u(2), _solution);
  TFunctionPtr<double>  p_prev = TFunction<double>::solution( p(), _previousSolution);
  TFunctionPtr<double> u1_prev = TFunction<double>::solution(u(1), _previousSolution);
  TFunctionPtr<double> u2_prev = TFunction<double>::solution(u(2), _previousSolution);

  TFunctionPtr<double> squaredSum = (p_current+p_prev) * (p_current+p_prev) + (u1_current+u1_prev) * (u1_current+u1_prev) + (u2_current + u2_prev) * (u2_current + u2_prev);
  // average would be each summand divided by 4
  double L2OfAverage = sqrt( 0.25 * squaredSum->integrate(_solution->mesh()));

  TFunctionPtr<double> squaredDiff = (p_current-p_prev) * (p_current-p_prev) + (u1_current-u1_prev) * (u1_current-u1_prev) + (u2_current - u2_prev) * (u2_current - u2_prev);

  double valSquared = squaredDiff->integrate(_solution->mesh());
  if (L2OfAverage < 1e-15) return sqrt(valSquared);

  return sqrt(valSquared) / L2OfAverage;
}

double OldroydBFormulationUW::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double OldroydBFormulationUW::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

int OldroydBFormulationUW::nonlinearIterationCount()
{
  return _nonlinearIterationCount;
}

double OldroydBFormulationUW::muS()
{
  return _muS;
}

double OldroydBFormulationUW::muP()
{
  return _muP;
}

Teuchos::RCP<ParameterFunction> OldroydBFormulationUW::lambda()
{
  return _lambda;
}

double OldroydBFormulationUW::alpha()
{
  return _alpha;
}

// ! set lambda during continuation
void OldroydBFormulationUW::setLambda(double lambda)
{
  _lambda->setValue(lambda);
}

RefinementStrategyPtr OldroydBFormulationUW::getRefinementStrategy()
{
  return _refinementStrategy;
}

void OldroydBFormulationUW::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void OldroydBFormulationUW::refine()
{
  _refinementStrategy->refine();
}

void OldroydBFormulationUW::hRefine()
{
  _hRefinementStrategy->refine();
}

void OldroydBFormulationUW::pRefine()
{
  _pRefinementStrategy->refine();
}

RHSPtr OldroydBFormulationUW::rhs(TFunctionPtr<double> f, bool excludeFluxesAndTraces)
{

  // TO DO : UPDATE THIS!
  RHSPtr rhs = RHS::rhs();

  TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);

  TFunctionPtr<double> p_prev;
  TFunctionPtr<double> u1_prev, u2_prev, u3_prev;
  TFunctionPtr<double> L11_prev, L12_prev, L13_prev, L21_prev, L22_prev, L23_prev, L31_prev, L32_prev, L33_prev;

  VarPtr q;
  VarPtr v1, v2, v3;
  VarPtr M1, M2, M3;

  switch (_spaceDim) {
          case 3:
          v3 = this->v(3);
          M3 = this->M(3);
          u3_prev = TFunction<double>::solution(this->u(3),backgroundFlowWeakReference);
          L13_prev = TFunction<double>::solution(this->L(1,3),backgroundFlowWeakReference);
          L23_prev = TFunction<double>::solution(this->L(2,3),backgroundFlowWeakReference);
          L31_prev = TFunction<double>::solution(this->L(1,3),backgroundFlowWeakReference);
          L32_prev = TFunction<double>::solution(this->L(2,3),backgroundFlowWeakReference);
          L33_prev = TFunction<double>::solution(this->L(3,3),backgroundFlowWeakReference);
          case 2:
          q = this->q();
          v1 = this->v(1);
          v2 = this->v(2);
          M1 = this->M(1);
          M2 = this->M(2);
          p_prev = TFunction<double>::solution(this->p(),backgroundFlowWeakReference);
          u1_prev = TFunction<double>::solution(this->u(1),backgroundFlowWeakReference);
          u2_prev = TFunction<double>::solution(this->u(2),backgroundFlowWeakReference);
          L11_prev = TFunction<double>::solution(this->L(1,1),backgroundFlowWeakReference);
          L12_prev = TFunction<double>::solution(this->L(1,2),backgroundFlowWeakReference);
          L21_prev = TFunction<double>::solution(this->L(2,1),backgroundFlowWeakReference);
          L22_prev = TFunction<double>::solution(this->L(2,2),backgroundFlowWeakReference);
          break;

        default:
          break;
      }

  if (f != Teuchos::null)
  {
    rhs->addTerm( f->x() * v1 );
    rhs->addTerm( f->y() * v2 );
    if (_spaceDim == 3) rhs->addTerm( f->z() * v3 );
  }

  // subtract the stokesBF from the RHS (this doesn't work well for some reason)
  // rhs->addTerm( -_steadyStokesBF->testFunctional(backgroundFlowWeakReference, excludeFluxesAndTraces) );

  // STOKES part
  double muS = this->muS();

  // M1 terms:
  rhs->addTerm( -muS * u1_prev * M1->div()); // L1 = muS * du1/dx
  rhs->addTerm( -L11_prev * M1->x()); // (L1, M1)
  rhs->addTerm( -L12_prev * M1->y());
  if (_spaceDim == 3) rhs->addTerm( -L13_prev * M1->z());
  // rhs->addTerm(-mu * u1_hat, M1->dot_normal());

  // M2 terms:
  rhs->addTerm( -muS * u2_prev * M2->div());
  rhs->addTerm( -L21_prev * M2->x());
  rhs->addTerm( -L22_prev * M2->y());
  if (_spaceDim == 3) rhs->addTerm( -L23_prev * M2->z());
  // rhs->addTerm(-mu * u2_hat, M2->dot_normal());

  // M3:
  if (_spaceDim == 3)
  {
    rhs->addTerm( -muS * u3_prev * M3->div());
    rhs->addTerm( -L31_prev * M3->x());
    rhs->addTerm( -L32_prev * M3->y());
    rhs->addTerm( -L33_prev * M3->z());
    // rhs->addTerm(-mu * u3_hat, M3->dot_normal());
  }

  // v1:
  rhs->addTerm( -L11_prev * v1->dx()); // (L1, grad v1)
  rhs->addTerm( -L12_prev * v1->dy());
  if (_spaceDim==3) rhs->addTerm( -L13_prev * v1->dz());
  rhs->addTerm( p_prev * v1->dx() );
  // rhs->addTerm( sigma1n_hat, v1);

  // v2:

  rhs->addTerm( -L21_prev * v2->dx()); // (L2, grad v2)
  rhs->addTerm( -L22_prev * v2->dy());
  if (_spaceDim==3) rhs->addTerm( -L23_prev * v2->dz());
  rhs->addTerm( p_prev * v2->dy());
  // rhs->addTerm( sigma2n_hat, v2);

  // v3:
  if (_spaceDim > 2)
  {
    rhs->addTerm( -L31_prev * v3->dx()); // (L3, grad v3)
    rhs->addTerm( -L32_prev * v3->dy());
    rhs->addTerm( -L33_prev * v3->dz());
    rhs->addTerm( p_prev * v3->dz());
    // rhs->addTerm( sigma3n_hat, v3);
  }

  // q:
  if (_spaceDim > 0) rhs->addTerm( u1_prev * q->dx()); // (-u, grad q)
  if (_spaceDim > 1) rhs->addTerm( u2_prev * q->dy());
  if (_spaceDim > 2) rhs->addTerm( u3_prev * q->dz());

  // if (_spaceDim==2)
  // {
  //   // rhs->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
  // }
  // else if (_spaceDim==3)
  // {
  //   // rhs->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
  // }

  // if (_spaceTime)
  // {
  //   rhs = Teuchos::rcp( new BF(*rhs) );

  //   TFunctionPtr<double> n_spaceTime = TFunction<double>::normalSpaceTime();

  //   // v1:
  //   rhs->addTerm( u1_prev * v1->dt());

  //   // v2:
  //   rhs->addTerm( u2_prev * v2->dt());

  //   // v3:
  //   if (_spaceDim == 3)
  //   {
  //     rhs->addTerm( u3_prev * v3->dt());
  //   }

  //   if (!_includeVelocityTracesInFluxTerm)
  //   {
  //     rhs->addTerm(u1_hat * n_spaceTime->t(), v1);
  //     rhs->addTerm(u2_hat * n_spaceTime->t(), v2);
  //     if (_spaceDim == 3) rhs->addTerm(u3_hat * n_spaceTime->t(), v3);
  //   }
  // }

  // add the u L term:
  double Re = 1.0 / muS;
  double muP = this->muP();
  Teuchos::RCP<ParameterFunction> lambda = this->lambda();
  TFunctionPtr<double> lambdaFxn = lambda;
  double alpha = this->alpha();
  if (!_stokesOnly)
  {
    if (!_conservationFormulation)
    {
      for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
      {
        VarPtr vi = this->v(comp_i);

        for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
        {
          VarPtr uj = this->u(comp_j);
          TFunctionPtr<double> uj_prev = TFunction<double>::solution(uj,backgroundFlowWeakReference);
          VarPtr L_ij = this->L(comp_i, comp_j);
          TFunctionPtr<double> L_ij_prev = TFunction<double>::solution(L_ij, backgroundFlowWeakReference);
          rhs->addTerm((-Re * uj_prev * L_ij_prev) * vi);
        }
      }
    }
    else
    {
      if (_spaceDim == 2)
      {
        rhs->addTerm( u1_prev * u1_prev * v1->dx() );
        rhs->addTerm( u1_prev * u2_prev * v1->dy() );
        rhs->addTerm( u2_prev * u1_prev * v2->dx() );
        rhs->addTerm( u2_prev * u2_prev * v2->dy() );
      }
      else if (_spaceDim == 3)
      {
        rhs->addTerm( u1_prev * u1_prev * v1->dx() );
        rhs->addTerm( u1_prev * u2_prev * v1->dy() );
        rhs->addTerm( u1_prev * u3_prev * v1->dz() );

        rhs->addTerm( u2_prev * u1_prev * v2->dx() );
        rhs->addTerm( u2_prev * u2_prev * v2->dy() );
        rhs->addTerm( u2_prev * u3_prev * v2->dz() );

        rhs->addTerm( u3_prev * u1_prev * v3->dx() );
        rhs->addTerm( u3_prev * u2_prev * v3->dy() );
        rhs->addTerm( u3_prev * u3_prev * v3->dz() );
      }
    }
  }

  VarPtr T11, T12, T22, T13, T23, T33;
  TFunctionPtr<double> T11_prev, T12_prev, T22_prev, T13_prev, T23_prev, T33_prev;

  // new constitutive terms:
  switch (_spaceDim) {
          case 3:
            T13 = this->T(1,3);
            T23 = this->T(2,3);
            T33 = this->T(3,3);
            T13_prev = TFunction<double>::solution(T13,backgroundFlowWeakReference);
            T23_prev = TFunction<double>::solution(T23,backgroundFlowWeakReference);
            T33_prev = TFunction<double>::solution(T33,backgroundFlowWeakReference);

            rhs->addTerm( -T13_prev * v1->dz());
            rhs->addTerm( -T23_prev * v2->dz());
            rhs->addTerm( -T13_prev * v3->dx());
            rhs->addTerm( -T23_prev * v3->dy());
            rhs->addTerm( -T33_prev * v3->dz());
          case 2:
            T11 = this->T(1,1);
            T12 = this->T(1,2);
            T22 = this->T(2,2);
            T11_prev = TFunction<double>::solution(T11,backgroundFlowWeakReference);
            T12_prev = TFunction<double>::solution(T12,backgroundFlowWeakReference);
            T22_prev = TFunction<double>::solution(T22,backgroundFlowWeakReference);

            rhs->addTerm( -T11_prev * v1->dx());
            rhs->addTerm( -T12_prev * v1->dy());
            rhs->addTerm( -T12_prev * v2->dx());
            rhs->addTerm( -T22_prev * v2->dy());
            break;

          default:
            break;
        }


  // UPPER-CONVECTED MAXWELL EQUATION FOR T

  for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
  {
    for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
    {
      VarPtr T_ij = this->T(comp_i, comp_j);
      // VarPtr Tu_ijn_hat = this->Tun_hat(comp_i, comp_j);
      VarPtr L_ij = this->L(comp_i, comp_j);
      VarPtr S_ij = this->S(comp_i, comp_j);

      TFunctionPtr<double> T_ij_prev = TFunction<double>::solution(T_ij, backgroundFlowWeakReference);
      TFunctionPtr<double> L_ij_prev = TFunction<double>::solution(L_ij, backgroundFlowWeakReference);

      rhs->addTerm( -T_ij_prev * S_ij);
      //
      // rhs->addTerm( lambda * Tu_ijn_hat_prev * S_ij);
      //
      rhs->addTerm( 2 * muP * Re * L_ij_prev * S_ij);

      for (int comp_k=1; comp_k <= _spaceDim; comp_k++)
      {
        VarPtr u_k = this->u(comp_k);
        VarPtr L_ik = this->L(comp_i, comp_k);
        VarPtr T_kj = this->T(comp_k, comp_j);

        FunctionPtr u_k_prev = TFunction<double>::solution(u_k, backgroundFlowWeakReference);
        FunctionPtr L_ik_prev = TFunction<double>::solution(L_ik, backgroundFlowWeakReference);
        FunctionPtr T_kj_prev = TFunction<double>::solution(T_kj, backgroundFlowWeakReference);

        switch (comp_k) {
          case 1:
            rhs->addTerm( lambdaFxn * T_ij_prev * u_k_prev * S_ij->dx());
            break;
          case 2:
            rhs->addTerm( lambdaFxn * T_ij_prev * u_k_prev * S_ij->dy());
            break;
          case 3:
            rhs->addTerm( lambdaFxn * T_ij_prev * u_k_prev * S_ij->dz());
            break;

          default:
            break;
        }

        rhs->addTerm( 2.0 * lambdaFxn * Re * L_ik_prev * T_kj_prev * S_ij);

        // Giesekus model
        if (alpha > 0)
        {
          VarPtr T_ik = this->T(comp_i, comp_k);
          FunctionPtr T_ik_prev = TFunction<double>::solution(T_ik, _backgroundFlow);

          rhs->addTerm( - alpha * lambdaFxn / muP * T_ik_prev * T_kj_prev * S_ij);
        }
      }
    }
  }

  // cout << endl <<endl << rhs->linearTerm()->displayString() << endl;

  return rhs;
}

VarPtr OldroydBFormulationUW::L(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> LStrings = {{S_L11, S_L12, S_L13},{S_L21, S_L22, S_L23},{S_L31, S_L32, S_L33}};

  return _vf->fieldVar(LStrings[i-1][j-1]);
}

VarPtr OldroydBFormulationUW::u(int i)
{
  CHECK_VALID_COMPONENT(i);

  static const vector<string> uStrings = {S_U1,S_U2,S_U3};
  return _vf->fieldVar(uStrings[i-1]);
}

VarPtr OldroydBFormulationUW::T(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> TStrings = {{S_T11, S_T12, S_T13},{S_T12, S_T22, S_T23},{S_T13, S_T23, S_T33}};

  return _vf->fieldVar(TStrings[i-1][j-1]);
}

VarPtr OldroydBFormulationUW::p()
{
  return _vf->fieldVar(S_P);
}

// traces:
VarPtr OldroydBFormulationUW::sigman_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> sigmanStrings = {S_SIGMAN1_HAT,S_SIGMAN2_HAT,S_SIGMAN3_HAT};
  return _vf->fluxVar(sigmanStrings[i-1]);
}

VarPtr OldroydBFormulationUW::u_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> uHatStrings = {S_U1_HAT,S_U2_HAT,S_U3_HAT};
  return _vf->traceVar(uHatStrings[i-1]);
}

VarPtr OldroydBFormulationUW::Tun_hat(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> TunHatStrings = {{S_TUN11_HAT, S_TUN12_HAT, S_TUN13_HAT},{S_TUN12_HAT, S_TUN22_HAT, S_TUN23_HAT},{S_TUN13_HAT, S_TUN23_HAT, S_TUN33_HAT}};;
  return _vf->traceVar(TunHatStrings[i-1][j-1]);
}

// test variables:
VarPtr OldroydBFormulationUW::q()
{
  return _vf->testVar(S_Q, HGRAD);
}

VarPtr OldroydBFormulationUW::M(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  const static vector<string> MStrings = {S_M1,S_M2,S_M3};
  return _vf->testVar(MStrings[i-1], HDIV);
}

VarPtr OldroydBFormulationUW::v(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  const static vector<string> vStrings = {S_V1,S_V2,S_V3};
  return _vf->testVar(vStrings[i-1], HGRAD);
}
VarPtr OldroydBFormulationUW::S(int i, int j)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  TEUCHOS_TEST_FOR_EXCEPTION((j > _spaceDim) || (j < 1), std::invalid_argument, "j must be at least 1 and less than or equal to _spaceDim");
  const static vector<vector<string>> SStrings = {{S_S11, S_S12, S_S13},{S_S12, S_S22, S_S23},{S_S13, S_S23, S_S33}};
  return _vf->testVar(SStrings[i-1][j-1], HGRAD);
}

TRieszRepPtr<double> OldroydBFormulationUW::rieszResidual(FunctionPtr forcingFunction)
{
  // recompute residual with updated background flow
  // :: recall that the solution residual is the forcing term for the solution increment problem
  // _rhsForResidual = this->rhs(forcingFunction, false);
  LinearTermPtr residual = _rhsForResidual->linearTermCopy();
  residual->addTerm(-_oldroydBBF->testFunctional(_solnIncrement));
  RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(_solnIncrement->mesh(), _solnIncrement->ip(), residual));
  return rieszResidual;
}


// ! Saves the solution(s) and mesh to an HDF5 format.
void OldroydBFormulationUW::save(std::string prefixString)
{
  _backgroundFlow->mesh()->saveToHDF5(prefixString+".mesh");
  _backgroundFlow->saveToHDF5(prefixString+".soln");

  if (_streamSolution != Teuchos::null)
  {
    _streamSolution->mesh()->saveToHDF5(prefixString+"_stream.mesh");
    _streamSolution->saveToHDF5(prefixString + "_stream.soln");
  }
}

// ! set current time step used for transient solve
void OldroydBFormulationUW::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

// ! Returns the solution (at current time)
TSolutionPtr<double> OldroydBFormulationUW::solution()
{
  return _backgroundFlow;
}

TSolutionPtr<double> OldroydBFormulationUW::solutionIncrement()
{
  return _solnIncrement;
}

void OldroydBFormulationUW::solveForIncrement()
{
  // before we solve, clear out the solnIncrement:
  this->clearSolutionIncrement();
  // (this matters for iterative solvers; otherwise we'd start with a bad initial guess after the first Newton step)

  RHSPtr savedRHS = _solnIncrement->rhs();
  _solnIncrement->setRHS(_rhsForSolve);
  _solnIncrement->solve(_solver);
  // _solnIncrement->condensedSolve(_solver);
  _solnIncrement->setRHS(savedRHS);
}

void OldroydBFormulationUW::accumulate(double weight)
{
  bool allowEmptyCells = false;
  _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  _nonlinearIterationCount++;
}

void OldroydBFormulationUW::solveAndAccumulate(double weight)
{
  // before we solve, clear out the solnIncrement:
  this->clearSolutionIncrement();
  // (this matters for iterative solvers; otherwise we'd start with a bad initial guess after the first Newton step)

  RHSPtr savedRHS = _solnIncrement->rhs();
  _solnIncrement->setRHS(_rhsForSolve);
  _solnIncrement->solve(_solver);
  _solnIncrement->setRHS(savedRHS);
  // mesh->registerSolution(_backgroundFlow);

  bool allowEmptyCells = false;
  _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  _nonlinearIterationCount++;
}

// double OldroydBFormulationUW::computeG(FunctionPtr forcingFunction, double weight)
double OldroydBFormulationUW::computeG(double weight)
{
  // // evaluate linearization at current guess against solution increment
  // LinearTermPtr soln_functional = _oldroydBBF->testFunctional(_solnIncrement);

  // // TSolutionPtr<double> temporarySolnBackground = 

  // // accumulate background flow
  // bool allowEmptyCells = false;
  // _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  // // this->setForcingFunction(Teuchos::null);
  // // RHSPtr savedRHS = _solnIncrement->rhs();
  // // _solnIncrement->setRHS(_rhsForResidual);

  // // calculate Riesz representation of accumulated nonlinear residual
  // // RHSPtr nonlinearRHS = this->rhs(Teuchos::null, _neglectFluxesOnRHS);
  // // LinearTermPtr residual = nonlinearRHS->linearTermCopy();
  // LinearTermPtr residual = _rhsForResidual->linearTermCopy();
  // // LinearTermPtr residual = (_solnIncrement->rhs())->linearTermCopy();

  // // RieszRepPtr rieszResidual = this->rieszResidual(Teuchos::null);
  // RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(_solnIncrement->mesh(), _solnIncrement->ip(), residual));

  // // G(\Delta u) = b(\Delta u,psi)
  // rieszResidual->computeRieszRep();

  // TFunctionPtr<double> dG = TFunction<double>::zero();
  // map<int,VarPtr> testVars = _vf->testVars();
  // for (auto entry : testVars)
  // {
  //   VarPtr var = entry.second;
  //   FunctionPtr var_err = Teuchos::rcp(new RepFunction<double>(var,rieszResidual));
  //   map<int,FunctionPtr> var_errFxn;
  //   var_errFxn[var->ID()] = var_err;
  //   dG = dG + soln_functional->evaluate(var_errFxn);
  // }

  // // FunctionPtr q_err = Teuchos::rcp(new RepFunction<double>(this->q(),rieszResidual));
  // // map<int,FunctionPtr> q_errFxn;
  // // q_errFxn[(this->q())->ID()] = q_err;
  // // TFunctionPtr<double> dG = soln_functional->evaluate(q_errFxn);

  // // FunctionPtr vi_err, Mi_err, Sij_err;
  // // map<int,FunctionPtr> vi_errFxn, Mi_errFxn, Sij_errFxn;
  // // for (int i = 1; i <= _spaceDim; ++i)
  // // {
  // //   vi_err = Teuchos::rcp(new RepFunction<double>(this->v(i),rieszResidual));
  // //   vi_errFxn[(this->v(i))->ID()] = vi_err;
  // //   dG = dG + soln_functional->evaluate(vi_errFxn);

  // //   Mi_err = Teuchos::rcp(new RepFunction<double>(this->M(i),rieszResidual));
  // //   Mi_errFxn[(this->M(i))->ID()] = Mi_err;
  // //   dG = dG + soln_functional->evaluate(Mi_errFxn);
  // //   for (int j = i; j <= _spaceDim; ++j)
  // //   {
  // //     Sij_err = Teuchos::rcp(new RepFunction<double>(this->S(i,j),rieszResidual));
  // //     Sij_errFxn[(this->S(i,j))->ID()] = Sij_err;
  // //     dG = dG + soln_functional->evaluate(Sij_errFxn);
  // //   }
  // // }

  // double G = dG->integrate(_backgroundFlow->mesh());

  // // reset background flow
  // _backgroundFlow->addSolution(_solnIncrement, -weight, allowEmptyCells, _neglectFluxesOnRHS);
  // // _solnIncrement->clearComputedResiduals();
  // // _solnIncrement->setRHS(savedRHS);
  // // this->setForcingFunction(Teuchos::null);

  return 1.0-weight;
  // return G;

}


// ! Returns the solution (at previous time)
TSolutionPtr<double> OldroydBFormulationUW::solutionPreviousTimeStep()
{
  return _previousSolution;
}

// ! Solves iteratively
void OldroydBFormulationUW::solveIteratively(int maxIters, double cgTol, int azOutputLevel, bool suppressSuperLUOutput)
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

int OldroydBFormulationUW::spaceDim()
{
  return _spaceDim;
}

PoissonFormulation & OldroydBFormulationUW::streamFormulation()
{
  return *_streamFormulation;
}

VarPtr OldroydBFormulationUW::streamPhi()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamFormulation->phi();
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

TSolutionPtr<double> OldroydBFormulationUW::streamSolution()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamSolution;
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

SolverPtr OldroydBFormulationUW::getSolver()
{
  return _solver;
}

void OldroydBFormulationUW::setSolver(SolverPtr solver)
{
  _solver = solver;
}

// ! Returns the sum of the time steps taken thus far.
double OldroydBFormulationUW::getTime()
{
  return _time;
}

TFunctionPtr<double> OldroydBFormulationUW::getTimeFunction()
{
  return _t;
}

void OldroydBFormulationUW::turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver){
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


LinearTermPtr OldroydBFormulationUW::getTraction(int i)
{
  CHECK_VALID_COMPONENT(i);
  switch (i)
  {
    case 1:
      return _t1;
    case 2:
      return _t2;
    case 3:
      return _t3;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

TFunctionPtr<double> OldroydBFormulationUW::getPressureSolution()
{
  TFunctionPtr<double> p_soln = Function::solution(p(), _backgroundFlow);
  return p_soln;
}

const std::map<int,int> & OldroydBFormulationUW::getTrialVariablePolyOrderAdjustments()
{
  return _trialVariablePolyOrderAdjustments;
}

TFunctionPtr<double> OldroydBFormulationUW::getVelocitySolution()
{
  vector<FunctionPtr> u_components;
  for (int d=1; d<=_spaceDim; d++)
  {
    u_components.push_back(Function::solution(u(d), _backgroundFlow));
  }
  return Function::vectorize(u_components);
}

TFunctionPtr<double> OldroydBFormulationUW::getVorticity()
{
  LinearTermPtr u1_dy = (1.0 / _muS) * this->L(1,2);
  LinearTermPtr u2_dx = (1.0 / _muS) * this->L(2,1);

  TFunctionPtr<double> vorticity = Teuchos::rcp( new PreviousSolutionFunction<double>(_backgroundFlow, u2_dx - u1_dy) );
  return vorticity;
}

// VarPtr OldroydBFormulationUW::addFieldVar(std::string name)
// {
//   VarPtr var = _vf->fieldVar(name);
//   return var;
// }

// VarPtr OldroydBFormulationUW::addTraceVar(std::string name)
// {
//   VarPtr var = _vf->traceVar(name);
//   return var;
// }

// VarPtr OldroydBFormulationUW::addFluxVar(std::string name)
// {
//   VarPtr var = _vf->fluxVar(name);
//   return var;
// }

// VarPtr OldroydBFormulationUW::addTestVar(std::string name, Space fs)
// {
//   VarPtr var = _vf->testVar(name, fs);
//   return var;
// }


// TO COMPUTE DRAG COEFFICIENT around cylinder

// TFunctionPtr<double> OldroydBFormulationUW::friction(SolutionPtr soln)
// {
//   if (_spaceDim == 2)
//   {
//     // friction is given by (sigma n) x n (that's a cross product)
//     TFunctionPtr<double> n = Function::normal();
//     // LinearTermPtr f_lt = n->y() * (sigma11->times_normal_x() + sigma12->times_normal_y())
//     //                    - n->x() * (sigma21->times_normal_x() + sigma22->times_normal_y());
//     LinearTermPtr f_lt = n->y() * this->sigman_hat(1) - n->x() * this->sigman_hat(2);

//     TFunctionPtr<double> f = Teuchos::rcp( new PreviousSolutionFunction<double>(soln, f_lt) );
//     return f;
//   }
//   else
//   {
//     cout << "ERROR: this function is only supported on 2D solutions.  Returning null.\n";
//     return Teuchos::null;
//   }
// }
