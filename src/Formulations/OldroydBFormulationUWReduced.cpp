//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  OldroydBFormulationUWReduced.cpp
//  Camellia
//
//  Created by Brendan Keith, April 2018
//
//

#include "OldroydBFormulationUWReduced.h"

#include "ConstantScalarFunction.h"
#include "Constraint.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "H1ProjectionFormulation.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"
#include "LagrangeConstraints.h"


using namespace Camellia;

const string OldroydBFormulationUWReduced::S_U1 = "u_1";
const string OldroydBFormulationUWReduced::S_U2 = "u_2";
const string OldroydBFormulationUWReduced::S_U3 = "u_3";
const string OldroydBFormulationUWReduced::S_L11 = "L_{11}";
const string OldroydBFormulationUWReduced::S_L12 = "L_{12}";
const string OldroydBFormulationUWReduced::S_L13 = "L_{13}";
const string OldroydBFormulationUWReduced::S_L21 = "L_{21}";
const string OldroydBFormulationUWReduced::S_L22 = "L_{22}";
const string OldroydBFormulationUWReduced::S_L23 = "L_{23}";
const string OldroydBFormulationUWReduced::S_L31 = "L_{31}";
const string OldroydBFormulationUWReduced::S_L32 = "L_{32}";
const string OldroydBFormulationUWReduced::S_L33 = "L_{33}";
const string OldroydBFormulationUWReduced::S_T11 = "T_{11}";
const string OldroydBFormulationUWReduced::S_T12 = "T_{12}";
const string OldroydBFormulationUWReduced::S_T13 = "T_{13}";
const string OldroydBFormulationUWReduced::S_T22 = "T_{22}";
const string OldroydBFormulationUWReduced::S_T23 = "T_{23}";
const string OldroydBFormulationUWReduced::S_T33 = "T_{33}";
const string OldroydBFormulationUWReduced::S_P = "p";

const string OldroydBFormulationUWReduced::S_U1_HAT = "\\widehat{u}_1";
const string OldroydBFormulationUWReduced::S_U2_HAT = "\\widehat{u}_2";
const string OldroydBFormulationUWReduced::S_U3_HAT = "\\widehat{u}_3";
const string OldroydBFormulationUWReduced::S_SIGMAN1_HAT = "\\widehat{\\sigma}_{1n}";
const string OldroydBFormulationUWReduced::S_SIGMAN2_HAT = "\\widehat{\\sigma}_{2n}";
const string OldroydBFormulationUWReduced::S_SIGMAN3_HAT = "\\widehat{\\sigma}_{3n}";
const string OldroydBFormulationUWReduced::S_TUN11_HAT = "\\hat{(T\\otimes u)_{n_{11}}}";
const string OldroydBFormulationUWReduced::S_TUN12_HAT = "\\hat{(T\\otimes u)_{n_{12}}}";
const string OldroydBFormulationUWReduced::S_TUN13_HAT = "\\hat{(T\\otimes u)_{n_{13}}}";
const string OldroydBFormulationUWReduced::S_TUN22_HAT = "\\hat{(T\\otimes u)_{n_{22}}}";
const string OldroydBFormulationUWReduced::S_TUN23_HAT = "\\hat{(T\\otimes u)_{n_{23}}}";
const string OldroydBFormulationUWReduced::S_TUN33_HAT = "\\hat{(T\\otimes u)_{n_{33}}}";

const string OldroydBFormulationUWReduced::S_V1 = "v_1";
const string OldroydBFormulationUWReduced::S_V2 = "v_2";
const string OldroydBFormulationUWReduced::S_V3 = "v_3";
const string OldroydBFormulationUWReduced::S_M1 = "M_{1}";
const string OldroydBFormulationUWReduced::S_M2 = "M_{2}";
const string OldroydBFormulationUWReduced::S_M3 = "M_{3}";
const string OldroydBFormulationUWReduced::S_S11 = "S_{11}";
const string OldroydBFormulationUWReduced::S_S12 = "S_{12}";
const string OldroydBFormulationUWReduced::S_S13 = "S_{13}";
const string OldroydBFormulationUWReduced::S_S22 = "S_{22}";
const string OldroydBFormulationUWReduced::S_S23 = "S_{23}";
const string OldroydBFormulationUWReduced::S_S33 = "S_{33}";

// OldroydBFormulationUWReduced OldroydBFormulationUWReduced::steadyFormulation(int spaceDim, double mu, bool useConformingTraces)
// {
//   Teuchos::ParameterList parameters;

//   parameters.set("spaceDim", spaceDim);
//   parameters.set("mu",mu);
//   parameters.set("useConformingTraces",useConformingTraces);
//   parameters.set("useTimeStepping", false);
//   parameters.set("useSpaceTime", false);

//   return OldroydBFormulationUWReduced(parameters);
// }

// OldroydBFormulationUWReduced OldroydBFormulationUWReduced::spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces,
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

//   return OldroydBFormulationUWReduced(parameters);
// }

// OldroydBFormulationUWReduced OldroydBFormulationUWReduced::timeSteppingFormulation(int spaceDim, double mu, double dt,
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

//   return OldroydBFormulationUWReduced(parameters);
// }

OldroydBFormulationUWReduced::OldroydBFormulationUWReduced(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  double rho = parameters.get<double>("rho",1.0); // density
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
  _temporalPolyOrder = temporalPolyOrder;
  _rho = rho;
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

  TEUCHOS_TEST_FOR_EXCEPTION((spaceDim != 2), std::invalid_argument, "spaceDim must be 2 ");
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

  _vf = VarFactory::varFactory();
  u1 = _vf->fieldVar(S_U1);
  u2 = _vf->fieldVar(S_U2);

  vector<VarPtr> u(spaceDim);
  u[0] = u1;
  u[1] = u2;

  p = _vf->fieldVar(S_P);

  vector<vector<VarPtr>> L(spaceDim,vector<VarPtr>(spaceDim));
  L11 = _vf->fieldVar(S_L11);
  L12 = _vf->fieldVar(S_L12);
  L21 = _vf->fieldVar(S_L21);
  // L22 = L11;
  L[0][0] = L11;
  L[0][1] = L12;
  L[1][0] = L21;
  L[1][1] = L11;

  vector<vector<VarPtr>> T(spaceDim,vector<VarPtr>(spaceDim));
  T11 = _vf->fieldVar(S_T11);
  T12 = _vf->fieldVar(S_T12);
  T22 = _vf->fieldVar(S_T22);
  T[0][0] = T11;
  T[0][1] = T12;
  T[1][0] = T12;
  T[1][1] = T22;

  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
  u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);

  TFunctionPtr<double> n = TFunction<double>::normal();

  // Too complicated at the moment to define where these other trace variables comes from
  sigma1n_hat = _vf->fluxVar(S_SIGMAN1_HAT);
  sigma2n_hat = _vf->fluxVar(S_SIGMAN2_HAT);

  Tu11n_hat = _vf->fluxVar(S_TUN11_HAT);
  Tu12n_hat = _vf->fluxVar(S_TUN12_HAT);
  Tu22n_hat = _vf->fluxVar(S_TUN22_HAT);

  v1 = _vf->testVar(S_V1, HGRAD);
  v2 = _vf->testVar(S_V2, HGRAD);

  M1 = _vf->testVar(S_M1, HDIV);
  M2 = _vf->testVar(S_M2, HDIV);

  vector<vector<VarPtr>> S(spaceDim,vector<VarPtr>(spaceDim));
  S11 = _vf->testVar(S_S11, HGRAD);
  S12 = _vf->testVar(S_S12, HGRAD);
  S22 = _vf->testVar(S_S22, HGRAD);
  S[0][0] = S11;
  S[0][1] = S12;
  S[1][0] = S12;
  S[1][1] = S22;

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
  _steadyStokesBF->addTerm(-_muS * u1_hat, M1->dot_normal());

  // M2 terms:
  _steadyStokesBF->addTerm(_muS * u2, M2->div());
  _steadyStokesBF->addTerm(L21, M2->x());
  _steadyStokesBF->addTerm(-1.0 * L11, M2->y());
  _steadyStokesBF->addTerm(-_muS * u2_hat, M2->dot_normal());

  // v1:
  _steadyStokesBF->addTerm(L11, v1->dx()); // (L1, grad v1)
  _steadyStokesBF->addTerm(L12, v1->dy());
  _steadyStokesBF->addTerm( - p, v1->dx() );
  _steadyStokesBF->addTerm( sigma1n_hat, v1);

  // v2:
  _steadyStokesBF->addTerm(L21, v2->dx()); // (L2, grad v2)
  _steadyStokesBF->addTerm(-1.0 * L11, v2->dy());
  _steadyStokesBF->addTerm( - p, v2->dy());
  _steadyStokesBF->addTerm( sigma2n_hat, v2);

  _oldroydBBF = _steadyStokesBF;

  // NONLINEAR TERMS //

  vector<int> H1Order;
  H1Order = {spatialPolyOrder+1};

  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, _oldroydBBF, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;

  _backgroundFlow = TSolution<double>::solution(mesh);
  _solnIncrement = TSolution<double>::solution(mesh);


  // CONSERVATION  OF MOMENTUM

  // convective terms:
  // vector<FunctionPtr> L_prev, u_prev;

  double Re = _rho / _muS;
  // double Re = 1.0 / _muS;
  // TFunctionPtr<double> p_prev = TFunction<double>::solution(this->p(), _backgroundFlow);

  if (!_stokesOnly)
  {
    if (!_conservationFormulation)
    {
      for (int comp_i=1; comp_i <= _spaceDim; comp_i++)
      {
        VarPtr v_i = this->v(comp_i);

        for (int comp_j=1; comp_j <= _spaceDim; comp_j++)
        {
          VarPtr u_j = this->u(comp_j);
          VarPtr L_ij = this->L(comp_i, comp_j);

          double mult_ij = 1.0;
          if (comp_i == 2 && comp_j == 2)
          {
            mult_ij =-1.0;
          }

          FunctionPtr u_prev_j = TFunction<double>::solution(u_j, _backgroundFlow);
          FunctionPtr L_prev_ij = TFunction<double>::solution(L_ij, _backgroundFlow);

          _oldroydBBF->addTerm( mult_ij * Re * L_prev_ij * u_j, v_i);
          _oldroydBBF->addTerm( mult_ij * Re * u_prev_j * L_ij, v_i);
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
  _oldroydBBF->addTerm(T11, v1->dx());
  _oldroydBBF->addTerm(T12, v1->dy());
  _oldroydBBF->addTerm(T12, v2->dx());
  _oldroydBBF->addTerm(T22, v2->dy());


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

      double mult_ij = 1.0;
      if (comp_i == 2 && comp_j == 2)
      {
        mult_ij =-1.0;
      }

      FunctionPtr T_prev_ij = TFunction<double>::solution(T_ij, _backgroundFlow);

      _oldroydBBF->addTerm( T_ij, S_ij);
      //
      _oldroydBBF->addTerm( lambdaFxn * Tu_ijn_hat, S_ij);
      //
      _oldroydBBF->addTerm( mult_ij * -2 / _muS * _muP * L_ij, S_ij);

      for (int comp_k=1; comp_k <= _spaceDim; comp_k++)
      {
        VarPtr u_k = this->u(comp_k);
        VarPtr L_ik = this->L(comp_i, comp_k);
        VarPtr T_kj = this->T(comp_k, comp_j);

        double mult_ik = 1.0;
        if (comp_i == 2 && comp_k == 2)
        {
          mult_ik =-1.0;
        }

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

          default:
            break;
        }
        //
        _oldroydBBF->addTerm( mult_ik * -2 * lambdaFxn / _muS * L_prev_ik * T_kj, S_ij);
        _oldroydBBF->addTerm( mult_ik * -2 * lambdaFxn / _muS * T_prev_kj * L_ik, S_ij);

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
  _t1 = n->x() * (2 * L11 - p)       + n->y() * (L11 + L21);
  _t2 = n->x() * (L12 + L21) + n->y() * (-2 * L11 - p);


  // cout << endl << _oldroydBBF->displayString() << endl;

  // set the inner product to the graph norm:
  this->setIP( _oldroydBBF->graphNorm() );
  // setIP( _oldroydBBF->graphNorm() );

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
  TFunctionPtr<double> p_prev = TFunction<double>::solution(this->p(), _backgroundFlow);

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
      // CONSERVATION OF MOMENTUM (if Stokes)
      // if (_stokesOnly)
      // {
      //   // we are assuming that there is no body forcing in the problem.
      //   FunctionPtr x    = Function::xn(1);
      //   FunctionPtr y    = Function::yn(1);
      //   _solnIncrement->lagrangeConstraints()->addConstraint(sigma1n_hat == zero);
      //   _solnIncrement->lagrangeConstraints()->addConstraint(sigma2n_hat == zero);
      //   _solnIncrement->lagrangeConstraints()->addConstraint(-y*sigma1n_hat + x*sigma2n_hat == zero); // seems to upset convergence 
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

void OldroydBFormulationUWReduced::addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u)
{
  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);

  _solnIncrement->bc()->addDirichlet(u1_hat, inflowRegion, u->x());
  _solnIncrement->bc()->addDirichlet(u2_hat, inflowRegion, u->y());

}

void OldroydBFormulationUWReduced::addInflowViscoelasticStress(SpatialFilterPtr inflowRegion, TFunctionPtr<double> T11un, TFunctionPtr<double> T12un, TFunctionPtr<double> T22un)
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
}

void OldroydBFormulationUWReduced::addOutflowCondition(SpatialFilterPtr outflowRegion, double yMax, double muP, double lambda, bool usePhysicalTractions)
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

    if (pcRCP != Teuchos::null)   // i.e., we're not just adding to a prior PenaltyConstraints object
    {
      _solnIncrement->setFilter(pcRCP);
    }
  }
  else
  {
    TFunctionPtr<double> zero = TFunction<double>::zero();
    _solnIncrement->bc()->addDirichlet(this->sigman_hat(1), outflowRegion, zero);
    _solnIncrement->bc()->addDirichlet(this->u_hat(2), outflowRegion, zero);
  }
}

void OldroydBFormulationUWReduced::addPointPressureCondition(vector<double> vertex)
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

  if (_solnIncrement->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    _solnIncrement->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void OldroydBFormulationUWReduced::addWallCondition(SpatialFilterPtr wall)
{
  vector<double> zero(_spaceDim, 0.0);
  addInflowCondition(wall, TFunction<double>::constant(zero));
}

void OldroydBFormulationUWReduced::addSymmetryCondition(SpatialFilterPtr symmetryRegion)
{
  TFunctionPtr<double> zero = TFunction<double>::zero();
  _solnIncrement->bc()->addDirichlet(this->sigman_hat(1), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->Tun_hat(1,1), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->Tun_hat(1,2), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->Tun_hat(2,2), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->u_hat(2), symmetryRegion, zero);
  _solnIncrement->bc()->addDirichlet(this->T(1,2), symmetryRegion, zero);
}

void OldroydBFormulationUWReduced::addZeroMeanPressureCondition()
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

BFPtr OldroydBFormulationUWReduced::bf()
{
  return _oldroydBBF;
}

void OldroydBFormulationUWReduced::clearSolutionIncrement()
{
  _solnIncrement->clear(); // only clears the local cell coefficients, not the global solution vector
  if (_solnIncrement->getLHSVector().get() != NULL)
    _solnIncrement->getLHSVector()->PutScalar(0); // this clears global solution vector
  _solnIncrement->clearComputedResiduals();
}

void OldroydBFormulationUWReduced::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

FunctionPtr OldroydBFormulationUWReduced::convectiveTerm(int spaceDim, FunctionPtr u_exact)
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

void OldroydBFormulationUWReduced::setForcingFunction(FunctionPtr forcingFunction)
{
  // set the RHS:
  if (forcingFunction == Teuchos::null)
  {
    FunctionPtr scalarZero = Function::zero();
    if (_spaceDim == 1)
      forcingFunction = scalarZero;
    else if (_spaceDim == 2)
      forcingFunction = Function::vectorize(scalarZero, scalarZero);
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported space dimension");
  }

  _rhsForSolve = this->rhs(forcingFunction, _neglectFluxesOnRHS);
  _rhsForResidual = this->rhs(forcingFunction, false);
  _solnIncrement->setRHS(_rhsForSolve);
}

// void OldroydBFormulationUWReduced::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
//     TFunctionPtr<double> forcingFunction, int temporalPolyOrder)
// {
//   this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction, "", temporalPolyOrder);
// }

// void OldroydBFormulationUWReduced::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k,
//     TFunctionPtr<double> forcingFunction, int temporalPolyOrder)
// {
//   this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, forcingFunction, filePrefix, temporalPolyOrder);
// }

// void OldroydBFormulationUWReduced::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
//     TFunctionPtr<double> forcingFunction, string savedSolutionAndMeshPrefix, int temporalPolyOrder)
// {
//   _haveOutflowConditionsImposed = false;
//   BCPtr bc = BC::bc();

//   vector<int> H1Order {fieldPolyOrder + 1};
//   MeshPtr mesh;
//   if (savedSolutionAndMeshPrefix == "")
//   {
//     if (_spaceTime) H1Order.push_back(temporalPolyOrder+1);
//     mesh = Teuchos::rcp( new Mesh(meshTopo, _oldroydBBF, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
//     _solution = TSolution<double>::solution(mesh,bc);
//   }
//   else
//   {
//     mesh = MeshFactory::loadFromHDF5(_oldroydBBF, savedSolutionAndMeshPrefix+".mesh");
//     _solution = TSolution<double>::solution(mesh, bc);
//     _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
//   }

//   RHSPtr rhs = this->rhs(forcingFunction, _neglectFluxesOnRHS); // in transient case, this will refer to _previousSolution
//   IPPtr ip = _oldroydBBF->graphNorm();

// //  cout << "graph norm for Stokes BF:\n";
// //  ip->printInteractions();

//   _solution->setRHS(rhs);
//   _solution->setIP(ip);

//   mesh->registerSolution(_solution); // will project both time steps during refinements...

//   LinearTermPtr residual = rhs->linearTerm() - _oldroydBBF->testFunctional(_solution,false); // false: don't exclude boundary terms

//   double energyThreshold = 0.2;
//   _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );

//   double maxDouble = std::numeric_limits<double>::max();
//   double maxP = 20;
//   _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold, 0, 0, false ) );
//   _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold, maxDouble, maxP, true ) );

//   _time = 0;
//   _t->setTime(_time);

//   if (_spaceDim==2)
//   {
//     // finally, set up a stream function solve for 2D
//     _streamFormulation = Teuchos::rcp( new PoissonFormulation(_spaceDim,_useConformingTraces) );

//     MeshPtr streamMesh;
//     if (savedSolutionAndMeshPrefix == "")
//     {
//       MeshTopologyPtr streamMeshTopo = meshTopo->deepCopy();
//       streamMesh = Teuchos::rcp( new Mesh(streamMeshTopo, _streamFormulation->bf(), H1Order, delta_k) ) ;
//     }
//     else
//     {
//       streamMesh = MeshFactory::loadFromHDF5(_streamFormulation->bf(), savedSolutionAndMeshPrefix+"_stream.mesh");
//     }

//     mesh->registerObserver(streamMesh); // refine streamMesh whenever mesh is refined

//     LinearTermPtr u1_dy = (1.0 / _muS) * this->L(1,2);
//     LinearTermPtr u2_dx = (1.0 / _muS) * this->L(2,1);

//     TFunctionPtr<double> vorticity = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, u2_dx - u1_dy) );
//     RHSPtr streamRHS = RHS::rhs();
//     VarPtr q_stream = _streamFormulation->v();
//     streamRHS->addTerm( -vorticity * q_stream );
//     bool dontWarnAboutOverriding = true;
//     ((PreviousSolutionFunction<double>*) vorticity.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

//     /* Stream function phi is such that
//      *    d/dx phi = -u2
//      *    d/dy phi =  u1
//      * Therefore, psi = grad phi = (-u2, u1), and psi * n = u1 n2 - u2 n1
//      */

//     TFunctionPtr<double> u1_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, this->u(1) ) );
//     TFunctionPtr<double> u2_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, this->u(2) ) );
//     ((PreviousSolutionFunction<double>*) u1_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
//     ((PreviousSolutionFunction<double>*) u2_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

//     TFunctionPtr<double> n = TFunction<double>::normal();

//     BCPtr streamBC = BC::bc();
//     VarPtr phi = _streamFormulation->u();
//     streamBC->addZeroMeanConstraint(phi);

//     VarPtr psi_n = _streamFormulation->sigma_n_hat();
//     streamBC->addDirichlet(psi_n, SpatialFilter::allSpace(), u1_soln * n->y() - u2_soln * n->x());

//     IPPtr streamIP = _streamFormulation->bf()->graphNorm();
//     _streamSolution = TSolution<double>::solution(streamMesh,streamBC,streamRHS,streamIP);

//     if (savedSolutionAndMeshPrefix != "")
//     {
//       _streamSolution->loadFromHDF5(savedSolutionAndMeshPrefix + "_stream.soln");
//     }
//   }
// }

bool OldroydBFormulationUWReduced::isSpaceTime() const
{
  return _spaceTime;
}

bool OldroydBFormulationUWReduced::isSteady() const
{
  return !_timeStepping && !_spaceTime;
}


bool OldroydBFormulationUWReduced::isTimeStepping() const
{
  return _timeStepping;
}

void OldroydBFormulationUWReduced::setIP(IPPtr ip)
{
  _solnIncrement->setIP(ip);
}

double OldroydBFormulationUWReduced::relativeL2NormOfTimeStep()
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

double OldroydBFormulationUWReduced::L2NormSolution()
{
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double OldroydBFormulationUWReduced::L2NormSolutionIncrement()
{
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

int OldroydBFormulationUWReduced::nonlinearIterationCount()
{
  return _nonlinearIterationCount;
}

double OldroydBFormulationUWReduced::rho()
{
  return _rho;
}

double OldroydBFormulationUWReduced::muS()
{
  return _muS;
}

double OldroydBFormulationUWReduced::muP()
{
  return _muP;
}

Teuchos::RCP<ParameterFunction> OldroydBFormulationUWReduced::lambda()
{
  return _lambda;
}

double OldroydBFormulationUWReduced::alpha()
{
  return _alpha;
}

// ! set lambda during continuation
void OldroydBFormulationUWReduced::setLambda(double lambda)
{
  _lambda->setValue(lambda);
}

RefinementStrategyPtr OldroydBFormulationUWReduced::getRefinementStrategy()
{
  return _refinementStrategy;
}

void OldroydBFormulationUWReduced::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void OldroydBFormulationUWReduced::refine()
{
  _refinementStrategy->refine();
}

void OldroydBFormulationUWReduced::hRefine()
{
  _hRefinementStrategy->refine();
}

void OldroydBFormulationUWReduced::pRefine()
{
  _pRefinementStrategy->refine();
}

RHSPtr OldroydBFormulationUWReduced::rhs(TFunctionPtr<double> f, bool excludeFluxesAndTraces)
{

  // TO DO : UPDATE THIS!
  RHSPtr rhs = RHS::rhs();

  TSolutionPtr<double> backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);

  TFunctionPtr<double> p_prev;
  TFunctionPtr<double> u1_prev, u2_prev, u3_prev;
  TFunctionPtr<double> L11_prev, L12_prev, L13_prev, L21_prev, L22_prev, L23_prev, L31_prev, L32_prev, L33_prev;

  VarPtr v1, v2, v3;
  VarPtr M1, M2, M3;

  switch (_spaceDim) {
          case 2:
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
  }

  // subtract the stokesBF from the RHS (this doesn't work well for some reason)
  // rhs->addTerm( -_steadyStokesBF->testFunctional(backgroundFlowWeakReference, excludeFluxesAndTraces) );

  // STOKES part
  double muS = this->muS();

  // M1 terms:
  rhs->addTerm( -muS * u1_prev * M1->div()); // L1 = muS * du1/dx
  rhs->addTerm( -L11_prev * M1->x()); // (L1, M1)
  rhs->addTerm( -L12_prev * M1->y());

  // M2 terms:
  rhs->addTerm( -muS * u2_prev * M2->div());
  rhs->addTerm( -L21_prev * M2->x());
  rhs->addTerm(  L22_prev * M2->y());

  // v1:
  rhs->addTerm( -L11_prev * v1->dx()); // (L1, grad v1)
  rhs->addTerm( -L12_prev * v1->dy());
  rhs->addTerm( p_prev * v1->dx() );

  // v2:

  rhs->addTerm( -L21_prev * v2->dx()); // (L2, grad v2)
  rhs->addTerm(  L22_prev * v2->dy());
  rhs->addTerm( p_prev * v2->dy());

  // add the u L term:
  double Re = this->rho() / muS;
  // double Re = 1.0 / muS;
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

          double mult_ij = 1.0;
          if (comp_i == 2 && comp_j == 2)
          {
            mult_ij =-1.0;
          }

          TFunctionPtr<double> L_ij_prev = TFunction<double>::solution(L_ij, backgroundFlowWeakReference);
          rhs->addTerm((-Re * mult_ij * uj_prev * L_ij_prev) * vi);
        }
      }
    }
    else
    {
      rhs->addTerm( u1_prev * u1_prev * v1->dx() );
      rhs->addTerm( u1_prev * u2_prev * v1->dy() );
      rhs->addTerm( u2_prev * u1_prev * v2->dx() );
      rhs->addTerm( u2_prev * u2_prev * v2->dy() );
    }
  }

  VarPtr T11, T12, T22, T13, T23, T33;
  TFunctionPtr<double> T11_prev, T12_prev, T22_prev, T13_prev, T23_prev, T33_prev;

  // new constitutive terms:
  switch (_spaceDim) {
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

      double mult_ij = 1.0;
      if (comp_i == 2 && comp_j == 2)
      {
        mult_ij =-1.0;
      }

      rhs->addTerm( mult_ij * 2.0 * muP / muS * L_ij_prev * S_ij);

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

          default:
            break;
        }

        double mult_ik = 1.0;
        if (comp_i == 2 && comp_k == 2)
        {
          mult_ik =-1.0;
        }


        rhs->addTerm( mult_ik * 2.0 * lambdaFxn / muS * L_ik_prev * T_kj_prev * S_ij);

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

VarPtr OldroydBFormulationUWReduced::L(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> LStrings = {{S_L11, S_L12, S_L13},{S_L21, S_L22, S_L23},{S_L31, S_L32, S_L33}};

  if (i == 2 && j == 2)
  {
    return _vf->fieldVar(LStrings[0][0]);
  }
  else
  {
    return _vf->fieldVar(LStrings[i-1][j-1]);
  }
}

VarPtr OldroydBFormulationUWReduced::u(int i)
{
  CHECK_VALID_COMPONENT(i);

  static const vector<string> uStrings = {S_U1,S_U2,S_U3};
  return _vf->fieldVar(uStrings[i-1]);
}

VarPtr OldroydBFormulationUWReduced::T(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> TStrings = {{S_T11, S_T12, S_T13},{S_T12, S_T22, S_T23},{S_T13, S_T23, S_T33}};

  return _vf->fieldVar(TStrings[i-1][j-1]);
}

VarPtr OldroydBFormulationUWReduced::p()
{
  return _vf->fieldVar(S_P);
}

// traces:
VarPtr OldroydBFormulationUWReduced::sigman_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> sigmanStrings = {S_SIGMAN1_HAT,S_SIGMAN2_HAT,S_SIGMAN3_HAT};
  return _vf->fluxVar(sigmanStrings[i-1]);
}

VarPtr OldroydBFormulationUWReduced::u_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> uHatStrings = {S_U1_HAT,S_U2_HAT,S_U3_HAT};
  return _vf->traceVar(uHatStrings[i-1]);
}

VarPtr OldroydBFormulationUWReduced::Tun_hat(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> TunHatStrings = {{S_TUN11_HAT, S_TUN12_HAT, S_TUN13_HAT},{S_TUN12_HAT, S_TUN22_HAT, S_TUN23_HAT},{S_TUN13_HAT, S_TUN23_HAT, S_TUN33_HAT}};;
  return _vf->traceVar(TunHatStrings[i-1][j-1]);
}

// test variables:
VarPtr OldroydBFormulationUWReduced::M(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  const static vector<string> MStrings = {S_M1,S_M2,S_M3};
  return _vf->testVar(MStrings[i-1], HDIV);
}

VarPtr OldroydBFormulationUWReduced::v(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  const static vector<string> vStrings = {S_V1,S_V2,S_V3};
  return _vf->testVar(vStrings[i-1], HGRAD);
}
VarPtr OldroydBFormulationUWReduced::S(int i, int j)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  TEUCHOS_TEST_FOR_EXCEPTION((j > _spaceDim) || (j < 1), std::invalid_argument, "j must be at least 1 and less than or equal to _spaceDim");
  const static vector<vector<string>> SStrings = {{S_S11, S_S12, S_S13},{S_S12, S_S22, S_S23},{S_S13, S_S23, S_S33}};
  return _vf->testVar(SStrings[i-1][j-1], HGRAD);
}

TRieszRepPtr<double> OldroydBFormulationUWReduced::rieszResidual(FunctionPtr forcingFunction)
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
void OldroydBFormulationUWReduced::save(std::string prefixString)
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
void OldroydBFormulationUWReduced::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

// ! Returns the solution (at current time)
TSolutionPtr<double> OldroydBFormulationUWReduced::solution()
{
  return _backgroundFlow;
}

TSolutionPtr<double> OldroydBFormulationUWReduced::solutionIncrement()
{
  return _solnIncrement;
}

void OldroydBFormulationUWReduced::solveForIncrement()
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

void OldroydBFormulationUWReduced::accumulate(double weight)
{
  bool allowEmptyCells = false;
  _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  _nonlinearIterationCount++;
}

void OldroydBFormulationUWReduced::solveAndAccumulate(double weight)
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

// ! Returns the solution (at previous time)
TSolutionPtr<double> OldroydBFormulationUWReduced::solutionPreviousTimeStep()
{
  return _previousSolution;
}

// ! Solves iteratively
void OldroydBFormulationUWReduced::solveIteratively(int maxIters, double cgTol, int azOutputLevel, bool suppressSuperLUOutput)
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

int OldroydBFormulationUWReduced::spaceDim()
{
  return _spaceDim;
}

PoissonFormulation & OldroydBFormulationUWReduced::streamFormulation()
{
  return *_streamFormulation;
}

VarPtr OldroydBFormulationUWReduced::streamPhi()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamFormulation->u();
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

TSolutionPtr<double> OldroydBFormulationUWReduced::streamSolution()
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

SolverPtr OldroydBFormulationUWReduced::getSolver()
{
  return _solver;
}

void OldroydBFormulationUWReduced::setSolver(SolverPtr solver)
{
  _solver = solver;
}

// ! Returns the sum of the time steps taken thus far.
double OldroydBFormulationUWReduced::getTime()
{
  return _time;
}

TFunctionPtr<double> OldroydBFormulationUWReduced::getTimeFunction()
{
  return _t;
}

void OldroydBFormulationUWReduced::turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver){
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


LinearTermPtr OldroydBFormulationUWReduced::getTraction(int i)
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

TFunctionPtr<double> OldroydBFormulationUWReduced::getPressureSolution()
{
  TFunctionPtr<double> p_soln = Function::solution(p(), _backgroundFlow);
  return p_soln;
}

const std::map<int,int> & OldroydBFormulationUWReduced::getTrialVariablePolyOrderAdjustments()
{
  return _trialVariablePolyOrderAdjustments;
}

TFunctionPtr<double> OldroydBFormulationUWReduced::getVelocitySolution()
{
  vector<FunctionPtr> u_components;
  for (int d=1; d<=_spaceDim; d++)
  {
    u_components.push_back(Function::solution(u(d), _backgroundFlow));
  }
  return Function::vectorize(u_components);
}

TFunctionPtr<double> OldroydBFormulationUWReduced::getVorticity()
{
  FunctionPtr u1_dy = (1.0 / _muS) * Function::solution(this->L(1,2), _backgroundFlow);
  FunctionPtr u2_dx = (1.0 / _muS) * Function::solution(this->L(2,1), _backgroundFlow);
  
  FunctionPtr vorticity = u2_dx - u1_dy;
  
  return vorticity;
}
