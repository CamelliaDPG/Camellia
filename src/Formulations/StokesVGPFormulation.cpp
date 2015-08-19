//
//  StokesVGPFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "StokesVGPFormulation.h"

#include "Constraint.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"
#include "SimpleFunction.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;

const string StokesVGPFormulation::S_U1 = "u_1";
const string StokesVGPFormulation::S_U2 = "u_2";
const string StokesVGPFormulation::S_U3 = "u_3";
const string StokesVGPFormulation::S_P = "p";
const string StokesVGPFormulation::S_SIGMA1 = "\\sigma_{1}";
const string StokesVGPFormulation::S_SIGMA2 = "\\sigma_{2}";
const string StokesVGPFormulation::S_SIGMA3 = "\\sigma_{3}";

const string StokesVGPFormulation::S_U1_HAT = "\\widehat{u}_1";
const string StokesVGPFormulation::S_U2_HAT = "\\widehat{u}_2";
const string StokesVGPFormulation::S_U3_HAT = "\\widehat{u}_3";
const string StokesVGPFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string StokesVGPFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string StokesVGPFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string StokesVGPFormulation::S_V1 = "v_1";
const string StokesVGPFormulation::S_V2 = "v_2";
const string StokesVGPFormulation::S_V3 = "v_3";
const string StokesVGPFormulation::S_TAU1 = "\\tau_{1}";
const string StokesVGPFormulation::S_TAU2 = "\\tau_{2}";
const string StokesVGPFormulation::S_TAU3 = "\\tau_{3}";
const string StokesVGPFormulation::S_Q = "q";

StokesVGPFormulation StokesVGPFormulation::steadyFormulation(int spaceDim, double mu, bool useConformingTraces)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  
  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation StokesVGPFormulation::spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", true);
  
  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation StokesVGPFormulation::timeSteppingFormulation(int spaceDim, double mu, double dt,
                                                                   bool useConformingTraces)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  parameters.set("dt", dt);
  
  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation::StokesVGPFormulation(Teuchos::ParameterList &parameters)
{
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  double mu = parameters.get<double>("mu",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  
  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double dt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);

  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _mu = mu;
  _dt = ParameterFunction::parameterFunction(dt);
  _t = ParameterFunction::parameterFunction(0);
  
  _theta = ParameterFunction::parameterFunction(0.5); // Crank-Nicolson
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;
  
  if ((spaceDim != 2) && (spaceDim != 3))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
  }
  
  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr p;
  VarPtr sigma1, sigma2, sigma3;
  
  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr t1n, t2n, t3n;
  
  // tests
  VarPtr v1, v2, v3;
  VarPtr tau1, tau2, tau3;
  VarPtr q;
  
  _vf = VarFactory::varFactory();
  u1 = _vf->fieldVar(S_U1);
  u2 = _vf->fieldVar(S_U2);
  if (spaceDim==3) u3 = _vf->fieldVar(S_U3);
  
  p = _vf->fieldVar(S_P);
  
  sigma1 = _vf->fieldVar(S_SIGMA1, VECTOR_L2);
  sigma2 = _vf->fieldVar(S_SIGMA2, VECTOR_L2);
  if (spaceDim==3)
  {
    sigma3 = _vf->fieldVar(S_SIGMA3, VECTOR_L2);
  }
  
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  
  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  if (spaceDim > 0) u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
  if (spaceDim > 1) u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
  if (spaceDim > 2) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);
  
  TFunctionPtr<double> n = TFunction<double>::normal();
  TFunctionPtr<double> n_parity = n * TFunction<double>::sideParity();
  
  LinearTermPtr t1n_lt, t2n_lt, t3n_lt;
  FunctionPtr minus_n_parity = - n_parity; // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)

  if (spaceDim > 0) t1n_lt = p * n_parity->x() + sigma1 * minus_n_parity;
  if (spaceDim > 1) t2n_lt = p * n_parity->y() + sigma2 * minus_n_parity;
  if (spaceDim > 2) t3n_lt = p * n_parity->z() + sigma3 * minus_n_parity;

  if (!_spaceTime)
  {
    if (spaceDim > 0) t1n = _vf->fluxVar(S_TN1_HAT, t1n_lt);
    if (spaceDim > 1) t2n = _vf->fluxVar(S_TN2_HAT, t2n_lt);
    if (spaceDim > 2) t3n = _vf->fluxVar(S_TN3_HAT, t3n_lt);
  }
  else
  {
    if (spaceDim > 0) t1n = _vf->fluxVarSpaceOnly(S_TN1_HAT, t1n_lt);
    if (spaceDim > 1) t2n = _vf->fluxVarSpaceOnly(S_TN2_HAT, t2n_lt);
    if (spaceDim > 2) t3n = _vf->fluxVarSpaceOnly(S_TN3_HAT, t3n_lt);
  }
  
  if (spaceDim > 0) v1 = _vf->testVar(S_V1, HGRAD);
  if (spaceDim > 1) v2 = _vf->testVar(S_V2, HGRAD);
  if (spaceDim > 2) v3 = _vf->testVar(S_V3, HGRAD);
  
  if (spaceDim > 0) tau1 = _vf->testVar(S_TAU1, HDIV);
  if (spaceDim > 1) tau2 = _vf->testVar(S_TAU2, HDIV);
  if (spaceDim > 2) tau3 = _vf->testVar(S_TAU3, HDIV);
  
  q = _vf->testVar(S_Q, HGRAD);
  
  _steadyStokesBF = Teuchos::rcp( new BF(_vf) );
  // tau1 terms:
  _steadyStokesBF->addTerm(u1, tau1->div());
  _steadyStokesBF->addTerm((1.0/_mu) * sigma1, tau1); // (sigma1, tau1)
  _steadyStokesBF->addTerm(-u1_hat, tau1->dot_normal());
  
  // tau2 terms:
  _steadyStokesBF->addTerm(u2, tau2->div());
  _steadyStokesBF->addTerm((1.0/_mu) * sigma2, tau2);
  _steadyStokesBF->addTerm(-u2_hat, tau2->dot_normal());

  // tau3:
  if (spaceDim > 2)
  {
    _steadyStokesBF->addTerm(u3, tau3->div());
    _steadyStokesBF->addTerm((1.0/_mu) * sigma3, tau3);
    _steadyStokesBF->addTerm(-u3_hat, tau3->dot_normal());
  }
  
  // v1:
  _steadyStokesBF->addTerm(sigma1, v1->grad()); // (sigma1, grad v1)
  _steadyStokesBF->addTerm( - p, v1->dx() );
  _steadyStokesBF->addTerm( t1n, v1);
  
  // v2:
  _steadyStokesBF->addTerm(sigma2, v2->grad()); // (sigma2, grad v2)
  _steadyStokesBF->addTerm( - p, v2->dy());
  _steadyStokesBF->addTerm( t2n, v2);
  
  // v3:
  if (spaceDim > 2)
  {
    _steadyStokesBF->addTerm(sigma3, v3->grad()); // (mu sigma3, grad v3)
    _steadyStokesBF->addTerm( - p, v3->dz());
    _steadyStokesBF->addTerm( t3n, v3);
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
  
  if (!_spaceTime && !_timeStepping)
  {
    _stokesBF = _steadyStokesBF;
  }
  else if (_timeStepping)
  {
    _stokesBF = Teuchos::rcp( new BF(_vf) );
    // v1
    // tau1 terms:
    _stokesBF->addTerm(_theta * u1, tau1->div());
    _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma1), tau1); // (sigma1, tau1)
    _stokesBF->addTerm(-u1_hat, tau1->dot_normal());
    
    // tau2 terms:
    if (spaceDim > 1)
    {
      _stokesBF->addTerm(_theta * u2, tau2->div());
      _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma2), tau2);
      _stokesBF->addTerm(-u2_hat, tau2->dot_normal());
    }
    
    // tau3:
    if (spaceDim==3)
    {
      _stokesBF->addTerm(_theta * u3, tau3->div());
      _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma3), tau3);
      _stokesBF->addTerm(-u3_hat, tau3->dot_normal());
    }
    
    TFunctionPtr<double> dtFxn = _dt; // cast to allow use of TFunctionPtr<double> operator overloads
    // v1:
    _stokesBF->addTerm(u1 / dtFxn, v1);
    _stokesBF->addTerm(_theta * sigma1, v1->grad()); // (mu sigma1, grad v1)
    _stokesBF->addTerm(_theta * (-p), v1->dx() );
    _stokesBF->addTerm( t1n, v1);
    
    // v2:
    if (spaceDim > 1)
    {
      _stokesBF->addTerm(u2 / dtFxn, v2);
      _stokesBF->addTerm(_theta * sigma2, v2->grad()); // (mu sigma2, grad v2)
      _stokesBF->addTerm(_theta * (-p), v2->dy());
      _stokesBF->addTerm( t2n, v2);
    }
    
    // v3:
    if (spaceDim > 2)
    {
      _stokesBF->addTerm(u3 / dtFxn, v3);
      _stokesBF->addTerm(_theta * sigma3, v3->grad()); // (mu sigma3, grad v3)
      _stokesBF->addTerm(_theta * (- p), v3->dz());
      _stokesBF->addTerm( t3n, v3);
    }
    
    // q:
    if (spaceDim > 0) _stokesBF->addTerm(_theta * (-u1),q->dx()); // (-u, grad q)
    if (spaceDim > 1) _stokesBF->addTerm(_theta * (-u2),q->dy());
    if (spaceDim > 2) _stokesBF->addTerm(_theta * (-u3), q->dz());
    
    if (spaceDim==2)
    {
      _stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
    }
    else if (spaceDim==3)
    {
      _stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
    }
  }
  else if (_spaceTime)
  {
    _stokesBF = Teuchos::rcp( new BF(*_steadyStokesBF) );
    
    TFunctionPtr<double> n_spaceTime = TFunction<double>::normalSpaceTime();
    
    // v1:
    _stokesBF->addTerm(-u1, v1->dt());
    _stokesBF->addTerm(u1_hat * n_spaceTime->t(), v1);
    
    // v2:
    _stokesBF->addTerm(-u2, v2->dt());
    _stokesBF->addTerm(u2_hat * n_spaceTime->t(), v2);
    
    // v3:
    if (_spaceDim == 3)
    {
      _stokesBF->addTerm(-u3, v3->dt());
      _stokesBF->addTerm(u3_hat * n_spaceTime->t(), v3);
    }
  }
  
  // define tractions (used in outflow conditions)
  // definition of traction: _mu * ( (\nabla u) + (\nabla u)^T ) n - p n
  //                      = (sigma + sigma^T) n - p n
  if (spaceDim == 2)
  {
    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x());
    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p);
  }
  else
  {
    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x()) + n->z() * (sigma1->z() + sigma3->x());
    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p)       + n->z() * (sigma2->z() + sigma3->y());
    _t3 = n->x() * (sigma1->z() + sigma3->x()) + n->y() * (sigma2->z() + sigma3->y()) + n->z() * (2 * sigma3->z() - p);
  }
}
//
//StokesVGPFormulation::StokesVGPFormulation(int spaceDim, bool useConformingTraces, double mu,
//    bool transient, double dt)
//{
//  _spaceDim = spaceDim;
//  _useConformingTraces = useConformingTraces;
//  _mu = mu;
//  _dt = ParameterFunction::parameterFunction(dt);
//  _t = ParameterFunction::parameterFunction(0);
//
//  _theta = ParameterFunction::parameterFunction(0.5); // Crank-Nicolson
//  _transient = transient;
//
//  if ((spaceDim != 2) && (spaceDim != 3))
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
//  }
//
//  // declare all possible variables -- will only create the ones we need for spaceDim
//  // fields
//  VarPtr u1, u2, u3;
//  VarPtr p;
//  VarPtr sigma1, sigma2, sigma3;
//
//  // traces
//  VarPtr u1_hat, u2_hat, u3_hat;
//  VarPtr t1n, t2n, t3n;
//
//  // tests
//  VarPtr v1, v2, v3;
//  VarPtr tau1, tau2, tau3;
//  VarPtr q;
//
//  _vf = VarFactory::varFactory();
//  u1 = _vf->fieldVar(S_U1);
//  u2 = _vf->fieldVar(S_U2);
//  if (spaceDim==3) u3 = _vf->fieldVar(S_U3);
//
//  p = _vf->fieldVar(S_P);
//
//  sigma1 = _vf->fieldVar(S_SIGMA1, VECTOR_L2);
//  sigma2 = _vf->fieldVar(S_SIGMA2, VECTOR_L2);
//  if (spaceDim==3)
//  {
//    sigma3 = _vf->fieldVar(S_SIGMA3, VECTOR_L2);
//  }
//
//  Space uHatSpace = useConformingTraces ? HGRAD : L2;
//
//  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
//  u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
//  u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
//  if (spaceDim==3) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);
//
//  TFunctionPtr<double> n = TFunction<double>::normal();
//  TFunctionPtr<double> n_parity = n * TFunction<double>::sideParity();
//
//  LinearTermPtr t1n_lt, t2n_lt, t3n_lt;
//  FunctionPtr minus_n_parity = - n_parity; // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
//  t1n_lt = p * n_parity->x() + sigma1 * minus_n_parity ;
//  t2n_lt = p * n_parity->y() + sigma2 * minus_n_parity;
//  if (spaceDim==3)
//  {
//    t3n_lt = p * n_parity->z() + sigma3 * minus_n_parity;
//  }
//  t1n = _vf->fluxVar(S_TN1_HAT, t1n_lt);
//  t2n = _vf->fluxVar(S_TN2_HAT, t2n_lt);
//  if (spaceDim==3) t3n = _vf->fluxVar(S_TN3_HAT, t3n_lt);
//
//  v1 = _vf->testVar(S_V1, HGRAD);
//  v2 = _vf->testVar(S_V2, HGRAD);
//  if (spaceDim==3) v3 = _vf->testVar(S_V3, HGRAD);
//
//  tau1 = _vf->testVar(S_TAU1, HDIV);
//  tau2 = _vf->testVar(S_TAU2, HDIV);
//  if (spaceDim==3)
//  {
//    tau3 = _vf->testVar(S_TAU3, HDIV);
//  }
//
//  q = _vf->testVar(S_Q, HGRAD);
//
//  _steadyStokesBF = Teuchos::rcp( new BF(_vf) );
//  // v1
//  // tau1 terms:
//  _steadyStokesBF->addTerm(u1, tau1->div());
//  _steadyStokesBF->addTerm((1.0/_mu) * sigma1, tau1); // (sigma1, tau1)
//  _steadyStokesBF->addTerm(-u1_hat, tau1->dot_normal());
//
//  // tau2 terms:
//  _steadyStokesBF->addTerm(u2, tau2->div());
//  _steadyStokesBF->addTerm((1.0/_mu) * sigma2, tau2);
//  _steadyStokesBF->addTerm(-u2_hat, tau2->dot_normal());
//
//  // tau3:
//  if (spaceDim==3)
//  {
//    _steadyStokesBF->addTerm(u3, tau3->div());
//    _steadyStokesBF->addTerm((1.0/_mu) * sigma3, tau3);
//    _steadyStokesBF->addTerm(-u3_hat, tau3->dot_normal());
//  }
//
//  // v1:
//  _steadyStokesBF->addTerm(sigma1, v1->grad()); // (mu sigma1, grad v1)
//  _steadyStokesBF->addTerm( - p, v1->dx() );
//  _steadyStokesBF->addTerm( t1n, v1);
//
//  // v2:
//  _steadyStokesBF->addTerm(sigma2, v2->grad()); // (mu sigma2, grad v2)
//  _steadyStokesBF->addTerm( - p, v2->dy());
//  _steadyStokesBF->addTerm( t2n, v2);
//
//  // v3:
//  if (spaceDim==3)
//  {
//    _steadyStokesBF->addTerm(sigma3, v3->grad()); // (mu sigma3, grad v3)
//    _steadyStokesBF->addTerm( - p, v3->dz());
//    _steadyStokesBF->addTerm( t3n, v3);
//  }
//
//  // q:
//  _steadyStokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
//  _steadyStokesBF->addTerm(-u2,q->dy());
//  if (spaceDim==3) _steadyStokesBF->addTerm(-u3, q->dz());
//
//  if (spaceDim==2)
//  {
//    _steadyStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
//  }
//  else if (spaceDim==3)
//  {
//    _steadyStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
//  }
//
//  if (!_transient)
//  {
//    _stokesBF = _steadyStokesBF;
//  }
//  else
//  {
//    _stokesBF = Teuchos::rcp( new BF(_vf) );
//    // v1
//    // tau1 terms:
//    _stokesBF->addTerm(_theta * u1, tau1->div());
//    _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma1), tau1); // (sigma1, tau1)
//    _stokesBF->addTerm(-u1_hat, tau1->dot_normal());
//
//    // tau2 terms:
//    _stokesBF->addTerm(_theta * u2, tau2->div());
//    _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma2), tau2);
//    _stokesBF->addTerm(-u2_hat, tau2->dot_normal());
//
//    // tau3:
//    if (spaceDim==3)
//    {
//      _stokesBF->addTerm(_theta * u3, tau3->div());
//      _stokesBF->addTerm(_theta * ((1.0/_mu) * sigma3), tau3);
//      _stokesBF->addTerm(-u3_hat, tau3->dot_normal());
//    }
//
//    TFunctionPtr<double> dtFxn = _dt; // cast to allow use of TFunctionPtr<double> operator overloads
//    // v1:
//    _stokesBF->addTerm(u1 / dtFxn, v1);
//    _stokesBF->addTerm(_theta * sigma1, v1->grad()); // (mu sigma1, grad v1)
//    _stokesBF->addTerm(_theta * (-p), v1->dx() );
//    _stokesBF->addTerm( t1n, v1);
//
//    // v2:
//    _stokesBF->addTerm(u2 / dtFxn, v2);
//    _stokesBF->addTerm(_theta * sigma2, v2->grad()); // (mu sigma2, grad v2)
//    _stokesBF->addTerm(_theta * (-p), v2->dy());
//    _stokesBF->addTerm( t2n, v2);
//
//    // v3:
//    if (spaceDim==3)
//    {
//      _stokesBF->addTerm(u3 / dtFxn, v3);
//      _stokesBF->addTerm(_theta * sigma3, v3->grad()); // (mu sigma3, grad v3)
//      _stokesBF->addTerm(_theta * (- p), v3->dz());
//      _stokesBF->addTerm( t3n, v3);
//    }
//
//    // q:
//    _stokesBF->addTerm(_theta * (-u1),q->dx()); // (-u, grad q)
//    _stokesBF->addTerm(_theta * (-u2),q->dy());
//    if (spaceDim==3) _stokesBF->addTerm(_theta * (-u3), q->dz());
//
//    if (spaceDim==2)
//    {
//      _stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
//    }
//    else if (spaceDim==3)
//    {
//      _stokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
//    }
//  }
//
//  // define tractions (used in outflow conditions)
//  // definition of traction: _mu * ( (\nabla u) + (\nabla u)^T ) n - p n
//  //                      = (sigma + sigma^T) n - p n
//  if (spaceDim == 2)
//  {
//    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x());
//    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p);
//  }
//  else
//  {
//    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x()) + n->z() * (sigma1->z() + sigma3->x());
//    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p)       + n->z() * (sigma2->z() + sigma3->y());
//    _t3 = n->x() * (sigma1->z() + sigma3->x()) + n->y() * (sigma2->z() + sigma3->y()) + n->z() * (2 * sigma3->z() - p);
//  }
//}

void StokesVGPFormulation::addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u)
{
  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (_spaceDim==3) u3_hat = this->u_hat(3);

  if (! _timeStepping)
  {
    _solution->bc()->addDirichlet(u1_hat, inflowRegion, u->x());
    _solution->bc()->addDirichlet(u2_hat, inflowRegion, u->y());
    if (_spaceDim==3) _solution->bc()->addDirichlet(u3_hat, inflowRegion, u->z());
  }
  else
  {
    TFunctionPtr<double> u1_hat_prev, u2_hat_prev, u3_hat_prev;
    TSolutionPtr<double> prevSolnWeakRef = Teuchos::rcp( _previousSolution.get(), false ); // avoid circular references

    u1_hat_prev = TFunction<double>::solution(this->u_hat(1), prevSolnWeakRef);
    u2_hat_prev = TFunction<double>::solution(this->u_hat(2), prevSolnWeakRef);
    if (_spaceDim==3) u3_hat_prev = TFunction<double>::solution(this->u_hat(3), prevSolnWeakRef);
    TFunctionPtr<double> thetaFxn = _theta; // cast to allow use of TFunctionPtr<double> operator overloads
    _solution->bc()->addDirichlet(u1_hat, inflowRegion, thetaFxn * u->x() + (1.-thetaFxn)* u1_hat_prev);
    _solution->bc()->addDirichlet(u2_hat, inflowRegion, thetaFxn * u->y() + (1.-thetaFxn)* u2_hat_prev);
    if (_spaceDim==3) _solution->bc()->addDirichlet(u3_hat, inflowRegion, thetaFxn * u->z() + (1.-thetaFxn)* u3_hat_prev);
  }
}

void StokesVGPFormulation::addOutflowCondition(SpatialFilterPtr outflowRegion)
{
//  for (int d=0; d<spaceDim; d++) {
//    VarPtr tn_hat = this->tn_hat(d+1);
//    _solution->bc()->addDirichlet(tn_hat, outflowRegion, TFunction<double>::zero());
//  }

  _haveOutflowConditionsImposed = true;

  // point pressure and zero-mean pressures are not compatible with outflow conditions:
  VarPtr p = this->p();
  if (_solution->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    cout << "Removing zero-mean constraint on pressure by virtue of outflow condition.\n";
    _solution->bc()->removeZeroMeanConstraint(p->ID());
  }

  if (_solution->bc()->singlePointBC(p->ID()))
  {
    cout << "Removing zero-point condition on pressure by virtue of outflow condition.\n";
    _solution->bc()->removeSinglePointBC(p->ID());
  }

  // my favorite way to do outflow conditions is via penalty constraints imposing a zero traction
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter_incr = _solution->filter();

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
    _solution->setFilter(pcRCP);
  }
}

void StokesVGPFormulation::addPointPressureCondition()
{
  if (_haveOutflowConditionsImposed)
  {
    cout << "ERROR: can't add pressure point condition if there are outflow conditions imposed.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
  }

  VarPtr p = this->p();

  vector<double> vertex0 = _solution->mesh()->getTopology()->getVertex(0);
  if (_spaceTime) // then the last coordinate is time; drop it
  {
    vertex0.pop_back();
  }
  _solution->bc()->addSpatialPointBC(p->ID(), 0.0, vertex0);

  if (_solution->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    _solution->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void StokesVGPFormulation::addWallCondition(SpatialFilterPtr wall)
{
  vector<double> zero(_spaceDim, 0.0);
  addInflowCondition(wall, TFunction<double>::constant(zero));
}

void StokesVGPFormulation::addZeroMeanPressureCondition()
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

  _solution->bc()->addZeroMeanConstraint(p);

  if (_solution->bc()->singlePointBC(p->ID()))
  {
    _solution->bc()->removeSinglePointBC(p->ID());
  }
}

BFPtr StokesVGPFormulation::bf()
{
  return _stokesBF;
}

TFunctionPtr<double> StokesVGPFormulation::forcingFunction(TFunctionPtr<double> u_exact, TFunctionPtr<double> p_exact)
{
  // f1 and f2 are those for Navier-Stokes, but without the u \cdot \grad u term
  TFunctionPtr<double> u1_exact = u_exact->x();
  TFunctionPtr<double> u2_exact = u_exact->y();
  TFunctionPtr<double> u3_exact = u_exact->z();

  TFunctionPtr<double> f;

  if (_spaceDim == 2)
  {
    TFunctionPtr<double> f1, f2;
    f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    if (_spaceTime)
    {
      f1 = f1 + u1_exact->dt();
      f2 = f2 + u2_exact->dt();
    }
    else if (_timeStepping)
    {
      // leave as is, for now.  The rhs() method will add some terms; this may be the right thing.
    }
    f = TFunction<double>::vectorize(f1, f2);
  }
  else
  {
    TFunctionPtr<double> f1, f2, f3;
    f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz());
    f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz());
    f3 = p_exact->dz() - _mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz());
    if (_spaceTime)
    {
      f1 = f1 + u1_exact->dt();
      f2 = f2 + u2_exact->dt();
      f3 = f3 + u3_exact->dt();
    }
    else if (_timeStepping)
    {
      // leave as is, for now.  The rhs() method will add some terms; this may be the right thing.
    }
    
    f = TFunction<double>::vectorize(f1, f2, f3);
  }
  return f;
}

void StokesVGPFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction)
{
  this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction, "");
}

void StokesVGPFormulation::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction)
{
  this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, forcingFunction, filePrefix);
}

void StokesVGPFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, string savedSolutionAndMeshPrefix)
{
  _haveOutflowConditionsImposed = false;
  BCPtr bc = BC::bc();

  int H1Order = fieldPolyOrder + 1;
  MeshPtr mesh;
  if (savedSolutionAndMeshPrefix == "")
  {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _stokesBF, H1Order, delta_k) ) ;
    _solution = TSolution<double>::solution(mesh,bc);
    if (_timeStepping) _previousSolution = TSolution<double>::solution(mesh,bc);
  }
  else
  {
    mesh = MeshFactory::loadFromHDF5(_stokesBF, savedSolutionAndMeshPrefix+".mesh");
    _solution = TSolution<double>::solution(mesh, bc);
    _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    if (_timeStepping)
    {
      _previousSolution = TSolution<double>::solution(mesh,bc);
      _previousSolution->loadFromHDF5(savedSolutionAndMeshPrefix+"_previous.soln");
    }
  }

  RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
  IPPtr ip = _stokesBF->graphNorm();
  
//  cout << "graph norm for Stokes BF:\n";
//  ip->printInteractions();

  _solution->setRHS(rhs);
  _solution->setIP(ip);

  mesh->registerSolution(_solution); // will project both time steps during refinements...
  if (_timeStepping) mesh->registerSolution(_previousSolution);

  LinearTermPtr residual = rhs->linearTerm() - _stokesBF->testFunctional(_solution,false); // false: don't exclude boundary terms

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

    LinearTermPtr u1_dy = (1.0 / _mu) * this->sigma(1)->y();
    LinearTermPtr u2_dx = (1.0 / _mu) * this->sigma(2)->x();

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

bool StokesVGPFormulation::isSpaceTime() const
{
  return _spaceTime;
}

bool StokesVGPFormulation::isSteady() const
{
  return !_timeStepping && !_spaceTime;
}


bool StokesVGPFormulation::isTimeStepping() const
{
  return _timeStepping;
}

double StokesVGPFormulation::L2NormOfTimeStep()
{
  TFunctionPtr<double>  p_current = TFunction<double>::solution( p(), _solution);
  TFunctionPtr<double> u1_current = TFunction<double>::solution(u(1), _solution);
  TFunctionPtr<double> u2_current = TFunction<double>::solution(u(2), _solution);
  TFunctionPtr<double>  p_prev = TFunction<double>::solution( p(), _previousSolution);
  TFunctionPtr<double> u1_prev = TFunction<double>::solution(u(1), _previousSolution);
  TFunctionPtr<double> u2_prev = TFunction<double>::solution(u(2), _previousSolution);

  TFunctionPtr<double> squaredDiff = (p_current-p_prev) * (p_current-p_prev) + (u1_current-u1_prev) * (u1_current-u1_prev) + (u2_current - u2_prev) * (u2_current - u2_prev);
  double valSquared = squaredDiff->integrate(_solution->mesh());
  return sqrt(valSquared);
}

double StokesVGPFormulation::mu()
{
  return _mu;
}

VarPtr StokesVGPFormulation::p()
{
  return _vf->fieldVar(S_P);
}

RefinementStrategyPtr StokesVGPFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void StokesVGPFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void StokesVGPFormulation::refine()
{
  _refinementStrategy->refine();
}

void StokesVGPFormulation::hRefine()
{
  _hRefinementStrategy->refine();
}

void StokesVGPFormulation::pRefine()
{
  _pRefinementStrategy->refine();
}

RHSPtr StokesVGPFormulation::rhs(TFunctionPtr<double> f)
{
  RHSPtr rhs = RHS::rhs();

  VarPtr v1 = this->v(1);
  VarPtr v2 = this->v(2);
  VarPtr v3;
  if (_spaceDim==3) v3 = this->v(3);

  if (f != Teuchos::null)
  {
    rhs->addTerm( f->x() * v1 );
    rhs->addTerm( f->y() * v2 );
    if (_spaceDim == 3) rhs->addTerm( f->z() * v3 );
  }

  if (_timeStepping)
  {
    TFunctionPtr<double> u1_prev, u2_prev, u3_prev;
    u1_prev = TFunction<double>::solution(this->u(1), _previousSolution);
    u2_prev = TFunction<double>::solution(this->u(2), _previousSolution);
    if (_spaceDim==3) u3_prev = TFunction<double>::solution(this->u(3), _previousSolution);
    TFunctionPtr<double> dtFxn = _dt; // cast to allow use of TFunctionPtr<double> operator overloads
    rhs->addTerm(u1_prev / dtFxn * v1);
    rhs->addTerm(u2_prev / dtFxn * v2);
    if (_spaceDim==3) rhs->addTerm(u3_prev / dtFxn * v3);

    bool excludeFluxesAndTraces = true;
    LinearTermPtr prevTimeStepFunctional = _steadyStokesBF->testFunctional(_previousSolution,excludeFluxesAndTraces);
    TFunctionPtr<double> thetaFxn = _theta; // cast to allow use of TFunctionPtr<double> operator overloads
    rhs->addTerm((thetaFxn - 1.0) * prevTimeStepFunctional);
  }
  else if (_spaceTime)
  {
    //
  }

  return rhs;
}

VarPtr StokesVGPFormulation::sigma(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->fieldVar(S_SIGMA1);
  case 2:
    return _vf->fieldVar(S_SIGMA2);
  case 3:
    return _vf->fieldVar(S_SIGMA3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::u(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->fieldVar(S_U1);
  case 2:
    return _vf->fieldVar(S_U2);
  case 3:
    return _vf->fieldVar(S_U3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// traces:
VarPtr StokesVGPFormulation::tn_hat(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->fluxVar(S_TN1_HAT);
  case 2:
    return _vf->fluxVar(S_TN2_HAT);
  case 3:
    return _vf->fluxVar(S_TN3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::u_hat(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->traceVar(S_U1_HAT);
  case 2:
    return _vf->traceVar(S_U2_HAT);
  case 3:
    return _vf->traceVar(S_U3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// test variables:
VarPtr StokesVGPFormulation::tau(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->testVar(S_TAU1, HDIV);
  case 2:
    return _vf->testVar(S_TAU2, HDIV);
  case 3:
    return _vf->testVar(S_TAU3, HDIV);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");

}

// ! Saves the solution(s) and mesh to an HDF5 format.
void StokesVGPFormulation::save(std::string prefixString)
{
  _solution->mesh()->saveToHDF5(prefixString+".mesh");
  _solution->saveToHDF5(prefixString+".soln");
  if (_timeStepping)
  {
    _previousSolution->saveToHDF5(prefixString+"_previous.soln");
  }
  if (_streamSolution != Teuchos::null)
  {
    _streamSolution->mesh()->saveToHDF5(prefixString+"_stream.mesh");
    _streamSolution->saveToHDF5(prefixString + "_stream.soln");
  }
}

// ! set current time step used for transient solve
void StokesVGPFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

// ! Returns the solution (at current time)
TSolutionPtr<double> StokesVGPFormulation::solution()
{
  return _solution;
}

// ! Returns the solution (at previous time)
TSolutionPtr<double> StokesVGPFormulation::solutionPreviousTimeStep()
{
  return _previousSolution;
}

// ! Solves
void StokesVGPFormulation::solve()
{
  _solution->solve();
}

int StokesVGPFormulation::spaceDim()
{
  return _spaceDim;
}

PoissonFormulation & StokesVGPFormulation::streamFormulation()
{
  return *_streamFormulation;
}

VarPtr StokesVGPFormulation::streamPhi()
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

TSolutionPtr<double> StokesVGPFormulation::streamSolution()
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

// ! Takes a time step (assumes you have called solve() first)
void StokesVGPFormulation::takeTimeStep()
{
  ConstantScalarFunction<double>* dtValueFxn = dynamic_cast<ConstantScalarFunction<double>*>(_dt->getValue().get());

  double dt = dtValueFxn->value(0);
  _time += dt;
  _t->setValue(_time);

  // if we implemented some sort of value-replacement in Solution, that would be more efficient than this:
  _previousSolution->clear();
  _previousSolution->addSolution(_solution, 1.0);
}

// ! Returns the sum of the time steps taken thus far.
double StokesVGPFormulation::getTime()
{
  return _time;
}

TFunctionPtr<double> StokesVGPFormulation::getTimeFunction()
{
  return _t;
}

VarPtr StokesVGPFormulation::v(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->testVar(S_V1, HGRAD);
  case 2:
    return _vf->testVar(S_V2, HGRAD);
  case 3:
    return _vf->testVar(S_V3, HGRAD);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}
