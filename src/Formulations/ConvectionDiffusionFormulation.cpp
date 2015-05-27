//
//  ConvectionDiffusionFormulation.cpp
//  Camellia
//
//  Created by Truman Ellis on 5/11/15.
//
//

#include "ConvectionDiffusionFormulation.h"

using namespace Camellia;

const string ConvectionDiffusionFormulation::s_u = "u";
const string ConvectionDiffusionFormulation::s_sigma = "sigma";

const string ConvectionDiffusionFormulation::s_uhat = "uhat";
const string ConvectionDiffusionFormulation::s_tc = "tc";

const string ConvectionDiffusionFormulation::s_v = "v";
const string ConvectionDiffusionFormulation::s_tau = "tau";

ConvectionDiffusionFormulation::ConvectionDiffusionFormulation(int spaceDim, bool useConformingTraces, FunctionPtr beta, double epsilon)
{
  _spaceDim = spaceDim;
  _epsilon = epsilon;
  _beta = beta;

  FunctionPtr zero = Function::constant(1);
  FunctionPtr one = Function::constant(1);

  Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
  Space uhat_space = useConformingTraces ? HGRAD : L2;
  Space vSpace = (spaceDim > 1) ? VECTOR_L2 : L2;

  // fields
  VarPtr u;
  VarPtr sigma;

  // traces
  VarPtr uhat, tc;

  // tests
  VarPtr v;
  VarPtr tau;

  _vf = VarFactory::varFactory();
  u = _vf->fieldVar(s_u);
  sigma = _vf->fieldVar(s_sigma, vSpace);

  if (spaceDim > 1)
    uhat = _vf->traceVar(s_uhat, u, uhat_space);
  else
    uhat = _vf->fluxVar(s_uhat, u, uhat_space); // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)

  TFunctionPtr<double> n = TFunction<double>::normal();
  TFunctionPtr<double> parity = TFunction<double>::sideParity();

  if (spaceDim > 1)
    tc = _vf->fluxVar(s_tc, (_beta*u-sigma) * (n * parity));
  else
    tc = _vf->fluxVar(s_tc, _beta*u-sigma);

  v = _vf->testVar(s_v, HGRAD);
  tau = _vf->testVar(s_tau, tauSpace);

  _bf = Teuchos::rcp( new BF(_vf) );

  if (spaceDim==1)
  {
    // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
    _bf->addTerm(sigma/_epsilon, tau);
    _bf->addTerm(u, tau->dx());
    _bf->addTerm(-uhat, tau);

    _bf->addTerm(-_beta*u + sigma, v->dx());
    _bf->addTerm(tc, v);
  }
  else
  {
    _bf->addTerm(sigma/_epsilon, tau);
    _bf->addTerm(u, tau->div());
    _bf->addTerm(-uhat, tau->dot_normal());

    _bf->addTerm(-_beta*u + sigma, v->grad());
    _bf->addTerm(tc, v);
  }

  _ips["Graph"] = _bf->graphNorm();

  if (spaceDim > 1)
  {
    _ips["Robust"] = Teuchos::rcp(new IP);
    _ips["Robust"]->addTerm(tau->div());
    _ips["Robust"]->addTerm(_beta*v->grad());
    _ips["Robust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
    _ips["Robust"]->addTerm(sqrt(_epsilon)*v->grad());
    _ips["Robust"]->addTerm(_beta*v->grad());
    _ips["Robust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  }
  else
  {
    _ips["Robust"] = Teuchos::rcp(new IP);
    _ips["Robust"]->addTerm(tau->dx());
    _ips["Robust"]->addTerm(_beta*v->dx());
    _ips["Robust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
    _ips["Robust"]->addTerm(sqrt(_epsilon)*v->dx());
    _ips["Robust"]->addTerm(_beta*v->dx());
    _ips["Robust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  }

  if (spaceDim > 1)
  {
    _ips["CoupledRobust"] = Teuchos::rcp(new IP);
    _ips["CoupledRobust"]->addTerm(tau->div()-_beta*v->grad());
    _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
    _ips["CoupledRobust"]->addTerm(sqrt(_epsilon)*v->grad());
    _ips["CoupledRobust"]->addTerm(_beta*v->grad());
    _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  }
  else
  {
    _ips["CoupledRobust"] = Teuchos::rcp(new IP);
    _ips["CoupledRobust"]->addTerm(tau->dx()-_beta*v->dx());
    _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
    _ips["CoupledRobust"]->addTerm(sqrt(_epsilon)*v->dx());
    _ips["CoupledRobust"]->addTerm(_beta*v->dx());
    _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  }
}

VarFactoryPtr ConvectionDiffusionFormulation::vf()
{
  return _vf;
}

BFPtr ConvectionDiffusionFormulation::bf()
{
  return _bf;
}

IPPtr ConvectionDiffusionFormulation::ip(string normName)
{
  return _ips.at(normName);
}

// field variables:
VarPtr ConvectionDiffusionFormulation::u()
{
  return _vf->fieldVar(s_u);
}

VarPtr ConvectionDiffusionFormulation::sigma()
{
  return _vf->fieldVar(s_sigma);
}

// traces:
VarPtr ConvectionDiffusionFormulation::tc()
{
  return _vf->fluxVar(s_tc);
}

VarPtr ConvectionDiffusionFormulation::uhat()
{
  return _vf->traceVar(s_uhat);
}

// test variables:
VarPtr ConvectionDiffusionFormulation::v()
{
  return _vf->testVar(s_v, HGRAD);
}

VarPtr ConvectionDiffusionFormulation::tau()
{
  if (_spaceDim > 1)
    return _vf->testVar(s_tau, HDIV);
  else
    return _vf->testVar(s_tau, HGRAD);
}
