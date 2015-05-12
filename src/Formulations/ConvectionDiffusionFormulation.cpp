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

ConvectionDiffusionFormulation::ConvectionDiffusionFormulation(int spaceDim, bool useConformingTraces, FunctionPtr beta, double epsilon) {
  _spaceDim = spaceDim;
  _epsilon = epsilon;
  _beta = beta;

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

  VarFactoryPtr vf = VarFactory::varFactory();
  u = vf->fieldVar(s_u);
  sigma = vf->fieldVar(s_sigma, vSpace);

  if (spaceDim > 1)
    uhat = vf->traceVar(s_uhat, u, uhat_space);
  else
    uhat = vf->fluxVar(s_uhat, u, uhat_space); // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)

  TFunctionPtr<double> n = TFunction<double>::normal();
  TFunctionPtr<double> parity = TFunction<double>::sideParity();

  if (spaceDim > 1)
    tc = vf->fluxVar(s_tc, (_beta*u-sigma) * (n * parity));
  else
    tc = vf->fluxVar(s_tc, _beta*u-sigma);

  v = vf->testVar(s_v, HGRAD);
  tau = vf->testVar(s_tau, tauSpace);

  _convectionDiffusionBF = Teuchos::rcp( new BF(vf) );

  if (spaceDim==1) {
    // for spaceDim==1, the "normal" component is in the flux-ness of uhat (it's a plus or minus 1)
    _convectionDiffusionBF->addTerm(sigma/_epsilon, tau);
    _convectionDiffusionBF->addTerm(u, tau->dx());
    _convectionDiffusionBF->addTerm(-uhat, tau);

    _convectionDiffusionBF->addTerm(-_beta*u + sigma, v->dx());
    _convectionDiffusionBF->addTerm(tc, v);
  } else {
    _convectionDiffusionBF->addTerm(sigma/_epsilon, tau);
    _convectionDiffusionBF->addTerm(u, tau->div());
    _convectionDiffusionBF->addTerm(-uhat, tau->dot_normal());

    _convectionDiffusionBF->addTerm(-_beta*u + sigma, v->grad());
    _convectionDiffusionBF->addTerm(tc, v);
  }
}

BFPtr ConvectionDiffusionFormulation::bf() {
  return _convectionDiffusionBF;
}

// field variables:
VarPtr ConvectionDiffusionFormulation::u() {
  VarFactoryPtr vf = _convectionDiffusionBF->varFactory();
  return vf->fieldVar(s_u);
}

VarPtr ConvectionDiffusionFormulation::sigma() {
  VarFactoryPtr vf = _convectionDiffusionBF->varFactory();
  return vf->fieldVar(s_sigma);
}

// traces:
VarPtr ConvectionDiffusionFormulation::tc() {
  VarFactoryPtr vf = _convectionDiffusionBF->varFactory();
  return vf->fluxVar(s_tc);
}

VarPtr ConvectionDiffusionFormulation::uhat() {
  VarFactoryPtr vf = _convectionDiffusionBF->varFactory();
  return vf->traceVar(s_uhat);
}

// test variables:
VarPtr ConvectionDiffusionFormulation::v() {
  VarFactoryPtr vf = _convectionDiffusionBF->varFactory();
  return vf->testVar(s_v, HGRAD);
}

VarPtr ConvectionDiffusionFormulation::tau() {
  VarFactoryPtr vf = _convectionDiffusionBF->varFactory();
  if (_spaceDim > 1)
    return vf->testVar(s_tau, HDIV);
  else
    return vf->testVar(s_tau, HGRAD);
}
