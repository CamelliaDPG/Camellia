//
//  StokesVGPFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "StokesVGPFormulation.h"

const string StokesVGPFormulation::S_U1 = "u_1";
const string StokesVGPFormulation::S_U2 = "u_2";
const string StokesVGPFormulation::S_U3 = "u_3";
const string StokesVGPFormulation::S_P = "p";
const string StokesVGPFormulation::S_SIGMA1 = "\\widehat{\\sigma}_{1}";
const string StokesVGPFormulation::S_SIGMA2 = "\\widehat{\\sigma}_{2}";
const string StokesVGPFormulation::S_SIGMA3 = "\\widehat{\\sigma}_{3}";

const string StokesVGPFormulation::S_U1_HAT = "\\widehat{u}_1";
const string StokesVGPFormulation::S_U2_HAT = "\\widehat{u}_2";
const string StokesVGPFormulation::S_U3_HAT = "\\widehat{u}_3";
const string StokesVGPFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string StokesVGPFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string StokesVGPFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string StokesVGPFormulation::S_V1 = "v_1";
const string StokesVGPFormulation::S_V2 = "v_2";
const string StokesVGPFormulation::S_V3 = "v_3";
const string StokesVGPFormulation::S_TAU1 = "\\widehat{\\tau}_{1}";
const string StokesVGPFormulation::S_TAU2 = "\\widehat{\\tau}_{2}";
const string StokesVGPFormulation::S_TAU3 = "\\widehat{\\tau}_{3}";

StokesVGPFormulation::StokesVGPFormulation(int spaceDim, bool useConformingTraces, double mu) {
  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _mu = mu;
  
  if ((spaceDim != 2) && (spaceDim != 3)) {
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
  
  VarFactory vf;
  u1 = vf.fieldVar(S_U1);
  u2 = vf.fieldVar(S_U2);
  if (spaceDim==3) u3 = vf.fieldVar(S_U3);
  
  p = vf.fieldVar(S_P);
  
  sigma1 = vf.fieldVar(S_SIGMA1, VECTOR_L2);
  sigma2 = vf.fieldVar(S_SIGMA2, VECTOR_L2);
  if (spaceDim==3) {
    sigma3 = vf.fieldVar(S_SIGMA3, VECTOR_L2);
  }
  
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  
  u1_hat = vf.traceVar(S_U1_HAT, 1.0 * u1, uHatSpace);
  u2_hat = vf.traceVar(S_U2_HAT, 1.0 * u2, uHatSpace);
  if (spaceDim==3) u3_hat = vf.traceVar(S_U3_HAT, 1.0 * u3, uHatSpace);
  
  FunctionPtr n = Function::normal();
  LinearTermPtr t1n_lt, t2n_lt, t3n_lt;
  t1n_lt = sigma1 * n - p * n->x();
  t2n_lt = sigma2 * n - p * n->y();
  if (spaceDim==3) {
    t3n_lt = sigma3 * n - p * n->z();
  }
  t1n = vf.fluxVar(S_TN1_HAT, t1n_lt);
  t2n = vf.fluxVar(S_TN2_HAT, t2n_lt);
  if (spaceDim==3) t3n = vf.fluxVar(S_TN3_HAT, t3n_lt);
  
  v1 = vf.testVar(S_V1, HGRAD);
  v2 = vf.testVar(S_V2, HGRAD);
  if (spaceDim==3) v3 = vf.testVar(S_V3, HGRAD);
  
  tau1 = vf.testVar(S_TAU1, HDIV);
  tau2 = vf.testVar(S_TAU2, HDIV);
  if (spaceDim==3) {
    tau3 = vf.testVar(S_TAU3, HDIV);
  }
  
  _stokesBF = Teuchos::rcp( new BF(vf) );
  // v1
  // tau1 terms:
  _stokesBF->addTerm(u1, tau1->div());
  _stokesBF->addTerm((1.0/_mu) * sigma1, tau1); // (sigma1, tau1)
  _stokesBF->addTerm(-u1_hat, tau1->dot_normal());
  
  // tau2 terms:
  _stokesBF->addTerm(u2, tau2->div());
  _stokesBF->addTerm((1.0/_mu) * sigma2, tau2);
  _stokesBF->addTerm(-u2_hat, tau2->dot_normal());
  
  // tau3:
  if (spaceDim==3) {
    _stokesBF->addTerm(u3, tau3->div());
    _stokesBF->addTerm((1.0/_mu) * sigma3, tau3);
    _stokesBF->addTerm(-u3_hat, tau3->dot_normal());
  }
  
  // v1:
  _stokesBF->addTerm(sigma1, v1->grad()); // (mu sigma1, grad v1)
  _stokesBF->addTerm( - p, v1->dx() );
  _stokesBF->addTerm( t1n, v1);
  
  // v2:
  _stokesBF->addTerm(sigma2, v2->grad()); // (mu sigma2, grad v2)
  _stokesBF->addTerm( - p, v2->dy());
  _stokesBF->addTerm( t2n, v2);
  
  // v3:
  if (spaceDim==3) {
    _stokesBF->addTerm(sigma3, v3->grad()); // (mu sigma3, grad v3)
    _stokesBF->addTerm( - p, v3->dz());
    _stokesBF->addTerm( t3n, v3);
  }
}

BFPtr StokesVGPFormulation::bf() {
  return _stokesBF;
}

VarPtr StokesVGPFormulation::p() {
  VarFactory vf = _stokesBF->varFactory();
  return vf.fieldVar(S_P);
}

VarPtr StokesVGPFormulation::sigma(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.fieldVar(S_SIGMA1);
    case 2:
      return vf.fieldVar(S_SIGMA2);
    case 3:
      return vf.fieldVar(S_SIGMA3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::u(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.fieldVar(S_U1);
    case 2:
      return vf.fieldVar(S_U2);
    case 3:
      return vf.fieldVar(S_U3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// traces:
VarPtr StokesVGPFormulation::tn_hat(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.fluxVar(S_TN1_HAT);
    case 2:
      return vf.fluxVar(S_TN2_HAT);
    case 3:
      return vf.fluxVar(S_TN3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::u_hat(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.traceVar(S_U1_HAT);
    case 2:
      return vf.traceVar(S_U2_HAT);
    case 3:
      return vf.traceVar(S_U3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// test variables:
VarPtr StokesVGPFormulation::tau(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.testVar(S_TAU1, HDIV);
    case 2:
      return vf.testVar(S_TAU2, HDIV);
    case 3:
      return vf.testVar(S_TAU3, HDIV);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr StokesVGPFormulation::v(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.testVar(S_V1, HGRAD);
    case 2:
      return vf.testVar(S_V2, HGRAD);
    case 3:
      return vf.testVar(S_V3, HGRAD);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}