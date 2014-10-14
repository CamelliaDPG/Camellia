//
//  PressurelessStokesFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/9/14.
//
//

#include "PressurelessStokesFormulation.h"

const string PressurelessStokesFormulation::S_U1 = "u_1";
const string PressurelessStokesFormulation::S_U2 = "u_2";
const string PressurelessStokesFormulation::S_U3 = "u_3";
const string PressurelessStokesFormulation::S_SIGMA11 = "\\sigma_{11}";
const string PressurelessStokesFormulation::S_SIGMA12 = "\\sigma_{12}";
const string PressurelessStokesFormulation::S_SIGMA13 = "\\sigma_{13}";
const string PressurelessStokesFormulation::S_SIGMA22 = "\\sigma_{22}";
const string PressurelessStokesFormulation::S_SIGMA23 = "\\sigma_{23}";
const string PressurelessStokesFormulation::S_SIGMA33 = "\\sigma_{33}";

const string PressurelessStokesFormulation::S_U1_HAT = "\\widehat{u}_1";
const string PressurelessStokesFormulation::S_U2_HAT = "\\widehat{u}_2";
const string PressurelessStokesFormulation::S_U3_HAT = "\\widehat{u}_3";
const string PressurelessStokesFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string PressurelessStokesFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string PressurelessStokesFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string PressurelessStokesFormulation::S_V1 = "v_1";
const string PressurelessStokesFormulation::S_V2 = "v_2";
const string PressurelessStokesFormulation::S_V3 = "v_3";
const string PressurelessStokesFormulation::S_TAU11 = "\\tau_{11}";
const string PressurelessStokesFormulation::S_TAU12 = "\\tau_{12}";
const string PressurelessStokesFormulation::S_TAU13 = "\\tau_{13}";
const string PressurelessStokesFormulation::S_TAU22 = "\\tau_{22}";
const string PressurelessStokesFormulation::S_TAU23 = "\\tau_{23}";
const string PressurelessStokesFormulation::S_TAU33 = "\\tau_{33}";

PressurelessStokesFormulation::PressurelessStokesFormulation(int spaceDim) {
  _spaceDim = spaceDim;
  
  if ((spaceDim != 2) && (spaceDim != 3)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
  }
  
  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr sigma11, sigma12, sigma13;
  VarPtr sigma22, sigma23;
  VarPtr sigma33;
  
  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr t1n, t2n, t3n;
  
  // tests
  VarPtr v1, v2, v3;
  VarPtr tau11, tau12, tau13;
  VarPtr tau22, tau23;
  VarPtr tau33;
  
  VarFactory vf;
  u1 = vf.fieldVar(S_U1);
  u2 = vf.fieldVar(S_U2);
  if (spaceDim==3) u3 = vf.fieldVar(S_U3);
  
  sigma11 = vf.fieldVar(S_SIGMA11);
  sigma12 = vf.fieldVar(S_SIGMA12);
  sigma22 = vf.fieldVar(S_SIGMA22);
  if (spaceDim==3) {
    sigma13 = vf.fieldVar(S_SIGMA13);
    sigma23 = vf.fieldVar(S_SIGMA23);
    sigma33 = vf.fieldVar(S_SIGMA33);
  }
  
  u1_hat = vf.traceVar(S_U1_HAT, 1.0 * u1, L2);
  u2_hat = vf.traceVar(S_U2_HAT, 1.0 * u2, L2);
  if (spaceDim==3) u3_hat = vf.traceVar(S_U3_HAT, 1.0 * u3, L2);
  
  FunctionPtr n = Function::normal();
  LinearTermPtr sigma1n, sigma2n, sigma3n;
  if (spaceDim==2) {
    sigma1n = sigma11 * n->x() + sigma12 * n->y();
    sigma2n = sigma12 * n->x() + sigma22 * n->y();
  } else {
    sigma1n = sigma11 * n->x() + sigma12 * n->y() + sigma13 * n->z();
    sigma2n = sigma12 * n->x() + sigma22 * n->y() + sigma23 * n->z();
    sigma3n = sigma13 * n->x() + sigma23 * n->y() + sigma33 * n->z();
  }
  t1n = vf.fluxVar(S_TN1_HAT, sigma1n);
  t2n = vf.fluxVar(S_TN2_HAT, sigma2n);
  if (spaceDim==3) t3n = vf.fluxVar(S_TN3_HAT, sigma3n);
  
  v1 = vf.testVar(S_V1, HGRAD);
  v2 = vf.testVar(S_V2, HGRAD);
  if (spaceDim==3) v3 = vf.testVar(S_V3, HGRAD);
  
  tau11 = vf.testVar(S_TAU11, HGRAD);
  tau12 = vf.testVar(S_TAU12, HGRAD);
  tau22 = vf.testVar(S_TAU22, HGRAD);
  if (spaceDim==3) {
    tau13 = vf.testVar(S_TAU13, HGRAD);
    tau23 = vf.testVar(S_TAU23, HGRAD);
    tau33 = vf.testVar(S_TAU33, HGRAD);
  }
  
  _stokesBF = Teuchos::rcp( new BF(vf) );
  // v1
  _stokesBF->addTerm(-sigma11, v1->dx());
  _stokesBF->addTerm(-sigma12, v1->dy());
  if (spaceDim==3) {
    _stokesBF->addTerm(-sigma13, v1->dz());
  }
  _stokesBF->addTerm(t1n, v1);
  
  // v2
  _stokesBF->addTerm(-sigma12, v2->dx());
  _stokesBF->addTerm(-sigma22, v2->dy());
  if (spaceDim==3) {
    _stokesBF->addTerm(-sigma23, v2->dz());
  }
  _stokesBF->addTerm(t2n, v2);
  
  // v3
  if (spaceDim==3) {
    _stokesBF->addTerm(-sigma13, v3->dx());
    _stokesBF->addTerm(-sigma23, v3->dy());
    _stokesBF->addTerm(-sigma33, v3->dz());
    _stokesBF->addTerm(t3n, v3);
  }
  
  LinearTermPtr p; // pressure term, the negative weighted trace of tensor sigma
  if (spaceDim==2) {
    p = -0.5 * sigma11 + -0.5 * sigma22;
  } else {
    p = -(1.0/3.0) * sigma11 + -(1.0/3.0) * sigma22 + -(1.0/3.0) * sigma33;
  }
  
  LinearTermPtr tau1n, tau2n, tau3n;
  LinearTermPtr div_tau1, div_tau2, div_tau3;
  if (spaceDim==2) {
    tau1n = tau11 * n->x() + tau12 * n->y();
    tau2n = tau12 * n->x() + tau22 * n->y();
    
    div_tau1 = tau11->dx() + tau12->dy();
    div_tau2 = tau12->dx() + tau22->dy();
  } else {
    tau1n = tau11 * n->x() + tau12 * n->y() + tau13 * n->z();
    tau2n = tau12 * n->x() + tau22 * n->y() + tau23 * n->z();
    tau3n = tau13 * n->x() + tau23 * n->y() + tau33 * n->z();
    
    div_tau1 = tau11->dx() + tau12->dy() + tau13->dz();
    div_tau2 = tau12->dx() + tau22->dy() + tau23->dz();
    div_tau3 = tau13->dx() + tau23->dy() + tau33->dz();
  }
  
  // tau1j
  _stokesBF->addTerm(sigma11, tau11);
  _stokesBF->addTerm(sigma12, tau12);
  if (spaceDim==3) _stokesBF->addTerm(sigma13, tau13);
  _stokesBF->addTerm(2 * u1, div_tau1);
  _stokesBF->addTerm(-2 * u1_hat, tau1n);
  _stokesBF->addTerm(p, tau11);
  
  // tau2j
  _stokesBF->addTerm(sigma12, tau12);
  _stokesBF->addTerm(sigma22, tau22);
  if (spaceDim==3) _stokesBF->addTerm(sigma23, tau23);
  _stokesBF->addTerm(2 * u2, div_tau2);
  _stokesBF->addTerm(-2 * u2_hat, tau2n);
  _stokesBF->addTerm(p, tau22);
  
  // tau3j
  if (spaceDim==3) {
    _stokesBF->addTerm(sigma13, tau13);
    _stokesBF->addTerm(sigma23, tau23);
    _stokesBF->addTerm(sigma33, tau33);
    _stokesBF->addTerm(2 * u3, div_tau3);
    _stokesBF->addTerm(-2 * u3_hat, tau3n);
    _stokesBF->addTerm(p, tau33);
  }
}

BFPtr PressurelessStokesFormulation::bf() {
  return _stokesBF;
}

LinearTermPtr PressurelessStokesFormulation::p() {
  VarPtr sigma11 = this->sigma(1, 1);
  VarPtr sigma22 = this->sigma(2, 2);
  
  LinearTermPtr p; // pressure term, the negative weighted trace of tensor sigma
  if (_spaceDim==2) {
    p = -0.5 * sigma11 + -0.5 * sigma22;
  } else {
    VarPtr sigma33 = this->sigma(3, 3);
    p = -(1.0/3.0) * sigma11 + -(1.0/3.0) * sigma22 + -(1.0/3.0) * sigma33;
  }
  return p;
}

VarPtr PressurelessStokesFormulation::sigma(int i, int j) {
  if (i > j) { // swap them
    int k = i;
    i = j;
    j = k;
  }
  if (j > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i and j must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      switch (j) {
        case 1:
          return vf.fieldVar(S_SIGMA11);
        case 2:
          return vf.fieldVar(S_SIGMA12);
        case 3:
          return vf.fieldVar(S_SIGMA13);
      }
    case 2:
      switch (j) {
        case 2:
          return vf.fieldVar(S_SIGMA22);
        case 3:
          return vf.fieldVar(S_SIGMA23);
      }
    case 3:
      switch (j) {
        case 3:
          return vf.fieldVar(S_SIGMA23);
      }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled (i,j) pair");
}

VarPtr PressurelessStokesFormulation::u(int i) {
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
VarPtr PressurelessStokesFormulation::tn_hat(int i) {
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

VarPtr PressurelessStokesFormulation::u_hat(int i) {
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
VarPtr PressurelessStokesFormulation::tau(int i, int j) {
  if (i > j) { // swap them
    int k = i;
    i = j;
    j = k;
  }
  if (j > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i and j must be less than or equal to _spaceDim");
  }
  VarFactory vf = _stokesBF->varFactory();
  switch (i) {
    case 1:
      switch (j) {
        case 1:
          return vf.testVar(S_TAU11, HGRAD);
        case 2:
          return vf.testVar(S_TAU12, HGRAD);
        case 3:
          return vf.testVar(S_TAU13, HGRAD);
      }
    case 2:
      switch (j) {
        case 2:
          return vf.testVar(S_TAU22, HGRAD);
        case 3:
          return vf.testVar(S_TAU23, HGRAD);
      }
    case 3:
      switch (j) {
        case 3:
          return vf.testVar(S_TAU23, HGRAD);
      }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled (i,j) pair");
}

VarPtr PressurelessStokesFormulation::v(int i) {
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