//
//  PoissonFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/16/14.
//
//

#include "PoissonFormulation.h"

const string PoissonFormulation::S_PHI = "\\phi";
const string PoissonFormulation::S_PSI = "\\psi";

const string PoissonFormulation::S_PHI_HAT = "\\widehat{\\phi}";
const string PoissonFormulation::S_PSI_N_HAT = "\\widehat{\\psi}_n";

const string PoissonFormulation::S_Q = "q";
const string PoissonFormulation::S_TAU = "\\tau";

PoissonFormulation::PoissonFormulation(int spaceDim, bool useConformingTraces) {
  _spaceDim = spaceDim;
  
  Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
  Space phi_hat_space = useConformingTraces ? HGRAD : L2;
  Space psiSpace = (spaceDim > 1) ? VECTOR_L2 : L2;
  
  // fields
  VarPtr phi;
  VarPtr psi;
  
  // traces
  VarPtr phi_hat, psi_n_hat;
  
  // tests
  VarPtr q;
  VarPtr tau;
  
  VarFactory vf;
  phi = vf.fieldVar(S_PHI);
  psi = vf.fieldVar(S_PSI, psiSpace);

  if (spaceDim > 1)
    phi_hat = vf.traceVar(S_PHI_HAT, phi, phi_hat_space);
  else
    phi_hat = vf.fluxVar(S_PHI_HAT, phi, phi_hat_space); // for spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
  
  FunctionPtr n = Function::normal();
  if (spaceDim > 1)
    psi_n_hat = vf.fluxVar(S_PSI_N_HAT, psi * n);
  else
    psi_n_hat = vf.fluxVar(S_PSI_N_HAT, psi);
  
  q = vf.testVar(S_Q, HGRAD);
  tau = vf.testVar(S_TAU, tauSpace);
  
  _poissonBF = Teuchos::rcp( new BF(vf) );

  if (spaceDim==1) {
    // for spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
    _poissonBF->addTerm(phi, tau->dx());
    _poissonBF->addTerm(psi, tau);
    _poissonBF->addTerm(-phi_hat, tau);
    
    _poissonBF->addTerm(-psi, q->dx());
    _poissonBF->addTerm(psi_n_hat, q);
  } else {
    _poissonBF->addTerm(phi, tau->div());
    _poissonBF->addTerm(psi, tau);
    _poissonBF->addTerm(-phi_hat, tau->dot_normal());
    
    _poissonBF->addTerm(-psi, q->grad());
    _poissonBF->addTerm(psi_n_hat, q);
  }
}

BFPtr PoissonFormulation::bf() {
  return _poissonBF;
}

// field variables:
VarPtr PoissonFormulation::phi() {
  VarFactory vf = _poissonBF->varFactory();
  return vf.fieldVar(S_PHI);
}

VarPtr PoissonFormulation::psi() {
  VarFactory vf = _poissonBF->varFactory();
  return vf.fieldVar(S_PSI);
}

// traces:
VarPtr PoissonFormulation::psi_n_hat() {
  VarFactory vf = _poissonBF->varFactory();
  return vf.fluxVar(S_PSI_N_HAT);
}

VarPtr PoissonFormulation::phi_hat() {
  VarFactory vf = _poissonBF->varFactory();
  return vf.traceVar(S_PHI_HAT);
}

// test variables:
VarPtr PoissonFormulation::q() {
  VarFactory vf = _poissonBF->varFactory();
  return vf.testVar(S_Q, HGRAD);
}

VarPtr PoissonFormulation::tau() {
  VarFactory vf = _poissonBF->varFactory();
  if (_spaceDim > 1)
    return vf.testVar(S_TAU, HDIV);
  else
    return vf.testVar(S_TAU, HGRAD);
}