//
//  H1ProjectionFormulation.cpp
//  Camellia
//
//  Created by Brendan Keith 08/16.
//
//

#include "H1ProjectionFormulation.h"

using namespace Camellia;

const string H1ProjectionFormulation::S_PHI = "\\phi";
const string H1ProjectionFormulation::S_PSI = "\\psi";

const string H1ProjectionFormulation::S_PHI_HAT = "\\widehat{\\phi}";
const string H1ProjectionFormulation::S_PSI_N_HAT = "\\widehat{\\psi}_n";

const string H1ProjectionFormulation::S_Q = "q";
const string H1ProjectionFormulation::S_TAU = "\\tau";

H1ProjectionFormulation::H1ProjectionFormulation(int spaceDim, bool useConformingTraces, H1ProjectionFormulationChoice formulationChoice)
{
  _spaceDim = spaceDim;

  if (formulationChoice == ULTRAWEAK)
  {
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

    VarFactoryPtr vf = VarFactory::varFactory();
    phi = vf->fieldVar(S_PHI);
    psi = vf->fieldVar(S_PSI, psiSpace);

    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    
    if (spaceDim > 1)
      phi_hat = vf->traceVar(S_PHI_HAT, -phi, phi_hat_space);
    else
      phi_hat = vf->fluxVar(S_PHI_HAT, -phi * (Function::normal_1D() * parity), phi_hat_space); // for spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)

    TFunctionPtr<double> n = TFunction<double>::normal();

    if (spaceDim > 1)
      psi_n_hat = vf->fluxVar(S_PSI_N_HAT, -psi * (n * parity));
    else
      psi_n_hat = vf->fluxVar(S_PSI_N_HAT, -psi * (Function::normal_1D() * parity));

    q = vf->testVar(S_Q, HGRAD);
    tau = vf->testVar(S_TAU, tauSpace);

    _H1ProjectionBF = Teuchos::rcp( new BF(vf) );

    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
      _H1ProjectionBF->addTerm(psi, tau);
      _H1ProjectionBF->addTerm(phi, tau->dx());
      _H1ProjectionBF->addTerm(-phi_hat, tau);

      _H1ProjectionBF->addTerm(psi, q->dx());
      _H1ProjectionBF->addTerm(phi, q);
      _H1ProjectionBF->addTerm(-psi_n_hat, q);
    }
    else
    {
      _H1ProjectionBF->addTerm(psi, tau);
      _H1ProjectionBF->addTerm(phi, tau->div());
      _H1ProjectionBF->addTerm(-phi_hat, tau->dot_normal());

      _H1ProjectionBF->addTerm(psi, q->grad());
      _H1ProjectionBF->addTerm(phi, q);
      _H1ProjectionBF->addTerm(-psi_n_hat, q);
    }
  }
  else if ((formulationChoice == PRIMAL) || (formulationChoice == CONTINUOUS_GALERKIN))
  {
    // field
    VarPtr phi;
    
    // flux
    VarPtr psi_n_hat;
    
    // tests
    VarPtr q;
    
    VarFactoryPtr vf = VarFactory::varFactory();
    phi = vf->fieldVar(S_PHI, HGRAD);
    
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();
    
    if (formulationChoice == PRIMAL)
    {
      if (spaceDim > 1)
        psi_n_hat = vf->fluxVar(S_PSI_N_HAT, -phi->grad() * (n * parity));
      else
        psi_n_hat = vf->fluxVar(S_PSI_N_HAT, -phi->dx() * (Function::normal_1D() * parity));
    }
    q = vf->testVar(S_Q, HGRAD);
    
    _H1ProjectionBF = BF::bf(vf);
    _H1ProjectionBF->addTerm(phi->grad(), q->grad());
    _H1ProjectionBF->addTerm(phi, q);

    if (formulationChoice == CONTINUOUS_GALERKIN)
    {
      FunctionPtr boundaryIndicator = Function::meshBoundaryCharacteristic();
      _H1ProjectionBF->addTerm(-phi->grad() * n, boundaryIndicator * q);
    }
    else // primal
    {
      _H1ProjectionBF->addTerm(-psi_n_hat, q);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported H1ProjectionFormulationChoice");
  }
}

BFPtr H1ProjectionFormulation::bf()
{
  return _H1ProjectionBF;
}

// field variables:
VarPtr H1ProjectionFormulation::phi()
{
  VarFactoryPtr vf = _H1ProjectionBF->varFactory();
  return vf->fieldVar(S_PHI);
}

VarPtr H1ProjectionFormulation::psi()
{
  VarFactoryPtr vf = _H1ProjectionBF->varFactory();
  return vf->fieldVar(S_PSI);
}

// traces:
VarPtr H1ProjectionFormulation::psi_n_hat()
{
  VarFactoryPtr vf = _H1ProjectionBF->varFactory();
  return vf->fluxVar(S_PSI_N_HAT);
}

VarPtr H1ProjectionFormulation::phi_hat()
{
  VarFactoryPtr vf = _H1ProjectionBF->varFactory();
  return vf->traceVar(S_PHI_HAT);
}

// test variables:
VarPtr H1ProjectionFormulation::q()
{
  VarFactoryPtr vf = _H1ProjectionBF->varFactory();
  return vf->testVar(S_Q, HGRAD);
}

VarPtr H1ProjectionFormulation::tau()
{
  VarFactoryPtr vf = _H1ProjectionBF->varFactory();
  if (_spaceDim > 1)
    return vf->testVar(S_TAU, HDIV);
  else
    return vf->testVar(S_TAU, HGRAD);
}
