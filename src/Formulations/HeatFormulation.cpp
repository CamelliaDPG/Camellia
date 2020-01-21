//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  HeatFormulation.cpp
//  Camellia
//
//  Created by Stefan Henneking on 11/14/19.
//
//

#include "HeatFormulation.h"
#include "RHS.h"

using namespace Camellia;

const string HeatFormulation::S_U = "u";
const string HeatFormulation::S_SIGMA = "\\sigma";

const string HeatFormulation::S_U_HAT = "\\widehat{u}";
const string HeatFormulation::S_SIGMA_N_HAT = "\\widehat{\\sigma}_n";

const string HeatFormulation::S_V = "v";
const string HeatFormulation::S_TAU = "\\tau";

HeatFormulation::HeatFormulation(int spaceDim, bool useConformingTraces, double timeStep, double theta, HeatFormulationChoice formulationChoice, bool fieldImplicit)
{
  _spaceDim = spaceDim;

  if (formulationChoice == ULTRAWEAK)
  {
    Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
    Space u_hat_space = useConformingTraces ? HGRAD : L2;
    Space sigmaSpace = (spaceDim > 1) ? VECTOR_L2 : L2;

    // fields
    VarPtr u;
    VarPtr sigma;

    // traces
    VarPtr u_hat, sigma_n_hat;

    // tests
    VarPtr v;
    VarPtr tau;

    VarFactoryPtr vf = VarFactory::varFactory();
    u = vf->fieldVar(S_U);
    sigma = vf->fieldVar(S_SIGMA, sigmaSpace);

    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    
    if (!fieldImplicit) {
      if (spaceDim > 1)
        u_hat = vf->traceVar(S_U_HAT, u, u_hat_space);
      else
        // for spaceDim==1, the "normal" component is in the flux of u_hat (it's a plus or minus 1)
        u_hat = vf->fluxVar(S_U_HAT, u * (Function::normal_1D() * parity), u_hat_space);
    }
    
    TFunctionPtr<double> n = TFunction<double>::normal();
    
    if (theta > 0.0 && !fieldImplicit) {
      if (spaceDim > 1) sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, sigma * (n * parity));
      else              sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, sigma * (Function::normal_1D() * parity));
    }
    v = vf->testVar(S_V, HGRAD);
    tau = vf->testVar(S_TAU, tauSpace);

    _heatBF = Teuchos::rcp( new BF(vf) );

    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
      _heatBF->addTerm(u, tau->dx());
      _heatBF->addTerm(sigma, tau);
      if (!fieldImplicit) _heatBF->addTerm(-u_hat, tau);

      if (timeStep > 0.0) _heatBF->addTerm(u/timeStep, v);
      _heatBF->addTerm(theta*sigma, v->dx());
      if (theta > 0.0 && !fieldImplicit) _heatBF->addTerm(-theta*sigma_n_hat, v);
    }
    else
    {
      _heatBF->addTerm(u, tau->div());
      _heatBF->addTerm(sigma, tau);
      if (!fieldImplicit) _heatBF->addTerm(-u_hat, tau->dot_normal());

      if (timeStep > 0.0) _heatBF->addTerm(u/timeStep, v);
      _heatBF->addTerm(theta*sigma, v->grad());
      if (theta > 0.0 && !fieldImplicit) _heatBF->addTerm(-theta*sigma_n_hat, v);
    }
  }
  else if ((formulationChoice == PRIMAL) || (formulationChoice == CONTINUOUS_GALERKIN))
  {
    // field
    VarPtr u;
    
    // flux
    VarPtr sigma_n_hat;
    
    // tests
    VarPtr v;
    
    VarFactoryPtr vf = VarFactory::varFactory();
    u = vf->fieldVar(S_U, HGRAD);
    
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();
    
    if (formulationChoice == PRIMAL && theta > 0.0 && !fieldImplicit)
    {
      if (spaceDim > 1)
        sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, u->grad() * (n * parity)); // TODO: WHY ARE WE PASSING A LINEAR TERM INSTEAD OF A SPACE?
      else
        sigma_n_hat = vf->fluxVar(S_SIGMA_N_HAT, u->dx() * (Function::normal_1D() * parity));
    }
    v = vf->testVar(S_V, HGRAD);
    
    _heatBF = BF::bf(vf);
    if (timeStep > 0.0) _heatBF->addTerm(u/timeStep, v);
    if (theta > 0.0) _heatBF->addTerm(theta*u->grad(), v->grad());

    if (formulationChoice == CONTINUOUS_GALERKIN && theta > 0.0)
    {
      FunctionPtr boundaryIndicator = Function::meshBoundaryCharacteristic();
      _heatBF->addTerm(-theta*u->grad() * n, boundaryIndicator * v); // what about 'parity'
    }
    else if (theta > 0.0 && !fieldImplicit) // primal (neither explicit scheme nor field implicit scheme)
    {
      _heatBF->addTerm(-theta*sigma_n_hat, v);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported HeatFormulationChoice");
  }
}

BFPtr HeatFormulation::bf()
{
  return _heatBF;
}

RHSPtr HeatFormulation::rhs(FunctionPtr forcingFunction)
{
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(forcingFunction * v());
  return rhs;
}

// field variables:
VarPtr HeatFormulation::u()
{
  VarFactoryPtr vf = _heatBF->varFactory();
  return vf->fieldVar(S_U);
}

VarPtr HeatFormulation::sigma()
{
  VarFactoryPtr vf = _heatBF->varFactory();
  return vf->fieldVar(S_SIGMA);
}

// traces:
VarPtr HeatFormulation::sigma_n_hat()
{
  VarFactoryPtr vf = _heatBF->varFactory();
  return vf->fluxVar(S_SIGMA_N_HAT);
}

VarPtr HeatFormulation::u_hat()
{
  VarFactoryPtr vf = _heatBF->varFactory();
  return vf->traceVar(S_U_HAT);
}

// test variables:
VarPtr HeatFormulation::v()
{
  VarFactoryPtr vf = _heatBF->varFactory();
  return vf->testVar(S_V, HGRAD);
}

VarPtr HeatFormulation::tau()
{
  VarFactoryPtr vf = _heatBF->varFactory();
  if (_spaceDim > 1)
    return vf->testVar(S_TAU, HDIV);
  else
    return vf->testVar(S_TAU, HGRAD);
}
