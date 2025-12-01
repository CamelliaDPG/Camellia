//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  PoissonFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/16/14.
//
//

#include "PoissonFormulation.h"
#include "RHS.h"

using namespace Camellia;

const string PoissonFormulation::S_U = "u";
const string PoissonFormulation::S_SIGMA = "\\sigma";

const string PoissonFormulation::S_U_HAT = "\\widehat{u}";
const string PoissonFormulation::S_SIGMA_N_HAT = "\\widehat{\\sigma}_n";

const string PoissonFormulation::S_V = "v";
const string PoissonFormulation::S_TAU = "\\tau";

PoissonFormulation::PoissonFormulation(int spaceDim, bool useConformingTraces, PoissonFormulationChoice formulationChoice,
                                       const std::string &scalarVarName, const std::string &vectorVarName,
                                       const std::string &scalarTraceName, const std::string &vectorTraceName,
                                       bool reverseVectorVarSign)
:
_scalarVarName(scalarVarName),
_vectorVarPrefix(vectorVarName),
_scalarTraceName(scalarTraceName),
_vectorTraceName(vectorTraceName)
{
  _spaceDim = spaceDim;

  if (formulationChoice == ULTRAWEAK)
  {
    Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
    Space u_hat_space = useConformingTraces ? HGRAD : L2;
    Space sigmaSpace = L2;

    // fields
    VarPtr u;
    std::vector<VarPtr> sigma(_spaceDim);

    // traces
    VarPtr u_hat, sigma_n_hat;

    // tests
    VarPtr v;
    VarPtr tau;

    VarFactoryPtr vf = VarFactory::varFactory();
    u = vf->fieldVar(_scalarVarName);
    for (int d=0; d<_spaceDim; d++)
    {
      sigma[d] = vf->fieldVar(_vectorVarPrefix + std::to_string(d), sigmaSpace);
    }

    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    
    if (spaceDim > 1)
      u_hat = vf->traceVar(_scalarTraceName, u, u_hat_space);
    else
      u_hat = vf->fluxVar(_scalarTraceName, u * (Function::normal_1D() * parity), u_hat_space); // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)

    TFunctionPtr<double> n = TFunction<double>::normal();

    if (spaceDim > 1)
    {
      LinearTermPtr sigma_n;
      for (int d=0; d<_spaceDim; d++)
      {
        FunctionPtr n_d = n->spatialComponent(d+1);
        sigma_n = sigma_n + sigma[d] * n_d;
      }
      sigma_n_hat = vf->fluxVar(_vectorTraceName, sigma_n * parity);
    }
    else
      sigma_n_hat = vf->fluxVar(_vectorTraceName, sigma[0] * (Function::normal_1D() * parity));

    v = vf->testVar(S_V, HGRAD);
    tau = vf->testVar(S_TAU, tauSpace);

    _poissonBF = Teuchos::rcp( new BF(vf) );

    double sigmaDefWeight = reverseVectorVarSign ? -1 : 1;
    
    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
      _poissonBF->addTerm(u, tau->dx());
      _poissonBF->addTerm(sigmaDefWeight * sigma[0], tau);
      _poissonBF->addTerm(-u_hat, tau);

      _poissonBF->addTerm(-sigma[0], v->dx());
      _poissonBF->addTerm(sigma_n_hat, v);
    }
    else
    {
      _poissonBF->addTerm(u, tau->div());
      for (int d=0; d<_spaceDim; d++)
      {
        _poissonBF->addTerm(sigmaDefWeight * sigma[d], tau->spatialComponent(d+1));
      }
      _poissonBF->addTerm(-u_hat, tau->dot_normal());

      
      for (int d=0; d<_spaceDim; d++)
      {
        _poissonBF->addTerm(-sigma[d], v->di(d+1));
      }
      _poissonBF->addTerm(sigma_n_hat, v);
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
    u = vf->fieldVar(_scalarVarName, HGRAD);
    
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();
    
    if (formulationChoice == PRIMAL)
    {
      if (spaceDim > 1)
        sigma_n_hat = vf->fluxVar(_vectorTraceName, u->grad() * (n * parity));
      else
        sigma_n_hat = vf->fluxVar(_vectorTraceName, u->dx() * (Function::normal_1D() * parity));
    }
    v = vf->testVar(S_V, HGRAD);
    
    _poissonBF = BF::bf(vf);
    _poissonBF->addTerm(-u->grad(), v->grad());

    if (formulationChoice == CONTINUOUS_GALERKIN)
    {
      FunctionPtr boundaryIndicator = Function::meshBoundaryCharacteristic();
      _poissonBF->addTerm(u->grad() * n, boundaryIndicator * v);
    }
    else // primal
    {
      _poissonBF->addTerm(sigma_n_hat, v);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported PoissonFormulationChoice");
  }
}

BFPtr PoissonFormulation::bf()
{
  return _poissonBF;
}

RHSPtr PoissonFormulation::rhs(FunctionPtr forcingFunction)
{
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(forcingFunction * v());
  return rhs;
}

// field variables:
VarPtr PoissonFormulation::u()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->fieldVar(_scalarVarName);
}

VarPtr PoissonFormulation::sigma(const int &d)
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->fieldVar(_vectorVarPrefix + std::to_string(d));
}

// traces:
VarPtr PoissonFormulation::sigma_n_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->fluxVar(_vectorTraceName);
}

VarPtr PoissonFormulation::u_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->traceVar(_scalarTraceName);
}

// test variables:
VarPtr PoissonFormulation::v()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->testVar(S_V, HGRAD);
}

VarPtr PoissonFormulation::tau()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  if (_spaceDim > 1)
    return vf->testVar(S_TAU, HDIV);
  else
    return vf->testVar(S_TAU, HGRAD);
}
