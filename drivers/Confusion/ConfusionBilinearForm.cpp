#include "ConfusionBilinearForm.h"

#include "Function.h"

// trial variable names:
const string ConfusionBilinearForm::S_SIGMA_1 = "\\sigma_1";
const string ConfusionBilinearForm::S_SIGMA_2 = "\\sigma_2";
const string ConfusionBilinearForm::S_U = "u";
const string ConfusionBilinearForm::S_U_HAT = "\\hat{u }";
const string ConfusionBilinearForm::S_BETA_N_U_MINUS_SIGMA_HAT = "\\widehat{\\beta_n u - \\sigma_n}";

// test variable names:
const string ConfusionBilinearForm::S_TAU = "\\tau";
const string ConfusionBilinearForm::S_V = "v";

int ConfusionBilinearForm::U_ID = 2, ConfusionBilinearForm::SIGMA_1_ID = 3, ConfusionBilinearForm::SIGMA_2_ID = 3;

int ConfusionBilinearForm::V_ID = 1;

BFPtr ConfusionBilinearForm::confusionBF(double eps, FunctionPtr beta)
{
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr tau = vf->testVar(S_TAU, HDIV);
  VarPtr v = vf->testVar(S_V, HGRAD);

  VarPtr u_hat = vf->traceVar(S_U_HAT);
  VarPtr beta_n_u_minus_sigma_hat = vf->fluxVar(S_BETA_N_U_MINUS_SIGMA_HAT);

  VarPtr u = vf->fieldVar(S_U);
  VarPtr sigma1 = vf->fieldVar(S_SIGMA_1);
  VarPtr sigma2 = vf->fieldVar(S_SIGMA_2);

  U_ID = u->ID();
  V_ID = v->ID();
  SIGMA_1_ID = sigma1->ID();
  SIGMA_2_ID = sigma2->ID();

  BFPtr bf = Teuchos::rcp( new BF(vf) );

  // tau terms:
  bf->addTerm(sigma1 / eps, tau->x());
  bf->addTerm(sigma2 / eps, tau->y());
  bf->addTerm(u, tau->div());
  bf->addTerm(-u_hat, tau->dot_normal());

  // v terms:
  bf->addTerm( sigma1, v->dx() );
  bf->addTerm( sigma2, v->dy() );
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_minus_sigma_hat, v);

  return bf;
}

BFPtr ConfusionBilinearForm::confusionBF(double epsilon, double beta_x, double beta_y)
{
  FunctionPtr beta = Function::vectorize(Function::constant(beta_x), Function::constant(beta_y));

  return confusionBF(epsilon, beta);
}
