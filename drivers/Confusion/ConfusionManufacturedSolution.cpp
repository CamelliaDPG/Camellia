#include "ConfusionBilinearForm.h"
#include "ConfusionManufacturedSolution.h"

ConfusionManufacturedSolution::ConfusionManufacturedSolution(double epsilon, double beta_x, double beta_y) {
  _epsilon = epsilon;
  _beta_x  = beta_x;
  _beta_y  = beta_y;
  
  // set the class variables from ExactSolution:
//  _bc = Teuchos::rcp(this,false);  // false: don't let the RCP own the memory
//  _rhs = Teuchos::rcp(this,false);
  
  BFPtr bf = ConfusionBilinearForm::confusionBF(epsilon,beta_x,beta_y);
  
  _bilinearForm = bf;
  
  VarFactory vf = bf->varFactory();
  
  VarPtr u = vf.fieldVar(ConfusionBilinearForm::S_U);
  VarPtr sigma1 = vf.fieldVar(ConfusionBilinearForm::S_SIGMA_1);
  VarPtr sigma2 = vf.fieldVar(ConfusionBilinearForm::S_SIGMA_2);
  
  _u_hat = vf.traceVar(ConfusionBilinearForm::S_U_HAT);
  _beta_n_u_minus_sigma_hat = vf.fluxVar(ConfusionBilinearForm::S_BETA_N_U_MINUS_SIGMA_HAT);
  
  _v = vf.testVar(ConfusionBilinearForm::S_V, HGRAD);
  
  FunctionPtr u_exact = this->u();
  FunctionPtr sigma_exact = epsilon * u_exact->grad();
  
  FunctionPtr u_exact_laplacian = u_exact->dx()->dx() + u_exact->dy()->dy();
  
  _rhs = RHS::rhs();
  FunctionPtr f = - _epsilon * u_exact_laplacian + _beta_x * u_exact->dx() + _beta_y * u_exact->dy();
  _rhs->addTerm( f * _v );
  
  _bc = BC::bc();
  _bc->addDirichlet(_u_hat, SpatialFilter::allSpace(), u_exact);
  
  FunctionPtr beta = Function::vectorize(Function::constant(_beta_x), Function::constant(_beta_y));
  FunctionPtr n = Function::normal();
  FunctionPtr one_skeleton = Function::meshSkeletonCharacteristic(); // allows restriction to skeleton
  FunctionPtr sigma_flux_exact = beta * ( n * u_exact - sigma_exact * one_skeleton);
  
  this->setSolutionFunction(u, u_exact);
  this->setSolutionFunction(sigma1, sigma_exact->x());
  this->setSolutionFunction(sigma2, sigma_exact->y());
  this->setSolutionFunction(_u_hat, u_exact);
  this->setSolutionFunction(_beta_n_u_minus_sigma_hat, sigma_flux_exact);
}

int ConfusionManufacturedSolution::H1Order() {
  // -1 for non-polynomial solution...
  return -1;
}

FunctionPtr ConfusionManufacturedSolution::u() {
  // DPG Part III, section 5.1 (Egger and Schoeberl) solution choice:
  // u =   (x + (exp[(beta_x * x)/eps] - 1)/(1-exp[beta_x/eps]))
  //     * (y + (exp[(beta_y * y)/eps] - 1)/(1-exp[beta_y/eps]))

  FunctionPtr exp_beta_x = Teuchos::rcp( new Exp_ax(_beta_x / _epsilon) );
  FunctionPtr exp_beta_y = Teuchos::rcp( new Exp_ay(_beta_y / _epsilon) );
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  
  FunctionPtr f = x + (exp_beta_x - 1.0)/(1.0 - exp(_beta_x/_epsilon));
  f  = f * (y + (exp_beta_y - 1.0)/(1.0 - exp(_beta_y/_epsilon)) );
  return f;
}