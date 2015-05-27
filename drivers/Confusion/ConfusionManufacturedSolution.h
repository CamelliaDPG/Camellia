#ifndef DPG_CONFUSION_MANUFACTURED_SOLUTION
#define DPG_CONFUSION_MANUFACTURED_SOLUTION

#include "ExactSolution.h"
#include "BC.h"
#include "RHS.h"

#include "Var.h"

class ConfusionManufacturedSolution : public ExactSolution<double>
{
private:
  double _epsilon, _beta_x, _beta_y;

  VarPtr _u_hat, _beta_n_u_minus_sigma_hat; // trial

  VarPtr _v; // test var
protected:
  FunctionPtr u();
public:
  ConfusionManufacturedSolution(double epsilon, double beta_x, double beta_y);

  // ExactSolution:
  virtual int H1Order(); // polyOrder+1, for polynomial solutions...

  static Teuchos::RCP<ExactSolution<double>> confusionManufacturedSolution(double epsilon, double beta_x, double beta_y);
};
#endif
