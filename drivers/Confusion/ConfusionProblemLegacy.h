#ifndef DPG_CONFUSION_PROBLEM
#define DPG_CONFUSION_PROBLEM

#include "BC.h"
#include "RHS.h"

#include "BF.h"

#include "Var.h"

#include "Function.h"

class ConfusionProblemLegacy : public RHS, public BC {
 private:
  BFPtr _cbf;
  
  double _beta_x, _beta_y;
  
  VarPtr _u_hat, _beta_n_u_minus_sigma_hat; // trial
  
  VarPtr _v; // test var
 public:
  ConfusionProblemLegacy(BFPtr cbf, double beta_x, double beta_y);
    
  // RHS:
  bool nonZeroRHS(int testVarID);
  
  void rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values);
  
  // BC
  bool bcsImposed(int varID);
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
			FieldContainer<bool> &imposeHere);
};
#endif
