#ifndef DPG_CONFUSION_PROBLEM
#define DPG_CONFUSION_PROBLEM

#include "BC.h"
#include "RHS.h"

#include "ConfusionBilinearForm.h"

class ConfusionProblem : public RHS, public BC {
 private:
  Teuchos::RCP<ConfusionBilinearForm> _cbf;
 public:
  ConfusionProblem(Teuchos::RCP<ConfusionBilinearForm> cbf);
    
  // RHS:
  bool nonZeroRHS(int testVarID);
  
  void rhs(int testVarID, FieldContainer<double> &physicalPoints, FieldContainer<double> &values);
  
  // BC
  bool bcsImposed(int varID);
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
			FieldContainer<bool> &imposeHere);
};
#endif
