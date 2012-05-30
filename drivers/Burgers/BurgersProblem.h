#ifndef DPG_BURGERS_PROBLEM
#define DPG_BURGERS_PROBLEM

#include "BC.h"
#include "RHS.h"
#include "Constraints.h"
#include "BasisCache.h"

#include "BurgersBilinearForm.h"

class BurgersProblem : public RHS, public BC, public Constraints {
private:
  Teuchos::RCP<BurgersBilinearForm> _bf;
  double tol;
public:
  BurgersProblem( Teuchos::RCP<BurgersBilinearForm> bf);
  // RHS:
  
  vector<EOperatorExtended> operatorsForTestID(int testID);
  
  bool nonZeroRHS(int testVarID);
  
//  void rhs(int testVarID, int operatorIndex, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values);  
  void rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, FieldContainer<double> &values);
    
  // BC
  bool bcsImposed(int varID);
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere);

  void getConstraints(FieldContainer<double> &physicalPoints, 
                              FieldContainer<double> &unitNormals,
                              vector<map<int,FieldContainer<double > > > &constraintCoeffs,
                              vector<FieldContainer<double > > &constraintValues);
};

#endif
