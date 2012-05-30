#ifndef CARTER_PLATE_PROBLEM
#define CARTER_PLATE_PROBLEM

#include "BC.h"
#include "RHS.h"
#include "Constraints.h"
#include "BasisCache.h"

#include "NavierStokesBilinearForm.h"

class CarterPlateProblem : public RHS, public BC, public Constraints {
private:
  Teuchos::RCP<NavierStokesBilinearForm> _bf;
  double tol;
public:
  CarterPlateProblem( Teuchos::RCP<NavierStokesBilinearForm> bf);
  // RHS:
  
  vector<EOperatorExtended> operatorsForTestID(int testID);
  
  bool nonZeroRHS(int testVarID);
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
