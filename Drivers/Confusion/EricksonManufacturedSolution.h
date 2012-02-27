#ifndef DPG_ERICKSON_MANUFACTURED_SOLUTION
#define DPG_ERICKSON_MANUFACTURED_SOLUTION

#include <Sacado.hpp>  // for automatic differentiation
#include "ExactSolution.h"
#include "BC.h"
#include "RHS.h"
#include "Constraints.h"

class EricksonManufacturedSolution : public ExactSolution, public RHS, public BC, public Constraints, public AbstractFunction {
private:
  double _epsilon, _beta_x, _beta_y;
protected:
  template <typename T>
  const T u(T &x, T &y);  // in 2 dimensions
public:
  EricksonManufacturedSolution(double epsilon, double beta_x, double beta_y);
  
  // ExactSolution:
  virtual int H1Order(); // polyOrder+1, for polynomial solutions...
  virtual double solutionValue(int trialID,
                               FieldContainer<double> &physicalPoint);
  virtual double solutionValue(int trialID,
                               FieldContainer<double> &physicalPoint,
                               FieldContainer<double> &unitNormal);
  
  // RHS:
  virtual bool nonZeroRHS(int testVarID);
  virtual void rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values);
  
  // BC
  virtual bool bcsImposed(int varID); // returns true if there are any BCs anywhere imposed on varID
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere);

  // Constraints
  virtual void getConstraints(FieldContainer<double> &physicalPoints, 
			      FieldContainer<double> &unitNormals,
			      vector<map<int,FieldContainer<double > > > &constraintCoeffs,
			      vector<FieldContainer<double > > &constraintValues);

  // AbstractFunction
  virtual void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints);
  
};
#endif
