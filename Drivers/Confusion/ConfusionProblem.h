#ifndef DPG_CONFUSION_PROBLEM
#define DPG_CONFUSION_PROBLEM

#include "BC.h"
#include "RHS.h"

#include "ConfusionBilinearForm.h"

class ConfusionProblem : public RHS, public BC {
public:
  ConfusionProblem() : RHS(), BC() {
    
  }
    
  // RHS:
  bool nonZeroRHS(int testVarID) {
    return testVarID == ConfusionBilinearForm::V;
  }
  
  void rhs(int testVarID, FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    values.resize(numCells,numPoints);
    //    values.initialize(1.0);
    values.initialize(0.0);
  }
  
  // BC
  bool bcsImposed(int varID) {
    return varID == ConfusionBilinearForm::U_HAT;
  }
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    double tol = 1e-14;
    double x_cut = .50;
    double y_cut = .50;
    TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument, "spaceDim != 2" );
    for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
        double x = physicalPoints(cellIndex, ptIndex, 0);
        double y = physicalPoints(cellIndex, ptIndex, 1);
	
        if ( (abs(x) < 1e-14) && (y<y_cut) ) { // x basically 0 ==> u = 1 - y	  
	  dirichletValues(cellIndex,ptIndex) = 1.0 - y/y_cut;
	  //          dirichletValues(cellIndex,ptIndex) = exp(-1.0/(1.0-y*y)); // bump function
        } else if ( (abs(y) < 1e-14) &&  (x<x_cut) ) { // y basically 0 ==> u = 1 - x
	  dirichletValues(cellIndex,ptIndex) = 1.0 - x/x_cut;
	  //          dirichletValues(cellIndex,ptIndex) = exp(-1.0/(1.0-x*x)); // bump function
        } else {
          dirichletValues(cellIndex,ptIndex) = 0.0;
        }
        imposeHere(cellIndex,ptIndex) = true;
      }
    }
  }
};
#endif
