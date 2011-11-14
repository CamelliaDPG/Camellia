#ifndef DPG_CONFUSION_PROBLEM_FIRST_TIMESTEP
#define DPG_CONFUSION_PROBLEM_FIRST_TIMESTEP

#include "BC.h"
#include "RHS.h"

#include "TransientConfusionBilinearForm.h"
#include "Solution.h"

class ConfusionProblemFirstTimestep : public RHS, public BC {
 private:
  Teuchos::RCP< TransientConfusionBilinearForm > _cbf;
 public:
 ConfusionProblemFirstTimestep(Teuchos::RCP< TransientConfusionBilinearForm > cbf) : RHS(), BC() {    
    _cbf = cbf;
  }
 
  // RHS:
  bool nonZeroRHS(int testVarID) {
    return testVarID == TransientConfusionBilinearForm::V;
  }
  
  void rhs(int testVarID, FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    values.resize(numCells,numPoints);
    //    values.initialize(0.0);
    for (int cellIndex=0;cellIndex<numCells;cellIndex++){
      for (int pointIndex=0;pointIndex<numPoints;pointIndex++){
	double x = physicalPoints(cellIndex,pointIndex,0);
	double y = physicalPoints(cellIndex,pointIndex,1);
	double dt = _cbf->get_dt();
	values(cellIndex,pointIndex) = x*(1-x)*y*(1-y)/dt;
      }
    }
  }
  
  // BC
  bool bcsImposed(int varID) {
    return varID == TransientConfusionBilinearForm::U_HAT;
  }
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    double tol = 1e-14;
    double x_cut = 0.0;//zeros all around
    double y_cut = 0.0;//zeros all around
    TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument, "spaceDim != 2" );
    for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
        double x = physicalPoints(cellIndex, ptIndex, 0);
        double y = physicalPoints(cellIndex, ptIndex, 1);
        if ( (abs(x) < 1e-14) && (y<y_cut) ) { // x basically 0 ==> u = 1 - y	  
          dirichletValues(cellIndex,ptIndex) = 1.0 - y/y_cut;
        } else if ( (abs(y) < 1e-14) &&  (x<x_cut) ) { // y basically 0 ==> u = 1 - x
          dirichletValues(cellIndex,ptIndex) = 1.0 - x/x_cut;
        } else {
          dirichletValues(cellIndex,ptIndex) = 0.0;
        }
        imposeHere(cellIndex,ptIndex) = true;
      }
    }
  }
};
#endif
