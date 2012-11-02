#ifndef DPG_CONFECTION_CONSTRAINT_BC
#define DPG_CONFECTION_CONSTRAINT_BC

#include "Constraints.h"
#include "ConfusionBilinearForm.h"
#include "Teuchos_RCP.hpp"

class ConfectionBCConstraints : public Constraints {
 private: 
  Teuchos::RCP<ConfusionBilinearForm> _confusionBilinearForm;
  double tol;
 public:
  ConfectionBCConstraints(Teuchos::RCP< ConfusionBilinearForm > bfs){
    _confusionBilinearForm = bfs;
    tol = 1e-8;
  }

  virtual void getConstraints(FieldContainer<double> &physicalPoints, 
		      FieldContainer<double> &unitNormals,
		      vector<map<int,FieldContainer<double > > > &constraintCoeffs,
		      vector<FieldContainer<double > > &constraintValues){
    
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);       
    map<int,FieldContainer<double> > outflowConstraint;
    FieldContainer<double> uCoeffs(numCells,numPoints);
    FieldContainer<double> beta_sigmaCoeffs(numCells,numPoints);
    FieldContainer<double> outflowValues(numCells,numPoints);

    // default to no constraints, apply on outflow only
    uCoeffs.initialize(0.0);
    beta_sigmaCoeffs.initialize(0.0);
    outflowValues.initialize(0.0);
    
    for (int cellIndex=0;cellIndex<numCells;cellIndex++){
      for (int pointIndex=0;pointIndex<numPoints;pointIndex++){
	double x = physicalPoints(cellIndex,pointIndex,0);
	double y = physicalPoints(cellIndex,pointIndex,1);
	vector<double> beta = _confusionBilinearForm->getBeta(x,y);
	double beta_n = beta[0]*unitNormals(cellIndex,pointIndex,0)+beta[1]*unitNormals(cellIndex,pointIndex,1);

	
	if ((abs(x-1.0) < tol) || (abs(y-1.0) < tol)) { // if on outflow boundary
	  TEUCHOS_TEST_FOR_EXCEPTION(beta_n < 0,std::invalid_argument,"Inflow condition on boundary");
	  
	  // this combo isolates sigma_n
	  //	  uCoeffs(cellIndex,pointIndex) = 1.0;
	  uCoeffs(cellIndex,pointIndex) = beta_n;
	  beta_sigmaCoeffs(cellIndex,pointIndex) = -1.0;	    
	}
	
      }
    }
    outflowConstraint[ConfusionBilinearForm::U_HAT] = uCoeffs;
    outflowConstraint[ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT] = beta_sigmaCoeffs;	        
    constraintCoeffs.push_back(outflowConstraint); // only one constraint on outflow
    
  }
};

#endif
