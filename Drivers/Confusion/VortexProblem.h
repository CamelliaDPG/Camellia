#ifndef DPG_CONFECTION_VORTEX
#define DPG_CONFECTION_VORTEX

#include "BC.h"
#include "RHS.h"
#include "Constraints.h"

#include "ConfusionBilinearForm.h"

class VortexProblem : public RHS, public BC, public Constraints {
 private:
  Teuchos::RCP<ConfusionBilinearForm> _cbf;
  double tol;
 public:
 VortexProblem( Teuchos::RCP<ConfusionBilinearForm> cbf) : RHS(), BC(), Constraints() {
    _cbf = cbf;
    tol = 1e-14;
  }
  
  // RHS:
  bool nonZeroRHS(int testVarID) {
    return false; //testVarID == ConfusionBilinearForm::V;
  }
  
  void rhs(int testVarID, FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    values.resize(numCells,numPoints);
    values.initialize(0.0);
  }
  
  // BC
  bool bcsImposed(int varID) {
    //    return varID == ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT;
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
	double beta_n = _cbf->getBeta(x,y)[0]*unitNormals(cellIndex,ptIndex,0)+_cbf->getBeta(x,y)[1]*unitNormals(cellIndex,ptIndex,1);

	// inflow 
	double u0=0.0;
	// points are already on boundary; just test if they're inflow or outflow
	if (beta_n < 0){
	  double pi = 2.0*acos(0.0);
	  //	  u0 = sin(2.0*pi*x)+sin(2.0*pi*y);
	  u0 = abs(x*y + (1-x)*(1-y) -.5);
	  if (bcsImposed(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT)){
	    dirichletValues(cellIndex,ptIndex) = beta_n*u0;
	  } else {
	    dirichletValues(cellIndex,ptIndex) = u0;
	  }

	  imposeHere(cellIndex,ptIndex) = true;
	} else {
	  imposeHere(cellIndex,ptIndex) = false;
	}
	
      }
    }
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
	vector<double> beta = _cbf->getBeta(x,y);
	double beta_n = beta[0]*unitNormals(cellIndex,pointIndex,0)+beta[1]*unitNormals(cellIndex,pointIndex,1);
	
	bool isOnABoundary=false;
	if ((abs(x) < 1e-12) || (abs(y) < 1e-12)){
	  isOnABoundary = true;
	}
	if ((abs(x-1.0) < 1e-12) || (abs(y-1.0) < 1e-12)){
	  isOnABoundary = true;
	}
	if ( (beta_n > 0.0) && isOnABoundary) {
	  // this combo isolates sigma_n
	  uCoeffs(cellIndex,pointIndex) = beta_n;
	  beta_sigmaCoeffs(cellIndex,pointIndex) = -1.0;	    
	  outflowValues(cellIndex,pointIndex) = 0.0;//beta_n*_cbf->getEpsilon();
	}
	
      }
    }
    //    outflowConstraint[ConfusionBilinearForm::U_HAT] = beta_sigmaCoeffs;
    outflowConstraint[ConfusionBilinearForm::U_HAT] = uCoeffs;
    outflowConstraint[ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT] = beta_sigmaCoeffs;
    constraintCoeffs.push_back(outflowConstraint); // only one constraint on outflow
    constraintValues.push_back(outflowValues); // only one constraint on outflow
    
  }

};
#endif
