//
//  BurgersProblem.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/20/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "BurgersProblem.h"
#include "Solution.h"

BurgersProblem::BurgersProblem( Teuchos::RCP<BurgersBilinearForm> bf) : RHS(), BC(), Constraints() {
  _bf = bf;
  tol = 1e-14;
}

// RHS:

vector<EOperatorExtended> BurgersProblem::operatorsForTestID(int testID){
  vector<EOperatorExtended> ops;    
  if (testID==BurgersBilinearForm::V){
    ops.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);
  } else if (testID==BurgersBilinearForm::TAU){
    ops.push_back(IntrepidExtendedTypes::OPERATOR_DIV);
  }
  return ops;    
}

bool BurgersProblem::nonZeroRHS(int testVarID) {
  return ((testVarID == BurgersBilinearForm::V)||(testVarID == BurgersBilinearForm::TAU));
  //    return false;
}

void BurgersProblem::rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, FieldContainer<double> &values)
{
  FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
  
  int numCells = points.dimension(0);
  int numPoints = points.dimension(1);
  int spaceDim = points.dimension(2);
  
  /*
   if (testVarID==BurgersBilinearForm::V){
   values.resize(numCells,numPoints,spaceDim);
   values.initialize(0.0);
   } else if (testVarID==BurgersBilinearForm::TAU){
   values.resize(numCells,numPoints);
   values.initialize(0.0);
   }
   return;
   */
  
  FieldContainer<double> solnValues(numCells,numPoints);
  _bf->getBackgroundFlow()->solutionValues(solnValues,BurgersBilinearForm::U,basisCache);
  
  if (testVarID==BurgersBilinearForm::V){
    values.resize(numCells,numPoints,spaceDim);
    values.initialize(0.0);    
    FieldContainer<double> beta = _bf->getBeta(points);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++){
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++){
        double x = points(cellIndex,ptIndex,0);
        double y = points(cellIndex,ptIndex,1);
        double u = solnValues(cellIndex,ptIndex);
        
        //sign is positive - opposite from applyBilinearFormData
        values(cellIndex,ptIndex,0) = beta(cellIndex,ptIndex,0)/2.0; // making it rank 2 will automatically dot it with the gradient
        values(cellIndex,ptIndex,1) = beta(cellIndex,ptIndex,1);
        
        values(cellIndex,ptIndex,0) *= u;
        values(cellIndex,ptIndex,1) *= u;
      }
    }
    
  } else if (testVarID==BurgersBilinearForm::TAU){ // should be against divergence of tau
    values.resize(numCells,numPoints);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++){
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++){
        double u = solnValues(cellIndex,ptIndex);
        values(cellIndex,ptIndex) = -u; 
      }
    }      
    
  } 
}


// BC
bool BurgersProblem::bcsImposed(int varID) {
  return varID == BurgersBilinearForm::U_HAT;
}

void BurgersProblem::imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                              FieldContainer<double> &unitNormals,
                              FieldContainer<double> &dirichletValues,
                              FieldContainer<bool> &imposeHere) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  double tol = 1e-14;
  TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument, "spaceDim != 2" );
  
  FieldContainer<double> beta = _bf->getBeta(physicalPoints);
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
      double x = physicalPoints(cellIndex, ptIndex, 0);
      double y = physicalPoints(cellIndex, ptIndex, 1);
      //	double beta_n = _bf->getBeta(x,y)[0]*unitNormals(cellIndex,ptIndex,0)+_bf->getBeta(x,y)[1]*unitNormals(cellIndex,ptIndex,1);
      double beta_n = 0;
      for (int i = 0;i<spaceDim;i++){
        beta_n += beta(cellIndex,ptIndex,i)*unitNormals(cellIndex,ptIndex,i);
      }
      
      double u0;	
      if (abs(y-1.0)>tol){ // if we're not at the top outflow boundary
        u0 = 1.0-2.0*x;
      }	
      if (abs(y-1.0)>tol){ // if not at top boundary
        dirichletValues(cellIndex,ptIndex) = u0; 
        //	  dirichletValues(cellIndex,ptIndex) = 0.0;	  
        imposeHere(cellIndex,ptIndex) = true; // test by imposing all zeros
      }
    }
  }
}

void BurgersProblem::getConstraints(FieldContainer<double> &physicalPoints, 
                                    FieldContainer<double> &unitNormals,
                                    vector<map<int,FieldContainer<double > > > &constraintCoeffs,
                                    vector<FieldContainer<double > > &constraintValues) {
  
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
  
  FieldContainer<double> beta = _bf->getBeta(physicalPoints);
  for (int cellIndex=0;cellIndex<numCells;cellIndex++){
    for (int pointIndex=0;pointIndex<numPoints;pointIndex++){
      double x = physicalPoints(cellIndex,pointIndex,0);
      double y = physicalPoints(cellIndex,pointIndex,1);
      //	vector<double> beta = _bf->getBeta(x,y);
      //	double beta_n = beta[0]*unitNormals(cellIndex,pointIndex,0)+beta[1]*unitNormals(cellIndex,pointIndex,1); // MODIFY THIS TO ACCT FOR DIFF IN NONLINEAR FLUXES?
      double beta_n = 0;
      for (int i = 0;i<spaceDim;i++){
        beta_n += beta(cellIndex,pointIndex,i)*unitNormals(cellIndex,pointIndex,i);
      }
      
      
      bool isOnABoundary=false;
      if ((abs(x) < 1e-12) || (abs(y) < 1e-12)){
        isOnABoundary = true;
      }
      if ((abs(x-1.0) < 1e-12) || (abs(y-1.0) < 1e-12)){
        isOnABoundary = true;
      }
      if (isOnABoundary && (abs(y-1.0)<tol)) { // top boundary
        // this combo isolates sigma_n
        uCoeffs(cellIndex,pointIndex) = beta_n;
        beta_sigmaCoeffs(cellIndex,pointIndex) = -1.0;	    
      }	
    }
  }
  outflowConstraint[BurgersBilinearForm::U_HAT] = uCoeffs;
  outflowConstraint[BurgersBilinearForm::BETA_N_U_MINUS_SIGMA_HAT] = beta_sigmaCoeffs;
  constraintCoeffs.push_back(outflowConstraint); // only one constraint on outflow
  constraintValues.push_back(outflowValues); // only one constraint on outflow    
}
