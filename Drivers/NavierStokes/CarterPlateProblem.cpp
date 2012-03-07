//
//  BurgersProblem.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/20/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "CarterPlateProblem.h"
#include "Solution.h"

CarterPlateProblem::CarterPlateProblem( Teuchos::RCP<NavierStokesBilinearForm> bf) : RHS(), BC(), Constraints() {
  _bf = bf;
  tol = 1e-14;
}

// RHS:

vector<EOperatorExtended> CarterPlateProblem::operatorsForTestID(int testID){
  vector<EOperatorExtended> ops;    
  if (testID==NavierStokesBilinearForm::V){
    ops.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);
  } else if (testID==NavierStokesBilinearForm::TAU){
    ops.push_back(IntrepidExtendedTypes::OPERATOR_DIV);
  }
  return ops;    
}

bool CarterPlateProblem::nonZeroRHS(int testVarID) {
  return false;
}

void CarterPlateProblem::rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, FieldContainer<double> &values)
{
  FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
  
  int numCells = points.dimension(0);
  int numPoints = points.dimension(1);
  int spaceDim = points.dimension(2);
  
  /*  
  FieldContainer<double> solnValues(numCells,numPoints);
  _bf->getBackgroundFlow()->solutionValues(solnValues,NavierStokesBilinearForm::U,basisCache);

  if (testVarID==NavierStokesBilinearForm::V){
    values.resize(numCells,numPoints,spaceDim);
    values.initialize(0.0);    
    FieldContainer<double> beta = _bf->getBeta(basisCache);
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
    
  } else if (testVarID==NavierStokesBilinearForm::TAU){ // should be against divergence of tau
    values.resize(numCells,numPoints);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++){
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++){
        double u = solnValues(cellIndex,ptIndex);
        values(cellIndex,ptIndex) = -u; 
      }
    }      
    
  } 
  */
}


// BC
bool CarterPlateProblem::bcsImposed(int varID) {
  //  return varID == NavierStokesBilinearForm::U_HAT;
  return ((varID == NavierStokesBilinearForm::F1_N)||
	  (varID == NavierStokesBilinearForm::F2_N)||
	  (varID == NavierStokesBilinearForm::F3_N)||
	  (varID == NavierStokesBilinearForm::F4_N)||
	  (varID == NavierStokesBilinearForm::U1)||
	  (varID == NavierStokesBilinearForm::U2)||
	  (varID == NavierStokesBilinearForm::T)); 

}

void CarterPlateProblem::imposeBC(int varID, FieldContainer<double> &physicalPoints, 
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

      
    }
  }

}

void CarterPlateProblem::getConstraints(FieldContainer<double> &physicalPoints, 
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

      bool isOnABoundary=false;
    }
  }
  outflowConstraint[NavierStokesBilinearForm::U1_HAT] = uCoeffs;
  outflowConstraint[NavierStokesBilinearForm::F1_N] = beta_sigmaCoeffs;
  constraintCoeffs.push_back(outflowConstraint); // only one constraint on outflow
  constraintValues.push_back(outflowValues); // only one constraint on outflow    
}
