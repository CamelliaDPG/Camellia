//
//  ConfusionProblem.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/16/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "ConfusionProblem.h"

ConfusionProblem::ConfusionProblem(Teuchos::RCP<ConfusionBilinearForm> cbf) : RHS(), BC() {
  _cbf = cbf;  
}

// RHS:
bool ConfusionProblem::nonZeroRHS(int testVarID) {
  return testVarID == ConfusionBilinearForm::V;
}

void ConfusionProblem::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  values.resize(numCells,numPoints);
  values.initialize(1.0);
  //values.initialize(0.0);
}

// BC
bool ConfusionProblem::bcsImposed(int varID) {
  return (varID == ConfusionBilinearForm::U_HAT || varID==ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT);
//  return varID == ConfusionBilinearForm::U_HAT;
}

void ConfusionProblem::imposeBC(int varID, FieldContainer<double> &physicalPoints, 
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
      double beta_n = unitNormals(cellIndex, ptIndex, 0)*_cbf->getBeta(x,y)[0]
                    + unitNormals(cellIndex, ptIndex, 1)*_cbf->getBeta(x,y)[1];
      
      double u0 = 0.0;
      if ( (abs(x) < 1e-14) && (y<y_cut) ) { // x basically 0 ==> u = 1 - y	  
        u0 = 1.0 - y/y_cut;
      } else if ( (abs(y) < 1e-14) &&  (x<x_cut) ) { // y basically 0 ==> u = 1 - x
        u0 = 1.0 - x/x_cut;
      } 
      
      imposeHere(cellIndex,ptIndex) = false;
      if (bcsImposed(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT) && varID==ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT){
        dirichletValues(cellIndex,ptIndex) = beta_n*u0;
        if (abs(y) < 1e-14 || abs(x) < 1e-14) { // only impose this var on the inflow
          imposeHere(cellIndex,ptIndex) = true;
        } 
      } else {
        dirichletValues(cellIndex,ptIndex) = u0;
        imposeHere(cellIndex,ptIndex) = true;
      }       
      
      // if outflow, always apply wall BC
      if (abs(y-1.0)<1e-14 || abs(x-1.0)<1e-14){
        if (varID==ConfusionBilinearForm::U_HAT){
          dirichletValues(cellIndex,ptIndex) = 0.0;
          imposeHere(cellIndex,ptIndex) = true;
        } 
      }
      
    }
  }
}
