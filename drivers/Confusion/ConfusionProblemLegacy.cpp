//
//  ConfusionProblemLegacy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/16/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "ConfusionProblemLegacy.h"

#include "ConfusionBilinearForm.h"

ConfusionProblemLegacy::ConfusionProblemLegacy(BFPtr cbf, double beta_x, double beta_y) : RHS(true), BC(true) { // true: legacy subclass of RHS, legacy subclass of BC
  _cbf = cbf;
  VarFactory vf = cbf->varFactory();
  
  _beta_x = beta_x;
  _beta_y = beta_y;
  
  _u_hat = vf.traceVar(ConfusionBilinearForm::S_U_HAT);
  _beta_n_u_minus_sigma_hat = vf.fluxVar(ConfusionBilinearForm::S_BETA_N_U_MINUS_SIGMA_HAT);
  
  _v = vf.testVar(ConfusionBilinearForm::S_V, HGRAD);
}

// RHS:
bool ConfusionProblemLegacy::nonZeroRHS(int testVarID) {
  return testVarID == _v->ID();
}

void ConfusionProblemLegacy::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  values.resize(numCells,numPoints);
  values.initialize(1.0);
  //values.initialize(0.0);
}

// BC
bool ConfusionProblemLegacy::bcsImposed(int varID) {
  return (varID == _u_hat->ID() || varID==_beta_n_u_minus_sigma_hat->ID());
//  return varID == ConfusionBilinearForm::U_HAT;
}

void ConfusionProblemLegacy::imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                                FieldContainer<double> &unitNormals,
                                FieldContainer<double> &dirichletValues,
                                FieldContainer<bool> &imposeHere) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  double tol = 1e-14;
  double x_cut = .50;
  double y_cut = .50;
  TEUCHOS_TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument, "spaceDim != 2" );
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
      double x = physicalPoints(cellIndex, ptIndex, 0);
      double y = physicalPoints(cellIndex, ptIndex, 1);
      double beta_n = unitNormals(cellIndex, ptIndex, 0)* _beta_x
                    + unitNormals(cellIndex, ptIndex, 1)* _beta_y;
      
      double u0 = 0.0;
      if ( (abs(x) < 1e-14) && (y<y_cut) ) { // x basically 0 ==> u = 1 - y	  
        u0 = 1.0 - y/y_cut;
      } else if ( (abs(y) < 1e-14) &&  (x<x_cut) ) { // y basically 0 ==> u = 1 - x
        u0 = 1.0 - x/x_cut;
      } 
      
      imposeHere(cellIndex,ptIndex) = false;
      if (bcsImposed(_beta_n_u_minus_sigma_hat->ID()) && varID==_beta_n_u_minus_sigma_hat->ID()){
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
        if (varID==_u_hat->ID()){
          dirichletValues(cellIndex,ptIndex) = 0.0;
          imposeHere(cellIndex,ptIndex) = true;
        } 
      }
      
    }
  }
}
