//
//  NonlinearSolveStrategy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "NonlinearSolveStrategy.h"

NonlinearSolveStrategy::NonlinearSolveStrategy(Teuchos::RCP<Solution> backgroundFlow, Teuchos::RCP<Solution> solution, Teuchos::RCP<NonlinearStepSize> stepSize, double relativeEnergyTolerance) {
  _backgroundFlow = backgroundFlow;
  _solution = solution;
  _stepSize = stepSize;
  _relativeEnergyTolerance = relativeEnergyTolerance;
}

void NonlinearSolveStrategy::solve(bool printToConsole) {
  Teuchos::RCP< Mesh > mesh = _solution->mesh();
  // initialize energyError stuff
  const map<int, double>* energyError;
  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
  vector< Teuchos::RCP< Element > >::iterator activeElemIt;
  
  int i = 0;    
  double prevError = 0.0;
  bool converged = false;
  while (!converged) { // while energy error has not stabilized
    
    _solution->solve(false);
    
    // see if energy error has stabilized
    energyError = &(_solution->energyError());
    double totalError = 0.0;
    
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      double cellEnergyError = energyError->find(current_element->cellID())->second;
      totalError += cellEnergyError*cellEnergyError;
    }
    double relErrorDiff = abs(totalError-prevError)/max(totalError,prevError);
    if (printToConsole){
      cout << "on iter = " << i  << ", relative change in energy error is " << relErrorDiff;
      if (abs(relErrorDiff - 1.0) < 0.1) { // for large rel. error, print more detail...
        cout << "\t(totalError: " << totalError << "; prevError: " << prevError << ")";
      }
      cout << endl;
    }
    
    if (relErrorDiff < _relativeEnergyTolerance) {
      converged = true;
    } else {
      prevError = totalError; // reset previous error and continue
    } 
    
    double stepLength = _stepSize->stepSize(_solution,_backgroundFlow);
    _backgroundFlow->addSolution(_solution,stepLength);
    
    i++;            
  }

}