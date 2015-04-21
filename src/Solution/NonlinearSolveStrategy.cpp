//
//  NonlinearSolveStrategy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "NonlinearSolveStrategy.h"

using namespace Camellia;

NonlinearSolveStrategy::NonlinearSolveStrategy(TSolutionPtr<double> backgroundFlow, TSolutionPtr<double> solution, Teuchos::RCP<NonlinearStepSize> stepSize, double relativeEnergyTolerance) {
  _backgroundFlow = backgroundFlow;
  _solution = solution;
  _stepSize = stepSize;
  _relativeEnergyTolerance = relativeEnergyTolerance;
  _usePicardIteration = false; // Newton-Raphson by default
}

void NonlinearSolveStrategy::setUsePicardIteration(bool value) {
  _usePicardIteration = value;
}

void NonlinearSolveStrategy::solve(bool printToConsole) {
  Teuchos::RCP< Mesh > mesh = _solution->mesh();

  vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
  vector< Teuchos::RCP< Element > >::iterator activeElemIt;

  int i = 0;
  double prevError = 0.0;
  bool converged = false;
  while (!converged) { // while energy error has not stabilized

    _solution->solve(false);

    double totalErrorSquareRoot = _solution->energyErrorTotal();
    double totalError = totalErrorSquareRoot * totalErrorSquareRoot; // NVR 9-17-14: this is the energy error squared.  Is that what we want??

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

    if ( ! _usePicardIteration ) {
      double stepLength = _stepSize->stepSize(_solution,_backgroundFlow);
      _backgroundFlow->addSolution(_solution,stepLength);
    } else {
      _backgroundFlow->setSolution(_solution);
    }

    i++;
  }

}
