//
//  TimeMarchingProblem.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/27/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "TimeMarchingProblem.h"
#include "RHS.h"
#include "Solution.h"

#include "SerialDenseWrapper.h"

using namespace Intrepid;
using namespace Camellia;

TimeMarchingProblem::TimeMarchingProblem(BFPtr bilinearForm,
                                         Teuchos::RCP<RHS> rhs) : RHS(true), BF(true) { // true: legacy subclass
  _bilinearForm = bilinearForm;
  _rhs = rhs;
  _dt = 1.0;
  _testIDs = bilinearForm->testIDs();
  _trialIDs = bilinearForm->trialIDs();
}

void TimeMarchingProblem::trialTestOperators(int trialID, int testID,
                                                  vector<Camellia::EOperator> &trialOps,
                                                  vector<Camellia::EOperator> &testOps) {
  // each (trial,test) pair gets one extra operator, a VALUE on each, belonging to the time marching
  _bilinearForm->trialTestOperators(trialID,testID,trialOps,testOps);

  if (hasTimeDerivative(trialID,testID)) {
    trialOps.insert(trialOps.begin(),  Camellia::OP_VALUE);
    testOps.insert(testOps.begin(),  Camellia::OP_VALUE);
  }
}


void TimeMarchingProblem::applyBilinearFormData(FieldContainer<double> &trialValues,
                                                     FieldContainer<double> &testValues,
                                                     int trialID, int testID, int operatorIndex,
                                                     Teuchos::RCP<BasisCache> basisCache) {
  if (hasTimeDerivative(trialID,testID)) {
    operatorIndex--;
  }
  if (operatorIndex >= 0) { // then this belongs to the non-time-marching BilinearForm
    _bilinearForm->applyBilinearFormData(trialValues,testValues,trialID,testID,operatorIndex-1,basisCache);
  } else {
    this->timeLHS(trialValues,trialID,basisCache);
    SerialDenseWrapper::multiplyFCByWeight(trialValues, 1.0 / _dt);
  }
}

Camellia::EFunctionSpace TimeMarchingProblem::functionSpaceForTest(int testID) {
  return _bilinearForm->functionSpaceForTest(testID);
}

Camellia::EFunctionSpace TimeMarchingProblem::functionSpaceForTrial(int trialID) {
  return _bilinearForm->functionSpaceForTrial(trialID);
}

bool TimeMarchingProblem::isFluxOrTrace(int trialID) {
  return _bilinearForm->isFluxOrTrace(trialID);
}

bool TimeMarchingProblem::nonZeroRHS(int testVarID) {
  return _rhs->nonZeroRHS(testVarID);
}

vector<Camellia::EOperator> TimeMarchingProblem::operatorsForTestID(int testID) {
  // check whether there's time derivative-interaction with any trial function
  vector< Camellia::EOperator > ops = _rhs->operatorsForTestID(testID);
  if ( testHasTimeDerivative(testID) ) {
    ops.insert(ops.begin(), Camellia::OP_VALUE);
  }
  return ops;
}

bool TimeMarchingProblem::testHasTimeDerivative(int testID) {
  bool testHasTimeDerivative = false;
  vector<int> trialIDs = _bilinearForm->trialIDs();
  for (vector<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
    int trialID = *trialIDIt;
    if (hasTimeDerivative(trialID,testID)) {
      testHasTimeDerivative = true;
      break;
    }
  }
  return testHasTimeDerivative;
}

void TimeMarchingProblem::rhs(int testID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache,
                              FieldContainer<double> &values) {
  if (testHasTimeDerivative(testID)) {
    operatorIndex--;
  }
  if (operatorIndex >= 0) {
    _rhs->rhs(testID,operatorIndex,basisCache,values);
  } else {
    this->timeRHS(values,testID,basisCache);
    SerialDenseWrapper::multiplyFCByWeight(values, 1.0 / _dt);
  }
}

void TimeMarchingProblem::setTimeStepSize(double dt) {
  _dt = dt;
}

void TimeMarchingProblem::timeLHS(FieldContainer<double> trialValues, int trialID, Teuchos::RCP<BasisCache> basisCache) {
  // default implementation leaves trialValues as they are.
}

void TimeMarchingProblem::timeRHS(FieldContainer<double> testValues, int testID, Teuchos::RCP<BasisCache> basisCache) {
  // default implementation leaves testValues as they are.
}

double TimeMarchingProblem::timeStepSize() {
  return _dt;
}

const string & TimeMarchingProblem::testName(int testID) {
  return _bilinearForm->testName(testID);
}

const string & TimeMarchingProblem::trialName(int trialID) {
  return _bilinearForm->trialName(trialID);
}

SolutionPtr TimeMarchingProblem::previousTimeSolution() {
  return _previousTimeSolution;
}

void TimeMarchingProblem::setPreviousTimeSolution(SolutionPtr previousSolution) {
  _previousTimeSolution = previousSolution;
}
