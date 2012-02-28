//
//  TimeMarchingProblem.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/27/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "TimeMarchingProblem.h"

TimeMarchingProblem::TimeMarchingProblem(Teuchos::RCP<BilinearForm> bilinearForm,
                                                   Teuchos::RCP<RHS> rhs) {
  _bilinearForm = bilinearForm;
  _rhs = rhs;
  _dt = 1.0;
}

void TimeMarchingProblem::trialTestOperators(int trialID, int testID, 
                                                  vector<EOperatorExtended> &trialOps,
                                                  vector<EOperatorExtended> &testOps) {
  // each (trial,test) pair gets one extra operator, a VALUE on each, belonging to the time marching
  _bilinearForm->trialTestOperators(testID1,testID2,trialOps,testOps);
  
  // TODO: add a mechanism to specify *which* trialIDs get time derivatives
  trialOps.insert(trialOps.begin(), IntrepidExtendedTypes::OPERATOR_VALUE);
  testOps.insert(testOps.begin(), IntrepidExtendedTypes::OPERATOR_VALUE);
}


void TimeMarchingProblem::applyBilinearFormData(FieldContainer<double> &trialValues, 
                                                     FieldContainer<double> &testValues, 
                                                     int trialID, int testID, int operatorIndex,
                                                     Teuchos::RCP<BasisCache> basisCache) {
  if (operatorIndex > 0) { // then this belongs to the non-time-marching BilinearForm
    _bilinearForm->applyBilinearFormData(trialValues,testValues,trialID,testID,operatorIndex-1,basisCache);
  } else {
    this->timeLHS(trialValues,trialID);
    BilinearForm::multiplyFCByWeight(trialValues, 1.0 / _dt);
  }
}

EFunctionSpaceExtended TimeMarchingProblem::functionSpaceForTest(int testID) {
  return _bilinearForm->functionSpaceForTest(testID);
}

EFunctionSpaceExtended TimeMarchingProblem::functionSpaceForTrial(int trialID) {
  return _bilinearForm->functionSpaceForTrial(trialID);
}

bool TimeMarchingProblem::isFluxOrTrace(int trialID) {
  return _bilinearForm->isFluxOrTrace(trialID);
}

bool TimeMarchingProblem::nonZeroRHS(int testVarID) {
  return _rhs->nonZeroRHS(testVarID);
}

vector<EOperatorExtended> TimeMarchingProblem::operatorsForTestID(int testID) {
  return _rhs->operatorsForTestID(testID);
}

void TimeMarchingProblem::rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, 
                                   FieldContainer<double> &values) {
  
}


void TimeMarchingProblem::setTimeStepSize(double dt) {
  _dt = dt;
}

void TimeMarchingProblem::timeLHS(FieldContainer<double> trialValues, int trialID) {
  // default implementation leaves trialValues as they are.
}

double TimeMarchingProblem::timeStepSize() {
  return _dt;
}