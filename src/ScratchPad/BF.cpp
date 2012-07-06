//
//  BF.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "BF.h"
#include "VarFactory.h"
#include "BilinearFormUtility.h"
#include "Function.h"
#include "PreviousSolutionFunction.h"

BF::BF( VarFactory varFactory ) { // copies (note that external changes in VarFactory won't be registered by BF)
  _varFactory = varFactory;
  // set super's ID containers:
  _trialIDs = _varFactory.trialIDs();
  _testIDs = _varFactory.testIDs();
}

BF::BF( VarFactory varFactory, VarFactory::BubnovChoice choice ) {
  _varFactory = varFactory.getBubnovFactory(choice);
  _trialIDs = _varFactory.trialIDs();
  _testIDs = _varFactory.testIDs();
}

void BF::addTerm( LinearTermPtr trialTerm, LinearTermPtr testTerm ) {
  _terms.push_back( make_pair( trialTerm, testTerm ) );
}

void BF::addTerm( VarPtr trialVar, LinearTermPtr testTerm ) {
  addTerm( Teuchos::rcp( new LinearTerm(trialVar) ), testTerm );
}

void BF::addTerm( VarPtr trialVar, VarPtr testVar ) {
  addTerm( Teuchos::rcp( new LinearTerm(trialVar) ), Teuchos::rcp( new LinearTerm(testVar) ) );
}

void BF::addTerm( LinearTermPtr trialTerm, VarPtr testVar) {
  addTerm( trialTerm, Teuchos::rcp( new LinearTerm(testVar) ) );
}

// BilinearForm implementation:
const string & BF::testName(int testID) {
  return _varFactory.test(testID)->name();
}
const string & BF::trialName(int trialID) {
  return _varFactory.trial(trialID)->name();
}

EFunctionSpaceExtended BF::functionSpaceForTest(int testID) {
  return efsForSpace(_varFactory.test(testID)->space());
}

EFunctionSpaceExtended BF::functionSpaceForTrial(int trialID) {
  return efsForSpace(_varFactory.trial(trialID)->space());
}

bool BF::isFluxOrTrace(int trialID) {
  VarType varType = _varFactory.trial(trialID)->varType();
  return (varType == FLUX) || (varType == TRACE);
}

void BF::printTrialTestInteractions() {
  cout << "BF::printTrialTestInteractions() not yet implemented.\n";
}

void BF::stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                         FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {
  // stiffness is sized as (C, FTest, FTrial)
  stiffness.initialize(0.0);
  basisCache->setCellSideParities(cellSideParities);
  
  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
       btIt != _terms.end(); btIt++) {
    BilinearTerm bt = *btIt;
    LinearTermPtr trialTerm = btIt->first;
    LinearTermPtr testTerm = btIt->second;
    trialTerm->integrate(stiffness, elemType->trialOrderPtr,
                         testTerm,  elemType->testOrderPtr, basisCache);
  }
  BilinearFormUtility::checkForZeroRowsAndColumns("BF stiffness", stiffness);
}

LinearTermPtr BF::testFunctional(SolutionPtr trialSolution) {
  LinearTermPtr functional = Teuchos::rcp(new LinearTerm());
  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
       btIt != _terms.end(); btIt++) {
    BilinearTerm bt = *btIt;
    LinearTermPtr trialTerm = btIt->first;
    LinearTermPtr testTerm = btIt->second;
    FunctionPtr trialValue = Teuchos::rcp( new PreviousSolutionFunction(trialSolution, trialTerm) );
    functional = functional + trialValue * testTerm;
  }
  return functional;
}