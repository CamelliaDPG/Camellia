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
#include "LinearTerm.h"

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
  ostringstream bfStream;
  bool first = true;
  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
       btIt != _terms.end(); btIt++) {
    if (! first ) {
      bfStream << " + ";
    }
    BilinearTerm bt = *btIt;
    LinearTermPtr trialTerm = btIt->first;
    LinearTermPtr testTerm = btIt->second;
    bfStream << "( " << trialTerm->displayString() << ", " << testTerm->displayString() << ")";
    first = false;
  }
  string bfString = bfStream.str();
  cout << bfString << endl;
}

void BF::stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                         FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {
  stiffnessMatrix(stiffness, elemType, cellSideParities, basisCache, true); // default to checking
}

// can override check for zero cols (i.e. in hessian matrix)
void BF::stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                         FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache,
			 bool checkForZeroCols) {
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
  if (checkForZeroCols){
    bool checkRows = false; // zero rows just mean a test basis function won't get used, which is fine
    bool checkCols = true; // zero columns mean that a trial basis function doesn't enter the computation, which is bad
    if (! BilinearFormUtility::checkForZeroRowsAndColumns("BF stiffness", stiffness, checkRows, checkCols) ) {
      cout << "trial ordering:\n" << *(elemType->trialOrderPtr);
      //    cout << "test ordering:\n" << *(elemType->testOrderPtr);
      //    cout << "stiffness:\n" << stiffness;
    }
  }
}

// No cellSideParities required, no checking of columns, integrates in a bubnov fashion
void BF::bubnovStiffness(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
			 FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {
  // stiffness is sized as (C, FTrial, FTrial)
  stiffness.initialize(0.0);
  basisCache->setCellSideParities(cellSideParities);

  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
       btIt != _terms.end(); btIt++) {
    BilinearTerm bt = *btIt;
    LinearTermPtr trialTerm = btIt->first;
    LinearTermPtr testTerm = btIt->second;
    trialTerm->integrate(stiffness, elemType->trialOrderPtr,
                         testTerm,  elemType->trialOrderPtr, basisCache);
  }
 
}

IPPtr BF::graphNorm() {
  typedef pair< FunctionPtr, VarPtr > LinearSummand;
  map<int, LinearTermPtr> testTermsForVarID;
  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
       btIt != _terms.end(); btIt++) {
    BilinearTerm bt = *btIt;
    LinearTermPtr trialTerm = btIt->first;
    LinearTermPtr testTerm = btIt->second;
    vector< LinearSummand > summands = trialTerm->summands();
    for ( vector< LinearSummand >::iterator lsIt = summands.begin(); lsIt != summands.end(); lsIt++) {
      VarPtr trialVar = lsIt->second;
      if (trialVar->varType() == FIELD) {
        if (trialVar->op() != OP_VALUE) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BF::graphNorm() doesn't support non-value ops on field variables");
        }
        FunctionPtr f = lsIt->first;
        if (testTermsForVarID.find(trialVar->ID()) == testTermsForVarID.end()) {
          testTermsForVarID[trialVar->ID()] = Teuchos::rcp( new LinearTerm );
        }
        testTermsForVarID[trialVar->ID()]->addTerm( f * testTerm );
      }
    }
  }
  IPPtr ip = Teuchos::rcp( new IP );
  for ( map<int, LinearTermPtr>::iterator testTermIt = testTermsForVarID.begin();
       testTermIt != testTermsForVarID.end(); testTermIt++ ) {
    ip->addTerm( testTermIt->second );
  }
  // L^2 terms:
  map< int, VarPtr > testVars = _varFactory.testVars();
  for ( map< int, VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++) {
    ip->addTerm( testVarIt->second );
  }
  
  return ip;
}

IPPtr BF::l2Norm() {
  // L2 norm on test space:
  IPPtr ip = Teuchos::rcp( new IP );
  map< int, VarPtr > testVars = _varFactory.testVars();
  for ( map< int, VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++) {
    ip->addTerm( testVarIt->second );
  }
  return ip;
}

IPPtr BF::naiveNorm() {
  IPPtr ip = Teuchos::rcp( new IP );
  map< int, VarPtr > testVars = _varFactory.testVars();
  for ( map< int, VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++) {
    VarPtr var = testVarIt->second;
    ip->addTerm( var );
    // HGRAD, HCURL, HDIV, L2, CONSTANT_SCALAR, VECTOR_HGRAD, VECTOR_L2
    if ( (var->space() == HGRAD) || (var->space() == VECTOR_HGRAD) ) {
      ip->addTerm( var->grad() );
    } else if ( (var->space() == L2) || (var->space() == VECTOR_L2) ) {
      // do nothing (we already added the L2 term
    } else if (var->space() == HCURL) {
      ip->addTerm( var->curl() );
    } else if (var->space() == HDIV) {
      ip->addTerm( var->div() );
    }
  }
  return ip;  
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
