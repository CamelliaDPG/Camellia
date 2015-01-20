//
//  VarFactory.cpp
//  Camellia
//
//  Created by Nathan Roberts on 1/7/15.
//  Copyright (c) 2015 __MyCompanyName__. All rights reserved.
//

#include "VarFactory.h"

using namespace std;

VarFactory::VarFactory() {
  _nextTestID = 0;
  _nextTrialID = 0;
}

// protected constructor:
VarFactory::VarFactory(const map< string, VarPtr > &trialVars, const map< string, VarPtr > &testVars,
           const map< int, VarPtr > &trialVarsByID, const map< int, VarPtr > &testVarsByID,
           int nextTrialID, int nextTestID) {
  _trialVars = trialVars;
  _testVars = testVars;
  _trialVarsByID = trialVarsByID;
  _testVarsByID = testVarsByID;
  _nextTrialID = nextTrialID;
  _nextTestID = nextTestID;
}

void VarFactory::addTestVar(VarPtr var) {
  _testVars[var->name()] = var;
  _testVarsByID[var->ID()] = _testVars[var->name()];
  _nextTestID = max(var->ID(), _nextTestID);
}

void VarFactory::addTrialVar(VarPtr var) {
  _trialVars[var->name()] = var;
  _trialVarsByID[var->ID()] = _trialVars[var->name()];
  _nextTrialID = max(var->ID(), _nextTrialID);
}

VarFactory VarFactory::getBubnovFactory(BubnovChoice choice) {
  VarFactory factory;
  if (choice == BUBNOV_TRIAL) {
    factory = VarFactory(_trialVars, _trialVars, _trialVarsByID, _trialVarsByID, _nextTrialID, _nextTrialID);
  } else {
    factory = VarFactory(_testVars, _testVars, _testVarsByID, _testVarsByID, _nextTestID, _nextTestID);
  }
  return factory;
}

int VarFactory::getTestID(int IDarg) {
  if (IDarg == -1) {
    IDarg = _nextTestID++;
  } else {
    _nextTestID = max(IDarg + 1, _nextTestID); // ensures that automatically assigned IDs don't conflict with manually assigned ones…
  }
  return IDarg;
}

int VarFactory::getTrialID(int IDarg) {
  if (IDarg == -1) {
    IDarg = _nextTrialID++;
  } else {
    _nextTrialID = max(IDarg + 1, _nextTrialID); // ensures that automatically assigned IDs don't conflict with manually assigned ones…
  }
  return IDarg;
}

// accessors:
VarPtr VarFactory::test(int testID) {
  map< int, VarPtr >::iterator testIt = _testVarsByID.find(testID);
  if (testIt == _testVarsByID.end()) {
    // return null, then
    return Teuchos::rcp((Var*)NULL);
  }
  return testIt->second;
}
VarPtr VarFactory::trial(int trialID) {
  map< int, VarPtr >::iterator trialIt = _trialVarsByID.find(trialID);
  if (trialIt == _trialVarsByID.end()) {
    // return null, then
    return Teuchos::rcp((Var*)NULL);
  }
  return trialIt->second;
}

vector<int> VarFactory::testIDs() {
  vector<int> testIDs;
  for ( map< int, VarPtr >::iterator testIt = _testVarsByID.begin();
       testIt != _testVarsByID.end(); testIt++) {
    testIDs.push_back( testIt->first );
  }
  return testIDs;
}

vector<int> VarFactory::trialIDs() {
  vector<int> trialIDs;
  for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
       trialIt != _trialVarsByID.end(); trialIt++) {
    trialIDs.push_back( trialIt->first );
  }
  return trialIDs;
}

// when there are other scratchpads (e.g. BilinearFormScratchPad), we'll want to share
// the variables.  The basic function of the factory is to assign unique test/trial IDs.

VarPtr VarFactory::testVar(string name, Space fs, int ID) {
  if ( _testVars.find(name) != _testVars.end() ) {
    return _testVars[name];
  }
  
  ID = getTestID(ID);
  int rank = VarFunctionSpaces::rankForSpace(fs);
  
  _testVars[name] = Teuchos::rcp( new Var( ID, rank, name,
                                          Camellia::OP_VALUE, fs, TEST) );
  _testVarsByID[ID] = _testVars[name];
  return _testVarsByID[ID];
}
VarPtr VarFactory::fieldVar(string name, Space fs, int ID) {
  if (_trialVars.find(name) != _trialVars.end()) {
    return _trialVars[name];
  }
  ID = getTrialID(ID);
  int rank = VarFunctionSpaces::rankForSpace(fs);
  _trialVars[name] = Teuchos::rcp( new Var( ID, rank, name,
                                           Camellia::OP_VALUE, fs, FIELD) );
  _trialVarsByID[ID] = _trialVars[name];
  return _trialVarsByID[ID];
}

VarPtr VarFactory::fluxVar(string name, LinearTermPtr termTraced, Space fs, int ID) {
  if (_trialVars.find(name) != _trialVars.end()) {
    return _trialVars[name];
  }
  int rank = VarFunctionSpaces::rankForSpace(fs);
  ID = getTrialID(ID);
  _trialVars[name] = Teuchos::rcp( new Var( ID, rank, name,
                                           Camellia::OP_VALUE, fs, FLUX, termTraced) );
  _trialVarsByID[ID] = _trialVars[name];
  return _trialVarsByID[ID];
}

VarPtr VarFactory::fluxVar(string name, VarPtr termTraced, Space fs, int ID) {
  return fluxVar(name, 1.0 * termTraced, fs, ID);
}

VarPtr VarFactory::fluxVar(string name, Space fs, int ID) {
  return fluxVar(name, Teuchos::rcp((LinearTerm*)NULL), fs, ID);
}

VarPtr VarFactory::traceVar(string name, LinearTermPtr termTraced, Space fs, int ID) {
  if (_trialVars.find(name) != _trialVars.end() ) {
    return _trialVars[name];
  }
  int rank = ((fs == HGRAD) || (fs == L2) || (fs == CONSTANT_SCALAR)) ? 0 : 1;
  ID = getTrialID(ID);
  _trialVars[name] = Teuchos::rcp( new Var( ID, rank, name,
                                           Camellia::OP_VALUE, fs, TRACE, termTraced) );
  _trialVarsByID[ID] = _trialVars[name];
  return _trialVarsByID[ID];
}

VarPtr VarFactory::traceVar(string name, VarPtr termTraced, Space fs, int ID) {
  return traceVar(name, 1.0 * termTraced, fs, ID);
}

VarPtr VarFactory::traceVar(string name, Space fs, int ID) {
  return traceVar(name, Teuchos::rcp((LinearTerm*)NULL), fs, ID);
}

const map< int, VarPtr > & VarFactory::testVars() {
  return _testVarsByID;
}

const map< int, VarPtr > & VarFactory::trialVars() {
  return _trialVarsByID;
}

vector< VarPtr > VarFactory::fieldVars() {
  vector< VarPtr > vars;
  
  for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
       trialIt != _trialVarsByID.end(); trialIt++) {
    if (trialIt->second->varType() == FIELD) {
      vars.push_back(trialIt->second);
    }
  }
  return vars;
}

vector< VarPtr > VarFactory::fluxVars() {
  vector< VarPtr > vars;
  
  for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
       trialIt != _trialVarsByID.end(); trialIt++) {
    if (trialIt->second->varType() == FLUX) {
      vars.push_back(trialIt->second);
    }
  }
  return vars;
}

vector< VarPtr > VarFactory::traceVars() {
  vector< VarPtr > vars;
  
  for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
       trialIt != _trialVarsByID.end(); trialIt++) {
    if (trialIt->second->varType() == TRACE) {
      vars.push_back(trialIt->second);
    }
  }
  return vars;
}

VarFactory VarFactory::trialSubFactory(vector< VarPtr > &trialVars) {
  // returns a new VarFactory with the same test space, and a subspace of the trial space
  VarFactory subFactory;
  for (vector< VarPtr >::iterator trialVarIt=trialVars.begin(); trialVarIt != trialVars.end(); trialVarIt++) {
    VarPtr trialVar = *trialVarIt;
    subFactory.addTrialVar(trialVar);
  }
  for (map<int, VarPtr>::iterator testVarIt=_testVarsByID.begin(); testVarIt != _testVarsByID.end(); testVarIt++) {
    VarPtr testVar = testVarIt->second;
    subFactory.addTestVar(testVar);
  }
  return subFactory;
}