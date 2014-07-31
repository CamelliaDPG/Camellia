//
//  VarFactory.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_VarFactory_h
#define Camellia_VarFactory_h

#include "Var.h"
#include "BilinearForm.h"
#include "LinearTerm.h"

class VarFactory {
  map< string, VarPtr > _testVars;
  map< string, VarPtr > _trialVars;
  map< int, VarPtr > _testVarsByID;
  map< int, VarPtr > _trialVarsByID;
  int _nextTrialID;
  int _nextTestID;
  
  int getTestID(int IDarg) {
    if (IDarg == -1) {
      IDarg = _nextTestID++;
    } else {
      _nextTestID = max(IDarg + 1, _nextTestID); // ensures that automatically assigned IDs don't conflict with manually assigned ones…
    }
    return IDarg;
  }
  
  int getTrialID(int IDarg) {
    if (IDarg == -1) {
      IDarg = _nextTrialID++;
    } else {
      _nextTrialID = max(IDarg + 1, _nextTrialID); // ensures that automatically assigned IDs don't conflict with manually assigned ones…
    }
    return IDarg;
  }
protected:
  VarFactory(const map< string, VarPtr > &trialVars, const map< string, VarPtr > &testVars,
             const map< int, VarPtr > &trialVarsByID, const map< int, VarPtr > &testVarsByID,
             int nextTrialID, int nextTestID) {
    _trialVars = trialVars;
    _testVars = testVars;
    _trialVarsByID = trialVarsByID;
    _testVarsByID = testVarsByID;
    _nextTrialID = nextTrialID;
    _nextTestID = nextTestID;
  }
  void addTestVar(VarPtr var) {
    _testVars[var->name()] = var;
    _testVarsByID[var->ID()] = _testVars[var->name()];
    _nextTestID = max(var->ID(), _nextTestID);
  }
  void addTrialVar(VarPtr var) {
    _trialVars[var->name()] = var;
    _trialVarsByID[var->ID()] = _trialVars[var->name()];
    _nextTrialID = max(var->ID(), _nextTrialID);
  }
public:
  enum BubnovChoice { BUBNOV_TRIAL, BUBNOV_TEST };
  
  VarFactory() {
    _nextTestID = 0;
    _nextTrialID = 0;
  }
  VarFactory getBubnovFactory(BubnovChoice choice) {
    VarFactory factory;
    if (choice == BUBNOV_TRIAL) {
      factory = VarFactory(_trialVars, _trialVars, _trialVarsByID, _trialVarsByID, _nextTrialID, _nextTrialID);
    } else {
      factory = VarFactory(_testVars, _testVars, _testVarsByID, _testVarsByID, _nextTestID, _nextTestID);
    }
    return factory;
  }
  // accessors:
  VarPtr test(int testID) {
    map< int, VarPtr >::iterator testIt = _testVarsByID.find(testID);
    if (testIt == _testVarsByID.end()) {
      // return null, then
      return Teuchos::rcp((Var*)NULL);
    }
    return testIt->second;
  }
  VarPtr trial(int trialID) {
    map< int, VarPtr >::iterator trialIt = _trialVarsByID.find(trialID);
    if (trialIt == _trialVarsByID.end()) {
      // return null, then
      return Teuchos::rcp((Var*)NULL);
    }
    return trialIt->second;
  }
  
  vector<int> testIDs() {
    vector<int> testIDs;
    for ( map< int, VarPtr >::iterator testIt = _testVarsByID.begin();
         testIt != _testVarsByID.end(); testIt++) {
      testIDs.push_back( testIt->first );
    }
    return testIDs;
  }
  
  vector<int> trialIDs() {
    vector<int> trialIDs;
    for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
         trialIt != _trialVarsByID.end(); trialIt++) {
      trialIDs.push_back( trialIt->first );
    }
    return trialIDs;
  }
  
  // when there are other scratchpads (e.g. BilinearFormScratchPad), we'll want to share
  // the variables.  The basic function of the factory is to assign unique test/trial IDs.
  
  VarPtr testVar(string name, Space fs, int ID = -1) {
    if ( _testVars.find(name) != _testVars.end() ) {
      return _testVars[name];
    }
    
    ID = getTestID(ID);
    int rank = VarFunctionSpaces::rankForSpace(fs);
    
    _testVars[name] = Teuchos::rcp( new Var( ID, rank, name, 
                                             IntrepidExtendedTypes::OP_VALUE, fs, TEST) );
    _testVarsByID[ID] = _testVars[name];
    return _testVarsByID[ID];
  }
  VarPtr fieldVar(string name, Space fs = L2, int ID = -1) {
    if (_trialVars.find(name) != _trialVars.end()) {
      return _trialVars[name];
    }
    ID = getTrialID(ID);
    int rank = VarFunctionSpaces::rankForSpace(fs);
    _trialVars[name] = Teuchos::rcp( new Var( ID, rank, name,
                                              IntrepidExtendedTypes::OP_VALUE, fs, FIELD) );
    _trialVarsByID[ID] = _trialVars[name];
    return _trialVarsByID[ID];
  }
  
  VarPtr fluxVar(string name, LinearTermPtr termTraced, Space fs = L2, int ID = -1) { // trace of HDIV  (implemented as L2 on boundary)
    if (_trialVars.find(name) != _trialVars.end()) {
      return _trialVars[name];
    }
    int rank = VarFunctionSpaces::rankForSpace(fs);
    ID = getTrialID(ID);
    _trialVars[name] = Teuchos::rcp( new Var( ID, rank, name,
                                             IntrepidExtendedTypes::OP_VALUE, fs, FLUX, termTraced) );
    _trialVarsByID[ID] = _trialVars[name];
    return _trialVarsByID[ID];
  }

  VarPtr fluxVar(string name, VarPtr termTraced, Space fs = L2, int ID = -1) { // trace of HDIV  (implemented as L2 on boundary)
    return fluxVar(name, 1.0 * termTraced, fs, ID);
  }
  
  VarPtr fluxVar(string name, Space fs = L2, int ID = -1) { // trace of HDIV  (implemented as L2 on boundary)
    return fluxVar(name, Teuchos::rcp((LinearTerm*)NULL), fs, ID);
  }
  
  VarPtr traceVar(string name, LinearTermPtr termTraced, Space fs = HGRAD, int ID = -1) { // trace of HGRAD (implemented as HGRAD on boundary)
    if (_trialVars.find(name) != _trialVars.end() ) {
      return _trialVars[name];
    }
    int rank = ((fs == HGRAD) || (fs == L2) || (fs == CONSTANT_SCALAR)) ? 0 : 1;
    ID = getTrialID(ID);
    _trialVars[name] = Teuchos::rcp( new Var( ID, rank, name, 
                                              IntrepidExtendedTypes::OP_VALUE, fs, TRACE, termTraced) );
    _trialVarsByID[ID] = _trialVars[name];
    return _trialVarsByID[ID];
  }

  VarPtr traceVar(string name, VarPtr termTraced, Space fs = HGRAD, int ID = -1) {
    return traceVar(name, 1.0 * termTraced, fs, ID);
  }
  
  VarPtr traceVar(string name, Space fs = HGRAD, int ID = -1) {
    return traceVar(name, Teuchos::rcp((LinearTerm*)NULL), fs, ID);
  }
  
  const map< int, VarPtr > & testVars() {
    return _testVarsByID;
  }
  
  const map< int, VarPtr > & trialVars() {
    return _trialVarsByID;
  }
  
  vector< VarPtr > fieldVars() {
    vector< VarPtr > vars;
    
    for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
         trialIt != _trialVarsByID.end(); trialIt++) {
      if (trialIt->second->varType() == FIELD) {
        vars.push_back(trialIt->second);
      }
    }
    return vars;
  }
  
  vector< VarPtr > fluxVars() {
    vector< VarPtr > vars;
    
    for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
         trialIt != _trialVarsByID.end(); trialIt++) {
      if (trialIt->second->varType() == FLUX) {
        vars.push_back(trialIt->second);
      }
    }
    return vars;
  }
  
  vector< VarPtr > traceVars() {
    vector< VarPtr > vars;
    
    for ( map< int, VarPtr >::iterator trialIt = _trialVarsByID.begin();
         trialIt != _trialVarsByID.end(); trialIt++) {
      if (trialIt->second->varType() == TRACE) {
        vars.push_back(trialIt->second);
      }
    }
    return vars;
  }
  
  VarFactory trialSubFactory(vector< VarPtr > &trialVars) {
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
};

#endif
