//
//  RHSEasy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RHSEasy.h"

void RHSEasy::addTerm( LinearTermPtr rhsTerm ) {
  TEUCHOS_TEST_FOR_EXCEPTION( rhsTerm->termType() != TEST, std::invalid_argument, "RHS should only involve test functions (no trials)");
  TEUCHOS_TEST_FOR_EXCEPTION( rhsTerm->rank() != 0, std::invalid_argument, "RHSEasy only handles scalar terms.");
  _terms.push_back( rhsTerm );
  set<int> testIDs = rhsTerm->varIDs();
  _testIDs.insert(testIDs.begin(),testIDs.end());
}

void RHSEasy::addTerm( VarPtr v ) {
  addTerm( Teuchos::rcp( new LinearTerm( v ) ) );
}

// at a conceptual/design level, this method isn't necessary
bool RHSEasy::nonZeroRHS(int testVarID) {
  return (_testIDs.find(testVarID) != _testIDs.end());
}

void RHSEasy::integrateAgainstStandardBasis(FieldContainer<double> &rhsVector, 
                                            Teuchos::RCP<DofOrdering> testOrdering, 
                                            BasisCachePtr basisCache) {
  // rhsVector: (numCells, numTestDofs)
  rhsVector.initialize(0.0);
  
  for (vector< LinearTermPtr >::iterator ltIt = _terms.begin(); ltIt != _terms.end(); ltIt++) {
    LinearTermPtr lt = *ltIt;
    lt->integrate(rhsVector, testOrdering, basisCache);
  }
}