//
//  RHSEasy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RHSEasy.h"

void RHSEasy::addTerm( LinearTermPtr rhsTerm ) {
  TEST_FOR_EXCEPTION( rhsTerm->termType() != TEST, std::invalid_argument, "RHS should only involve test functions (no trials)");
  TEST_FOR_EXCEPTION( rhsTerm->rank() != 0, std::invalid_argument, "RHSEasy only handles scalar terms.");
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
  int numCells = rhsVector.dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  
  for (vector< LinearTermPtr >::iterator ltIt = _terms.begin(); ltIt != _terms.end(); ltIt++) {
    LinearTermPtr lt = *ltIt;
    
    set<int> testIDs = lt->varIDs();
    
    Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testBasis;
    
    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(numPoints);
    FieldContainer<double> values;
    
    for (set<int>::iterator testIt = testIDs.begin(); testIt != testIDs.end(); testIt++) {
      int testID = *testIt;
      testBasis = testOrdering->getBasis(testID);
      int basisCardinality = testBasis->getCardinality();
      ltValueDim[1] = basisCardinality;
      values.resize(ltValueDim);
      lt->values(values, testID, testBasis, basisCache, true); // true: applyCubatureWeights
      vector<int> testDofIndices = testOrdering->getDofIndices(testID,0);
      // compute integrals:
      for (int cellIndex = 0; cellIndex<numCells; cellIndex++) {
        for (int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++) {
          int testDofIndex = testDofIndices[basisOrdinal];
          for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
            rhsVector(cellIndex,testDofIndex) += values(cellIndex,basisOrdinal,ptIndex);
          }
        }
      }
    }
  }
}
