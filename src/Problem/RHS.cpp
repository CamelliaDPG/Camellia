//
//  RHS.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RHS.h"
#include "BasisCache.h"

void RHS::integrateAgainstStandardBasis(FieldContainer<double> &rhsVector, 
                                        Teuchos::RCP<DofOrdering> testOrdering, 
                                        BasisCachePtr basisCache) {
  // rhsVector dimensions are: (numCells, # testOrdering Dofs)
  
  // steps:
  // 0. Set up Cubature
  // 3. For each optimalTestFunction
  //   a. Apply the value operators to the basis in the DofOrdering, at the cubature points
  //   b. weight with Jacobian/Piola transform and cubature weights
  //   c. Pass the result to RHS to get resultant values at each point
  //   d. Sum up (integrate) and place in rhsVector according to DofOrdering indices
  
  // 0. Set up Cubature
  
  shards::CellTopology cellTopo = basisCache->cellTopology();
  
  unsigned numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  unsigned spaceDim = cellTopo.getDimension();
    
  TEUCHOS_TEST_FOR_EXCEPTION( ( testOrdering->totalDofs() != rhsVector.dimension(1) ),
                     std::invalid_argument,
                     "testOrdering->totalDofs() (=" << testOrdering->totalDofs() << ") and rhsVector.dimension(1) (=" << rhsVector.dimension(1) << ") do not match.");
  
  vector<int> testIDs = testOrdering->getVarIDs();
  vector<int>::iterator testIterator;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testBasis;
  
  FieldContainer<double> rhsPointValues; // the rhs method will resize...	
  
  rhsVector.initialize(0.0);
  
  for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
    int testID = *testIterator;
    
    vector<EOperatorExtended> testOperators = this->operatorsForTestID(testID);
    int operatorIndex = -1;
    for (vector<EOperatorExtended>::iterator testOpIt=testOperators.begin();
         testOpIt != testOperators.end(); testOpIt++) {
      operatorIndex++;
      IntrepidExtendedTypes::EOperatorExtended testOperator = *testOpIt;
      bool notZero = this->nonZeroRHS(testID);
      if (notZero) { // compute the integral(s)
        
        testBasis = testOrdering->getBasis(testID);
        
        Teuchos::RCP< const FieldContainer<double> > testValuesTransformedWeighted;
        
        testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator);
        FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();
        
        vector<int> testDofIndices = testOrdering->getDofIndices(testID,0);
        
        this->rhs(testID,operatorIndex,basisCache,rhsPointValues);
        
        //   d. Sum up (integrate)
        // to integrate, first multiply the testValues (C,F,P) or (C,F,P,D)
        //               by the rhsPointValues (C,P) or (C,P,D), respectively, and then sum.
        int numPoints = rhsPointValues.dimension(1);
        for (unsigned k=0; k < numCells; k++) {
          for (int i=0; i < testBasis->getCardinality(); i++) {
            int testDofIndex = testDofIndices[i];
            for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
              if (rhsPointValues.rank() == 2) {
                rhsVector(k,testDofIndex) += (*testValuesTransformedWeighted)(k,i,ptIndex) * rhsPointValues(k,ptIndex);
              } else {
                for (int d=0; d<spaceDim; d++) {
                  rhsVector(k,testDofIndex) += (*testValuesTransformedWeighted)(k,i,ptIndex,d) * rhsPointValues(k,ptIndex,d);
                }
              }
            }
          }
        }
      }
    }
  }
  // cout << "rhsVector: " << endl << rhsVector;
}

void RHS::integrateAgainstOptimalTests(FieldContainer<double> &rhsVector,
                                       const FieldContainer<double> &optimalTestWeights,
                                       Teuchos::RCP<DofOrdering> testOrdering,
                                       BasisCachePtr basisCache) {
  // rhsVector dimensions are: (numCells, # trialOrdering Dofs) == (numCells, # optimal test functions)
  // optimalTestWeights dimensions are: (numCells, numTrial, numTest) -- numTrial is the optTest index
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != rhsVector.dimension(1) ),
                     std::invalid_argument,
                     "optimalTestWeights.dimension(1) (=" << optimalTestWeights.dimension(1) << ") and rhsVector.dimension(1) (=" << rhsVector.dimension(1) << ") do not match.");
  
  // can represent this as the multiplication of optTestWeights matrix against the standard-basis RHS vector
  unsigned numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  
  unsigned numTrialDofs = optimalTestWeights.dimension(1);
  unsigned numTestDofs = testOrdering->totalDofs();
  
  FieldContainer<double> rhsVectorStandardBasis(numCells,numTestDofs);
  
  integrateAgainstStandardBasis(rhsVectorStandardBasis,testOrdering,basisCache);
  
  rhsVector.initialize(0.0);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int trialDofIndex=0; trialDofIndex<numTrialDofs; trialDofIndex++) {
      for (int testDofIndex=0; testDofIndex<numTestDofs; testDofIndex++) {
        rhsVector(cellIndex,trialDofIndex) += optimalTestWeights(cellIndex,trialDofIndex,testDofIndex) 
                                              * rhsVectorStandardBasis(cellIndex,testDofIndex);
      }
    }
  }
}