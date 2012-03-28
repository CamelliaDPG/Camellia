//
//  RHS.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RHS.h"
#include "BasisCache.h"
#include "BilinearFormUtility.h" // just for the weightCellBasisValues thing -- may be better to move that here

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
    
  TEST_FOR_EXCEPTION( ( testOrdering->totalDofs() != rhsVector.dimension(1) ),
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
      EOperatorExtended testOperator = *testOpIt;
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
  //cout << "rhsVector: " << endl << rhsVector;
}

void RHS::integrateAgainstOptimalTests(FieldContainer<double> &rhsVector,
                                       const FieldContainer<double> &optimalTestWeights,
                                       Teuchos::RCP<DofOrdering> testOrdering,
                                       BasisCachePtr basisCache) {
  // rhsVector dimensions are: (numCells, # trialOrdering Dofs) == (numCells, # optimal test functions)
  // optimalTestWeights dimensions are: (numCells, numTrial, numTest) -- numTrial is the optTest index
  
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
  
  int numOptTestFunctions = optimalTestWeights.dimension(1); // should also == numTrialDofs
  
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != rhsVector.dimension(1) ),
                     std::invalid_argument,
                     "optimalTestWeights.dimension(1) (=" << optimalTestWeights.dimension(1) << ") and rhsVector.dimension(1) (=" << rhsVector.dimension(1) << ") do not match.");
  
  vector<int> testIDs = testOrdering->getVarIDs();
  vector<int>::iterator testIterator;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testBasis;
  
  FieldContainer<double> rhsPointValues; // the rhs method will resize...	
  
  rhsVector.initialize(0.0);
  
  for (int optTestIndex=0; optTestIndex < numOptTestFunctions; optTestIndex++) {
    FieldContainer<double> weights(numCells,testOrdering->totalDofs());
    for (unsigned i=0; i<numCells; i++) {
      for (int j=0; j<testOrdering->totalDofs(); j++) {
        weights(i,j) = optimalTestWeights(i,optTestIndex,j);
      }
    }
    //cout << "for optTestIndex " << optTestIndex << ", weights: " << endl << weights;
    for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
      int testID = *testIterator;
      
      vector<EOperatorExtended> testOperators = operatorsForTestID(testID);
      int operatorIndex = -1;
      for (vector<EOperatorExtended>::iterator testOpIt=testOperators.begin();
           testOpIt != testOperators.end(); testOpIt++) {
        operatorIndex++;
        EOperatorExtended testOperator = *testOpIt;
        bool notZero = nonZeroRHS(testID);
        if (notZero) { // compute the integral(s)
          
          testBasis = testOrdering->getBasis(testID);
          
          Teuchos::RCP< const FieldContainer<double> > testValuesTransformedWeighted;
          
          testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator);
          FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();
          
          int testDofOffset = testOrdering->getDofIndex(testID,0);
          // note that weightCellBasisValues does depend on contiguous test basis dofs...
          // (this is the plan, since there shouldn't be any kind of identification between different test dofs,
          //  especially since test functions live only inside the cell)
          FieldContainer<double> testValuesTransformedWeightedWeighted = *testValuesTransformedWeighted;
          BilinearFormUtility::weightCellBasisValues(testValuesTransformedWeightedWeighted, weights, testDofOffset);
          
          rhs(testID,operatorIndex,basisCache,rhsPointValues);
          
          //cout << "rhsPointValues for testID " << testID << ":" << endl << rhsPointValues;
          
          //cout << "d." << endl;
          //   d. Sum up (integrate)
          // to integrate, first multiply the testValues (C,F,P) or (C,F,P,D)
          //               by the rhsPointValues (C,P) or (C,P,D), respectively, and then sum.
          int numPoints = rhsPointValues.dimension(1);
          for (unsigned k=0; k < numCells; k++) {
            for (int i=0; i < testBasis->getCardinality(); i++) {
              for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
                if (rhsPointValues.rank() == 2) {
                  rhsVector(k,optTestIndex) += testValuesTransformedWeightedWeighted(k,i,ptIndex) * rhsPointValues(k,ptIndex);
                } else {
                  for (int d=0; d<spaceDim; d++) {
                    rhsVector(k,optTestIndex) += testValuesTransformedWeightedWeighted(k,i,ptIndex,d) * rhsPointValues(k,ptIndex,d);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  //cout << "rhsVector: " << endl << rhsVector;
}