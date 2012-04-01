// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "L2InnerProduct.h"
#include "BilinearFormUtility.h"
#include "BasisCache.h"
#include "AbstractFunction.h"

using namespace IntrepidExtendedTypes;

void L2InnerProduct::operators(int testID1, int testID2, 
			       vector<EOperatorExtended> &testOp1,
			       vector<EOperatorExtended> &testOp2) {
  testOp1.clear();
  testOp2.clear();
  if (testID1 == testID2) { //decouple the test inner products for each test function
    testOp1.push_back( IntrepidExtendedTypes::OP_VALUE);
    testOp2.push_back( IntrepidExtendedTypes::OP_VALUE);
  }
}

void L2InnerProduct::applyInnerProductData(FieldContainer<double> &testValues1, 
					   FieldContainer<double> &testValues2,
					   int testID1, int testID2, int operatorIndex,
					   const FieldContainer<double>& physicalPoints) {
  // empty implementation -- no weights needed...
}

void L2InnerProduct::computeInnerProductMatrix(FieldContainer<double> &innerProduct,
					       Teuchos::RCP<DofOrdering> dofOrdering,
					       shards::CellTopology &cellTopo,
					       FieldContainer<double>& physicalCellNodes){

  // much of this code is the same as what's in the volume integration in computrialiffness...
  unsigned numCells = physicalCellNodes.dimension(0);
  unsigned numNodesPerElem = physicalCellNodes.dimension(1);
  unsigned spaceDim = physicalCellNodes.dimension(2);
  
  // Check that cellTopo and physicalCellNodes agree
  TEST_FOR_EXCEPTION( ( numNodesPerElem != cellTopo.getNodeCount() ),
		      std::invalid_argument,
		      "Second dimension of physicalCellNodes and cellTopo.getNodeCount() do not match.");
  TEST_FOR_EXCEPTION( ( spaceDim != cellTopo.getDimension() ),
		      std::invalid_argument,
		      "Third dimension of physicalCellNodes and cellTopo.getDimension() do not match.");
  
  // Set up Basis cache
  int cubDegree = 2*dofOrdering->maxBasisDegree();
  BasisCache basisCache(physicalCellNodes, cellTopo, cubDegree);
  
  vector<int> trialIDs = _bilinearForm->trialIDs();
  vector<int>::iterator trialIterator1;
  vector<int>::iterator trialIterator2;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trial1Basis, trial2Basis;

  innerProduct.initialize(0.0);
  FieldContainer<double> physicalCubaturePoints = basisCache.getPhysicalCubaturePoints();
  
  for (trialIterator1= trialIDs.begin(); trialIterator1 != trialIDs.end(); trialIterator1++) {
    int trialID1 = *trialIterator1;
    for (trialIterator2= trialIDs.begin(); trialIterator2 != trialIDs.end(); trialIterator2++) {
      int trialID2 = *trialIterator2;
      
      vector<EOperatorExtended> trial1Operators;
      vector<EOperatorExtended> trial2Operators;
      
      operators(trialID1,trialID2,trial1Operators,trial2Operators);
      
      // check dimensions
      TEST_FOR_EXCEPTION( ( trial1Operators.size() != trial2Operators.size() ),
			  std::invalid_argument,
			  "trial1Operators.size() and trial2Operators.size() do not match.");
      
      vector<EOperatorExtended>::iterator op1It;
      vector<EOperatorExtended>::iterator op2It = trial2Operators.begin();
      int operatorIndex = 0;
      for (op1It=trial1Operators.begin(); op1It != trial1Operators.end(); op1It++) {
        IntrepidExtendedTypes::EOperatorExtended op1 = *(op1It);
        IntrepidExtendedTypes::EOperatorExtended op2 = *(op2It);
        FieldContainer<double> trial1Values; // these will be resized inside applyOperator..
        FieldContainer<double> trial2Values; // derivative values
        
        trial1Basis = dofOrdering->getBasis(trialID1);
        trial2Basis = dofOrdering->getBasis(trialID2);
        
        int numDofs1 = trial1Basis->getCardinality();
        int numDofs2 = trial2Basis->getCardinality();
        
        FieldContainer<double> miniMatrix( numCells, numDofs1, numDofs2 );
      
        Teuchos::RCP< const FieldContainer<double> > trial1ValuesTransformedWeighted, trial2ValuesTransformed;
        
        trial1ValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trial1Basis,op1);
        trial2ValuesTransformed = basisCache.getTransformedValues(trial2Basis,op2);
        
        FieldContainer<double> innerProductDataAppliedToTrial2 = *trial2ValuesTransformed; // copy first
        FieldContainer<double> innerProductDataAppliedToTrial1 = *trial1ValuesTransformedWeighted; // copy first

        applyInnerProductData(innerProductDataAppliedToTrial1, innerProductDataAppliedToTrial2, 
                              trialID1, trialID2, operatorIndex, physicalCubaturePoints);

        FunctionSpaceTools::integrate<double>(miniMatrix,innerProductDataAppliedToTrial1,
                                              innerProductDataAppliedToTrial2,COMP_CPP);
      
        int trial1DofOffset = dofOrdering->getDofIndex(trialID1,0);
        int trial2DofOffset = dofOrdering->getDofIndex(trialID2,0);
        
        // there may be a more efficient way to do this copying:
        for (int i=0; i < numDofs1; i++) {
          for (int j=0; j < numDofs2; j++) {
            for (unsigned k=0; k < numCells; k++) {
              innerProduct(k,i+trial1DofOffset,j+trial2DofOffset) += miniMatrix(k,i,j);
            }
          }
        }        
        op2It++;
        operatorIndex++;
      }
      
    }
  }
}


void L2InnerProduct::computeInnerProductVector(FieldContainer<double> &innerProduct,
					       Teuchos::RCP<DofOrdering> dofOrdering,
					       shards::CellTopology &cellTopo,
					       FieldContainer<double>& physicalCellNodes,
					       Teuchos::RCP<AbstractFunction> fxn){

  // much of this code is the same as what's in the volume integration in computrialiffness...
  unsigned numCells = physicalCellNodes.dimension(0);
  unsigned numNodesPerElem = physicalCellNodes.dimension(1);
  unsigned spaceDim = physicalCellNodes.dimension(2);
  
  // Check that cellTopo and physicalCellNodes agree
  TEST_FOR_EXCEPTION( ( numNodesPerElem != cellTopo.getNodeCount() ),
		      std::invalid_argument,
		      "Second dimension of physicalCellNodes and cellTopo.getNodeCount() do not match.");
  TEST_FOR_EXCEPTION( ( spaceDim != cellTopo.getDimension() ),
		      std::invalid_argument,
		      "Third dimension of physicalCellNodes and cellTopo.getDimension() do not match.");
  
  // Set up Basis cache
  int cubDegree = 2*dofOrdering->maxBasisDegree();
  BasisCache basisCache(physicalCellNodes, cellTopo, cubDegree);
  
  vector<int> trialIDs = _bilinearForm->trialIDs();
  vector<int>::iterator trialIterator1;
  vector<int>::iterator trialIterator2;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trial1Basis, trial2Basis;

  innerProduct.initialize(0.0);
  FieldContainer<double> physicalCubaturePoints = basisCache.getPhysicalCubaturePoints();
  int numFieldTrialIDs = 0;
  for (trialIterator1= trialIDs.begin(); trialIterator1 != trialIDs.end(); trialIterator1++) {
    if (!_bilinearForm->isFluxOrTrace(*trialIterator1)){
      numFieldTrialIDs++;
    }
  }  
  
  for (trialIterator1= trialIDs.begin(); trialIterator1 != trialIDs.end(); trialIterator1++) {
    int trialID1 = *trialIterator1;

    FieldContainer<double> fxnValues(numFieldTrialIDs); // to be resized inside fxn->computeValues(...)
    fxn->getValues(fxnValues,physicalCubaturePoints); 
    if (fxnValues.rank()==0){
      //      FieldContainer<double> a (_numCells, numCubPoints, _spaceDim);      
    }

    for (trialIterator2= trialIDs.begin(); trialIterator2 != trialIDs.end(); trialIterator2++) {
      int trialID2 = *trialIterator2;
     
      vector<EOperatorExtended> trial1Operators;
      vector<EOperatorExtended> trial2Operators;
      
      operators(trialID1,trialID2,trial1Operators,trial2Operators);
      
      // check dimensions
      TEST_FOR_EXCEPTION( ( trial1Operators.size() != trial2Operators.size() ),
			  std::invalid_argument,
			  "trial1Operators.size() and trial2Operators.size() do not match.");
      
      vector<EOperatorExtended>::iterator op1It;
      vector<EOperatorExtended>::iterator op2It = trial2Operators.begin();
      int operatorIndex = 0;
      for (op1It=trial1Operators.begin(); op1It != trial1Operators.end(); op1It++) {
        IntrepidExtendedTypes::EOperatorExtended op1 = *(op1It);
        IntrepidExtendedTypes::EOperatorExtended op2 = *(op2It);
        
        trial1Basis = dofOrdering->getBasis(trialID1);
	//        trial2Basis = dofOrdering->getBasis(trialID2);
        
        int numDofs1 = trial1Basis->getCardinality();
	//        int numDofs2 = trial2Basis->getCardinality();
        
	//        FieldContainer<double> miniMatrix( numCells, numDofs1, numDofs2 );
        FieldContainer<double> miniVector( numCells, numDofs1 );
      
        Teuchos::RCP< const FieldContainer<double> > trial1ValuesTransformedWeighted, trial2ValuesTransformed;
        
        trial1ValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trial1Basis,op1);
	// trial2ValuesTransformed = basisCache.getTransformedValues(trial2Basis,op2);
	// (replace with call to fxn.getValues or fxnValues field container)
	// fxn.transformValues
	cout << "Trial values: "<< *trial1ValuesTransformedWeighted;
        //(C cells, F index of the basis fxn, P pts, D solution components) 
	//(C,F=0,P,D) (for the function-to-project)
        FieldContainer<double> innerProductDataAppliedToTrial2 = *trial2ValuesTransformed; // copy first
        FieldContainer<double> innerProductDataAppliedToTrial1 = *trial1ValuesTransformedWeighted; // copy first

        applyInnerProductData(innerProductDataAppliedToTrial1, innerProductDataAppliedToTrial2, 
                              trialID1, trialID2, operatorIndex, physicalCubaturePoints); 

        FunctionSpaceTools::integrate<double>(miniVector,innerProductDataAppliedToTrial1,
                                              innerProductDataAppliedToTrial2,COMP_CPP);
      
        int trial1DofOffset = dofOrdering->getDofIndex(trialID1,0);
	//        int trial2DofOffset = dofOrdering->getDofIndex(trialID2,0);
        
        // there may be a more efficient way to do this copying:
        for (int i=0; i < numDofs1; i++) {
	  //          for (int j=0; j < numDofs2; j++) {
	  for (unsigned k=0; k < numCells; k++) {
	    //              innerProduct(k,i+trial1DofOffset,j+trial2DofOffset) += miniVector(k,i,j);
	    innerProduct(k,i+trial1DofOffset) += miniVector(k,i);
	  }
	  //	}
        }        
        op2It++;
        operatorIndex++;
      }
      
    }
  }
}
