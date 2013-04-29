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

#include "BilinearFormUtility.h"

#include "DPGInnerProduct.h"
#include "BasisCache.h"

using namespace IntrepidExtendedTypes;

DPGInnerProduct::DPGInnerProduct(Teuchos::RCP< BilinearForm > bfs) {
  _bilinearForm = bfs;
}

void DPGInnerProduct::applyInnerProductData(FieldContainer<double> &testValues1,
                                            FieldContainer<double> &testValues2,
                                            int testID1, int testID2, int operatorIndex,
                                            Teuchos::RCP<BasisCache> basisCache) {
  applyInnerProductData(testValues1, testValues2, testID1, testID2, operatorIndex, basisCache->getPhysicalCubaturePoints());  
}

void DPGInnerProduct::computeInnerProductMatrix(FieldContainer<double> &innerProduct,
                                                Teuchos::RCP<DofOrdering> dofOrdering, shards::CellTopology &cellTopo,
                                                FieldContainer<double>& physicalCellNodes) {
  Teuchos::RCP<shards::CellTopology> cellTopoPtr = Teuchos::rcp( new shards::CellTopology(cellTopo.getCellTopologyData() ) );
  Teuchos::RCP<ElementType> elemTypePtr = Teuchos::rcp( new ElementType(dofOrdering,dofOrdering, cellTopoPtr) );
  Teuchos::RCP<Mesh> nullMeshPtr = Teuchos::rcp( (Mesh*) NULL );
  BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr, nullMeshPtr,true));
  ipBasisCache->setPhysicalCellNodes(physicalCellNodes,vector<int>(), false);
  computeInnerProductMatrix(innerProduct,dofOrdering,ipBasisCache);
}

void DPGInnerProduct::computeInnerProductMatrix(FieldContainer<double> &innerProduct,
                                                Teuchos::RCP<DofOrdering> dofOrdering, 
                                                Teuchos::RCP<BasisCache> basisCache) {
  // much of this code is the same as what's in the volume integration in computeStiffness...
  FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
  
  unsigned numCells = physicalCubaturePoints.dimension(0);
  unsigned spaceDim = physicalCubaturePoints.dimension(2);
  
  shards::CellTopology cellTopo = basisCache->cellTopology();
    
  vector<int> testIDs = _bilinearForm->testIDs();
  vector<int>::iterator testIterator1;
  vector<int>::iterator testIterator2;
  
  BasisPtr test1Basis, test2Basis;

  innerProduct.initialize(0.0);
  
  for (testIterator1= testIDs.begin(); testIterator1 != testIDs.end(); testIterator1++) {
    int testID1 = *testIterator1;
    for (testIterator2= testIDs.begin(); testIterator2 != testIDs.end(); testIterator2++) {
      int testID2 = *testIterator2;
      
      vector<EOperatorExtended> test1Operators;
      vector<EOperatorExtended> test2Operators;
      
      operators(testID1,testID2,test1Operators,test2Operators);
      
      // check dimensions
      TEUCHOS_TEST_FOR_EXCEPTION( ( test1Operators.size() != test2Operators.size() ),
                         std::invalid_argument,
                         "test1Operators.size() and test2Operators.size() do not match.");
      
      vector<EOperatorExtended>::iterator op1It;
      vector<EOperatorExtended>::iterator op2It = test2Operators.begin();
      int operatorIndex = 0;
      for (op1It=test1Operators.begin(); op1It != test1Operators.end(); op1It++) {
        IntrepidExtendedTypes::EOperatorExtended op1 = *(op1It);
        IntrepidExtendedTypes::EOperatorExtended op2 = *(op2It);
        FieldContainer<double> test1Values; // these will be resized inside applyOperator..
        FieldContainer<double> test2Values; // derivative values
        
        test1Basis = dofOrdering->getBasis(testID1);
        test2Basis = dofOrdering->getBasis(testID2);
        
        int numDofs1 = test1Basis->getCardinality();
        int numDofs2 = test2Basis->getCardinality();
        
        FieldContainer<double> miniMatrix( numCells, numDofs1, numDofs2 );
      
        Teuchos::RCP< const FieldContainer<double> > test1ValuesTransformedWeighted, test2ValuesTransformed;
        
        test1ValuesTransformedWeighted = basisCache->getTransformedWeightedValues(test1Basis,op1);
        test2ValuesTransformed = basisCache->getTransformedValues(test2Basis,op2);
        
        FieldContainer<double> innerProductDataAppliedToTest2 = *test2ValuesTransformed; // copy first
        FieldContainer<double> innerProductDataAppliedToTest1 = *test1ValuesTransformedWeighted; // copy first

        //cout << "rank of test2ValuesTransformed: " << test2ValuesTransformed->rank() << endl;
        applyInnerProductData(innerProductDataAppliedToTest1, innerProductDataAppliedToTest2, 
                              testID1, testID2, operatorIndex, basisCache);

        FunctionSpaceTools::integrate<double>(miniMatrix,innerProductDataAppliedToTest1,
                                              innerProductDataAppliedToTest2,COMP_CPP);
      
        int test1DofOffset = dofOrdering->getDofIndex(testID1,0);
        int test2DofOffset = dofOrdering->getDofIndex(testID2,0);
        
        // there may be a more efficient way to do this copying:
        for (int i=0; i < numDofs1; i++) {
          for (int j=0; j < numDofs2; j++) {
            for (unsigned k=0; k < numCells; k++) {
              innerProduct(k,i+test1DofOffset,j+test2DofOffset) += miniMatrix(k,i,j);
            }
          }
        }
        
        op2It++;
        operatorIndex++;
      }
      
    }
  }
}

bool DPGInnerProduct::hasBoundaryTerms() {
  return false;
}

void DPGInnerProduct::printInteractions() {
  cout << "Inner product: test interactions\n";
  vector<int> testIDs = _bilinearForm->testIDs();
  for (vector<int>::iterator testIt = testIDs.begin(); testIt != testIDs.end(); testIt++) {
    int testID = *testIt;
    cout << endl << "****** Interactions with test variable " << _bilinearForm->testName(testID) << " ******* " << endl;
    bool first = true;
    for (vector<int>::iterator testIt2 = testIDs.begin(); testIt2 != testIDs.end(); testIt2++) {
      int testID2 = *testIt2;
      vector<EOperatorExtended> ops1, ops2;
      operators(testID, testID2, ops1, ops2);
      int numOps = ops1.size();
      for (int i=0; i<numOps; i++) {
        if ( ! first) cout << " + ";
        cout << _bilinearForm->operatorName(ops1[i]) << " " << _bilinearForm->testName(testID) << " ";
        cout << _bilinearForm->operatorName(ops2[i]) << " " << _bilinearForm->testName(testID2);
        first = false;
      }
    }
    cout << endl;
  }
}