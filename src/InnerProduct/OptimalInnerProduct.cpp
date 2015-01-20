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


/*
 *  OptimalInnerProduct.cpp
 *
 *  Created by Nathan Roberts on 7/22/11.
 *
 */

#include "OptimalInnerProduct.h"
#include "SerialDenseWrapper.h"

typedef pair<IntrepidExtendedTypes::EOperator, int > OpOpIndexPair;

OptimalInnerProduct::OptimalInnerProduct(Teuchos::RCP< BilinearForm > bf) : IP(bf) {
  _beta = 1; // TODO: allow this to be controlled from outside
  // TODO: replace the cout with an ostringstream, and save the string so that we can return description on request...
  vector<int> trialIDs = bf->trialIDs();
  vector<int> testIDs =  bf->testIDs();
  for (vector<int>::iterator trialIt = trialIDs.begin();
       trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    cout << "*********** optimal test Interactions for trial variable " << _bilinearForm->trialName(trialID) << "************" << endl;
    vector<int> myTestIDs;
    vector<vector<IntrepidExtendedTypes::EOperator> > ops;
    for (vector<int>::iterator testIt=testIDs.begin();
         testIt != testIDs.end(); testIt++) {
      int testID = *testIt;
      vector<IntrepidExtendedTypes::EOperator> trialOperators, testOperators;
      bf->trialTestOperators(trialID, testID, trialOperators, testOperators);
      vector<IntrepidExtendedTypes::EOperator>::iterator trialOpIt, testOpIt;
      testOpIt = testOperators.begin();
      // NVR moved next two lines outside the for loop below 11/14/11
      vector<IntrepidExtendedTypes::EOperator> testOps;
      myTestIDs.push_back(testID);
      for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++) {
        IntrepidExtendedTypes::EOperator op1 = *trialOpIt;
        IntrepidExtendedTypes::EOperator op2 = *testOpIt;

        // there is a live combination
        // op1 will always be value
        if (( op1 !=  IntrepidExtendedTypes::OP_VALUE)
         && ( op1 != IntrepidExtendedTypes::OP_VECTORIZE_VALUE)
         && ( !_bilinearForm->isFluxOrTrace(trialID)) ) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::invalid_argument,
                             "OptimalInnerProduct assumes OP_VALUE for trialIDs.")
        }
        if ( _bilinearForm->isFluxOrTrace(trialID) ) {
          // boundary value: push inside the element (take L2 norm there)
          op2 =  IntrepidExtendedTypes::OP_VALUE;
        }
        testOps.push_back(op2);
        testOpIt++;
      }
      ops.push_back(testOps); // Nate moved this line 11-14-11 (from within above for loop)
    }

    bool first = true; // for printing debug code out...
    // determine the contribution corresponding to trialID, now that we have:
    //            ( op1( testID1 ) + op2 (testID2) + ... )
    // essentially, we want to square the sum of the operators, keeping track of the right indices so that
    // we can ask the bilinearForm for the appropriate weight.
    int test1Index = -1;
    for (vector<int>::iterator testIt1=myTestIDs.begin();
         testIt1 != myTestIDs.end(); testIt1++) {
      test1Index++;
      int testID1 = *testIt1;
      if (_bilinearForm->isFluxOrTrace(trialID)) {
        // we'll want to skip the regular computation of the square of the sum above, and instead
        // just take L^2 norms of each test function
        pair<int,int> key = make_pair(testID1, testID1);
        int opIndex = -1; // placeholder; should not be used
        OpOpIndexPair opPair = make_pair( IntrepidExtendedTypes::OP_VALUE,opIndex);
        pair<pair<OpOpIndexPair,OpOpIndexPair>, int> entry = make_pair( make_pair(opPair,opPair), trialID);
        _testCombos[key].push_back(entry);
        if ( ! first) cout << " + ";
        cout << _bilinearForm->testName(testID1) << " " << _bilinearForm->testName(testID1);
        first = false;
      } else {
        vector<IntrepidExtendedTypes::EOperator> ops1 = ops[test1Index];
        int test2Index = -1;
        for (vector<int>::iterator testIt2=myTestIDs.begin();
             testIt2 != myTestIDs.end(); testIt2++) {
          test2Index++;
          int testID2 = *testIt2;
          pair<int,int> key = make_pair(testID1, testID2);
          vector<IntrepidExtendedTypes::EOperator> ops2 = ops[test2Index];
          vector<IntrepidExtendedTypes::EOperator>::iterator op1It, op2It;
          int op1Index = -1;
          for (op1It = ops1.begin(); op1It != ops1.end(); op1It++) {
            IntrepidExtendedTypes::EOperator op1 = *op1It;
            op1Index++;
            int op2Index = -1;
            for (op2It = ops2.begin(); op2It != ops2.end(); op2It++) {
              IntrepidExtendedTypes::EOperator op2 = *op2It;
              op2Index++;
              OpOpIndexPair op1Pair = make_pair(op1,op1Index);
              OpOpIndexPair op2Pair = make_pair(op2,op2Index);
              pair<pair<OpOpIndexPair,OpOpIndexPair>, int> entry = make_pair( make_pair(op1Pair,op2Pair), trialID);
              _testCombos[key].push_back(entry);
              if ( ! first) cout << " + ";
              cout << _bilinearForm->operatorName(op1) << _bilinearForm->testName(testID1) << " ";
              cout << _bilinearForm->operatorName(op2) << _bilinearForm->testName(testID2);
              first = false;
            }
          }
        }
      }
    }
    cout << endl;
  } // end of trialID loop
}


void OptimalInnerProduct::operators(int testID1, int testID2, 
               vector<IntrepidExtendedTypes::EOperator> &testOp1,
               vector<IntrepidExtendedTypes::EOperator> &testOp2) {
  testOp1.clear();
  testOp2.clear();
  pair<int, int> key = make_pair(testID1,testID2);
  if ( _testCombos.find(key) != _testCombos.end() ) {
    vector< pair<pair<OpOpIndexPair,OpOpIndexPair>, int> > entries = _testCombos[key];
    vector< pair<pair<OpOpIndexPair,OpOpIndexPair>, int> >::iterator entryIt;
    for (entryIt = entries.begin(); entryIt != entries.end(); entryIt++) {
      pair<pair<OpOpIndexPair,OpOpIndexPair>, int> entry = *entryIt;
      pair<OpOpIndexPair,OpOpIndexPair> opOpPair = entry.first;
      testOp1.push_back(opOpPair.first.first);
      testOp2.push_back(opOpPair.second.first);
    }
  }
}

//void vectorizeFC(FieldContainer<double> &testValues, int spaceDim) {
//  Teuchos::Array<int> dimensions;
//  testValues.dimensions(dimensions);
//  dimensions.push_back(spaceDim);
//  FieldContainer<double> vectorValues(dimensions);
//  int numEntries = testValues.size();
//  for (int i=0; i<numEntries; i++) {
//    for (int dim = 0; dim < spaceDim; dim++) {
//      vectorValues[spaceDim*i + dim] = testValues[i];
//    }
//  }
//  testValues = vectorValues;
//}

void OptimalInnerProduct::applyInnerProductData(FieldContainer<double> &testValues1,
                                                FieldContainer<double> &testValues2,
                                                int testID1, int testID2, int operatorIndex,
                                                const FieldContainer<double>& physicalPoints) {
  pair<int, int> key = make_pair(testID1,testID2);
  if ( _testCombos.find(key) != _testCombos.end() ) {
    vector< pair<pair<OpOpIndexPair,OpOpIndexPair>, int> > entries = _testCombos[key];
    vector< pair<pair<OpOpIndexPair,OpOpIndexPair>, int> >::iterator entryIt;
    pair<pair<OpOpIndexPair,OpOpIndexPair>, int> entry = _testCombos[key][operatorIndex];
    int trialID = entry.second;
    pair<OpOpIndexPair,OpOpIndexPair> opOpPair = entry.first;
    
    if (! _bilinearForm->isFluxOrTrace(trialID)) {
      // if it's a flux or trace, then we operate on the volume, not the boundary, so we can't use
      // applyBilinearFormData...
      
      // set up a dummy trialValues
      int numCells = testValues1.dimension(0);
      int numFields1 = testValues1.dimension(1);
      int numPoints = testValues1.dimension(2);
      int spaceDim = physicalPoints.dimension(1);
      FieldContainer<double> trialValues(numCells,numFields1,numPoints);
      trialValues.initialize(1.0);
      
      int opIndex1 = opOpPair.first.second;
      _bilinearForm->applyBilinearFormData(trialValues,testValues1,
                                           trialID, testID1, opIndex1, physicalPoints);
      
      // _bilinearForm->applyBilinearFormData(trialID, testID1, trialValues, testValues1, physicalPoints);
      // weight testValues with anything that's been placed in trialValues:
      if (trialValues.size() == testValues1.size() ) {
        for (int i=0; i<trialValues.size(); i++) {
          testValues1[i] *= trialValues[i];
        }
      } else {
        if (testValues1.size() / trialValues.size() != spaceDim) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialValues is an unexpected size relative to testValues.");
        }
        for (int i=0; i<trialValues.size(); i++) {
          for (int dim=0; dim<spaceDim; dim++) {
            testValues1[i*spaceDim+dim] *= trialValues[i];
          }
        }
      }

      // reset trialValues
      int numFields2 = testValues2.dimension(1);
      trialValues.resize(numCells,numFields2,numPoints);
      trialValues.initialize(1.0);
    
      int opIndex2 = opOpPair.second.second;
      _bilinearForm->applyBilinearFormData(trialValues,testValues2, 
                                           trialID, testID2, opIndex2, physicalPoints);
      //_bilinearForm->applyBilinearFormData(trialID, testID2, trialValues, testValues2, physicalPoints);
      
      // weight testValues with anything that's been placed in trialValues:
      if (trialValues.size() == testValues2.size() ) {
        for (int i=0; i<trialValues.size(); i++) {
          testValues2[i] *= trialValues[i];
        }
      } else {
        if (testValues2.size() / trialValues.size() != spaceDim) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialValues is an unexpected size relative to testValues.");
        }
        for (int i=0; i<trialValues.size(); i++) {
          for (int dim=0; dim<spaceDim; dim++) {
            testValues2[i*spaceDim+dim] *= trialValues[i];
          }
        }
      }
    } else if (_bilinearForm->isFluxOrTrace(trialID)) {
      // then weight by _beta
      SerialDenseWrapper::multiplyFCByWeight(testValues2, _beta); // TODO: determine whether it should be beta or beta^2 to be consistent with hpDPG code...
    }
    // when bilinear form gets vectors of operators, we'll consult 
    // opOpPair.first.second (where opOpPair = entry.first is as above) for the operatorIndex for testID1, i.e.:
    // opIndex1 = opOpPair.first.second;
    // opIndex2 = opOpPair.second.second;
    // _bilinearForm->applyBilinearFormData(trialID, testID1, opIndex1, testValues, physicalPoints);
    // _bilinearForm->applyBilinearFormData(trialID, testID2, opIndex2, testValues, physicalPoints);
  }
}