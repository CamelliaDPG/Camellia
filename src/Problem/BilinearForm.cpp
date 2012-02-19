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

#include "BilinearForm.h"

#include "Intrepid_FunctionSpaceTools.hpp"

static const string & S_OPERATOR_VALUE = "";
static const string & S_OPERATOR_GRAD = "\\nabla ";
static const string & S_OPERATOR_CURL = "\\nabla \\times ";
static const string & S_OPERATOR_DIV = "\\nabla \\cdot ";
static const string & S_OPERATOR_D1 = "D1 ";
static const string & S_OPERATOR_D2 = "D2 ";
static const string & S_OPERATOR_D3 = "D3 ";
static const string & S_OPERATOR_D4 = "D4 ";
static const string & S_OPERATOR_D5 = "D5 ";
static const string & S_OPERATOR_D6 = "D6 ";
static const string & S_OPERATOR_D7 = "D7 ";
static const string & S_OPERATOR_D8 = "D8 ";
static const string & S_OPERATOR_D9 = "D9 ";
static const string & S_OPERATOR_D10 = "D10 ";
static const string & S_OPERATOR_X = "{1 \\choose 0} \\cdot ";
static const string & S_OPERATOR_Y = "{0 \\choose 1} \\cdot ";
static const string & S_OPERATOR_Z = "\\bf{k} \\cdot ";
static const string & S_OPERATOR_DX = "\\frac{\\partial}{\\partial x} ";
static const string & S_OPERATOR_DY = "\\frac{\\partial}{\\partial y} ";
static const string & S_OPERATOR_DZ = "\\frac{\\partial}{\\partial z} ";
static const string & S_OPERATOR_CROSS_NORMAL = "\\times \\widehat{n} ";
static const string & S_OPERATOR_DOT_NORMAL = "\\cdot \\widehat{n} ";
static const string & S_OPERATOR_TIMES_NORMAL = " \\widehat{n} \\cdot ";
static const string & S_OPERATOR_VECTORIZE_VALUE = ""; // handle this one separately...
static const string & S_OPERATOR_UNKNOWN = "[UNKNOWN OPERATOR] ";

const vector< int > & BilinearForm::trialIDs() {
  return _trialIDs;
}

const vector< int > & BilinearForm::testIDs() {
  return _testIDs;
}

void BilinearForm::applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                         int trialID, int testID, int operatorIndex,
                                         FieldContainer<double> &points) {
  applyBilinearFormData(trialID,testID,trialValues,testValues,points);
}

void BilinearForm::applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                         int trialID, int testID, int operatorIndex,
                                         FieldContainer<double> &points, 
                                         const map<int, FieldContainer<double> > &previousSolutionValues) {
  applyBilinearFormData(trialValues, testValues, trialID, testID, operatorIndex, points);
}

void BilinearForm::trialTestOperators(int testID1, int testID2, 
                                      vector<EOperatorExtended> &testOps1,
                                      vector<EOperatorExtended> &testOps2) {
  EOperatorExtended testOp1, testOp2;
  testOps1.clear();
  testOps2.clear();
  if (trialTestOperator(testID1,testID2,testOp1,testOp2)) {
    testOps1.push_back(testOp1);
    testOps2.push_back(testOp2);
  }
}

void BilinearForm::multiplyFCByWeight(FieldContainer<double> & fc, double weight) {
  int size = fc.size();
  double *valuePtr = &fc[0]; // to make this as fast as possible, do some pointer arithmetic...
  for (int i=0; i<size; i++) {
    *valuePtr *= weight;
    valuePtr++;
  }
}


void BilinearForm::previousSolutionRequired(set<int> &trialIDs) {
  // default: just clear the set
  trialIDs.clear();
}

vector<int> BilinearForm::trialVolumeIDs() {
  vector<int> ids;
  for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++) {
    int trialID = *(trialIt);
    if ( ! isFluxOrTrace(trialID) ) {
      ids.push_back(trialID);
    }
  }
  return ids;
}

vector<int> BilinearForm::trialBoundaryIDs() {
  vector<int> ids;
  for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++) {
    int trialID = *(trialIt);
    if ( isFluxOrTrace(trialID) ) {
      ids.push_back(trialID);
    }
  }
  return ids;  
}

int BilinearForm::operatorRank(EOperatorExtended op, EFunctionSpaceExtended fs) {
  // returns the rank of basis functions in the function space fs when op is applied
  // 0 scalar, 1 vector
  int SCALAR = 0, VECTOR = 1;
  switch (op) {
    case IntrepidExtendedTypes::OPERATOR_VALUE:
      if (   (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) 
          || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL)
          || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_ONE) )
        return SCALAR; 
      else
        return VECTOR;
    case IntrepidExtendedTypes::OPERATOR_GRAD:
    case IntrepidExtendedTypes::OPERATOR_CURL:
      return VECTOR;
    case IntrepidExtendedTypes::OPERATOR_DIV:
    case IntrepidExtendedTypes::OPERATOR_X:
    case IntrepidExtendedTypes::OPERATOR_Y:
    case IntrepidExtendedTypes::OPERATOR_Z:
    case IntrepidExtendedTypes::OPERATOR_DX:
    case IntrepidExtendedTypes::OPERATOR_DY:
    case IntrepidExtendedTypes::OPERATOR_DZ:
      return SCALAR; 
    case IntrepidExtendedTypes::OPERATOR_CROSS_NORMAL:
      return VECTOR; 
    case IntrepidExtendedTypes::OPERATOR_DOT_NORMAL:
      return SCALAR; 
    case IntrepidExtendedTypes::OPERATOR_TIMES_NORMAL:
      return VECTOR; 
    case IntrepidExtendedTypes::OPERATOR_VECTORIZE_VALUE:
      return VECTOR;
    default:
      return -1;
  }
}

const string & BilinearForm::operatorName(EOperatorExtended op) {
  switch (op) {
    case IntrepidExtendedTypes::OPERATOR_VALUE:
      return S_OPERATOR_VALUE; 
      break;
    case IntrepidExtendedTypes::OPERATOR_GRAD:
      return S_OPERATOR_GRAD; 
      break;
    case IntrepidExtendedTypes::OPERATOR_CURL:
      return S_OPERATOR_CURL; 
      break;
    case IntrepidExtendedTypes::OPERATOR_DIV:
      return S_OPERATOR_DIV; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D1:
      return S_OPERATOR_D1; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D2:
      return S_OPERATOR_D2; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D3:
      return S_OPERATOR_D3; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D4:
      return S_OPERATOR_D4; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D5:
      return S_OPERATOR_D5; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D6:
      return S_OPERATOR_D6; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D7:
      return S_OPERATOR_D7; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D8:
      return S_OPERATOR_D8; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D9:
      return S_OPERATOR_D9; 
      break;
    case IntrepidExtendedTypes::OPERATOR_D10:
      return S_OPERATOR_D10; 
      break;
    case IntrepidExtendedTypes::OPERATOR_X:
      return S_OPERATOR_X; 
      break;
    case IntrepidExtendedTypes::OPERATOR_Y:
      return S_OPERATOR_Y; 
      break;
    case IntrepidExtendedTypes::OPERATOR_Z:
      return S_OPERATOR_Z; 
      break;
    case IntrepidExtendedTypes::OPERATOR_DX:
      return S_OPERATOR_DX; 
      break;
    case IntrepidExtendedTypes::OPERATOR_DY:
      return S_OPERATOR_DY; 
      break;
    case IntrepidExtendedTypes::OPERATOR_DZ:
      return S_OPERATOR_DZ; 
      break;
    case IntrepidExtendedTypes::OPERATOR_CROSS_NORMAL:
      return S_OPERATOR_CROSS_NORMAL; 
      break;
    case IntrepidExtendedTypes::OPERATOR_DOT_NORMAL:
      return S_OPERATOR_DOT_NORMAL; 
      break;
    case IntrepidExtendedTypes::OPERATOR_TIMES_NORMAL:
      return S_OPERATOR_TIMES_NORMAL; 
      break;
    case IntrepidExtendedTypes::OPERATOR_VECTORIZE_VALUE:
      return S_OPERATOR_VECTORIZE_VALUE; 
      break;
    default:
      return S_OPERATOR_UNKNOWN;
      break;
  }
}

void BilinearForm::printTrialTestInteractions() {
  for (vector<int>::iterator testIt = _testIDs.begin(); testIt != _testIDs.end(); testIt++) {
    int testID = *testIt;
    cout << endl << "b(U," << testName(testID) << ") &= " << endl;
    bool first = true;
    int spaceDim = 2;
    FieldContainer<double> point(1,2); // (0,0)
    FieldContainer<double> testValueScalar(1,1,1); // 1 cell, 1 basis function, 1 point...
    FieldContainer<double> testValueVector(1,1,1,spaceDim); // 1 cell, 1 basis function, 1 point, spaceDim dimensions...
    FieldContainer<double> trialValueScalar(1,1,1); // 1 cell, 1 basis function, 1 point...
    FieldContainer<double> trialValueVector(1,1,1,spaceDim); // 1 cell, 1 basis function, 1 point, spaceDim dimensions...
    FieldContainer<double> testValue, trialValue;
    for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++) {
      int trialID = *trialIt;
      vector<EOperatorExtended> trialOperators, testOperators;
      trialTestOperators(trialID, testID, trialOperators, testOperators);
      vector<EOperatorExtended>::iterator trialOpIt, testOpIt;
      testOpIt = testOperators.begin();
      int operatorIndex = 0;
      for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++) {
        EOperatorExtended opTrial = *trialOpIt;
        EOperatorExtended opTest = *testOpIt;
        int trialRank = operatorRank(opTrial, functionSpaceForTrial(trialID));
        int testRank = operatorRank(opTest, functionSpaceForTest(testID));
        trialValue = ( trialRank == 0 ) ? trialValueScalar : trialValueVector;
        testValue = (testRank == 0) ? testValueScalar : testValueVector;
        
        trialValue[0] = 1.0; testValue[0] = 1.0;
        FieldContainer<double> testWeight(1), trialWeight(1); // for storing values that come back from applyBilinearForm
        applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
        if ((trialRank==1) && (trialValue.rank() == 3)) { // vector that became a scalar (a dot product)
          trialWeight.resize(spaceDim);
          trialWeight[0] = trialValue[0];
          for (int dim=1; dim<spaceDim; dim++) {
            trialValue = trialValueVector;
            trialValue.initialize(0.0);
            testValue = (testRank == 0) ? testValueScalar : testValueVector;
            trialValue[dim] = 1.0;
            applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
            trialWeight[dim] = trialValue[0];
          }
        } else {
          trialWeight[0] = trialValue[0];
        }
        // same thing, but now for testWeight
        if ((testRank==1) && (testValue.rank() == 3)) { // vector that became a scalar (a dot product)
          testWeight.resize(spaceDim);
          testWeight[0] = trialValue[0];
          for (int dim=1; dim<spaceDim; dim++) {
            testValue = testValueVector;
            testValue.initialize(0.0);
            trialValue = (trialRank == 0) ? trialValueScalar : trialValueVector;
            testValue[dim] = 1.0;
            applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
            testWeight[dim] = testValue[0];
          }
        } else {
          testWeight[0] = testValue[0];
        }
        if ((testWeight.size() == 2) && (trialWeight.size() == 2)) { // both vector values (unsupported)
          TEST_FOR_EXCEPTION( true, std::invalid_argument, "unsupported form." );
        } else {
          // scalar & vector: combine into one, in testWeight
          if ( (trialWeight.size() + testWeight.size()) == 3) {
            FieldContainer<double> smaller = (trialWeight.size()==1) ? trialWeight : testWeight;
            FieldContainer<double> bigger =  (trialWeight.size()==2) ? trialWeight : testWeight;
            testWeight.resize(spaceDim);
            for (int dim=0; dim<spaceDim; dim++) {
              testWeight[dim] = smaller[0] * bigger[dim];
            }
          } else { // both scalars: combine into one, in testWeight
            testWeight[0] *= trialWeight[0];
          }
        }
        if (testWeight.size() == 1) { // scalar weight
          if ( testWeight[0] == -1.0 ) {
            cout << " - ";
          } else {
            if (testWeight[0] == 1.0) {
              if (! first) cout << " + ";
            } else {
              if (testWeight[0] < 0.0) {
                cout << testWeight[0] << " ";
              } else {
                cout << " + " << testWeight[0] << " ";
              }
            }
          }
          if (! isFluxOrTrace(trialID) ) {
            cout << "\\int_{K} " ;
          } else {
            cout << "\\int_{\\partial K} " ;            
          }
          cout << operatorName(opTrial) << trialName(trialID) << " ";
        } else { // 
          if (! first) cout << " + ";
          if (! isFluxOrTrace(trialID) ) {
            cout << "\\int_{K} " ;
          } else {
            cout << "\\int_{\\partial K} " ;
          }
          if (opTrial != OPERATOR_TIMES_NORMAL) {
            cout << " \\begin{bmatrix}";
            for (int dim=0; dim<spaceDim; dim++) {
              if (testWeight[dim] != 1.0) {
                cout << testWeight[0];
              }
              if (dim != spaceDim-1) {
                cout << " \\\\ ";
              }
            }
            cout << "\\end{bmatrix} ";
            cout << trialName(trialID);
            cout << " \\cdot ";
          } else if (opTrial == OPERATOR_TIMES_NORMAL) {
            if (testWeight.size() == 2) {
              cout << " {";
              if (testWeight[0] != 1.0) {
                cout << testWeight[0];
              }
              cout << " n_1 " << " \\choose ";
              if (testWeight[1] != 1.0) {
                cout << testWeight[1];
              }
              cout << " n_2 " << "} " << trialName(trialID) << " \\cdot ";
            } else {
              if (testWeight[0] != 1.0) {
                cout << testWeight[0] << " " << trialName(trialID) << operatorName(opTrial);
              } else {
                cout << trialName(trialID) << operatorName(opTrial);
              }
            }
          }
        }
        if ((opTest == OPERATOR_CROSS_NORMAL) || (opTest == OPERATOR_DOT_NORMAL)) {
          // reverse the order:
          cout << testName(testID) << operatorName(opTest);
        } else {
          cout << operatorName(opTest) << testName(testID);
        }
        first = false;
        testOpIt++;
        operatorIndex++;
      }
    }
    cout << endl << "\\\\";
  }
}