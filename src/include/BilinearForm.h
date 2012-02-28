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

#ifndef BILINEAR_FORM_SPECIFICATION
#define BILINEAR_FORM_SPECIFICATION

#include "Intrepid_Types.hpp"
#include "Intrepid_FieldContainer.hpp"

class BasisCache;

using namespace std;
using namespace Intrepid;

namespace IntrepidExtendedTypes {
  enum EOperatorExtended { // first 13 simply copied from EOperator
    OPERATOR_VALUE = 0,
    OPERATOR_GRAD,      // 1
    OPERATOR_CURL,      // 2
    OPERATOR_DIV,       // 3
    OPERATOR_D1,        // 4
    OPERATOR_D2,        // 5
    OPERATOR_D3,        // 6
    OPERATOR_D4,        // 7
    OPERATOR_D5,        // 8
    OPERATOR_D6,        // 9
    OPERATOR_D7,        // 10
    OPERATOR_D8,        // 11
    OPERATOR_D9,        // 12
    OPERATOR_D10,       // 13
    OPERATOR_X,         // 14 (pick up where EOperator left off...)
    OPERATOR_Y,         // 15
    OPERATOR_Z,         // 16
    OPERATOR_DX,        // 17
    OPERATOR_DY,        // 18
    OPERATOR_DZ,        // 19
    OPERATOR_CROSS_NORMAL,    // 20
    OPERATOR_DOT_NORMAL,      // 21
    OPERATOR_TIMES_NORMAL,    // 22
    OPERATOR_VECTORIZE_VALUE  // 23
  };

  enum EFunctionSpaceExtended { // all but the last two copied from EFunctionSpace
    FUNCTION_SPACE_HGRAD = 0,
    FUNCTION_SPACE_HCURL,
    FUNCTION_SPACE_HDIV,
    FUNCTION_SPACE_HVOL,
    FUNCTION_SPACE_VECTOR_HGRAD,
    FUNCTION_SPACE_TENSOR_HGRAD,
    FUNCTION_SPACE_ONE,
    CURL_HGRAD_FOR_CONSERVATION
  };
}

using namespace IntrepidExtendedTypes;

class BilinearForm {
public:
  virtual bool trialTestOperator(int trialID, int testID, 
                                 EOperatorExtended &trialOperator, EOperatorExtended &testOperator) { 
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override either trialTestOperator or trialTestOperators!");
  }; // specifies differential operators to apply to trial and test (bool = false if no test-trial term)
  
  virtual void trialTestOperators(int trialID, int testID, 
                                  vector<EOperatorExtended> &trialOps,
                                  vector<EOperatorExtended> &testOps); // default implementation calls trialTestOperator
  
  virtual void applyBilinearFormData(int trialID, int testID,
                                     FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                     const FieldContainer<double> &points) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override either some version of applyBilinearFormData!");
  }
  
  virtual void applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                     int trialID, int testID, int operatorIndex,
                                     const FieldContainer<double> &points); // default implementation calls operatorIndex-less version
  
  virtual void applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                     int trialID, int testID, int operatorIndex,
                                     Teuchos::RCP<BasisCache> basisCache);
  // default implementation calls BasisCache-less version
                           
  const vector< int > & trialIDs();
  const vector< int > & testIDs();
  
  virtual const string & testName(int testID) = 0;
  virtual const string & trialName(int trialID) = 0;
  
  virtual EFunctionSpaceExtended functionSpaceForTest(int testID) = 0;
  virtual EFunctionSpaceExtended functionSpaceForTrial(int trialID) = 0; 
  
  virtual bool isFluxOrTrace(int trialID) = 0;
  
  static void multiplyFCByWeight(FieldContainer<double> &fc, double weight); // belongs elsewhere...
  static const string & operatorName(EOperatorExtended op);
  static int operatorRank(EOperatorExtended op, EFunctionSpaceExtended fs);
  vector<int> trialVolumeIDs();
  vector<int> trialBoundaryIDs();
  
  void printTrialTestInteractions();
protected:
 
  vector< int > _trialIDs, _testIDs;
};
#endif
