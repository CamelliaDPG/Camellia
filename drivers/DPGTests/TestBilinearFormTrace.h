#ifndef TEST_BILINEAR_FORM_TRACE
#define TEST_BILINEAR_FORM_TRACE

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

/*
 This test bilinear form just has b(u,v) = Int_dK (u_hat, \tau \cdot n),
 where u_hat is a trace (belongs formally to H^(1/2)), and \tau is a test
 function, belonging to H(div,K).
 */

class TestBilinearFormTrace : public BilinearForm {
private:
  static const string & S_TEST;
  static const string & S_TRIAL;
  
public:
  TestBilinearFormTrace() {
    _testIDs.push_back(0);
    _trialIDs.push_back(0);
  }
  
  // implement the virtual methods declared in super:
  const string & testName(int testID) {
    return S_TEST;
  }
  const string & trialName(int trialID) {
    return S_TRIAL;
  }
  
  bool trialTestOperator(int trialID, int testID, 
                         Camellia::EOperator &trialOperator, Camellia::EOperator &testOperator) {
    trialOperator = Camellia::OP_VALUE;
    testOperator  = Camellia::OP_DOT_NORMAL;
    return true;
  }
  
  void applyBilinearFormData(int trialID, int testID,
                           FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                           const FieldContainer<double> &points) {
    // leave values as they are...             
  }
  
  Camellia::EFunctionSpace functionSpaceForTest(int testID) {
    return Camellia::FUNCTION_SPACE_HDIV;
  }
  
  Camellia::EFunctionSpace functionSpaceForTrial(int trialID) {
    return Camellia::FUNCTION_SPACE_HGRAD;
  }
  
  bool isFluxOrTrace(int trialID) {
    return true;
  }
  
};

const string & TestBilinearFormTrace::S_TEST  = "test";
const string & TestBilinearFormTrace::S_TRIAL = "trace";

#endif
