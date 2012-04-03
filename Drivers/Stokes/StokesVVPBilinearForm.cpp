
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
 *  StokesVVPBilinearForm.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "StokesVVPBilinearForm.h"

using namespace std;

// trial variable names:
static const string & S_OMEGA_HAT = "\\widehat{\\omega}";
static const string & S_P_HAT = "\\widehat{P}";
static const string & S_U_CROSS_N_HAT = "\\widehat{u}_{\\times n}";
static const string & S_U_N_HAT = "\\widehat{u}_n";
static const string & S_U1 = "u_1";
static const string & S_U2 = "u_2";
static const string & S_OMEGA = "\\omega";
static const string & S_P = "P";

static const string & S_DEFAULT_TRIAL = "invalid trial";

// test variable names:
static const string & S_Q_1 = "q_1";
static const string & S_V_1 = "v_1";
static const string & S_V_2 = "v_2";
static const string & S_DEFAULT_TEST = "invalid test";

StokesVVPBilinearForm::StokesVVPBilinearForm() {
  StokesVVPBilinearForm(1.0);
}

StokesVVPBilinearForm::StokesVVPBilinearForm(double mu) {
  _mu = mu;
  
  _testIDs.push_back(Q_1);
  _testIDs.push_back(V_1);
  _testIDs.push_back(V_2);
  
  _trialIDs.push_back(OMEGA_HAT);
  _trialIDs.push_back(P_HAT);
  _trialIDs.push_back(U_CROSS_N_HAT);
  _trialIDs.push_back(U_N_HAT);
  _trialIDs.push_back(P);
  _trialIDs.push_back(OMEGA);
  _trialIDs.push_back(U1);
  _trialIDs.push_back(U2);
}

const string & StokesVVPBilinearForm::testName(int testID) {
  switch (testID) {
    case Q_1:
      return S_Q_1;
      break;
    case V_1:
      return S_V_1;
      break;
    case V_2:
      return S_V_2;
      break;
    default:
      return S_DEFAULT_TEST;
  }
}

const string & StokesVVPBilinearForm::trialName(int trialID) {
  switch(trialID) {
    case OMEGA_HAT:
      return S_OMEGA_HAT;
      break;
    case P_HAT:
      return S_P_HAT;
      break;
    case U_CROSS_N_HAT:
      return S_U_CROSS_N_HAT;
      break;
    case U_N_HAT:
      return S_U_N_HAT;
      break;
    case U1:
      return S_U1;
      break;
    case U2:
      return S_U2;
      break;
    case OMEGA:
      return S_OMEGA;
      break;
    case P:
      return S_P;
      break;
    default:
      return S_DEFAULT_TRIAL;
  }
}

bool StokesVVPBilinearForm::trialTestOperator(int trialID, int testID, 
                                            EOperatorExtended &trialOperator, EOperatorExtended &testOperator) {
  trialOperator = OP_VALUE;
  bool returnValue = false; // unless we specify otherwise, trial and test don't interact
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case OMEGA_HAT:
          returnValue = true;
          testOperator = OP_CROSS_NORMAL;
          break;
        case P_HAT:
          returnValue = true;
          testOperator = OP_DOT_NORMAL;
          break;
        case OMEGA:
          returnValue = true;
          testOperator = OP_CURL; // with -1.0 weight
          break;
        case P:
          returnValue = true;
          testOperator = OP_DIV; // with -1.0 weight
          break;
        default:
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case U_CROSS_N_HAT:
          returnValue = true;
          testOperator = OP_VALUE;
          break;
        case U1:
          returnValue = true;
          testOperator = OP_DY; // with -1.0 weight
          break;
        case U2:
          returnValue = true;
          testOperator = OP_DX;
          break;
        case OMEGA:
          returnValue = true;
          testOperator = OP_VALUE;
          break;
        default:
          break;
      }
      break;
    case V_2:
      switch (trialID) {
        case U_N_HAT:
          returnValue = true;
          testOperator = OP_VALUE;
          break;
        case U1:
          returnValue = true;
          testOperator = OP_DX; // -1.0 weight
          break;
        case U2:
          returnValue = true;
          testOperator = OP_DY; // -1.0 weight
          break;
        default:
          break;
      }
      break;      
    default:
      break;
  }
  return returnValue;
}

void StokesVVPBilinearForm::applyBilinearFormData(int trialID, int testID, 
                                                  FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                                  const FieldContainer<double> &points) {
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case OMEGA_HAT:
        case P_HAT:
          // 1.0 weight -- do nothing
          break;
        case OMEGA:
        case P:
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case U1:
          multiplyFCByWeight(testValues,-1.0);
          break;
        case U_CROSS_N_HAT:
        case U2:
        case OMEGA:
          break;
        default:
          break;
      }
      break;
    case V_2:
      switch (trialID) {
        case U_N_HAT:
          // 1.0 weight -- do nothing
          break;
        case U1:
        case U2:
          // -1.0 weight
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
      break;      
    default:
      break;
  }
}

EFunctionSpaceExtended StokesVVPBilinearForm::functionSpaceForTest(int testID) {
  switch (testID) {
    case Q_1:
      return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD;
      break;
    case V_1:
    case V_2:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
      break;
    default:
      throw "Error: unknown testID";
  }
}

EFunctionSpaceExtended StokesVVPBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables and fluxes are all L2.
  // Traces (like PHI_HAT) are H1 if we use conforming traces.
  if ( (trialID == P_HAT) || (trialID == OMEGA_HAT) ) { // traces
    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  } else {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
  }
}

bool StokesVVPBilinearForm::isFluxOrTrace(int trialID) { // return true for flux or trace
  if ((OMEGA_HAT==trialID) || (P_HAT==trialID) 
      || (U_N_HAT==trialID) || (U_CROSS_N_HAT==trialID) ) {
    return true;
  } else {
    return false;
  }
}