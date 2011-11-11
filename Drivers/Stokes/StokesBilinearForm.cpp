
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
 *  StokesBilinearForm.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "StokesBilinearForm.h"

using namespace std;

// trial variable names:
static const string & S_U1_HAT = "\\widehat{u}_1";
static const string & S_U2_HAT = "\\widehat{u}_2";
static const string & S_SIGMA1_N_HAT = "\\widehat{\\sigma}_{2n}";
static const string & S_SIGMA2_N_HAT = "\\widehat{\\sigma}_{1n}";
static const string & S_U_N_HAT = "\\widehat{u}_n";
static const string & S_U1 = "u_1";
static const string & S_U2 = "u_2";
static const string & S_SIGMA_11 = "\\sigma_{11}";
static const string & S_SIGMA_21 = "\\sigma_{21}";
static const string & S_SIGMA_22 = "\\sigma_{22}";
static const string & S_OMEGA = "\\omega";
static const string & S_P = "P";

static const string & S_DEFAULT_TRIAL = "invalid trial";

// test variable names:
static const string & S_Q_1 = "q_1";
static const string & S_Q_2 = "q_2";
static const string & S_V_1 = "v_1";
static const string & S_V_2 = "v_2";
static const string & S_V_3 = "v_3";
static const string & S_ONE = "1";
static const string & S_DEFAULT_TEST = "invalid test";

StokesBilinearForm::StokesBilinearForm() {
  StokesBilinearForm(1.0);
}

StokesBilinearForm::StokesBilinearForm(double mu) {
  _mu = mu;
  
  _testIDs.push_back(Q_1);
  _testIDs.push_back(Q_2);
  _testIDs.push_back(V_1);
  _testIDs.push_back(V_2);
  _testIDs.push_back(V_3);
  
  _trialIDs.push_back(U1_HAT);
  _trialIDs.push_back(U2_HAT);
  _trialIDs.push_back(SIGMA1_N_HAT);
  _trialIDs.push_back(SIGMA2_N_HAT);
  _trialIDs.push_back(U_N_HAT);
  _trialIDs.push_back(U1);
  _trialIDs.push_back(U2);
  _trialIDs.push_back(SIGMA_11);
  _trialIDs.push_back(SIGMA_21);
  _trialIDs.push_back(SIGMA_22);
  _trialIDs.push_back(OMEGA);
  _trialIDs.push_back(P);  
}

const string & StokesBilinearForm::testName(int testID) {
  switch (testID) {
    case Q_1:
      return S_Q_1;
      break;
    case Q_2:
      return S_Q_2;
      break;
    case V_1:
      return S_V_1;
      break;
    case V_2:
      return S_V_2;
      break;
    case V_3:
      return S_V_3;
      break;
    case ONE:
      return S_ONE;
      break;
    default:
      return S_DEFAULT_TEST;
  }
}

const string & StokesBilinearForm::trialName(int trialID) {
  switch(trialID) {
    case U1_HAT:
      return S_U1_HAT;
      break;
    case U2_HAT:
      return S_U2_HAT;
      break;
    case SIGMA1_N_HAT:
      return S_SIGMA1_N_HAT;
      break;
    case SIGMA2_N_HAT:
      return S_SIGMA2_N_HAT;
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
    case SIGMA_11:
      return S_SIGMA_11;
      break;
    case SIGMA_21:
      return S_SIGMA_21;
      break;
    case SIGMA_22:
      return S_SIGMA_22;
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

bool StokesBilinearForm::trialTestOperator(int trialID, int testID, 
                                            EOperatorExtended &trialOperator, EOperatorExtended &testOperator) {
  trialOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
  bool returnValue = false; // unless we specify otherwise, trial and test don't interact
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case U1:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DIV;
          break;
        case SIGMA_11:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_X; // x component of q1 against psi1 (dot product)
          break;
        case SIGMA_21:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_Y; // y component of q1 against psi1 (dot product)
          break;
        case P:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_X;
          break;
        case OMEGA:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_Y;
          break;
        case U1_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DOT_NORMAL;
          break;
        default:
          break;
      }
      break;
    case Q_2:
      switch (trialID) {
        case U2:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DIV;
          break;
        case SIGMA_21:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_X; // x component of q1 against psi1 (dot product)
          break;
        case SIGMA_22:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_Y; // y component of q1 against psi1 (dot product)
          break;
        case P:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_Y;
          break;
        case OMEGA:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_X;
          break;
        case U2_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DOT_NORMAL;
          break;
        default:
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case SIGMA_11:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DX;
          break;
        case SIGMA_21:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DY;
          break;
        case SIGMA1_N_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
          break;
        default:
          break;
      }
      break;
    case V_2:
      switch (trialID) {
        case SIGMA_21:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DX;
          break;
        case SIGMA_22:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DY;
          break;
        case SIGMA2_N_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
          break;
        default:
          break;
      }
      break;
    case V_3:
      switch (trialID) {
        case U1:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DX;
          break;
        case U2:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DY;
          break;
        case U_N_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
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

void StokesBilinearForm::applyBilinearFormData(int trialID, int testID, 
                                               FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                               FieldContainer<double> &points) {
  
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case U1:
          // 1.0 weight -- do nothing
          break;
        case SIGMA_11:
        case SIGMA_21:
        case P:
          multiplyFCByWeight(testValues,1.0/(2.0*_mu));
          break;
        case OMEGA:
          // 1.0 weight -- do nothing
          break;
        case U1_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
      break;
    case Q_2:
      switch (trialID) {
        case U2:
          // 1.0 weight -- do nothing
          break;
        case SIGMA_21:
        case SIGMA_22:
        case P:
          multiplyFCByWeight(testValues,1.0/(2.0*_mu));
          break;
        case OMEGA:
          // -1.0 weight:
          multiplyFCByWeight(testValues,-1.0);
          break;
        case U2_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case SIGMA_11:
        case SIGMA_21:
          // 1.0 weight -- do nothing
          break;
        case SIGMA1_N_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
    case V_2:
      switch (trialID) {
        case SIGMA_21:
        case SIGMA_22:
          // 1.0 weight -- do nothing
          break;
        case SIGMA2_N_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
    case V_3:
      switch (trialID) {
        case U1:
        case U2:
          // 1.0 weight -- do nothing
          break;
        case U_N_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
        default:
          break;
      }
    default:
      break;
  }
}

EFunctionSpaceExtended StokesBilinearForm::functionSpaceForTest(int testID) {
  switch (testID) {
    case Q_1:
    case Q_2:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
      break;
    case V_1:
    case V_2:
    case V_3:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
      break;
    case ONE:
      return IntrepidExtendedTypes::FUNCTION_SPACE_ONE;
    default:
      throw "Error: unknown testID";
  }
}

EFunctionSpaceExtended StokesBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables, and fluxes, are all L2.
  // Traces (like PHI_HAT) are H1 if we use conforming traces.
  // For now, don’t use conforming (like last summer).
  return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
}

bool StokesBilinearForm::isFluxOrTrace(int trialID) {
  if ((U1_HAT==trialID) || (U2_HAT==trialID) 
      || (SIGMA1_N_HAT==trialID) || (SIGMA2_N_HAT==trialID) 
      || (U_N_HAT==trialID) ) {
    return true;
  } else {
    return false;
  }
}