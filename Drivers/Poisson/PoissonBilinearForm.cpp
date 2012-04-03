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

#include "PoissonBilinearForm.h"

// trial variable names:
static const string & S_PHI = "\\phi";
static const string & S_PSI_1 = "\\psi_1";
static const string & S_PSI_2 = "\\psi_2";
static const string & S_PHI_HAT = "\\hat{\\phi}";
static const string & S_PSI_HAT_N ="\\hat{\\psi}_n";
static const string & S_DEFAULT_TRIAL = "invalid trial";

// test variable names:
static const string & S_Q_1 = "q_1";
static const string & S_V_1 = "v_1";
static const string & S_DEFAULT_TEST = "invalid test";

PoissonBilinearForm::PoissonBilinearForm() {
  _testIDs.push_back(Q_1);
  _testIDs.push_back(V_1);
  
  _trialIDs.push_back(PHI_HAT);
  _trialIDs.push_back(PSI_HAT_N);
  _trialIDs.push_back(PHI);
  _trialIDs.push_back(PSI_1);
  _trialIDs.push_back(PSI_2);
}

const string & PoissonBilinearForm::testName(int testID) {
  switch (testID) {
    case Q_1:
      return S_Q_1;
    break;
    case V_1:
      return S_V_1;
    break;
    default:
      return S_DEFAULT_TEST;
  }
}

const string & PoissonBilinearForm::trialName(int trialID) {
  switch(trialID) {
    case PHI:
      return S_PHI;
    break;
    case PSI_1:
      return S_PSI_1;
    break;
    case PSI_2:
      return S_PSI_2;
    break;
    case PHI_HAT:
      return S_PHI_HAT;
    break;
    case PSI_HAT_N:
      return S_PSI_HAT_N;
    break;
    default:
      return S_DEFAULT_TRIAL;
  }
}

bool PoissonBilinearForm::trialTestOperator(int trialID, int testID, 
                         EOperatorExtended &trialOperator, EOperatorExtended &testOperator) {
  // being DPG, trialOperator will always be OP_VALUE
  trialOperator = IntrepidExtendedTypes::OP_VALUE;
  bool returnValue = false; // unless we specify otherwise, trial and test don't interact
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case PSI_1:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_X; // x component of q1 against psi1 (dot product)
          break;
        case PSI_2:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_Y; // y component of q1 against psi1 (dot product)
          break;
        case PHI:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_DIV;
          break;
        case PHI_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_DOT_NORMAL;
          break;
        default:
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case PSI_1:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_DX;
          break;
        case PSI_2:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_DY;
          break;
        case PSI_HAT_N:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OP_VALUE;
          break;
        default:
          break;
      }
    default:
      break;
  }
  return returnValue;    
}

void PoissonBilinearForm::applyBilinearFormData(int trialID, int testID, 
                           FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                           const FieldContainer<double> &points) {
  switch (testID) {
    case Q_1:
      // - (phi, div q1)_K + (phi_hat, q_1n)_dK - (psi, q1)
      switch (trialID) {
        case PHI:
          // negate
          multiplyFCByWeight(testValues,-1.0);
        break;
        case PSI_1:
          // negate
          multiplyFCByWeight(testValues,-1.0);
        break;
        case PSI_2:
          // negate
          multiplyFCByWeight(testValues,-1.0);         
        break;
        case PHI_HAT:
          // do nothing -- testTrialValuesAtPoints already has the right values 
        break;
      }
    break;
    case V_1:
      switch(trialID) {
      // -(psi, grad v1)_K + (psi_hat_n, v1)_dK
        case PHI:
          throw "Error: no (v1, phi) term";
        break;
        case PSI_1:
          // negate
          multiplyFCByWeight(testValues,-1.0);
        break;
        case PSI_2:
          // negate
          multiplyFCByWeight(testValues,-1.0);
        break;
        case PSI_HAT_N:
          // do nothing -- testTrialValuesAtPoints already has the right values 
        break;
      }
    break;
  }
}

EFunctionSpaceExtended PoissonBilinearForm::functionSpaceForTest(int testID) {
  switch (testID) {
    case Q_1:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
    break;
    case V_1:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    break;
    default:
      throw "Error: unknown testID";
  }
}

EFunctionSpaceExtended PoissonBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables and fluxes are all L2.
  if (trialID != PHI_HAT) {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
  } else {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  }
}

bool PoissonBilinearForm::isFluxOrTrace(int trialID) {
  if ((PHI_HAT==trialID) || (PSI_HAT_N==trialID)) {
    return true;
  } else {
    return false;
  }
}
