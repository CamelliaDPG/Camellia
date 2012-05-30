
/*
 *  StokesMathBilinearForm.cpp
 *
 *  Created by Nathan Roberts on 3/22/12.
 *
 */

#include "StokesMathBilinearForm.h"

using namespace std;

// trial variable names:
static const string & S_U1_HAT = "\\widehat{u}_1";
static const string & S_U2_HAT = "\\widehat{u}_2";
static const string & S_SIGMA1_N_HAT = "\\widehat{P - \\sigma_{1n}}";
static const string & S_SIGMA2_N_HAT = "\\widehat{P - \\sigma_{2n}}";
static const string & S_U_N_HAT = "\\widehat{u}_n";
static const string & S_U1 = "u_1";
static const string & S_U2 = "u_2";
static const string & S_SIGMA_11 = "\\sigma_{11}";
static const string & S_SIGMA_12 = "\\sigma_{12}";
static const string & S_SIGMA_21 = "\\sigma_{21}";
static const string & S_SIGMA_22 = "\\sigma_{22}";
static const string & S_P = "p";

static const string & S_DEFAULT_TRIAL = "invalid trial";

// test variable names:
static const string & S_Q_1 = "q_1";
static const string & S_Q_2 = "q_2";
static const string & S_V_1 = "v_1";
static const string & S_V_2 = "v_2";
static const string & S_V_3 = "v_3";
static const string & S_DEFAULT_TEST = "invalid test";

StokesMathBilinearForm::StokesMathBilinearForm(double mu) {
  _mu = mu;
  
  _testIDs.push_back(Q_1);
  _testIDs.push_back(Q_2);
  _testIDs.push_back(V_1);
  _testIDs.push_back(V_2);
  _testIDs.push_back(V_3);
  
  _trialIDs.push_back(U1_HAT);
  _trialIDs.push_back(U2_HAT);
  _trialIDs.push_back(SIGMA1_N_HAT); // really P - sigma_1n
  _trialIDs.push_back(SIGMA2_N_HAT); // really P - sigma_2n
  //  _trialIDs.push_back(U_N_HAT);  // U_N_HAT now expressed in terms of u1, u2 hat.
  _trialIDs.push_back(U1);
  _trialIDs.push_back(U2);
  _trialIDs.push_back(SIGMA_11);
  _trialIDs.push_back(SIGMA_12);
  _trialIDs.push_back(SIGMA_21);
  _trialIDs.push_back(SIGMA_22);
  _trialIDs.push_back(P);
}

const string & StokesMathBilinearForm::testName(int testID) {
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
    default:
      return S_DEFAULT_TEST;
  }
}

const string & StokesMathBilinearForm::trialName(int trialID) {
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
    case U1:
      return S_U1;
      break;
    case U2:
      return S_U2;
      break;
    case SIGMA_11:
      return S_SIGMA_11;
      break;
    case SIGMA_12:
      return S_SIGMA_12;
      break;
    case SIGMA_21:
      return S_SIGMA_21;
      break;
    case SIGMA_22:
      return S_SIGMA_22;
      break;
    case P:
      return S_P;
      break;
    default:
      return S_DEFAULT_TRIAL;
  }
}

bool StokesMathBilinearForm::trialTestOperator(int trialID, int testID, 
                                               EOperatorExtended &trialOperator, EOperatorExtended &testOperator) {
  trialOperator = OP_VALUE;
  bool returnValue = false; // unless we specify otherwise, trial and test don't interact
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case U1:
          returnValue = true;
          testOperator = OP_DIV;
          break;
        case SIGMA_11:
          returnValue = true;
          testOperator = OP_X; // x component of q1 against sigma1 (dot product)
          break;
        case SIGMA_12:
          returnValue = true;
          testOperator = OP_Y; // y component of q1 against sigma1 (dot product)
          break;
        case U1_HAT:
          returnValue = true;
          testOperator = OP_DOT_NORMAL;
          break;
      }
      break;
    case Q_2:
      switch (trialID) {
        case U2:
          returnValue = true;
          testOperator = OP_DIV;
          break;
        case SIGMA_21:
          returnValue = true;
          testOperator = OP_X; // x component of q2 against sigma2 (dot product)
          break;
        case SIGMA_22:
          returnValue = true;
          testOperator = OP_Y; // y component of q2 against sigma2 (dot product)
          break;
        case U2_HAT:
          returnValue = true;
          testOperator = OP_DOT_NORMAL;
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case SIGMA_11:
          returnValue = true;
          testOperator = OP_DX;
          break;
        case SIGMA_12:
          returnValue = true;
          testOperator = OP_DY;
          break;
        case P:
          returnValue = true;
          testOperator = OP_DX;
          break;
        case SIGMA1_N_HAT:
          returnValue = true;
          testOperator = OP_VALUE;
          break;
      }
      break;
    case V_2:
      switch (trialID) {
        case SIGMA_21:
          returnValue = true;
          testOperator = OP_DX;
          break;
        case SIGMA_22:
          returnValue = true;
          testOperator = OP_DY;
          break;
        case P:
          returnValue = true;
          testOperator = OP_DY;
          break;
        case SIGMA2_N_HAT:
          returnValue = true;
          testOperator = OP_VALUE;
          break;
      }
      break;
    case V_3:
      switch (trialID) {
        case U1:
          returnValue = true;
          testOperator = OP_DX;
          break;
        case U2:
          returnValue = true;
          testOperator = OP_DY;
          break;
        case U1_HAT:
          returnValue = true;
          // NOTE: right now, OPERATOR_TIMES_NORMAL_i doesn't seem to work for testOperators (because of assumptions built into BasisCache / BasisEvaluation, probably)
          // but conceptually changing the trialOperator makes more sense anyway, so we do that here
          trialOperator = OP_TIMES_NORMAL_X;
          testOperator = OP_VALUE;
          break;
        case U2_HAT:
          // NOTE: right now, OPERATOR_TIMES_NORMAL_i doesn't seem to work for testOperators (because of assumptions built into BasisCache / BasisEvaluation, probably)
          // but conceptually changing the trialOperator makes more sense anyway, so we do that here
          returnValue = true;
          trialOperator = OP_TIMES_NORMAL_Y;
          testOperator = OP_VALUE;
          break;
      }
      break;
  }
  return returnValue;
}

void StokesMathBilinearForm::applyBilinearFormData(int trialID, int testID, 
                                                   FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                                   const FieldContainer<double> &points) {
  
  switch (testID) {
    case Q_1:
      switch (trialID) {
        case U1:
        case SIGMA_11:
        case SIGMA_12:
          // 1.0 weight -- do nothing
          break;
        case U1_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
      }
      break;
    case Q_2:
      switch (trialID) {
        case U2:
        case SIGMA_21:
        case SIGMA_22:
          // 1.0 weight -- do nothing
          break;
        case U2_HAT:
          multiplyFCByWeight(testValues,-1.0);
          break;
      }
      break;
    case V_1:
      switch (trialID) {
        case SIGMA_11:
        case SIGMA_12:
          multiplyFCByWeight(testValues,_mu);
          break;
        case P:
          multiplyFCByWeight(testValues,-1.0);
          break;
        case SIGMA1_N_HAT:
          // 1.0 weight -- do nothing
          break;
      }
      break;
    case V_2:
      switch (trialID) {
        case SIGMA_21:
        case SIGMA_22:
          multiplyFCByWeight(testValues,_mu);
          break;
        case P:
          multiplyFCByWeight(testValues,-1.0);
          break;
        case SIGMA2_N_HAT:
          // 1.0 weight -- do nothing
          break;
      }
      break;
    case V_3:
      switch (trialID) {
        case U1:
        case U2:
          multiplyFCByWeight(testValues,-1.0);
          break;
        case U1_HAT:
        case U2_HAT:
          // 1.0 weight -- do nothing
          break;
      }
      break;
  }
}

EFunctionSpaceExtended StokesMathBilinearForm::functionSpaceForTest(int testID) {
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
    default:
      throw "Error: unknown testID";
  }
}

EFunctionSpaceExtended StokesMathBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables, and fluxes, are all L2.
  // Traces (like U_i_HAT) are H1 if we use conforming traces.
  if ( (trialID == U1_HAT) || (trialID == U2_HAT) ) {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  }
  return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
}

bool StokesMathBilinearForm::isFluxOrTrace(int trialID) {
  if ((U1_HAT==trialID) || (U2_HAT==trialID) 
      || (SIGMA1_N_HAT==trialID) || (SIGMA2_N_HAT==trialID) ) {
    return true;
  } else {
    return false;
  }
}