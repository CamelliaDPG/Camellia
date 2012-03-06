/*
 *  NavierStokesBilinearForm.cpp
 *
 *  Created by Nathan Roberts on 3/6/12.
 *
 */

#include "NavierStokesBilinearForm.h"

using namespace std;

// trial variable names:
static const string & S_U1_HAT = "\\widehat{u}_1";
static const string & S_U2_HAT = "\\widehat{u}_2";
static const string & S_T_HAT = "\\widehat{T}";
static const string & S_F1_N_HAT = "\\widehat{F}_{1n}";
static const string & S_F2_N_HAT = "\\widehat{F}_{1n}";
static const string & S_F3_N_HAT = "\\widehat{F}_{1n}";
static const string & S_F4_N_HAT = "\\widehat{F}_{1n}";
static const string & S_U1 = "u_1";
static const string & S_U2 = "u_2";
static const string & S_RHO = "\\rho";
static const string & S_T = "T";
static const string & S_TAU_11 = "\\tau_{11}";
static const string & S_TAU_21 = "\\tau_{21}";
static const string & S_TAU_22 = "\\tau_{22}";
static const string & S_Q_1 = "\\q_{1}";
static const string & S_Q_2 = "\\q_{2}";
static const string & S_OMEGA = "\\omega";

static const string & S_DEFAULT_TRIAL = "invalid trial";

// test variable names:
static const string & S_Q_1 = "q_1";
static const string & S_Q_2 = "q_2";
static const string & S_Q_3 = "q_3";
static const string & S_V_1 = "v_1";
static const string & S_V_2 = "v_2";
static const string & S_V_3 = "v_3";
static const string & S_V_4 = "v_4";
static const string & S_DEFAULT_TEST = "invalid test";

NavierStokesBilinearForm::NavierStokesBilinearForm(double Reyn, double Mach) {
  _Reyn = Reyn;
  _Mach = Mach;
  
  _testIDs.push_back(Q_1);
  _testIDs.push_back(Q_2);
  _testIDs.push_back(Q_3);
  _testIDs.push_back(V_1);
  _testIDs.push_back(V_2);
  _testIDs.push_back(V_3);
  _testIDs.push_back(V_4);
  
  _trialIDs.push_back(U1_HAT);
  _trialIDs.push_back(U2_HAT);
  _trialIDs.push_back(T_HAT);
  _trialIDs.push_back(F1_N);
  _trialIDs.push_back(F2_N);
  _trialIDs.push_back(F3_N);
  _trialIDs.push_back(F4_N);
  
  _trialIDs.push_back(U1);
  _trialIDs.push_back(U2);
  _trialIDs.push_back(RHO);
  _trialIDs.push_back(T);
  _trialIDs.push_back(TAU_11);
  _trialIDs.push_back(TAU_21);
  _trialIDs.push_back(TAU_22);
  _trialIDs.push_back(Q_1);
  _trialIDs.push_back(Q_2);
  _trialIDs.push_back(OMEGA);
  
  pair<int, int> trialTest;
  // first equation, x part:
  trialTest = make_pair(RHO,V_1);
  _backFlowInteractions_x[trialTest].push_back(U_1);
  trialTest = make_pair(U_1,V_1);
  _backFlowInteractions_x[trialTest].push_back(RHO);
  // first equation, y part:
  trialTest = make_pair(RHO,V_1);
  _backFlowInteractions_y[trialTest].push_back(U_2);
  trialTest = make_pair(U_2,V_1);
  _backFlowInteractions_y[trialTest].push_back(RHO);
  
  // second equation, 
}

const string & NavierStokesBilinearForm::testName(int testID) {
  switch (testID) {
    case Q_1:
      return S_Q_1;
      break;
    case Q_2:
      return S_Q_2;
      break;
      break;
    case Q_3:
      return S_Q_3;
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
    case V_4:
      return S_V_4;
      break;
    default:
      return S_DEFAULT_TEST;
  }
}

const string & NavierStokesBilinearForm::trialName(int trialID) {
  switch(trialID) {
    case U1_HAT:
      return S_U1_HAT;
      break;
    case U2_HAT:
      return S_U2_HAT;
      break;
    case T_HAT:
      return S_T_HAT;
      break;
    case F1_N:
      return S_F1_N;
      break;
    case F2_N:
      return S_F2_N;
      break;
    case F3_N:
      return S_F3_N;
      break;
    case F4_N:
      return S_F4_N;
      break;
    case U1:
      return S_U1;
      break;
    case U2:
      return S_U2;
      break;
    case RHO:
      return S_RHO;
      break;
    case T:
      return S_T;
      break;
    case TAU_11:
      return S_TAU_11;
      break;
    case TAU_21:
      return S_TAU_21;
      break;
    case TAU_22:
      return S_TAU_22;
      break;
    case Q_1:
      return S_Q_1;
      break;
    case Q_2:
      return S_Q_2;
      break;
    case OMEGA:
      return S_OMEGA;
      break;
    default:
      return S_DEFAULT_TRIAL;
  }
}

bool NavierStokesBilinearForm::trialTestOperator(int trialID, int testID, 
                                                 EOperatorExtended &trialOperator, EOperatorExtended &testOperator) {
  if ( isFluxOrTrace(trialID) {
    testOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
    switch (testID) {
      case V_1:
        return (trialID == F1_N);
      case V_2:
        return (trialID == F2_N);
      case V_3:
        return (trialID == F3_N);
      case V_4:
        return (trialID == F4_N);
    }
    testOperator = IntrepidExtendedTypes::OPERATOR_DOT_NORMAL;
    switch (testID) {
      case Q_1:
        return (trialID == U1_HAT);
      case Q_2:
        return (trialID == U2_HAT);
      case Q_3:
        return (trialID == T_HAT);
    }
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "unknown flux or trace");
  }
  
  trialOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
  pair<int, int> key = make_pair(trialID,testID);
  if (   (_backFlowInteractions_x.find(key) == _backFlowInteractions_x.end() )
      && (_backFlowInteractions_y.find(key) == _backFlowInteractions_y.end() ) ) {
    return false;
  }
  
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
    case V_2:
    case V_3:
    case V_4:
      returnValue = true;
      testOperator = IntrepidExtendedTypes::OPERATOR_GRAD;
      break;
    default:
      break;
  }
  return returnValue;
}

void NavierStokesBilinearForm::applyBilinearFormData(int trialID, int testID, 
                                                     FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                                     const FieldContainer<double> &points) {
  
  // TODO: handle the traces & fluxes separately
  
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
    case V_2:
    case V_3:
    case V_4:
      // have grad v -- want grad_v dot something else that depends on background flow
    { // block off to avoid compiler complaints about adding new variables inside switch/case
      // dimensions of testTrialValuesAtPoints should be (C,F,P,D)
      int numCells = testValues.dimension(0);
      int basisCardinality = testValues.dimension(1);
      int numPoints = testValues.dimension(2);
      int spaceDim = testValues.dimension(3);
      // because we change dimensions of the values, by dotting with beta, 
      // we'll need to copy the values and resize the original container
      FieldContainer<double> testValuesCopy = testValues;
      testValues.resize(numCells,basisCardinality,numPoints);
      TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument,
                         "NavierStokesBilinearForm only supports 2 dimensions right now.");
      
      FieldContainer<double> beta = getValuesToDot(trialID,testID,basisCache);
      FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            double x = points(cellIndex,ptIndex,0);
            double y = points(cellIndex,ptIndex,1);
            testValues(cellIndex,basisOrdinal,ptIndex)  = -beta(cellIndex,ptIndex,0) * testValuesCopy(cellIndex,basisOrdinal,ptIndex,0) + -beta(cellIndex,ptIndex,1) * testValuesCopy(cellIndex,basisOrdinal,ptIndex,1);
            
          }
        }
      }
    }
  }
}

FieldContainer<double> NavierStokesBilinearForm::getValuesToDot(int trialID, int testID, Teuchos::RCP<BasisCache> basisCache) {
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  FieldContainer<double> values(numCells,numPoints);
  _backgroundFlow->solutionValues(values, BurgersBilinearForm::U, basisCache);  
  FieldContainer<double> beta(numCells,numPoints,spaceDim);
  for (int cellIndex=0;cellIndex<numCells;cellIndex++){
    for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
      beta(cellIndex,ptIndex,0) = 1.0*values(cellIndex,ptIndex);
      //      beta(cellIndex,ptIndex,0) = 1.0;
      beta(cellIndex,ptIndex,1) = 1.0;
    }
  }
  return beta;
}
      
EFunctionSpaceExtended NavierStokesBilinearForm::functionSpaceForTest(int testID) {
  switch (testID) {
    case Q_1:
    case Q_2:
    case Q_3:
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

EFunctionSpaceExtended NavierStokesBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables, and fluxes, are all L2.
  // Traces (like PHI_HAT) are H1 if we use conforming traces.
  // For now, don’t use conforming (like last summer).
  return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
}

bool NavierStokesBilinearForm::isFluxOrTrace(int trialID) {
  if ((U1_HAT==trialID) || (U2_HAT==trialID) 
      || (SIGMA1_N_HAT==trialID) || (SIGMA2_N_HAT==trialID) 
      || (U_N_HAT==trialID) ) {
    return true;
  } else {
    return false;
  }
}