#include "ConfusionBilinearForm.h"

// trial variable names:
static const string & S_SIGMA_1 = "\\sigma_1";
static const string & S_SIGMA_2 = "\\sigma_2";
static const string & S_U = "u";
static const string & S_U_HAT = "\\hat{u }";
static const string & S_BETA_N_U_MINUS_SIGMA_HAT = "\\widehat{\\beta_n u - \\sigma_n}";
static const string & S_DEFAULT_TRIAL = "invalid trial";

// test variable names:
static const string & S_TAU = "\\tau";
static const string & S_V = "v";
static const string & S_DEFAULT_TEST = "invalid test";

ConfusionBilinearForm::ConfusionBilinearForm(double epsilon, double beta_x, double beta_y) {
  _epsilon = epsilon;
  _beta_x = beta_x;
  _beta_y = beta_y;
  
  _testIDs.push_back(TAU);
  _testIDs.push_back(V);
  
  _trialIDs.push_back(U_HAT);
  _trialIDs.push_back(BETA_N_U_MINUS_SIGMA_HAT);
  _trialIDs.push_back(U);
  _trialIDs.push_back(SIGMA_1);
  _trialIDs.push_back(SIGMA_2);
}

double ConfusionBilinearForm::getEpsilon(){
  return _epsilon;
}

void ConfusionBilinearForm::setEpsilon(double newEpsilon){
  _epsilon = newEpsilon;
}

vector<double> ConfusionBilinearForm::getBeta(double x, double y){
  vector<double> beta;
  //  beta.push_back(_beta_x);beta.push_back(_beta_y);
  double xn = (x-.5);
  double yn = (y-.5);
  double r  = 1.0; //sqrt(xn*xn+yn*yn);
  beta.push_back(-yn/r);
  beta.push_back(xn/r);  

  return beta;
}

const string & ConfusionBilinearForm::testName(int testID) {
  switch (testID) {
    case TAU:
      return S_TAU;
      break;
    case V:
      return S_V;
      break;
    default:
      return S_DEFAULT_TEST;
  }
}

const string & ConfusionBilinearForm::trialName(int trialID) {
  switch(trialID) {
    case U:
      return S_U;
      break;
    case U_HAT:
      return S_U_HAT;
      break;
    case SIGMA_1:
      return S_SIGMA_1;
      break;
    case SIGMA_2:
      return S_SIGMA_2;
      break;
    case BETA_N_U_MINUS_SIGMA_HAT:
      return S_BETA_N_U_MINUS_SIGMA_HAT;
      break;
    default:
      return S_DEFAULT_TRIAL;
  }
}

bool ConfusionBilinearForm::trialTestOperator(int trialID, int testID, 
                                            EOperatorExtended &trialOperator, EOperatorExtended &testOperator) {
  // being DPG, trialOperator will always be OPERATOR_VALUE
  trialOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
  bool returnValue = false; // unless we specify otherwise, trial and test don't interact
  switch (testID) {
    case TAU:
      switch (trialID) {
        case U:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DIV;
          break;
        case SIGMA_1:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_X; // x component of tau against sigma (dot product)
          break;
        case SIGMA_2:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_Y; // y component of tau against sigma (dot product)
          break;
        case U_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DOT_NORMAL;
          break;
        default:
          break;
      }
      break;
    case V:
      switch (trialID) {
        case U:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_GRAD;
          break;
        case SIGMA_1:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DX; // dot sigma with grad v
          break;
        case SIGMA_2:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_DY; // dot sigma with grad v
          break;
        case BETA_N_U_MINUS_SIGMA_HAT:
          returnValue = true;
          testOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
          break;
        default:
          break;
      }
    default:
      break;
  }
  return returnValue;    
}

void ConfusionBilinearForm::applyBilinearFormData(int trialID, int testID, 
                                                FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                                FieldContainer<double> &points) {
  switch (testID) {
    case TAU:
      // 1/eps (sigma, tau)_K + (u, div tau)_K - (u_hat, tau_n)_dK
      switch (trialID) {
        case U:
          // do nothing -- testTrialValuesAtPoints has the right values already
          break;
        case SIGMA_1:
          multiplyFCByWeight(testValues,1.0/_epsilon);
          break;
        case SIGMA_2:
          multiplyFCByWeight(testValues,1.0/_epsilon);          
          break;
        case U_HAT:
          // negate
          multiplyFCByWeight(testValues,-1.0);
          break;
      }
      break;
    case V:
      switch(trialID) {
          // (sigma, grad v)_K - (sigma_hat_n, v)_dK - (u, beta dot grad v) + (u_hat * n dot beta, v)_dK
        case U:
          // have grad v -- want grad_v dot beta
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
                             "ConfusionBilinearForm only supports 2 dimensions right now.");
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
		//                testValues(cellIndex,basisOrdinal,ptIndex)  = -_beta_x * testValuesCopy(cellIndex,basisOrdinal,ptIndex,0)
		//                                                            + -_beta_y * testValuesCopy(cellIndex,basisOrdinal,ptIndex,1);
		double x = points(cellIndex,ptIndex,0);
		double y = points(cellIndex,ptIndex,1);
                testValues(cellIndex,basisOrdinal,ptIndex)  = -getBeta(x,y)[0] * testValuesCopy(cellIndex,basisOrdinal,ptIndex,0)
		  + -getBeta(x,y)[1] * testValuesCopy(cellIndex,basisOrdinal,ptIndex,1);

              }
            }
          }
        }
          break;
        case SIGMA_1:
          // do nothing -- testTrialValuesAtPoints has the right values already
          break;
        case SIGMA_2:
          // do nothing -- testTrialValuesAtPoints has the right values already
          break;
        case U_HAT:
          // we have u_hat * n -- need to dot with beta
        { // block off to avoid compiler complaints about adding new variables inside switch/case
          // dimensions of testTrialValuesAtPoints should be (C,F,P,D)
          int numCells = trialValues.dimension(0);
          int basisCardinality = trialValues.dimension(1);
          int numPoints = trialValues.dimension(2);
          int spaceDim = trialValues.dimension(3);
          // because we change dimensions of the values, by dotting with beta, 
          // we'll need to copy the values and resize the original container
          FieldContainer<double> valuesCopy = trialValues;
          trialValues.resize(numCells,basisCardinality,numPoints);
          TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument,
                             "ConfusionBilinearForm only supports 2 dimensions right now.");
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
		double x = points(cellIndex,ptIndex,0);
		double y = points(cellIndex,ptIndex,1);
		
		//                trialValues(cellIndex,basisOrdinal,ptIndex)
		//                = valuesCopy(cellIndex,basisOrdinal,ptIndex,0)*_beta_x
		//                  + valuesCopy(cellIndex,basisOrdinal,ptIndex,1)*_beta_y;
                trialValues(cellIndex,basisOrdinal,ptIndex)
		  = valuesCopy(cellIndex,basisOrdinal,ptIndex,0)*getBeta(x,y)[0]
                  + valuesCopy(cellIndex,basisOrdinal,ptIndex,1)*getBeta(x,y)[1];
              }
            }
          }
        }
          break;
        case BETA_N_U_MINUS_SIGMA_HAT:
          // do nothing (1.0 weight)
          break;
      }
      break;
  }
}

EFunctionSpaceExtended ConfusionBilinearForm::functionSpaceForTest(int testID) {
  switch (testID) {
    case TAU:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
      break;
    case V:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
      break;
    default:
      throw "Error: unknown testID";
  }
}

EFunctionSpaceExtended ConfusionBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables and fluxes are all L2.
  if (trialID != U_HAT) {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
  } else {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  }
}

bool ConfusionBilinearForm::isFluxOrTrace(int trialID) {
  if ((U_HAT==trialID) || (BETA_N_U_MINUS_SIGMA_HAT==trialID)) {
    return true;
  } else {
    return false;
  }
}
