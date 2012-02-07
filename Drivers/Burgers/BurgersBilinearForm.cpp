#include "BurgersBilinearForm.h"
#include "Solution.h"

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

BurgersBilinearForm::BurgersBilinearForm(double epsilon) {
  _epsilon = epsilon;

  _backgroundFlow = Teuchos::rcp((Solution*)NULL); // initialize null pointer for now
  
  _testIDs.push_back(TAU);
  _testIDs.push_back(V);
  
  _trialIDs.push_back(U_HAT);
  _trialIDs.push_back(BETA_N_U_MINUS_SIGMA_HAT);
  _trialIDs.push_back(U);
  _trialIDs.push_back(SIGMA_1);
  _trialIDs.push_back(SIGMA_2);
}

void BurgersBilinearForm::setBackgroundFlow(Teuchos::RCP<Solution> flow){
  _backgroundFlow = flow;
}

double BurgersBilinearForm::getEpsilon(){
  return _epsilon;
}

void BurgersBilinearForm::setEpsilon(double newEpsilon){
  _epsilon = newEpsilon;
}

FieldContainer<double> BurgersBilinearForm::getBeta(FieldContainer<double> physicalPoints){
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  FieldContainer<double> allPoints = physicalPoints;
  allPoints.resize(numCells*numPoints,spaceDim);

  FieldContainer<double> values(numCells*numPoints);
  _backgroundFlow->solutionValues(values, BurgersBilinearForm::U, allPoints);  
  values.resize(numCells,numPoints);
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

vector<double> BurgersBilinearForm::getBeta(double x, double y){
  FieldContainer<double> values(1); // only one point on one cell
  FieldContainer<double> point(1,2); // one point, one cell, 2 space dim
  point(0,0) = x;
  point(0,1) = y;
  
  _backgroundFlow->solutionValues(values, BurgersBilinearForm::U, point);
  double u = values(0);

  vector<double> beta;  
  beta.push_back(u);beta.push_back(1.0); // (u*du)_x + (du)_y 

  return beta;
}

const string & BurgersBilinearForm::testName(int testID) {
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

const string & BurgersBilinearForm::trialName(int trialID) {
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

bool BurgersBilinearForm::trialTestOperator(int trialID, int testID, 
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

void BurgersBilinearForm::applyBilinearFormData(int trialID, int testID, 
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
                             "BurgersBilinearForm only supports 2 dimensions right now.");

	  FieldContainer<double> beta = getBeta(points);
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
		double x = points(cellIndex,ptIndex,0);
		double y = points(cellIndex,ptIndex,1);
		//                testValues(cellIndex,basisOrdinal,ptIndex)  = -getBeta(x,y)[0] * testValuesCopy(cellIndex,basisOrdinal,ptIndex,0) + -getBeta(x,y)[1] * testValuesCopy(cellIndex,basisOrdinal,ptIndex,1);
                testValues(cellIndex,basisOrdinal,ptIndex)  = -beta(cellIndex,ptIndex,0) * testValuesCopy(cellIndex,basisOrdinal,ptIndex,0) + -beta(cellIndex,ptIndex,1) * testValuesCopy(cellIndex,basisOrdinal,ptIndex,1);

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
                             "BurgersBilinearForm only supports 2 dimensions right now.");

	  FieldContainer<double> beta = getBeta(points);
	  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
	    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
	      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                trialValues(cellIndex,basisOrdinal,ptIndex)
		  = valuesCopy(cellIndex,basisOrdinal,ptIndex,0)*beta(cellIndex,ptIndex,0)
		  + valuesCopy(cellIndex,basisOrdinal,ptIndex,1)*beta(cellIndex,ptIndex,1);
		
		/*
		double x = points(cellIndex,ptIndex,0);
		double y = points(cellIndex,ptIndex,1);

                trialValues(cellIndex,basisOrdinal,ptIndex)
		  = valuesCopy(cellIndex,basisOrdinal,ptIndex,0)*getBeta(x,y)[0]
                  + valuesCopy(cellIndex,basisOrdinal,ptIndex,1)*getBeta(x,y)[1];
		*/
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

EFunctionSpaceExtended BurgersBilinearForm::functionSpaceForTest(int testID) {
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

EFunctionSpaceExtended BurgersBilinearForm::functionSpaceForTrial(int trialID) {
  // Field variables and fluxes are all L2.
  if (trialID != U_HAT) {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
  } else {
    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  }
}

bool BurgersBilinearForm::isFluxOrTrace(int trialID) {
  if ((U_HAT==trialID) || (BETA_N_U_MINUS_SIGMA_HAT==trialID)) {
    return true;
  } else {
    return false;
  }
}
