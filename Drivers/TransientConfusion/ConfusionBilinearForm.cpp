#include "ConfusionBilinearForm.h"
#include <vector>
using namespace std;

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

ConfusionBilinearForm::ConfusionBilinearForm(double epsilon, double beta_x, double beta_y, double dt) {
  _epsilon = epsilon;
  _beta_x = beta_x;
  _beta_y = beta_y;
  _dt     = dt;
  
  _testIDs.push_back(TAU);
  _testIDs.push_back(V);
  
  _trialIDs.push_back(U_HAT);
  _trialIDs.push_back(BETA_N_U_MINUS_SIGMA_HAT);
  _trialIDs.push_back(U);
  _trialIDs.push_back(SIGMA_1);
  _trialIDs.push_back(SIGMA_2);

  // define multiple
  _uvTestOperators.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);
  _uvTestOperators.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);
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

void ConfusionBilinearForm::trialTestOperators(int trialID, int testID, 
					       vector<EOperatorExtended> &trialOperators,
					       vector<EOperatorExtended> &testOperators){

  trialOperators.clear();
  testOperators.clear();
  // being DPG, trialOperator will always be OPERATOR_VALUE 
  trialOperators.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);
  
  bool returnValue = false; // unless we specify otherwise, trial and test don't interact
  switch (testID) {
  case TAU:
    switch (trialID) {
    case U:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_DIV);
      break;
    case SIGMA_1:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_X); // x component of tau against sigma (dot product)
      break;
    case SIGMA_2:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_Y); // y component of tau against sigma (dot product)
      break;
    case U_HAT:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_DOT_NORMAL);
      break;
    default:
      trialOperators.clear(); //necessary to have empty pair if no testOperator def'd
      break;
    }
    break;
  case V:
    switch (trialID) {
    case U:
      returnValue = true;
      testOperators = _uvTestOperators;
      break;
    case SIGMA_1:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_DX); // dot sigma with grad v
      break;
    case SIGMA_2:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_DY); // dot sigma with grad v
      break;
    case BETA_N_U_MINUS_SIGMA_HAT:
      returnValue = true;
      testOperators.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);
      break;
    default:
      trialOperators.clear(); //necessary to have empty pair if no testOperator def'd
      break;
    }
  default:
    // do not clear operator list here - if we break out of the inner switch loop, we may have added stuff to the operator lists. WE DON'T WANT TO REMOVE THAT.
    break;
  }
  //  return returnValue;    
}

//void ConfusionBilinearForm::applyBilinearFormData(int trialID, int testID, 
//						  FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
//						  FieldContainer<double> &points) {

// now use the operator-indexed one (for multiple operators)
void ConfusionBilinearForm::applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
						  int trialID, int testID, int operatorIndex,
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
      if (_uvTestOperators[operatorIndex]==IntrepidExtendedTypes::OPERATOR_GRAD)
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
	      testValues(cellIndex,basisOrdinal,ptIndex)  = -_beta_x * testValuesCopy(cellIndex,basisOrdinal,ptIndex,0)
		+ -_beta_y * testValuesCopy(cellIndex,basisOrdinal,ptIndex,1);
	    }
	  }
	}
      }
      else if(_uvTestOperators[operatorIndex]==IntrepidExtendedTypes::OPERATOR_VALUE){       // also now have a reaction term. 
	multiplyFCByWeight(testValues,_dt);
      }
      else{
	cout << "Getting an unexpected operator for transient confusion!" << endl;
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
	      trialValues(cellIndex,basisOrdinal,ptIndex)
                = valuesCopy(cellIndex,basisOrdinal,ptIndex,0)*_beta_x
		+ valuesCopy(cellIndex,basisOrdinal,ptIndex,1)*_beta_y;
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
