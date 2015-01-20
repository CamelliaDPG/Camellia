#include "TestBilinearFormDx.h"

TestBilinearFormDx::TestBilinearFormDx() : BF(true) { // true: legacy subclass
  _testIDs.push_back(0);
  _trialIDs.push_back(0);
}

// implement the virtual methods declared in super:
const string & TestBilinearFormDx::testName(int testID) {
  const static string S_TEST = "test";
  return S_TEST;
}
const string & TestBilinearFormDx::trialName(int trialID) {
  const static string S_TRIAL = "trial";
  return S_TRIAL;
}

bool TestBilinearFormDx::trialTestOperator(int trialID, int testID, 
                                           Camellia::EOperator &trialOperator,
                                           Camellia::EOperator &testOperator) {
  trialOperator = OP_DX;
  testOperator  = OP_DX;
  return true;
}

void TestBilinearFormDx::applyBilinearFormData(int trialID, int testID,
                                               FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                               const FieldContainer<double> &points) {
  // leave values as they are...             
}

Camellia::EFunctionSpace TestBilinearFormDx::functionSpaceForTest(int testID) {
  return Camellia::FUNCTION_SPACE_HGRAD;
}

Camellia::EFunctionSpace TestBilinearFormDx::functionSpaceForTrial(int trialID) {
  return Camellia::FUNCTION_SPACE_HGRAD;
}

bool TestBilinearFormDx::isFluxOrTrace(int trialID) {
  return false;
}