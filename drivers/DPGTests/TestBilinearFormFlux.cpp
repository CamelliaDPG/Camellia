#include "TestBilinearFormFlux.h"

/*
 This test bilinear form just has b(u,v) = Int_dK (u_hat, v ),
 where u_hat is a trace (belongs formally to H^(1/2)), and \tau is a test
 function, belonging to H(div,K).
 */

TestBilinearFormFlux::TestBilinearFormFlux() {
  _testIDs.push_back(0);
  _trialIDs.push_back(0);
}
  
  // implement the virtual methods declared in super:
const string & TestBilinearFormFlux::testName(int testID) {
    const static string S_TEST = "test";
    return S_TEST;
  }
const string & TestBilinearFormFlux::trialName(int trialID) {
  const static string S_TRIAL = "flux";
  return S_TRIAL;
}
  
bool TestBilinearFormFlux::trialTestOperator(int trialID, int testID, 
                                             IntrepidExtendedTypes::EOperator &trialOperator,
                                             IntrepidExtendedTypes::EOperator &testOperator) {
  trialOperator = OP_VALUE;
  testOperator  = OP_VALUE;
  return true;
}
  
void TestBilinearFormFlux::applyBilinearFormData(int trialID, int testID,
                                                 FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                                 const FieldContainer<double> &points) {
    // leave values as they are...             
}
  
IntrepidExtendedTypes::EFunctionSpace TestBilinearFormFlux::functionSpaceForTest(int testID) {
  return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
}
  
IntrepidExtendedTypes::EFunctionSpace TestBilinearFormFlux::functionSpaceForTrial(int trialID) {
  return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
}
  
bool TestBilinearFormFlux::isFluxOrTrace(int trialID) {
  return true;
}
