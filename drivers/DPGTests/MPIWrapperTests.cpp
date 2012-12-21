//
//  MPIWrapperTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#include "MPIWrapperTests.h"

#include "Teuchos_GlobalMPISession.hpp"

void MPIWrapperTests::setup() {
  
}

void MPIWrapperTests::teardown() {
  
}

void MPIWrapperTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testSimpleSum()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testentryWiseSum()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool MPIWrapperTests::testentryWiseSum() {
  bool success = true;
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int rank = Teuchos::GlobalMPISession::getRank();
  
  FieldContainer<double> expectedValues(2);
  for (int i=0; i<numProcs; i++) {
    expectedValues[0] += i*i;
    expectedValues[1] += 1;
  }
  FieldContainer<double> values(2);
  values[0] = rank*rank;
  values[1] = 1;
  
  MPIWrapper::entryWiseSum(values);
  double tol = 1e-16;
  
  double maxDiff = 0;
  if (! fcsAgree(values, expectedValues, tol, maxDiff) ) {
    success = false;
    cout << "MPIWrapperTests::testentryWiseSum() failed with maxDiff " << maxDiff << endl;
  }
  return success;
}

bool MPIWrapperTests::testSimpleSum() {
  bool success = true;
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int rank = Teuchos::GlobalMPISession::getRank();
  
  double expectedValue = 0;
  for (int i=0; i<numProcs; i++) {
    expectedValue += i;
  }
  FieldContainer<double> values(1);
  values[0] = rank;
  
  double sum = MPIWrapper::sum(values);
  double tol = 1e-16;
  if (abs(sum-expectedValue) > tol) {
    success = false;
    cout << "MPIWrapperTests::testSimpleSum() failed: expected " << expectedValue << " but got " << sum << endl;
  }
  return success;
}

std::string MPIWrapperTests::testSuiteName() {
  return "MPIWrapperTests";
}