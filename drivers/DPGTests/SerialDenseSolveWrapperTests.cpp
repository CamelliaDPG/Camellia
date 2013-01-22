//
//  SerialDenseSolveWrapperTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/19/13.
//
//

#include "SerialDenseSolveWrapperTests.h"
void SerialDenseSolveWrapperTests::setup() {
  
}

void SerialDenseSolveWrapperTests::teardown() {
  
}

void SerialDenseSolveWrapperTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testSimpleSolve()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool SerialDenseSolveWrapperTests::testSimpleSolve() {
  bool success = true;
  
  double tol = 1e-16;
  FieldContainer<double> A(2,2);
  FieldContainer<double> b(2);
  FieldContainer<double> x(2);
  
  FieldContainer<double> expected_x(2);
  
  // A = [ 1 1 ]
  //     [ 0 1 ]
  // x = [ 3 ]
  //     [ 4 ]
  // b = [ 7 ]
  //     [ 4 ]
  A(0,0) = 1.0;
  A(0,1) = 1.0;
  A(1,0) = 0.0;
  A(1,1) = 1.0;
  expected_x(0) = 3.0;
  expected_x(1) = 4.0;
  b(0) = 7.0;
  b(1) = 4.0;
  
  SerialDenseSolveWrapper::solveSystem(x, A, b);
  
  double maxDiff = 0;
  if (! fcsAgree(x, expected_x, tol, maxDiff)) {
    cout << "testSimpleSolve() failed: maxDiff " << maxDiff << " exceeds tol " << tol << endl;
    cout << "x: " << endl << x;
    cout << "expected_x: " << endl << expected_x;
  }
  
  return success;
}

std::string SerialDenseSolveWrapperTests::testSuiteName() {
  return "SerialDenseSolveWrapperTests";
}