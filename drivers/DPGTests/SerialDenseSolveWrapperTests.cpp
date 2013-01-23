//
//  SerialDenseSolveWrapperTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/19/13.
//
//

#include "SerialDenseSolveWrapperTests.h"
#include "SerialDenseWrapper.h"

void SerialDenseSolveWrapperTests::setup() {
  
}

void SerialDenseSolveWrapperTests::teardown() {
  
}

void SerialDenseSolveWrapperTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testMultiplyMatrices()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testSimpleSolve()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testSolveMultipleRHS()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool SerialDenseSolveWrapperTests::testMultiplyMatrices() {
  bool success = true;
  
  double tol = 1e-16;
  FieldContainer<double> A(2,2);
  FieldContainer<double> b(2,3);
  FieldContainer<double> x(2,3); 
  FieldContainer<double> expected_x(2,3);

  
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
  expected_x(0,0) = 11.0;
  expected_x(1,0) = 4.0;
  expected_x(0,1) = 11.0;
  expected_x(1,1) = 4.0;
  expected_x(0,2) = 11.0;
  expected_x(1,2) = 4.0;

  b(0,0) = 7.0;
  b(1,0) = 4.0;
  b(0,1) = 7.0;
  b(1,1) = 4.0;
  b(0,2) = 7.0;
  b(1,2) = 4.0;

  SerialDenseWrapper::multiply(x, A, b);

  double maxDiff = 0;
  if (! fcsAgree(x, expected_x, tol, maxDiff)) {
    cout << "testSolveMultipleRHS() failed: maxDiff " << maxDiff << " exceeds tol " << tol << endl;
    cout << "x: " << endl << x;
    cout << "expected_x: " << endl << expected_x;
    success = false;
    return success;
  }

  // test transposing each matrix X' = b'*A'
  FieldContainer<double> xT(3,2),expected_xT(3,2);
  expected_x(0,0) = 11.0;
  expected_x(0,1) = 4.0;
  expected_x(1,0) = 11.0;
  expected_x(1,1) = 4.0;
  expected_x(2,0) = 11.0;
  expected_x(2,1) = 4.0;

  SerialDenseWrapper::multiply(xT, b, A, 'T','T');

  maxDiff = 0;
  if (! fcsAgree(xT, expected_xT, tol, maxDiff)) {
    cout << "testSolveMultipleRHS() failed on getting transpose: maxDiff " << maxDiff << " exceeds tol " << tol << endl;
    cout << "xT: " << endl << xT;
    cout << "expected_xT: " << endl << expected_xT;
    success = false;
    return success;    
  }
  
  return success;
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
  
  SerialDenseWrapper::solveSystem(x, A, b);
  
  double maxDiff = 0;
  if (! fcsAgree(x, expected_x, tol, maxDiff)) {
    cout << "testSimpleSolve() failed: maxDiff " << maxDiff << " exceeds tol " << tol << endl;
    cout << "x: " << endl << x;
    cout << "expected_x: " << endl << expected_x;
    success = false;
    return success;    
  }
  
  return success;
}

bool SerialDenseSolveWrapperTests::testSolveMultipleRHS() {
  bool success = true;
  
  double tol = 1e-16;
  FieldContainer<double> A(2,2);
  FieldContainer<double> b(2,3);
  FieldContainer<double> x(2,3); 
  FieldContainer<double> expected_x(2,3);
  /*
  FieldContainer<double> b(2,1);
  FieldContainer<double> x(2,1); 
  FieldContainer<double> expected_x(2,1);
  */
  
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
  expected_x(0,0) = 3.0;
  expected_x(1,0) = 4.0;
  expected_x(0,1) = 3.0;
  expected_x(1,1) = 4.0;
  expected_x(0,2) = 3.0;
  expected_x(1,2) = 4.0;

  b(0,0) = 7.0;
  b(1,0) = 4.0;
  b(0,1) = 7.0;
  b(1,1) = 4.0;
  b(0,2) = 7.0;
  b(1,2) = 4.0;

  SerialDenseWrapper::solveSystemMultipleRHS(x, A, b);
  
  double maxDiff = 0;
  if (! fcsAgree(x, expected_x, tol, maxDiff)) {
    cout << "testSolveMultipleRHS() failed: maxDiff " << maxDiff << " exceeds tol " << tol << endl;
    cout << "x: " << endl << x;
    cout << "expected_x: " << endl << expected_x;
    success = false;
    return success;    
  }
  
  return success;
}

std::string SerialDenseSolveWrapperTests::testSuiteName() {
  return "SerialDenseSolveWrapperTests";
}
