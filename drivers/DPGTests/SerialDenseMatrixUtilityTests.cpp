//
//  SerialDenseMatrixUtilityTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/19/13.
//
//

#include "SerialDenseMatrixUtilityTests.h"
#include "SerialDenseWrapper.h"

void SerialDenseMatrixUtilityTests::setup() {
  
}

void SerialDenseMatrixUtilityTests::teardown() {
  
}

void SerialDenseMatrixUtilityTests::runTests(int &numTestsRun, int &numTestsPassed) {
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

  setup();
  if (testAddMatrices()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool SerialDenseMatrixUtilityTests::testAddMatrices() {
  bool success = true;
  
  double tol = 1e-16;
  FieldContainer<double> a(2,3);
  FieldContainer<double> b(2,3);
  FieldContainer<double> x(2,3); 
  FieldContainer<double> expected_x(2,3);
  
  // A = [ 1 1 ]
  //     [ 0 1 ]
  // x = [ 3 ]
  //     [ 4 ]
  // b = [ 7 ]
  //     [ 4 ]
  expected_x(0,0) = 8.0;
  expected_x(1,0) = 10.0;
  expected_x(0,1) = 12.0;
  expected_x(1,1) = 14.0;
  expected_x(0,2) = 16.0;
  expected_x(1,2) = 18.0;

  a(0,0) = 1.0;
  a(1,0) = 2.0;
  a(0,1) = 3.0;
  a(1,1) = 4.0;
  a(0,2) = 5.0;
  a(1,2) = 6.0;

  b(0,0) = 7.0;
  b(1,0) = 8.0;
  b(0,1) = 9.0;
  b(1,1) = 10.0;
  b(0,2) = 11.0;
  b(1,2) = 12.0;

  SerialDenseWrapper::add(x, a, b);

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
bool SerialDenseMatrixUtilityTests::testMultiplyMatrices() {
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

  SerialDenseWrapper::multiplyAndAdd(x, A, b, 'N', 'N', 1.0, 0.0);

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
  expected_xT(0,0) = 11.0;
  expected_xT(0,1) = 4.0;
  expected_xT(1,0) = 11.0;
  expected_xT(1,1) = 4.0;
  expected_xT(2,0) = 11.0;
  expected_xT(2,1) = 4.0;

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

bool SerialDenseMatrixUtilityTests::testSimpleSolve() {
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

bool SerialDenseMatrixUtilityTests::testSolveMultipleRHS() {
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
  // x = [ 3 1 5 ]
  //     [ 4 2 6 ]
  // b = [ 7 3 11 ]
  //     [ 4 2  6 ]
  A(0,0) = 1.0;
  A(0,1) = 1.0;
  A(1,0) = 0.0;
  A(1,1) = 1.0;
  expected_x(0,0) = 3.0;
  expected_x(1,0) = 4.0;
  expected_x(0,1) = 1.0;
  expected_x(1,1) = 2.0;
  expected_x(0,2) = 5.0;
  expected_x(1,2) = 6.0;

  b(0,0) = 7.0;
  b(1,0) = 4.0;
  b(0,1) = 3.0;
  b(1,1) = 2.0;
  b(0,2) = 11.0;
  b(1,2) = 6.0;

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

std::string SerialDenseMatrixUtilityTests::testSuiteName() {
  return "SerialDenseMatrixUtilityTests";
}
