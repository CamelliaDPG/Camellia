//
//  ParametricFunctionTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/16/13.
//
//

#include "ParametricFunctionTests.h"

void ParametricFunctionTests::setup() {
  
}

void ParametricFunctionTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testParametricFunctionRefinement()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool ParametricFunctionTests::testParametricFunctionRefinement() {
 // tests the kind of thing that will happen to parametric functions during mesh refinement
  bool success = true;
  
  ParametricFunctionPtr unitLine = ParametricFunction::line(0, 0, 1, 0);
  
  ParametricFunctionPtr halfLine = ParametricFunction::remapParameter(unitLine, 0, 0.5);
  
  double expected_x, expected_y, t, actual_x, actual_y;
  
  // t = 0
  t = 0;
  expected_x = 0; expected_y = 0;
  halfLine->value(t, actual_x, actual_y);
  
  if (actual_x != expected_x) {
    success = false;
    cout << "expected " << expected_x << " but actual x is " << actual_x << endl;
  }
  if (actual_y != expected_y) {
    success = false;
    cout << "expected " << expected_y << " but actual y is " << actual_y << endl;
  }
  
  // t = 1
  t = 1;
  expected_x = 0.5; expected_y = 0;
  halfLine->value(t, actual_x, actual_y);
  
  if (actual_x != expected_x) {
    success = false;
    cout << "expected " << expected_x << " but actual x is " << actual_x << endl;
  }
  if (actual_y != expected_y) {
    success = false;
    cout << "expected " << expected_y << " but actual y is " << actual_y << endl;
  }
  
  return allSuccess(success);
}

std::string ParametricFunctionTests::testSuiteName() {
  return "ParametricFunctionTests";
}
