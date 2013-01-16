//
//  ParametricFunctionTests.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/16/13.
//
//

#ifndef Camellia_debug_ParametricFunctionTests_h
#define Camellia_debug_ParametricFunctionTests_h

#include "ParametricFunction.h"
#include "TestSuite.h"

class ParametricFunctionTests : public TestSuite {
  void setup();
  void teardown() {}
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  bool testParametricFunctionRefinement(); // tests the kind of thing that will happen to parametric functions during mesh refinement
  
  std::string testSuiteName();
};

#endif
