//
//  SerialDenseSolveWrapperTests.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/19/13.
//
//

#ifndef __Camellia_debug__SerialDenseSolveWrapperTests__
#define __Camellia_debug__SerialDenseSolveWrapperTests__

#include "TestSuite.h"
#include "SerialDenseSolveWrapper.h"

class SerialDenseSolveWrapperTests : public TestSuite {
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testSimpleSolve();
  bool testSolveMultipleRHS();
  std::string testSuiteName();
};


#endif /* defined(__Camellia_debug__SerialDenseSolveWrapperTests__) */
