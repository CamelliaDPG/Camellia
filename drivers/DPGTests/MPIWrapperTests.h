//
//  MPIWrapperTests.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#ifndef __Camellia_MPIWrapperTests__
#define __Camellia_MPIWrapperTests__

#include "TestSuite.h"

// Teuchos includes
#include "MPIWrapper.h"

class MPIWrapperTests : public TestSuite {
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testSimpleSum();
  bool testentryWiseSum();
  
  std::string testSuiteName();
};

#endif /* defined(__Camellia_MPIWrapperTests__) */
