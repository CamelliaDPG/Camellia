//
//  GMGTests.h
//  Camellia
//
//  Created by Nate Roberts on 8/29/14.
//
//

#ifndef __Camellia__GMGTests__
#define __Camellia__GMGTests__

#include "TestSuite.h"
#include "Solution.h"

class GMGTests : public TestSuite {
  void setup();
  void teardown() {}
  
  SolutionPtr poissonExactSolution1D(int horizontalCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces);
  
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  // "identity" tests: fine and coarse mesh the same.
  bool testGMGOperatorIdentity();
  bool testGMGSolverIdentity();
  
  
  
  std::string testSuiteName();
};



#endif /* defined(__Camellia__GMGTests__) */
