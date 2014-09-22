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
  
public:
  static FunctionPtr getPhiExact(int spaceDim);
  
  static SolutionPtr poissonExactSolution(int horizontalCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces);
  static SolutionPtr poissonExactSolution(vector<int> numCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces);
  
  // a particular set of refinements (original computed for the Stokes cavity flow problem):
  // (starts with a 2x2 mesh on a unit domain, and refines from there)
  static SolutionPtr poissonExactSolutionRefined(int H1Order, FunctionPtr phi_exact, bool useH1Traces, int refinementSequenceOrdinal);
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  // "identity" tests: fine and coarse mesh the same.
  bool testGMGOperatorIdentity();
  bool testGMGSolverIdentity();
  
  // p-multigrid tests: fine and coarse mesh the same, except for polynomial order:
  bool testGMGOperatorP();
  
  bool testGMGSolverTwoGrid();
  bool testGMGSolverThreeGrid();
  
  std::string testSuiteName();
};



#endif /* defined(__Camellia__GMGTests__) */