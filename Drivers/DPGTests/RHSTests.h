//
//  RHSTests.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_RHSTests_h
#define Camellia_RHSTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

class Mesh;
class RHS;

class RHSTests : public TestSuite {
  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<RHS> _rhs;
  
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testComputeRHSLegacy(); // test copied from DPGTests
  bool testIntegrateAgainstStandardBasis();
  
  std::string testSuiteName();
};

#endif
