//
//  FunctionTests.h
//  Camellia
//
//  Created by Nathan Roberts on 4/9/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_FunctionTests_h
#define Camellia_FunctionTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"
#include "InnerProductScratchPad.h"
#include "BasisCache.h"

typedef Basis<double, FieldContainer<double> > DoubleBasis;
typedef Teuchos::RCP< DoubleBasis > BasisPtr;

/*
 For now, this is sort of a grab bag for tests against all the "new-style"
 (a.k.a. "ScratchPad") items.  There are some tests against these elsewhere
 (as of this writing, there's one against RHSEasy in RHSTests), and other
 places are probably the best spot for tests that compare the results of the
 old code to that of the new--as with RHS, the sensible place to add such tests
 is where we already test the old code.
 
 All that to say, the tests here are glommed together for convenience and
 quick test development.  Once they grow to a certain size, it may be better
 to split them apart...
 */

class FunctionTests : public TestSuite {  
  Teuchos::RCP<Mesh> _spectralConfusionMesh; // 1x1 mesh, H1 order = 1, pToAdd = 0
  Teuchos::RCP<BF> _confusionBF; // standard confusion bilinear form
  FieldContainer<double> _testPoints;
  ElementTypePtr _elemType;
  BasisCachePtr _basisCache;
  
  void setup();
  void teardown() {}
  bool functionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache);
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testThatLikeFunctionsAgree();
  
  bool testProductRule();
  
  bool testQuotientRule();
  
  std::string testSuiteName();
};


#endif
