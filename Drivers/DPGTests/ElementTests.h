//
//  ElementTests.h
//  Camellia
//
//  Created by Nathan Roberts on 2/15/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_ElementTests_h
#define Camellia_ElementTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"

typedef Basis<double, FieldContainer<double> > DoubleBasis;
typedef Teuchos::RCP< DoubleBasis > BasisPtr;

typedef Teuchos::RCP<Element> ElementPtr;

class ElementTests : public TestSuite {
  FieldContainer<double> _testPoints1D;
  
  Teuchos::RCP<Mesh> _mesh; // a 2x2 mesh refined in SW, and then in the SE of the SW
  ElementPtr _sw, _se, _nw, _ne, _sw_se, _sw_ne, _sw_se_se, _sw_se_ne;
  
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testNeighborPointMapping();
  bool testParentPointMapping();
  
  std::string testSuiteName();
};

#endif
