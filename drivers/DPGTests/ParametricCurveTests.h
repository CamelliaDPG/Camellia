//
//  ParametricCurveTests.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/16/13.
//
//

#ifndef Camellia_debug_ParametricCurveTests_h
#define Camellia_debug_ParametricCurveTests_h

#include "ParametricCurve.h"
#include "TestSuite.h"

class ParametricCurveTests : public TestSuite {
  void setup();
  void teardown() {}
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  bool testBubble();
  bool testCircularArc();
  bool testLine();
  bool testParametricCurveRefinement(); // tests the kind of thing that will happen to parametric curves during mesh refinement
  bool testPolygon();
  bool testProjectionBasedInterpolation();
  
  std::string testSuiteName();
};

#endif
