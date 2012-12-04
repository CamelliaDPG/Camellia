#ifndef CAMELLIA_HCONVERGENCE_STUDY_TESTS
#define CAMELLIA_HCONVERGENCE_STUDY_TESTS

#include "Solution.h"
#include "ExactSolution.h"

#include "PoissonExactSolution.h"
#include "ConfusionManufacturedSolution.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "TestSuite.h"

#include "SpatialFilter.h" // for testing
class HConvergenceStudyTests : public TestSuite {
private:
  void setup();
  void teardown();
public:
  HConvergenceStudyTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  bool testBestApproximationErrorComputation();
  string testSuiteName();
};


#endif