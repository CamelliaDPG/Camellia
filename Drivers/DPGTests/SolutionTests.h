#ifndef CAMELLIA_SOLUTION_TESTS
#define CAMELLIA_SOLUTION_TESTS

#include "Solution.h"
#include "ExactSolution.h"

#include "PoissonExactSolution.h"
#include "ConfusionManufacturedSolution.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "TestSuite.h"

class SolutionTests : public TestSuite {
private:
  FieldContainer<double> _testPoints;
  Teuchos::RCP< Solution > _confusionSolution1_2x2, _confusionSolution2_2x2, _poissonSolution;
  Teuchos::RCP< PoissonExactSolution > _poissonExactSolution;
  Teuchos::RCP< ConfusionManufacturedSolution > _confusionExactSolution;
  void setup();
  void teardown();
public:
  SolutionTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "SolutionTests"; }
  bool testAddSolution();
  bool testProjectFunction();
  bool testEnergyError();
};


#endif
