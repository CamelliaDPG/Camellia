#ifndef CAMELLIA_SOLUTION_TESTS
#define CAMELLIA_SOLUTION_TESTS

#include "Solution.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

class SolutionTests {
private:
  FieldContainer<double> _testPoints;
  Teuchos::RCP< Solution > _confusionSolution1_2x2, _confusionSolution2_2x2;
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  bool testAddSolution();
};


#endif