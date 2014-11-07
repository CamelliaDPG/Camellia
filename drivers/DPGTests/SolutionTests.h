#ifndef CAMELLIA_SOLUTION_TESTS
#define CAMELLIA_SOLUTION_TESTS

#include "Solution.h"
#include "ExactSolution.h"

#include "PoissonExactSolution.h"
#include "ConfusionManufacturedSolution.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "TestSuite.h"

#include "SpatialFilter.h" // for testing
class SolutionTests : public TestSuite {
private:
  FieldContainer<double> _testPoints;
  Teuchos::RCP< Solution > _confusionSolution1_2x2, _confusionSolution2_2x2, _poissonSolution, _confusionUnsolved;
  Teuchos::RCP< Solution > _poissonSolution_1x1; // single-element mesh
  Teuchos::RCP< Solution > _poissonSolution_1x1_unsolved; // single-element mesh, zero
  Teuchos::RCP< PoissonExactSolution > _poissonExactSolution;
  Teuchos::RCP< ConfusionManufacturedSolution > _confusionExactSolution;
  
  void setup();
  void teardown();
  
public:
  SolutionTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  static bool storageSizesAgree(Teuchos::RCP< Solution > sol1, Teuchos::RCP< Solution > sol2);
  string testSuiteName() { return "SolutionTests"; }
  bool testAddCondensedSolution();
  bool testAddSolution();
  bool testProjectFunction();
  bool testNewProjectFunction();
  bool testProjectSolutionOntoOtherMesh();
  bool testProjectVectorValuedSolution();
  bool testAddRefinedSolutions();
  bool testEnergyError();
  bool testHRefinementInitialization();
  bool testPRefinementInitialization();
  bool testSolutionEvaluationBasisCache();
  bool testScratchPadSolution();
  bool testSolutionsAreConsistent();
  bool testCondensationSolve();
  bool testCondensationSolveNonlinear();
  bool testCondensationSolveWithSinglePointConstraint();
  bool testCondensationSolveWithZeroMeanConstraint();
  
  static bool solutionCoefficientsAreConsistent(Teuchos::RCP<Solution> soln, bool printDetailsToConsole = false);
};

#endif
