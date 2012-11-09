#include "LocalConservationTests.h"

#include "BasicConfusionProblem.h"

void LocalConservationTests::SetUp()
{
  confusionProb.epsilon = 1.0;
  confusionProb.checkLocalConservation = true;
  confusionProb.printLocalConservation = false;
  confusionProb.enforceLocalConservation = true;
  vector<double> beta;
  beta.push_back(2.0);
  beta.push_back(1.0);
  confusionProb.defineVariables();
  confusionProb.beta = beta;
  confusionProb.defineBilinearForm(confusionProb.beta);
  confusionProb.setRobustZeroMeanIP(confusionProb.beta);
  confusionProb.defineRightHandSide();
  confusionProb.defineBoundaryConditions();
  confusionProb.defineMesh();
}

TEST_F(LocalConservationTests, TestZeroMeanTerm)
{
  EXPECT_TRUE(false);
}

TEST_F(LocalConservationTests, TestLocalConservation)
{
  char *argv[11] = {"./RunTests"};
  confusionProb.solveSteady(1, argv);
  double tol = 1e-14;
  EXPECT_LE(confusionProb.fluxImbalances[0], tol);
  EXPECT_LE(confusionProb.fluxImbalances[1], tol);
  EXPECT_LE(confusionProb.fluxImbalances[2], tol);
}
