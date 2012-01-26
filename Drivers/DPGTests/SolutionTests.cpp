
#include "SolutionTests.h"

#include "Intrepid_FieldContainer.hpp"
#include "Mesh.h"

#include "ConfusionBilinearForm.h"
#include "ConfusionManufacturedSolution.h"
#include "MathInnerProduct.h"

void SolutionTests::setup() {
  // first, build a simple mesh
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
  int sqrtElements = 2;
  
  double epsilon = 1e-2;
  double beta_x = 1.0, beta_y = 1.0;
  ConfusionManufacturedSolution exactSolution(epsilon,beta_x,beta_y); // 0 doesn't mean constant, but a particular solution...
  
  int H1Order = 3;
  int horizontalCells = 2; int verticalCells = 2;
  
  // before we hRefine, compute a solution for comparison after refinement
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+1);

  _confusionSolution1_2x2 = Teuchos::rcp( new Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip) );
  _confusionSolution2_2x2 = Teuchos::rcp( new Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip) );
  _confusionSolution1_2x2->solve();
  _confusionSolution2_2x2->solve();
  
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  double y[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  _testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[i];
    }
  }
}

void SolutionTests::teardown() {
  _confusionSolution1_2x2 = Teuchos::rcp( (Solution*)NULL );
  _confusionSolution2_2x2 = Teuchos::rcp( (Solution*)NULL );  
  _testPoints.resize(0);
}

void SolutionTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testAddSolution()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
}

bool SolutionTests::testAddSolution() {
  bool success = true;
  
  double weight = 3.141592;
  double tol = 1e-15;
  
  FieldContainer<double> expectedValuesU(_testPoints.dimension(0));
  FieldContainer<double> expectedValuesSIGMA1(_testPoints.dimension(0));
  FieldContainer<double> expectedValuesSIGMA2(_testPoints.dimension(0));
  _confusionSolution2_2x2->solutionValues(expectedValuesU, ConfusionBilinearForm::U, _testPoints);
  _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA1, ConfusionBilinearForm::SIGMA_1, _testPoints);
  _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA2, ConfusionBilinearForm::SIGMA_2, _testPoints);
  
  BilinearForm::multiplyFCByWeight(expectedValuesU, weight+1.0);
  BilinearForm::multiplyFCByWeight(expectedValuesSIGMA1, weight+1.0);
  BilinearForm::multiplyFCByWeight(expectedValuesSIGMA2, weight+1.0);
  
  _confusionSolution1_2x2->addSolution(_confusionSolution2_2x2, weight);
  FieldContainer<double> valuesU(_testPoints.dimension(0));
  FieldContainer<double> valuesSIGMA1(_testPoints.dimension(0));
  FieldContainer<double> valuesSIGMA2(_testPoints.dimension(0));
  
  _confusionSolution1_2x2->solutionValues(valuesU, ConfusionBilinearForm::U, _testPoints);
  _confusionSolution1_2x2->solutionValues(valuesSIGMA1, ConfusionBilinearForm::SIGMA_1, _testPoints);
  _confusionSolution1_2x2->solutionValues(valuesSIGMA2, ConfusionBilinearForm::SIGMA_2, _testPoints);
  
  for (int pointIndex=0; pointIndex < valuesU.size(); pointIndex++) {
    double diff = abs(valuesU[pointIndex] - expectedValuesU[pointIndex]);
    if (diff > tol) {
      success = false;
      cout << "expected value of U: " << expectedValuesU[pointIndex] << "; actual: " << valuesU[pointIndex] << endl;
    }
    
    diff = abs(valuesSIGMA1[pointIndex] - expectedValuesSIGMA1[pointIndex]);
    if (diff > tol) {
      success = false;
      cout << "expected value of SIGMA1: " << expectedValuesSIGMA1[pointIndex] << "; actual: " << valuesSIGMA1[pointIndex] << endl;
    }
    
    diff = abs(valuesSIGMA2[pointIndex] - expectedValuesSIGMA2[pointIndex]);
    if (diff > tol) {
      success = false;
      cout << "expected value of SIGMA2: " << expectedValuesSIGMA2[pointIndex] << "; actual: " << valuesSIGMA2[pointIndex] << endl;
    }
  }
  
  return success;
}