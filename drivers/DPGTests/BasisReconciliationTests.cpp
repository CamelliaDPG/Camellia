//
//  BasisReconciliationTests.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#include "BasisReconciliationTests.h"

#include "doubleBasisConstruction.h"

void BasisReconciliationTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testP()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testH()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

void BasisReconciliationTests::setup() {
  
}
void BasisReconciliationTests::teardown() {
  
}

bool BasisReconciliationTests::testH() {
  bool success = true;
  
  cout << "WARNING: BasisReconciliationTests::testH() not yet implemented.\n";
  
  return success;
}

bool BasisReconciliationTests::testP() {
  bool success = true;
  
  int fineOrder = 7;
  int coarseOrder = 5;
  BasisPtr fineBasis = Camellia::intrepidQuadHGRAD(fineOrder);
  BasisPtr coarseBasis = Camellia::intrepidQuadHGRAD(coarseOrder);
  
  // first question: does BasisReconciliation run to completion?
  BasisReconciliation br;
  FieldContainer<double> weights = br.constrainedWeights(fineBasis, coarseBasis);
  
  cout << "BasisReconciliation: weights: \n" << weights;
  cout << "(still need to check that the weights are correct!)\n";
  
  return success;
}