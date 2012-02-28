//
//  TimeMarchingBurgersProblem.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/28/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "TimeMarchingBurgersProblem.h"
#include "BurgersBilinearForm.h"
#include "Solution.h"

TimeMarchingBurgersProblem::TimeMarchingBurgersProblem(Teuchos::RCP<BurgersBilinearForm> bilinearForm,
                                                       Teuchos::RCP<RHS> rhs) : TimeMarchingProblem(bilinearForm,rhs) {
  
}

bool TimeMarchingBurgersProblem::hasTimeDerivative(int trialID, int testID) {
  if ( (trialID == BurgersBilinearForm::U) && (testID == BurgersBilinearForm::V) ) {
    return true;
  }
  return false;
}

void TimeMarchingBurgersProblem::timeLHS(FieldContainer<double> trialValues, int trialID, Teuchos::RCP<BasisCache> basisCache) {
  // leave as is: no scaling
}

void TimeMarchingBurgersProblem::timeRHS(FieldContainer<double> values, int testID, Teuchos::RCP<BasisCache> basisCache) {
  if (testID == BurgersBilinearForm::V) {
    int trialID = BurgersBilinearForm::U;
    Teuchos::RCP<Solution> backgroundFlow = ((BurgersBilinearForm*)_bilinearForm.get())->getBackgroundFlow();
    backgroundFlow->solutionValues(values, trialID, basisCache);
    Teuchos::Array<int> dimensions;
    values.dimensions(dimensions);
    FieldContainer<double> prevTimeValues(dimensions);
    previousTimeSolution()->solutionValues(prevTimeValues, trialID, basisCache);
    int numValues = prevTimeValues.size();
    for (int valIndex=0; valIndex<numValues; valIndex++) {
      values[valIndex] = prevTimeValues[valIndex] - values[valIndex];
    }
  }
}