//
//  TimeMarchingBurgersProblem.h
//  Camellia
//
//  Created by Nathan Roberts on 2/28/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_TimeMarchingBurgersProblem_h
#define Camellia_TimeMarchingBurgersProblem_h

#include "TimeMarchingProblem.h"

class BurgersBilinearForm;

class TimeMarchingBurgersProblem : public TimeMarchingProblem {
public:
  TimeMarchingBurgersProblem(Teuchos::RCP<BurgersBilinearForm> bilinearForm,
                              Teuchos::RCP<RHS> rhs);
  virtual bool hasTimeDerivative(int trialID, int testID);
  virtual void timeLHS(FieldContainer<double> trialValues, int trialID, Teuchos::RCP<BasisCache> basisCache);
  virtual void timeRHS(FieldContainer<double> values, int trialID, Teuchos::RCP<BasisCache> basisCache);
};

#endif
