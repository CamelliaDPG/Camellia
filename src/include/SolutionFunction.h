//
//  SolutionFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 1/25/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SolutionFunction_h
#define Camellia_SolutionFunction_h

#include "Solution.h"
#include "AbstractFunction.h"

class SolutionFunction : AbstractFunction {
  Teuchos::RCP<Solution> _solution;
  int _trialID;
public:
  SolutionFunction(Teuchos::RCP<Solution> soln, int trialID) {
    _solution = soln;
    _trialID = trialID;
  }
  
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    _solution->solutionValues(functionValues, trialID, physicalPoints);
  }
};

#endif
