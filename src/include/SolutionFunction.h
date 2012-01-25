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
public:
  SolutionFunction(Teuchos::RCP<Solution>, int trialID);
  void getValues(FieldContainer<double> &functionValues, FieldContainer<double> &physicalPoints);
};

#endif
