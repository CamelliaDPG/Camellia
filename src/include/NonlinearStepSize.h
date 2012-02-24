//
//  NonlinearStepSize.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_NonlinearStepSize_h
#define Camellia_NonlinearStepSize_h

#include "Solution.h"

class NonlinearStepSize {
  typedef Teuchos::RCP<Solution> SolutionPtr;
  double _fixedStepSize;
public:
  NonlinearStepSize(double fixedStepSize = 0.5) {
    _fixedStepSize = fixedStepSize;
  }
  virtual double stepSize(SolutionPtr u, SolutionPtr du) {
    return _fixedStepSize;
  }
};

#endif
