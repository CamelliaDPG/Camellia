//
//  NonlinearSolveStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_NonlinearSolveStrategy_h
#define Camellia_NonlinearSolveStrategy_h

#include "NonlinearStepSize.h"

namespace Camellia {
	class NonlinearSolveStrategy {
	  Teuchos::RCP<NonlinearStepSize> _stepSize;
	  Teuchos::RCP<Solution> _backgroundFlow, _solution;
	  double _relativeEnergyTolerance;
	  bool _usePicardIteration; // instead of Newton-Raphson (will just do background = new at each step)
	public:
	  NonlinearSolveStrategy(Teuchos::RCP<Solution> backgroundFlow, Teuchos::RCP<Solution> solution, Teuchos::RCP<NonlinearStepSize> stepSize, double relativeEnergyTolerance);
	  void setUsePicardIteration(bool value);
	  void solve(bool printToConsole=false);
	};
}

#endif
