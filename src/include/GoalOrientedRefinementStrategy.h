//
//  GoalOrientedRefinementStrategy.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 7/22/13.
//
//

#ifndef __Camellia_debug__GoalOrientedRefinementStrategy__
#define __Camellia_debug__GoalOrientedRefinementStrategy__

#include "TypeDefs.h"

#include <iostream>

#include "RefinementStrategy.h"

#include "LinearTerm.h"

#include "Solution.h"

namespace Camellia {
	class GoalOrientedRefinementStrategy : public RefinementStrategy {
	private:
	  LinearTermPtr _trialFunctional;
	public:
	  GoalOrientedRefinementStrategy( LinearTermPtr trialFunctional, SolutionPtr<double> solution, double relativeErrorThreshold, double min_h = 0);
	  virtual void refine(bool printToConsole=false);
	  void setTrialFunctional(LinearTermPtr trialFunctional);
	};
}

#endif /* defined(__Camellia_debug__GoalOrientedRefinementStrategy__) */
