//
//  LidDrivenFlowRefinementStrategy.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LidDrivenFlowRefinementStrategy_h
#define Camellia_debug_LidDrivenFlowRefinementStrategy_h

#include "Solution.h"
#include "RefinementStrategy.h"

class LidDrivenFlowRefinementStrategy : public RefinementStrategy {
  double _hmin;
public:
  LidDrivenFlowRefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold, double hmin) 
  : RefinementStrategy(solution,  relativeEnergyThreshold) { _hmin = hmin; }
  virtual void refineCells(vector<int> &cellsToRefine);
};


#endif
