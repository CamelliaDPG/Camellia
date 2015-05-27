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

#include "IndexType.h"

class LidDrivenFlowRefinementStrategy : public RefinementStrategy
{
  double _hmin;
  int _maxPolyOrder;
  bool _printToConsole;
  bool _symmetricRefinements; // any refinement done on the top, do on the bottom as well
  set<GlobalIndexType> symmetricCellIDs(set<GlobalIndexType> &cellIDs); // utility method for finding the symmetric counterparts for a set of cells
public:
  LidDrivenFlowRefinementStrategy( SolutionPtr solution, double relativeEnergyThreshold, double hmin, int maxPolyOrder, bool printToConsole=false)
    : RefinementStrategy(solution,relativeEnergyThreshold)
  {
    _hmin = hmin;
    _maxPolyOrder = maxPolyOrder;
    _printToConsole = printToConsole;
    _symmetricRefinements = false;
  }
  virtual void refineCells(vector<int> &cellsToRefine);
  void setSymmetricRefinements(bool value);
};


#endif
