//
//  ErrorPercentageRefinementStrategy.h
//  Camellia-debug
//
//  Created by Nate Roberts on 6/20/14.
//
//

#ifndef __Camellia_debug__ErrorPercentageRefinementStrategy__
#define __Camellia_debug__ErrorPercentageRefinementStrategy__

#include "RefinementStrategy.h"

#include "Solution.h"

namespace Camellia
{
template <typename Scalar>
class ErrorPercentageRefinementStrategy : public TRefinementStrategy<Scalar>
{
  double _percentageThreshold;
public:
  ErrorPercentageRefinementStrategy(TSolutionPtr<Scalar> soln, double percentageThreshold, double min_h = 0, int max_p = 10, bool preferPRefinements = false);
  virtual void refine(bool printToConsole);
};

extern template class ErrorPercentageRefinementStrategy<double>;
}


#endif /* defined(__Camellia_debug__ErrorPercentageRefinementStrategy__) */
