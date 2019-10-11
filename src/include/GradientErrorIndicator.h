//
//  GradientErrorIndicator.h
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#ifndef __Camellia__GradientErrorIndicator__
#define __Camellia__GradientErrorIndicator__

#include "ErrorIndicator.h"
#include "Solution.h"
#include "Var.h"

namespace Camellia
{
  template <typename Scalar>
  class GradientErrorIndicator : public ErrorIndicator
  {
    TSolutionPtr<Scalar> _solution;
    VarPtr _var;
    double _hPower; // h power to use in gradient approximation (default: 1.0 + spaceDim / 2.0).
    
    int _rieszRepCubatureEnrichment = 0;
  public:
    //! Uses gradient approximation as error indicator.
    GradientErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForGradient);

    //! Uses gradient approximation as error indicator.
    GradientErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForGradient, double hPower);
    
    //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
    virtual void measureError();
  };
}

#endif /* defined(__Camellia__GradientErrorIndicator__) */
