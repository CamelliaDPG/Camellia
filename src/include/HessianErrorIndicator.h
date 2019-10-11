//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  HessianErrorIndicator.h
//  Camellia
//
//  Created by Nate Roberts on 8/9/16.
//
//

#ifndef __Camellia__HessianErrorIndicator__
#define __Camellia__HessianErrorIndicator__

#include "ErrorIndicator.h"

#include "ErrorIndicator.h"
#include "Solution.h"
#include "Var.h"

namespace Camellia
{
  template <typename Scalar>
  class HessianErrorIndicator : public ErrorIndicator
  {
    TSolutionPtr<Scalar> _solution;
    VarPtr _var;
    double _hPower; // h power to use in hessian approximation (default: 2.0 + spaceDim / 2.0).
    
    int _rieszRepCubatureEnrichment = 0;
  public:
    //! Uses hessian approximation as error indicator.
    HessianErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForHessian);
    
    //! Uses hessian approximation as error indicator.
    HessianErrorIndicator(TSolutionPtr<Scalar> soln, VarPtr varForHessian, double hPower);
    
    //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
    virtual void measureError();
  };
}


#endif /* defined(__Camellia__HessianErrorIndicator__) */
