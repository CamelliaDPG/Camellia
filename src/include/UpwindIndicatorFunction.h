#ifndef UPWIND_INDICATOR_FUNCTION
#define UPWIND_INDICATOR_FUNCTION

#include "Function.h"
#include "TypeDefs.h"

namespace Camellia
{
  class UpwindIndicatorFunction : public TFunction<double>
  {
    std::vector<FunctionPtr> _beta;
    bool _upwind;
    std::vector<Intrepid::FieldContainer<double>> _valuesBuffers;
  public:
    UpwindIndicatorFunction(FunctionPtr beta, bool upwind);
    
    bool isZero(BasisCachePtr basisCache);
    
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    
    // ! returns an upwind indicator that functions as the DG "-" operator for the given
    // ! convective direction.
    static FunctionPtr minus(FunctionPtr beta);
    
    // ! returns an upwind indicator that functions as the DG "+" operator for the given
    // ! convective direction.
    static FunctionPtr plus(FunctionPtr beta);
    
    static FunctionPtr upwindIndicator(FunctionPtr beta, bool upwind);
  };
}

#endif