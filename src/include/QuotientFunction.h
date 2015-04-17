//
//  QuotientFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_QuotientFunction_h
#define Camellia_QuotientFunction_h

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class QuotientFunction : public Function<Scalar> {
    FunctionPtr<Scalar> _f, _scalarDivisor;
  public:
    QuotientFunction(FunctionPtr<Scalar> f, FunctionPtr<Scalar> scalarDivisor);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();
    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();
    FunctionPtr<Scalar> dt();
    std::string displayString();
  };
}

#endif
