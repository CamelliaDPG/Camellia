//
//  ConstantVectorFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ConstantVectorFunction_h
#define Camellia_ConstantVectorFunction_h

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class ConstantVectorFunction : public Function<Scalar> {
    std::vector<Scalar> _value;
  public:
    ConstantVectorFunction(std::vector<Scalar> value);
    bool isZero();

    FunctionPtr<Scalar> x();
    FunctionPtr<Scalar> y();
    FunctionPtr<Scalar> z();

    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    std::vector<Scalar> value();
  };
}

#endif
