//
//  SumFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/8/15.
//  Copyright (c) 2015 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SumFunction_h
#define Camellia_SumFunction_h

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class SumFunction : public Function<Scalar> {
    FunctionPtr<Scalar> _f1, _f2;
  public:
    SumFunction(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);

    FunctionPtr<Scalar> x();
    FunctionPtr<Scalar> y();
    FunctionPtr<Scalar> z();
    FunctionPtr<Scalar> t();

    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();
    FunctionPtr<Scalar> dt();

    FunctionPtr<Scalar> grad(int numComponents=-1); // gradient of sum is the sum of gradients
    FunctionPtr<Scalar> div();  // divergence of sum is sum of divergences

    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();
  };
}

#endif
