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

namespace Camellia
{
template <typename Scalar>
class SumFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f1, _f2;
public:
  SumFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);

  TFunctionPtr<Scalar> x();
  TFunctionPtr<Scalar> y();
  TFunctionPtr<Scalar> z();
  TFunctionPtr<Scalar> t();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  TFunctionPtr<Scalar> dt();

  TFunctionPtr<Scalar> grad(int numComponents=-1); // gradient of sum is the sum of gradients
  TFunctionPtr<Scalar> div();  // divergence of sum is sum of divergences

  void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();

  string displayString();
};
}

#endif
