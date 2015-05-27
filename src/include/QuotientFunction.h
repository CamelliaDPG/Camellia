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

namespace Camellia
{
template <typename Scalar>
class QuotientFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f, _scalarDivisor;
public:
  QuotientFunction(TFunctionPtr<Scalar> f, TFunctionPtr<Scalar> scalarDivisor);
  void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  virtual bool boundaryValueOnly();
  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  TFunctionPtr<Scalar> dt();
  std::string displayString();
};
}

#endif
