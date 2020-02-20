// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
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

namespace Camellia
{
template <typename Scalar>
class ConstantVectorFunction : public TFunction<Scalar>
{
  std::vector<Scalar> _value;
public:
  ConstantVectorFunction(std::vector<Scalar> value);
  string displayString() override;
  bool isZero() override;

  TFunctionPtr<Scalar> x() override;
  TFunctionPtr<Scalar> y() override;
  TFunctionPtr<Scalar> z() override;

  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  std::vector<Scalar> value();
};
}

#endif
