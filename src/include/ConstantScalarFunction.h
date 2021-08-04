// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  ConstantScalarFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ConstantScalarFunction_h
#define Camellia_ConstantScalarFunction_h

#include "SimpleFunction.h"

namespace Camellia
{
template <typename Scalar>
class ConstantScalarFunction : public TFunction<Scalar>
{
  double _value;
  string _stringDisplay;
public:
  ConstantScalarFunction(Scalar value);
  ConstantScalarFunction(Scalar value, string stringDisplay);
  string displayString() override;
  
  using TFunction<Scalar>::isZero;
  bool isZero() override;
  
  void scalarModifyFunctionValues(Camellia::FieldContainer<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache);
  
  using TFunction<Scalar>::scalarMultiplyFunctionValues;
  void scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override;
  
  using TFunction<Scalar>::scalarDivideFunctionValues;
  void scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override;
  void scalarMultiplyBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache) override;
  void scalarDivideBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache) override;
  
  using TFunction<Scalar>::values;
  // device path:
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  std::vector<Intrepid2::DimensionInfo> valuesDataDims(BasisCachePtr basisCache) override;
  void values(Intrepid2::Data<Scalar, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
  
  Scalar value();
  
  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;
  TFunctionPtr<Scalar> dt() override;
  
  TFunctionPtr<Scalar> di(int i) override;
};
}

#endif
