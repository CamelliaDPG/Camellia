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
class ConstantScalarFunction : public SimpleFunction<Scalar>
{
  double _value;
  string _stringDisplay;
public:
  ConstantScalarFunction(Scalar value);
  ConstantScalarFunction(Scalar value, string stringDisplay);
  string displayString() override;
  bool isZero() override;
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override;
  void scalarModifyFunctionValues(Camellia::FieldContainer<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache);
  void scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override;
  void scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override;
  void scalarMultiplyBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache) override;
  void scalarDivideBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache) override;
  
  // device path:
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;

  virtual Scalar value(double x) override;
  virtual Scalar value(double x, double y) override;
  virtual Scalar value(double x, double y, double z) override;

  using SimpleFunction<Scalar>::value; // avoid compiler warnings about the value() method below.
  Scalar value();

  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;  // Hmm... a design issue: if we implement dz() then grad() will return a 3D function, not what we want...  It may be that grad() should require a spaceDim argument.  I'm not sure.
  TFunctionPtr<Scalar> dt() override;
};
}

#endif
