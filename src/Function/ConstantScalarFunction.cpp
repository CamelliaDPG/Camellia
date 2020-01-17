//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ConstantScalarFunction.h"

#include "BasisCache.h"

using namespace Camellia;

template <typename Scalar>
ConstantScalarFunction<Scalar>::ConstantScalarFunction(Scalar value)
{
  _value = value;
  ostringstream valueStream;
  valueStream << value;
  _stringDisplay = valueStream.str();
}

template <typename Scalar>
ConstantScalarFunction<Scalar>::ConstantScalarFunction(Scalar value, string stringDisplay)
{
  _value = value;
  _stringDisplay = stringDisplay;
}

template <typename Scalar>
string ConstantScalarFunction<Scalar>::displayString()
{
  return _stringDisplay;
}

template <typename Scalar>
bool ConstantScalarFunction<Scalar>::isZero()
{
  return 0.0 == _value;
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView();
  Kokkos::deep_copy(dv, _value);
  values.checkinDeviceView();
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (_value != 1.0)
  {
    this->CHECK_VALUES_RANK(values);
    auto dv = values.checkoutDeviceView();
    auto value = _value;
    const int dim0 = dv.extent(0); // numCells
    const int dim1 = dv.extent(1); // numPoints
    Kokkos::parallel_for("ConstantScalarFunction_multiply", dim0, KOKKOS_LAMBDA (const int i0) {
      for (int i1 = 0; i1 < dim1; i1++) {
        dv(i0,i1) *= value;
      }
    });
    values.checkinDeviceView();
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (_value != 1.0)
  {
    this->CHECK_VALUES_RANK(values);
    auto dv = values.checkoutDeviceView();
    auto value = _value;
    const int dim0 = dv.extent(0); // numCells
    const int dim1 = dv.extent(1); // numPoints
    Kokkos::parallel_for("ConstantScalarFunction_divide", dim0, KOKKOS_LAMBDA (const int i0) {
      for (int i1 = 0; i1 < dim1; i1++) {
        dv(i0,i1) /= value;
      }
    });
    values.checkinDeviceView();
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  // we don't actually care about the shape of basisValues--just use the FunctionValues versions:
  scalarMultiplyFunctionValues(basisValues,basisCache);
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  scalarDivideFunctionValues(basisValues,basisCache);
}

template <typename Scalar>
Scalar ConstantScalarFunction<Scalar>::value(double x)
{
  return value();
}

template <typename Scalar>
Scalar ConstantScalarFunction<Scalar>::value(double x, double y)
{
  return value();
}

template <typename Scalar>
Scalar ConstantScalarFunction<Scalar>::value(double x, double y, double z)
{
  return value();
}

template <typename Scalar>
Scalar ConstantScalarFunction<Scalar>::value()
{
  return _value;
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantScalarFunction<Scalar>::dx()
{
  return TFunction<Scalar>::zero();
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantScalarFunction<Scalar>::dy()
{
  return TFunction<Scalar>::zero();
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantScalarFunction<Scalar>::dz()
{
  return TFunction<Scalar>::zero();
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantScalarFunction<Scalar>::dt()
{
  return TFunction<Scalar>::zero();
}

namespace Camellia
{
template class ConstantScalarFunction<double>;
}

