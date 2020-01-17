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
    auto dv = values.checkoutDeviceView();
    auto value = _value;
    
    const int dim0 = (dv.rank() > 0) ? dv.extent(0) : 1;
    const int dim1 = (dv.rank() > 1) ? dv.extent(1) : 1;
    const int dim2 = (dv.rank() > 2) ? dv.extent(2) : 1;
    const int dim3 = (dv.rank() > 3) ? dv.extent(3) : 1;
    const int dim4 = (dv.rank() > 4) ? dv.extent(4) : 1;
    const int dim5 = (dv.rank() > 5) ? dv.extent(5) : 1;
    const int dim6 = (dv.rank() > 6) ? dv.extent(6) : 1;
    
    Kokkos::parallel_for("ConstantScalarFunction_multiply", dim0, KOKKOS_LAMBDA (const int i0) {
      for (int i1 = 0; i1 < dim1; i1++) {
        for (int i2 = 0; i2 < dim2; i2++) {
          for (int i3 = 0; i3 < dim3; i3++) {
            for (int i4 = 0; i4 < dim4; i4++) {
              for (int i5 = 0; i5 < dim5; i5++) {
                for (int i6 = 0; i6 < dim6; i6++) {
                  dv.access(i0,i1,i2,i3,i4,i5,i6) *= value;
                }
              }
            }
          }
        }
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
    auto dv = values.checkoutDeviceView();
    auto value = _value;
    
    const int dim0 = (dv.rank() > 0) ? dv.extent(0) : 1;
    const int dim1 = (dv.rank() > 1) ? dv.extent(1) : 1;
    const int dim2 = (dv.rank() > 2) ? dv.extent(2) : 1;
    const int dim3 = (dv.rank() > 3) ? dv.extent(3) : 1;
    const int dim4 = (dv.rank() > 4) ? dv.extent(4) : 1;
    const int dim5 = (dv.rank() > 5) ? dv.extent(5) : 1;
    const int dim6 = (dv.rank() > 6) ? dv.extent(6) : 1;
    
    Kokkos::parallel_for("ConstantScalarFunction_divide", dim0, KOKKOS_LAMBDA (const int i0) {
      for (int i1 = 0; i1 < dim1; i1++) {
        for (int i2 = 0; i2 < dim2; i2++) {
          for (int i3 = 0; i3 < dim3; i3++) {
            for (int i4 = 0; i4 < dim4; i4++) {
              for (int i5 = 0; i5 < dim5; i5++) {
                for (int i6 = 0; i6 < dim6; i6++) {
                  dv.access(i0,i1,i2,i3,i4,i5,i6) /= value;
                }
              }
            }
          }
        }
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

