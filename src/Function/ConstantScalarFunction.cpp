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
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  if (modType == ASSIGN)
  {
    Kokkos::deep_copy(values, _value);
  }
  else
  {
    const int dim0 = values.extent_int(0);
    const int dim1 = values.extent_int(1);
    Kokkos::parallel_for("ConstantScalarFunction mod assignement", dim0, KOKKOS_LAMBDA (const int i0) {
      for (int i1 = 0; i1 < dim1; i1++) {
        auto & entry = values(i0, i1);
        assignValue(entry, modType, _value);
      }
    });
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarModifyFunctionValues(Camellia::FieldContainer<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  Teuchos::Array<int> dim(values.rank());
  values.dimensions(dim);
  
  const bool copyFromHost = (modType != ASSIGN);
  auto dv = values.checkoutDeviceView(copyFromHost);

  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
  Kokkos::Array<array_index_type,6> mdRangeEnd = {1,1,1,1,1,1}; // fully specifies all 6 MD dimensions
  int numRanksToCopy = (dim.size() <= 6) ? dim.size() : 6;
  for (int i=0; i<numRanksToCopy; i++)
  {
    mdRangeEnd[i] = dim[i];
  }
  Kokkos::Array<array_index_type,6> mdRangeBegin = {0,0,0,0,0,0};
  const int dim6 = dv.extent_int(6);

  Kokkos::MDRangePolicy< Kokkos::Rank<6> > mdRangePolicy(mdRangeBegin,mdRangeEnd);

  auto value = _value;
  Kokkos::parallel_for("ConstantScalarFunction_scalarModifyFunctionValues", mdRangePolicy,
  KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                 const array_index_type i3, const array_index_type i4, const array_index_type i5)
  {
    for (int i6 = 0; i6<dim6; i6++)
    {
      assignValue(dv(i0,i1,i2,i3,i4,i5,i6),modType,value);
    }
  });
  values.checkinDeviceView(true); // true: copy to host
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (_value != 1.0)
  {
    this->scalarModifyFunctionValues(values, MULTIPLY, basisCache);
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (_value != 1.0)
  {
    this->scalarModifyFunctionValues(values, DIVIDE, basisCache);
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

