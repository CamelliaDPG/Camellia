//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ConstantScalarFunction.h"

#include "BasisCache.h"

using namespace Camellia;

template <typename Scalar>
ConstantScalarFunction<Scalar>::ConstantScalarFunction(Scalar value)
: TFunction<Scalar>("ConstantScalarFunction",0,TFunction<Scalar>::VARIES_IN_NO_DIMENSION)
{
  _value = value;
  ostringstream valueStream;
  valueStream << value;
  _stringDisplay = valueStream.str();
}

template <typename Scalar>
ConstantScalarFunction<Scalar>::ConstantScalarFunction(Scalar value, string stringDisplay)
: TFunction<Scalar>("ConstantScalarFunction",0,TFunction<Scalar>::VARIES_IN_NO_DIMENSION)
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
  return 0.0 == this->value();
}

template <typename Scalar>
std::vector<Intrepid2::DimensionInfo> ConstantScalarFunction<Scalar>::valuesDataDims(BasisCachePtr basisCache)
{
  // get dimensions from base class
  std::vector<Intrepid2::DimensionInfo> dataDims = this->TFunction<Scalar>::valuesDataDims(basisCache);
  
  // override cell dimension: base class doesn't know that we're constant across cells
  const int numCells = dataDims[0].logicalExtent;
  auto cellDim  = TFunction<Scalar>::constantDimensionInfo(numCells);
  dataDims[0] = cellDim;
  
  return dataDims;
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::values(Intrepid2::Data<Scalar, Camellia::DeviceType> &values, BasisCachePtr basisCache)
{
  auto underlyingView1 = values.template getUnderlyingView<1>();
  
  const auto value = this->value();
  Kokkos::parallel_for("set values", 1,
  KOKKOS_LAMBDA (const int & /*entryOrdinal*/) {
    underlyingView1(0) = value;
  });
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("ConstantScalarFunction values()");
  if (modType == ASSIGN)
  {
    Kokkos::deep_copy(values, this->value());
  }
  else
  {
    auto value = this->value();
    const int dim6 = (values.rank() > 6) ? values.extent_int(6) : 1;
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    auto rangePolicy = FieldContainer<Scalar>::getMDRangePolicy6(values);
    Kokkos::parallel_for("ConstantScalarFunction mod assignment", rangePolicy,
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5) {
      for (int i6 = 0; i6 < dim6; i6++)
      {
        auto & entry = values.access(i0, i1, i2, i3, i4, i5, i6);
        assignValue(entry, modType, value);
      }
    });
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarModifyFunctionValues(Camellia::FieldContainer<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  const bool copyFromHost = (modType != ASSIGN);
  auto dv = values.checkoutDeviceView(copyFromHost);
  const int dim6 = (dv.rank() > 6) ? dv.extent_int(6) : 1;
  
  auto value = this->value();
  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
  Kokkos::parallel_for("ConstantScalarFunction_scalarModifyFunctionValues", values.getMDRangePolicy6(),
  KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                 const array_index_type i3, const array_index_type i4, const array_index_type i5)
  {
    for (int i6 = 0; i6<dim6; i6++)
    {
      assignValue(dv.access(i0,i1,i2,i3,i4,i5,i6),modType,value);
    }
  });
  values.checkinDeviceView(true); // true: copy to host
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (this->value() != 1.0)
  {
    this->scalarModifyFunctionValues(values, MULTIPLY, basisCache);
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (this->value() != 1.0)
  {
    this->scalarModifyFunctionValues(values, DIVIDE, basisCache);
  }
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  (void)basisCache;

  const Scalar value = this->value();
  if (value == 1.0)
  {
    return;
  }

  const bool overwritesValues = (value == 0.0);
  auto dv = basisValues.checkoutDeviceView(!overwritesValues);

  if (overwritesValues)
  {
    Kokkos::deep_copy(dv, 0.0);
  }
  else
  {
    auto data = dv.data();
    const int entryCount = basisValues.size();
    Kokkos::parallel_for("ConstantScalarFunction_scalarMultiplyBasisValues", entryCount,
    KOKKOS_LAMBDA (const int i)
    {
      data[i] *= value;
    });
  }
  basisValues.checkinDeviceView(true);
}

template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  (void)basisCache;

  const Scalar value = this->value();
  if (value == 1.0)
  {
    return;
  }

  auto dv = basisValues.checkoutDeviceView();
  auto data = dv.data();
  const int entryCount = basisValues.size();
  Kokkos::parallel_for("ConstantScalarFunction_scalarDivideBasisValues", entryCount,
  KOKKOS_LAMBDA (const int i)
  {
    data[i] /= value;
  });
  basisValues.checkinDeviceView(true);
}

template <typename Scalar>
Scalar ConstantScalarFunction<Scalar>::value() const
{
  return _value;
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantScalarFunction<Scalar>::di(int i)
{
  return TFunction<Scalar>::zero();
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

