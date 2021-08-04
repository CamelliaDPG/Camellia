//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ConstantVectorFunction.h"

#include "BasisCache.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
ConstantVectorFunction<Scalar>::ConstantVectorFunction(vector<Scalar> value) : TFunction<Scalar>("ConstantVectorFunction",1)
{
  _value = value;
}

template <typename Scalar>
string ConstantVectorFunction<Scalar>::displayString()
{
  ostringstream str;
  str << "(";
  for (int i=0; i<_value.size(); i++)
  {
    if (i > 0) str << ",";
    str << _value[i];
  }
  str << ")";
  return str.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantVectorFunction<Scalar>::x()
{
  return Teuchos::rcp( new ConstantScalarFunction<Scalar>( _value[0] ) );
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantVectorFunction<Scalar>::y()
{
  return Teuchos::rcp( new ConstantScalarFunction<Scalar>( _value[1] ) );
}

template <typename Scalar>
TFunctionPtr<Scalar> ConstantVectorFunction<Scalar>::z()
{
  if (_value.size() > 2)
  {
    return Teuchos::rcp( new ConstantScalarFunction<Scalar>( _value[2] ) );
  }
  else
  {
    return Teuchos::null;
  }
}

template <typename Scalar>
vector<Scalar> ConstantVectorFunction<Scalar>::value()
{
  return _value;
}

template <typename Scalar>
bool ConstantVectorFunction<Scalar>::isZero()
{
  for (int d=0; d < _value.size(); d++)
  {
    if (0.0 != _value[d])
    {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void ConstantVectorFunction<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("ConstantVectorFunction values()");
  const int dim2 = values.extent_int(2);
  TEUCHOS_TEST_FOR_EXCEPTION(dim2 > _value.size(), std::invalid_argument, "spaceDim is greater than length of vector...");
  
  Camellia::FieldContainer<Scalar> value(_value.size());
  for (int i = 0; i < _value.size(); i++) { value(i) = _value[i]; }
  auto dv_value = value.checkoutDeviceView();
  
  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<3> >::array_index_type;
  Kokkos::parallel_for("ConstantVectorFunction_values", Camellia::FieldContainer<Scalar>::getMDRangePolicy3(values),
  KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2) {
    assignValue(values(i0,i1,i2), modType, dv_value(i2));
  });
  value.checkinDeviceView(false);
}

template <typename Scalar>
int ConstantVectorFunction<Scalar>::numVectorComponents() const
{
  return _value.size();
}

namespace Camellia
{
template class ConstantVectorFunction<double>;
}
