//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ConstantVectorFunction.h"

#include "BasisCache.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
ConstantVectorFunction<Scalar>::ConstantVectorFunction(vector<Scalar> value) : TFunction<Scalar>(1)
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
void ConstantVectorFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  int spaceDim = values.dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > _value.size(), std::invalid_argument, "spaceDim is greater than length of vector...");
  
  Camellia::FieldContainer<Scalar> value(_value.size());
  for (int i = 0; i < _value.size(); i++) { value(i) = _value[i]; }
  auto dv_value = value.checkoutDeviceView();
  
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  const int dim0 = dv.extent(0); // (C) numCells
  const int dim1 = dv.extent(1); // (P) numPoints
  const int dim2 = dv.extent(2); // (D) spaceDim
  
  Kokkos::parallel_for("ConstantVectorFunction_values", dim0, KOKKOS_LAMBDA (const int i0) {
    for (int i1 = 0; i1 < dim1; i1++) {
      for (int i2 = 0; i2 < dim2; i2++) {
        dv(i0,i1,i2) = dv_value(i2);
      }
    }
  });
  value.checkinDeviceView(false);
  values.checkinDeviceView();
}
namespace Camellia
{
template class ConstantVectorFunction<double>;
}
