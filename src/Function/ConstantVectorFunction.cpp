#include "ConstantVectorFunction.h"

#include "BasisCache.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

template <typename Scalar>
ConstantVectorFunction<Scalar>::ConstantVectorFunction(vector<Scalar> value) : TFunction<Scalar>(1)
{
  _value = value;
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
void ConstantVectorFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  int spaceDim = values.dimension(2);
  if (spaceDim > _value.size())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim is greater than length of vector...");
  }
  // values are stored in (C,P,D) order, the important thing here being that we can do this:
  for (int i=0; i < values.size(); )
  {
    for (int d=0; d < spaceDim; d++)
    {
      values[i++] = _value[d];
    }
  }
}

namespace Camellia
{
template class ConstantVectorFunction<double>;
}
