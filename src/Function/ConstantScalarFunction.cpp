#include "ConstantScalarFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

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
void ConstantScalarFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  for (int i=0; i < values.size(); i++)
  {
    values[i] = _value;
  }
}
template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (_value != 1.0)
  {
    for (int i=0; i < values.size(); i++)
    {
      values[i] *= _value;
    }
  }
}
template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  if (_value != 1.0)
  {
    for (int i=0; i < values.size(); i++)
    {
      values[i] /= _value;
    }
  }
}
template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
{
  // we don't actually care about the shape of basisValues--just use the FunctionValues versions:
  scalarMultiplyFunctionValues(basisValues,basisCache);
}
template <typename Scalar>
void ConstantScalarFunction<Scalar>::scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache)
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

