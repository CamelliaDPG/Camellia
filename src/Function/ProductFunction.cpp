#include "ProductFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
string ProductFunction<Scalar>::displayString()
{
  ostringstream ss;
  ss << _f1->displayString() << " \\cdot " << _f2->displayString();
  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dx()
{
  if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dx() + _f2 * _f1->dx();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dy()
{
  if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dy() + _f2 * _f1->dy();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dz()
{
  if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dz() + _f2 * _f1->dz();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dt()
{
  if ( (_f1->dt().get() == NULL) || (_f2->dt().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dt() + _f2 * _f1->dt();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::x()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take x component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->x()))
  {
    return this->null();
  }
  return _f1 * _f2->x();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::y()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take y component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->y()))
  {
    return this->null();
  }
  return _f1 * _f2->y();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::z()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take z component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->z()))
  {
    return this->null();
  }
  return _f1 * _f2->z();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::t()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take t component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->t()))
  {
    return this->null();
  }
  return _f1 * _f2->t();
}

template <typename Scalar>
int ProductFunction<Scalar>::productRank(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  if (f1->rank() == f2->rank()) return 0;
  if (f1->rank() == 0) return f2->rank();
  if (f2->rank() == 0) return f1->rank();
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
  return -1;
}

template <typename Scalar>
ProductFunction<Scalar>::ProductFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2) : TFunction<Scalar>( productRank(f1,f2) )
{
  // for simplicity of values() code, ensure that rank of f1 ≤ rank of f2:
  if ( f1->rank() <= f2->rank() )
  {
    _f1 = f1;
    _f2 = f2;
  }
  else
  {
    _f1 = f2;
    _f2 = f1;
  }
  // the following should be false for all the automatic products.  Added the test for debugging...
  if ((_f1->isZero()) || (_f2->isZero()))
  {
    cout << "Warning: creating a ProductFunction where one of the multiplicands is zero." << endl;
  }
}

template <typename Scalar>
bool ProductFunction<Scalar>::boundaryValueOnly()
{
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

template <typename Scalar>
void ProductFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  if (( _f2->rank() > 0) && (this->rank() == 0))   // tensor product resulting in scalar value
  {
    _f2->valuesDottedWithTensor(values, _f1, basisCache);
  }
  else     // scalar multiplication by f1, then
  {
    _f2->values(values,basisCache);
    _f1->scalarMultiplyFunctionValues(values, basisCache);
  }
}

namespace Camellia
{
template class ProductFunction<double>;
}
