//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ProductFunction.h"

#include "BasisCache.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;

template <typename Scalar>
string ProductFunction<Scalar>::displayString()
{
  ostringstream ss;
  string times_string = " * ";
  if ((_f1->rank() == 1) && _f2->rank() == 1) times_string = " \\cdot ";
  
  ConstantScalarFunction<Scalar>* f1_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (_f1.get());
  if (f1_constant && (f1_constant->value() == -1.0))
  {
    // then just show "-f2"
    ss << "-" << _f2->displayString();
  }
  else
  {
    ss << _f1->displayString() << times_string << _f2->displayString();
  }
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
TFunctionPtr<Scalar> ProductFunction<Scalar>::evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f1, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_f2, valueMap);
  return f1 * f2;
}

template <typename Scalar>
bool ProductFunction<Scalar>::isZero(BasisCachePtr basisCache)
{
  return _f1->isZero(basisCache) || _f2->isZero(basisCache);
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::f1()
{
  return _f1;
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::f2()
{
  return _f2;
}

template <typename Scalar>
TLinearTermPtr<Scalar> ProductFunction<Scalar>::jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f1, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_f2, valueMap);
  auto df1 = _f1->jacobian(valueMap);
  auto df2 = _f2->jacobian(valueMap);
  return f1 * df2 + df1 * f2;
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
ProductFunction<Scalar>::ProductFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2) : TFunction<Scalar>("ProductFunction", productRank(f1,f2) )
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
std::vector<TFunctionPtr<Scalar>> ProductFunction<Scalar>::memberFunctions()
{
  return {{_f1, _f2}};
}

template <typename Scalar>
void ProductFunction<Scalar>::values(FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

template <typename Scalar>
void ProductFunction<Scalar>::values(typename FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  if (modType == ASSIGN)
  {
    if (( _f2->rank() > 0) && (this->rank() == 0)) // tensor product resulting in scalar value
    {
      _f2->valuesDottedWithTensor(values, _f1, basisCache);
    }
    else // scalar multiplication by f1, then
    {
      _f2->values(values, ValueModificationType::ASSIGN, basisCache);
      Kokkos::fence();
      _f1->scalarMultiplyFunctionValues(values, basisCache);
    }
  }
  else
  {
    auto valuesScratch = basisCache->checkoutScratchStorage<Scalar>(values);
    if (( _f2->rank() > 0) && (this->rank() == 0)) // tensor product resulting in scalar value
    {
      _f2->valuesDottedWithTensor(valuesScratch, _f1, basisCache);
    }
    else // scalar multiplication by f1, then
    {
      _f2->values(valuesScratch, ValueModificationType::ASSIGN, basisCache);
      Kokkos::fence();
      _f1->scalarMultiplyFunctionValues(valuesScratch, basisCache);
      
    }
    Kokkos::fence();

    const int dim6 = (values.rank() > 6) ? values.extent_int(6) : 1;
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    Kokkos::parallel_for("ProductFunction_values", FieldContainer<double>::getMDRangePolicy6(values),
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5)
    {
      for (int i6 = 0; i6 < dim6; i6++) {
        assignValue(values.access(i0,i1,i2,i3,i4,i5,i6), modType, valuesScratch.access(i0,i1,i2,i3,i4,i5,i6));
      }
    });
    basisCache->checkinScratchStorage<Scalar>(valuesScratch);
  }
}

namespace Camellia
{
template class ProductFunction<double>;
}
