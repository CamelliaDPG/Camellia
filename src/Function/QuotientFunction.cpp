//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "QuotientFunction.h"

#include "BasisCache.h"

using namespace Camellia;

template <typename Scalar>
QuotientFunction<Scalar>::QuotientFunction(TFunctionPtr<Scalar> f, TFunctionPtr<Scalar> scalarDivisor) : TFunction<Scalar>("QuotientFunction", f->rank() )
{
  if ( scalarDivisor->rank() != 0 )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
  }
  _f = f;
  _scalarDivisor = scalarDivisor;
  if (scalarDivisor->isZero())
  {
    cout << "WARNING: division by zero in QuotientFunction.\n";
  }
}

template <typename Scalar>
bool QuotientFunction<Scalar>::boundaryValueOnly()
{
  return _f->boundaryValueOnly() || _scalarDivisor->boundaryValueOnly();
}

template <typename Scalar>
string QuotientFunction<Scalar>::displayString()
{
  ostringstream ss;
  // use memberFunctions as a proxy for whether we should include parentheses (for sums and products, this does what we want)
  bool includeParentheses = _scalarDivisor->memberFunctions().size() > 1;
  if (includeParentheses)
    ss << _f->displayString() << " / (" << _scalarDivisor->displayString() << ")";
  else
    ss << _f->displayString() << " / " << _scalarDivisor->displayString();
  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_scalarDivisor, valueMap);
  return f1 / f2;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dx()
{
  if ( (_f->dx().get() == NULL) || (_scalarDivisor->dx().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dx() / _scalarDivisor - _f * _scalarDivisor->dx() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dy()
{
  if ( (_f->dy().get() == NULL) || (_scalarDivisor->dy().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dy() / _scalarDivisor - _f * _scalarDivisor->dy() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dz()
{
  if ( (_f->dz().get() == NULL) || (_scalarDivisor->dz().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dz() / _scalarDivisor - _f * _scalarDivisor->dz() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::dt()
{
  if ( (_f->dt().get() == NULL) || (_scalarDivisor->dt().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dt() / _scalarDivisor - _f * _scalarDivisor->dt() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
TLinearTermPtr<Scalar> QuotientFunction<Scalar>::jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_scalarDivisor, valueMap);
  auto df1 = _f->jacobian(valueMap);
  auto df2 = _scalarDivisor->jacobian(valueMap);
  return (1.0 / f2) * df1 - (f1 / ( f2 * f2 )) * df2;
}

template <typename Scalar>
std::vector<TFunctionPtr<Scalar>> QuotientFunction<Scalar>::memberFunctions()
{
  return {{_f, _scalarDivisor}};
}

template <typename Scalar>
void QuotientFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

template <typename Scalar>
void QuotientFunction<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType,  BasisCachePtr basisCache)
{
  if (modType == ASSIGN)
  {
    _f->values(values, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
  }
  else
  {
    auto quotientValues = FieldContainer<Scalar>::getDeviceView(values);
    _f->values(quotientValues, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _scalarDivisor->scalarDivideFunctionValues(quotientValues, basisCache);
    Kokkos::fence();
    const int dim6 = (values.rank() > 6) ? values.extent_int(6) : 1;
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    Kokkos::parallel_for("QuotientFunction_values", FieldContainer<Scalar>::getMDRangePolicy6(values),
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5)
    {
      for (int i6 = 0; i6 < dim6; i6++) {
        assignValue(values.access(i0,i1,i2,i3,i4,i5,i6), modType, quotientValues.access(i0,i1,i2,i3,i4,i5,i6));
      }
    });
    FieldContainer<Scalar>::recordDeviceViewDeallocation(quotientValues);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::t()
{
  return _f->t() / _scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::x()
{
  return _f->x() / _scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::y()
{
  return _f->y() / _scalarDivisor;
}

template <typename Scalar>
TFunctionPtr<Scalar> QuotientFunction<Scalar>::z()
{
  return _f->z() / _scalarDivisor;
}

namespace Camellia
{
template class QuotientFunction<double>;
}
