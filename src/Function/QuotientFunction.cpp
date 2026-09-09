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
TFunctionPtr<Scalar> QuotientFunction<Scalar>::di(int i) // 1 for dx(), 2 for dy(), 3 for dz()
{
  auto f_di = _f->di(i);
  auto scalarDivisor_di = _scalarDivisor->di(i);
  if ( (f_di == Teuchos::null) || (scalarDivisor_di == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return f_di / _scalarDivisor - _f * scalarDivisor_di / (_scalarDivisor * _scalarDivisor);
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
int QuotientFunction<Scalar>::numVectorComponents() const
{
  return _f->numVectorComponents();
}

template <typename Scalar>
std::vector<Intrepid2::DimensionInfo> QuotientFunction<Scalar>::valuesDataDims(BasisCachePtr basisCache)
{
  auto fDims = _f->valuesDataDims(basisCache);
  auto divisorDims = _scalarDivisor->valuesDataDims(basisCache);
  
  // if f is not scalar, we would expect the data dims to be of different sizes; right now,
  // we are only supporting Intrepid2::Data for scalar f.
  TEUCHOS_TEST_FOR_EXCEPTION(fDims.size() != divisorDims.size(), std::invalid_argument, "Only scalar f supported by valuesDataDims() at present");
  
  const int numDims = fDims.size();
  std::vector<Intrepid2::DimensionInfo> combinedDims(numDims);
  for (int d=0; d<numDims; d++)
  {
    combinedDims[d] = Intrepid2::combinedDimensionInfo(fDims[d], divisorDims[d]);
  }
  return combinedDims;
}

template <typename Scalar>
void QuotientFunction<Scalar>::values(Intrepid2::Data<Scalar, Camellia::DeviceType> &values, BasisCachePtr basisCache)
{
  auto fData       = _f->checkoutValuesData(basisCache);
  auto divisorData = _scalarDivisor->checkoutValuesData(basisCache);
  
  _f->values(fData,basisCache);
  _scalarDivisor->values(divisorData,basisCache);
  
  values.storeInPlaceQuotient(fData, divisorData);
  
  _f->checkinValuesData(fData);
  _scalarDivisor->checkinValuesData(divisorData);
}

template <typename Scalar>
void QuotientFunction<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType,  BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("QuotientFunction values()");
  if (modType == ASSIGN)
  {
    _f->values(values, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
  }
  else
  {
    auto quotientValues = basisCache->checkoutScratchStorage<Scalar>(values);
    _f->values(quotientValues, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _scalarDivisor->scalarDivideFunctionValues(quotientValues, basisCache);
    Kokkos::fence();
    assignValues<Scalar>(values, modType, quotientValues);
    basisCache->checkinScratchStorage<Scalar>(quotientValues);
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
