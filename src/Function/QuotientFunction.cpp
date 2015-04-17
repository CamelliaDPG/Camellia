#include "QuotientFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
QuotientFunction<Scalar>::QuotientFunction(FunctionPtr<Scalar> f, FunctionPtr<Scalar> scalarDivisor) : Function<Scalar>( f->rank() ) {
  if ( scalarDivisor->rank() != 0 ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
  }
  _f = f;
  _scalarDivisor = scalarDivisor;
  if (scalarDivisor->isZero()) {
    cout << "WARNING: division by zero in QuotientFunction.\n";
  }
}

template <typename Scalar>
bool QuotientFunction<Scalar>::boundaryValueOnly() {
  return _f->boundaryValueOnly() || _scalarDivisor->boundaryValueOnly();
}

template <typename Scalar>
string QuotientFunction<Scalar>::displayString() {
  ostringstream ss;
  ss << _f->displayString() << " / " << _scalarDivisor->displayString();
  return ss.str();
}

template <typename Scalar>
void QuotientFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
  this->CHECK_VALUES_RANK(values);
  _f->values(values,basisCache);
  _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
}

template <typename Scalar>
FunctionPtr<Scalar> QuotientFunction<Scalar>::dx() {
  if ( (_f->dx().get() == NULL) || (_scalarDivisor->dx().get() == NULL) ) {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dx() / _scalarDivisor - _f * _scalarDivisor->dx() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
FunctionPtr<Scalar> QuotientFunction<Scalar>::dy() {
  if ( (_f->dy().get() == NULL) || (_scalarDivisor->dy().get() == NULL) ) {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dy() / _scalarDivisor - _f * _scalarDivisor->dy() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
FunctionPtr<Scalar> QuotientFunction<Scalar>::dz() {
  if ( (_f->dz().get() == NULL) || (_scalarDivisor->dz().get() == NULL) ) {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dz() / _scalarDivisor - _f * _scalarDivisor->dz() / (_scalarDivisor * _scalarDivisor);
}

template <typename Scalar>
FunctionPtr<Scalar> QuotientFunction<Scalar>::dt() {
  if ( (_f->dt().get() == NULL) || (_scalarDivisor->dt().get() == NULL) ) {
    return this->null();
  }
  // otherwise, apply quotient rule:
  return _f->dt() / _scalarDivisor - _f * _scalarDivisor->dt() / (_scalarDivisor * _scalarDivisor);
}

template class QuotientFunction<double>;

