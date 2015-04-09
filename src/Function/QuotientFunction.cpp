#include "QuotientFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

QuotientFunction::QuotientFunction(FunctionPtr f, FunctionPtr scalarDivisor) : Function( f->rank() ) {
  if ( scalarDivisor->rank() != 0 ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
  }
  _f = f;
  _scalarDivisor = scalarDivisor;
  if (scalarDivisor->isZero()) {
    cout << "WARNING: division by zero in QuotientFunction.\n";
  }
}

bool QuotientFunction::boundaryValueOnly() {
  return _f->boundaryValueOnly() || _scalarDivisor->boundaryValueOnly();
}

string QuotientFunction::displayString() {
  ostringstream ss;
  ss << _f->displayString() << " / " << _scalarDivisor->displayString();
  return ss.str();
}

void QuotientFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  _f->values(values,basisCache);
  _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
}

FunctionPtr QuotientFunction::dx() {
  if ( (_f->dx().get() == NULL) || (_scalarDivisor->dx().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dx() / _scalarDivisor - _f * _scalarDivisor->dx() / (_scalarDivisor * _scalarDivisor);
}

FunctionPtr QuotientFunction::dy() {
  if ( (_f->dy().get() == NULL) || (_scalarDivisor->dy().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dy() / _scalarDivisor - _f * _scalarDivisor->dy() / (_scalarDivisor * _scalarDivisor);
}

FunctionPtr QuotientFunction::dz() {
  if ( (_f->dz().get() == NULL) || (_scalarDivisor->dz().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dz() / _scalarDivisor - _f * _scalarDivisor->dz() / (_scalarDivisor * _scalarDivisor);
}

FunctionPtr QuotientFunction::dt() {
  if ( (_f->dt().get() == NULL) || (_scalarDivisor->dt().get() == NULL) ) {
    return null();
  }
  // otherwise, apply quotient rule:
  return _f->dt() / _scalarDivisor - _f * _scalarDivisor->dt() / (_scalarDivisor * _scalarDivisor);
}
