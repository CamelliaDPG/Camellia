#include "ProductFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

string ProductFunction::displayString() {
  ostringstream ss;
  ss << _f1->displayString() << " \\cdot " << _f2->displayString();
  return ss.str();
}

FunctionPtr ProductFunction::dx() {
  if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dx() + _f2 * _f1->dx();
}

FunctionPtr ProductFunction::dy() {
  if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dy() + _f2 * _f1->dy();
}

FunctionPtr ProductFunction::dz() {
  if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dz() + _f2 * _f1->dz();
}

FunctionPtr ProductFunction::dt() {
  if ( (_f1->dt().get() == NULL) || (_f2->dt().get() == NULL) ) {
    return null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dt() + _f2 * _f1->dt();
}

FunctionPtr ProductFunction::x() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take x component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->x())) {
    return null();
  }
  return _f1 * _f2->x();
}

FunctionPtr ProductFunction::y() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take y component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->y())) {
    return null();
  }
  return _f1 * _f2->y();
}

FunctionPtr ProductFunction::z() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take z component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->z())) {
    return null();
  }
  return _f1 * _f2->z();
}

FunctionPtr ProductFunction::t() {
  if (this->rank() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take t component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (isNull(_f2->t())) {
    return null();
  }
  return _f1 * _f2->t();
}

int ProductFunction::productRank(FunctionPtr f1, FunctionPtr f2) {
  if (f1->rank() == f2->rank()) return 0;
  if (f1->rank() == 0) return f2->rank();
  if (f2->rank() == 0) return f1->rank();
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
  return -1;
}

ProductFunction::ProductFunction(FunctionPtr f1, FunctionPtr f2) : Function( productRank(f1,f2) ) {
  // for simplicity of values() code, ensure that rank of f1 ≤ rank of f2:
  if ( f1->rank() <= f2->rank() ) {
    _f1 = f1;
    _f2 = f2;
  } else {
    _f1 = f2;
    _f2 = f1;
  }
  // the following should be false for all the automatic products.  Added the test for debugging...
  if ((_f1->isZero()) || (_f2->isZero())) {
    cout << "Warning: creating a ProductFunction where one of the multiplicands is zero." << endl;
  }
}

bool ProductFunction::boundaryValueOnly() {
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

void ProductFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  if (( _f2->rank() > 0) && (this->rank() == 0)) { // tensor product resulting in scalar value
    _f2->valuesDottedWithTensor(values, _f1, basisCache);
  } else { // scalar multiplication by f1, then
    _f2->values(values,basisCache);
    _f1->scalarMultiplyFunctionValues(values, basisCache);
  }
}