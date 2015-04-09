#include "SumFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

SumFunction::SumFunction(FunctionPtr f1, FunctionPtr f2) : Function(f1->rank()) {
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "summands must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}

bool SumFunction::boundaryValueOnly() {
  // if either summand is BVO, then so is the sum...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

string SumFunction::displayString() {
  ostringstream ss;
  ss << "(" << _f1->displayString() << " + " << _f2->displayString() << ")";
  return ss.str();
}

void SumFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  _f1->values(values,basisCache);
  _f2->addToValues(values,basisCache);
}

FunctionPtr SumFunction::x() {
  if ( (_f1->x() == Teuchos::null) || (_f2->x() == Teuchos::null) ) {
    return null();
  }
  return _f1->x() + _f2->x();
}

FunctionPtr SumFunction::y() {
  if ( (_f1->y() == Teuchos::null) || (_f2->y() == Teuchos::null) ) {
    return null();
  }
  return _f1->y() + _f2->y();
}
FunctionPtr SumFunction::z() {
  if ( (_f1->z() == Teuchos::null) || (_f2->z() == Teuchos::null) ) {
    return Teuchos::null;
  }
  return _f1->z() + _f2->z();
}
FunctionPtr SumFunction::t() {
  if ( (_f1->t() == Teuchos::null) || (_f2->t() == Teuchos::null) ) {
    return Teuchos::null;
  }
  return _f1->t() + _f2->t();
}

FunctionPtr SumFunction::dx() {
  if ( (_f1->dx() == Teuchos::null) || (_f2->dx() == Teuchos::null) ) {
    return null();
  }
  return _f1->dx() + _f2->dx();
}

FunctionPtr SumFunction::dy() {
  if ( (_f1->dy() == Teuchos::null) || (_f2->dy() == Teuchos::null) ) {
    return Teuchos::null;
  }
  return _f1->dy() + _f2->dy();
}

FunctionPtr SumFunction::dz() {
  if ( (_f1->dz() == Teuchos::null) || (_f2->dz() == Teuchos::null) ) {
    return Teuchos::null;
  }
  return _f1->dz() + _f2->dz();
}

FunctionPtr SumFunction::dt() {
  if ( (_f1->dt() == Teuchos::null) || (_f2->dt() == Teuchos::null) ) {
    return Teuchos::null;
  }
  return _f1->dt() + _f2->dt();
}

FunctionPtr SumFunction::grad(int numComponents) {
  if ( (_f1->grad(numComponents) == Teuchos::null) || (_f2->grad(numComponents) == Teuchos::null) ) {
    return Teuchos::null;
  } else {
    return _f1->grad(numComponents) + _f2->grad(numComponents);
  }
}

FunctionPtr SumFunction::div() {
  if ( (_f1->div() == Teuchos::null) || (_f2->div() == Teuchos::null) ) {
    return Teuchos::null;
  } else {
    return _f1->div() + _f2->div();
  }
}