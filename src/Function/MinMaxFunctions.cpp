#include "MinMaxFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

MinFunction::MinFunction(FunctionPtr<double> f1, FunctionPtr<double> f2) : Function<double>(f1->rank()) {
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "both functions must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}

bool MinFunction::boundaryValueOnly() {
  // if either summand is BVO, then so is the min...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

string MinFunction::displayString() {
  ostringstream ss;
  ss << "\\min( " << _f1->displayString() << " , " << _f2->displayString() << " )";
  return ss.str();
}

void MinFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  this->CHECK_VALUES_RANK(values);
  Intrepid::FieldContainer<double> values2(values);
  _f1->values(values,basisCache);
  _f2->values(values2,basisCache);
  for(int i = 0; i < values.size(); i++) {
    values[i] = std::min(values[i],values2[i]);
  }
}

FunctionPtr<double> MinFunction::x() {
  if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) ) {
    return null();
  }
  return min(_f1->x(),_f2->x());
}

FunctionPtr<double> MinFunction::y() {
  if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) ) {
    return null();
  }
  return min(_f1->y(),_f2->y());
}
FunctionPtr<double> MinFunction::z() {
  if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) ) {
    return null();
  }
  return min(_f1->z(),_f2->z());
}

MaxFunction::MaxFunction(FunctionPtr<double> f1, FunctionPtr<double> f2) : Function<double>(f1->rank()) {
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "both functions must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}

bool MaxFunction::boundaryValueOnly() {
  // if either summand is BVO, then so is the max...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

string MaxFunction::displayString() {
  ostringstream ss;
  ss << "\\max( " << _f1->displayString() << " , " << _f2->displayString() << " )";
  return ss.str();
}

void MaxFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  this->CHECK_VALUES_RANK(values);
  Intrepid::FieldContainer<double> values2(values);
  _f1->values(values,basisCache);
  _f2->values(values2,basisCache);
  for(int i = 0; i < values.size(); i++) {
    values[i] = std::max(values[i],values2[i]);
  }
}

FunctionPtr<double> MaxFunction::x() {
  if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) ) {
    return null();
  }
  return max(_f1->x(),_f2->x());
}

FunctionPtr<double> MaxFunction::y() {
  if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) ) {
    return null();
  }
  return max(_f1->y(),_f2->y());
}
FunctionPtr<double> MaxFunction::z() {
  if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) ) {
    return null();
  }
  return max(_f1->z(),_f2->z());
}
