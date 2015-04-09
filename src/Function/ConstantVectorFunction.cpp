#include "ConstantVectorFunction.h"

#include "BasisCache.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

ConstantVectorFunction::ConstantVectorFunction(vector<double> value) : Function(1) {
  _value = value;
}

FunctionPtr ConstantVectorFunction::x() {
  return Teuchos::rcp( new ConstantScalarFunction( _value[0] ) );
}

FunctionPtr ConstantVectorFunction::y() {
  return Teuchos::rcp( new ConstantScalarFunction( _value[1] ) );
}

FunctionPtr ConstantVectorFunction::z() {
  if (_value.size() > 2) {
    return Teuchos::rcp( new ConstantScalarFunction( _value[2] ) );
  } else {
    return Teuchos::null;
  }
}

vector<double> ConstantVectorFunction::value() {
  return _value;
}

bool ConstantVectorFunction::isZero() {
  for (int d=0; d < _value.size(); d++) {
    if (0.0 != _value[d]) {
      return false;
    }
  }
  return true;
}

void ConstantVectorFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int spaceDim = values.dimension(2);
  if (spaceDim > _value.size()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim is greater than length of vector...");
  }
  // values are stored in (C,P,D) order, the important thing here being that we can do this:
  for (int i=0; i < values.size(); ) {
    for (int d=0; d < spaceDim; d++) {
      values[i++] = _value[d];
    }
  }
}