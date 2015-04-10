#include "ConstantScalarFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

ConstantScalarFunction::ConstantScalarFunction(double value) {
  _value = value;
  ostringstream valueStream;
  valueStream << value;
  _stringDisplay = valueStream.str();
}

ConstantScalarFunction::ConstantScalarFunction(double value, string stringDisplay) {
  _value = value;
  _stringDisplay = stringDisplay;
}

string ConstantScalarFunction::displayString() {
  return _stringDisplay;
}

bool ConstantScalarFunction::isZero() {
  return 0.0 == _value;
}

void ConstantScalarFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  for (int i=0; i < values.size(); i++) {
    values[i] = _value;
  }
}
void ConstantScalarFunction::scalarMultiplyFunctionValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_value != 1.0) {
    for (int i=0; i < values.size(); i++) {
      values[i] *= _value;
    }
  }
}
void ConstantScalarFunction::scalarDivideFunctionValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_value != 1.0) {
    for (int i=0; i < values.size(); i++) {
      values[i] /= _value;
    }
  }
}
void ConstantScalarFunction::scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // we don't actually care about the shape of basisValues--just use the FunctionValues versions:
  scalarMultiplyFunctionValues(basisValues,basisCache);
}
void ConstantScalarFunction::scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  scalarDivideFunctionValues(basisValues,basisCache);
}

double ConstantScalarFunction::value(double x) {
  return value();
}

double ConstantScalarFunction::value(double x, double y) {
  return value();
}

double ConstantScalarFunction::value(double x, double y, double z) {
  return value();
}

double ConstantScalarFunction::value() {
  return _value;
}

FunctionPtr ConstantScalarFunction::dx() {
  return Function::zero();
}

FunctionPtr ConstantScalarFunction::dy() {
  return Function::zero();
}

FunctionPtr ConstantScalarFunction::dz() {
  return Function::zero();
}

FunctionPtr ConstantScalarFunction::dt() {
  return Function::zero();
}