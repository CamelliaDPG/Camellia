//
//  ParameterFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/1/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "ParameterFunction.h"

using namespace Intrepid;
using namespace Camellia;

ParameterFunction::ParameterFunction(double value) : Function(0) {
  setValue(value);
}

ParameterFunction::ParameterFunction(FunctionPtr fxn) : Function(fxn->rank()) {
  setValue(fxn);
}

FunctionPtr ParameterFunction::getValue() const {
  return _fxn;
}

void ParameterFunction::setValue(FunctionPtr fxn) {
  if ((_fxn.get() == NULL) || (fxn->rank() == _fxn->rank())) {
    _fxn = fxn;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ParameterFunction can't change rank!");
  }
}

void ParameterFunction::setValue(double value) {
  _fxn = Function::constant(value);
}

FunctionPtr ParameterFunction::x() {
  return _fxn->x();
}
FunctionPtr ParameterFunction::y() {
  return _fxn->y();
}
FunctionPtr ParameterFunction::z() {
  return _fxn->z();
}

FunctionPtr ParameterFunction::dx() {
  return _fxn->dx();
}
FunctionPtr ParameterFunction::dy() {
  return _fxn->dy();
}
FunctionPtr ParameterFunction::dz() {
  return _fxn->dz();
}

FunctionPtr ParameterFunction::grad(int numComponents) {
  return _fxn->grad(numComponents);
}
FunctionPtr ParameterFunction::div() {
  return _fxn->div();
}

void ParameterFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  _fxn->values(values, basisCache);
}
bool ParameterFunction::boundaryValueOnly() {
  return _fxn->boundaryValueOnly();
}

string ParameterFunction::displayString() {
  return _fxn->displayString();
}

ParameterFunctionPtr ParameterFunction::parameterFunction(double value) {
  return Teuchos::rcp( new ParameterFunction(value) );
}
ParameterFunctionPtr ParameterFunction::parameterFunction(FunctionPtr fxn) {
  return Teuchos::rcp( new ParameterFunction(fxn) );
}