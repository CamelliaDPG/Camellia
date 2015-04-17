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

ParameterFunction::ParameterFunction(double value) : Function<double>(0) {
  setValue(value);
}

ParameterFunction::ParameterFunction(FunctionPtr<double> fxn) : Function<double>(fxn->rank()) {
  setValue(fxn);
}

FunctionPtr<double> ParameterFunction::getValue() const {
  return _fxn;
}

void ParameterFunction::setValue(FunctionPtr<double> fxn) {
  if ((_fxn.get() == NULL) || (fxn->rank() == _fxn->rank())) {
    _fxn = fxn;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ParameterFunction can't change rank!");
  }
}

void ParameterFunction::setValue(double value) {
  _fxn = Function::constant(value);
}

FunctionPtr<double> ParameterFunction::x() {
  return _fxn->x();
}
FunctionPtr<double> ParameterFunction::y() {
  return _fxn->y();
}
FunctionPtr<double> ParameterFunction::z() {
  return _fxn->z();
}

FunctionPtr<double> ParameterFunction::dx() {
  return _fxn->dx();
}
FunctionPtr<double> ParameterFunction::dy() {
  return _fxn->dy();
}
FunctionPtr<double> ParameterFunction::dz() {
  return _fxn->dz();
}

FunctionPtr<double> ParameterFunction::grad(int numComponents) {
  return _fxn->grad(numComponents);
}
FunctionPtr<double> ParameterFunction::div() {
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

Teuchos::RCP<ParameterFunction> ParameterFunction::parameterFunction(double value) {
  return Teuchos::rcp( new ParameterFunction(value) );
}
Teuchos::RCP<ParameterFunction> ParameterFunction::parameterFunction(FunctionPtr<double> fxn) {
  return Teuchos::rcp( new ParameterFunction(fxn) );
}
