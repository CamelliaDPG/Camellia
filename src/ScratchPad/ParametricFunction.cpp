//
//  ParametricFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "ParametricFunction.h"

void ParametricFunction::value(double t, double &x) {
  if (_underlyingFxn.get()) {
    _underlyingFxn->value(remap(t), x);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unimplemented method");
  }
}

void ParametricFunction::value(double t, double &x, double &y) {
  if (_underlyingFxn.get()) {
    _underlyingFxn->value(remap(t), x, y);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unimplemented method");
  }
}

void ParametricFunction::value(double t, double &x, double &y, double &z) {
  if (_underlyingFxn.get()) {
    _underlyingFxn->value(remap(t), x, y, z);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unimplemented method");
  }
}

ParametricFunctionPtr ParametricFunction::remappedParametricFunction(ParametricFunctionPtr fxn, double t0, double t1) {
  double t0_underlying = fxn->remap(t0);
  double t1_underlying = fxn->remap(t1);
  return Teuchos::rcp( new ParametricFunction(fxn->underlyingFunction(), t0_underlying, t1_underlying) );
}