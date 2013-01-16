//
//  ParametricFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "ParametricFunction.h"

class ParametricLine : public ParametricFunction {
  double _x0, _y0, _x1, _y1;
public:
  ParametricLine(double x0, double y0, double x1, double y1) {
    _x0 = x0;
    _y0 = y0;
    _x1 = x1;
    _y1 = y1;
  }
  void value(double t, double &x, double &y) {
    x = t * (_x1-_x0) + _x0;
    y = t * (_y1-_y0) + _y0;
  }
};

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

ParametricFunctionPtr ParametricFunction::line(double x0, double y0, double x1, double y1) {
  return Teuchos::rcp(new ParametricLine(x0,y0,x1,y1));
}

ParametricFunctionPtr ParametricFunction::remapParameter(ParametricFunctionPtr fxn, double t0, double t1) {
  double t0_underlying = fxn->remap(t0);
  double t1_underlying = fxn->remap(t1);
  ParametricFunctionPtr underlyingFxn = (fxn->underlyingFunction().get()==NULL) ? fxn : fxn->underlyingFunction();
  return Teuchos::rcp( new ParametricFunction(underlyingFxn, t0_underlying, t1_underlying) );
}