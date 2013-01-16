//
//  ParametricFunction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_ParametricFunction_h
#define Camellia_debug_ParametricFunction_h

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

class ParametricFunction;
typedef Teuchos::RCP<ParametricFunction> ParametricFunctionPtr;

class ParametricFunction {  
  // for now, we don't yet subclass Function.  In order to do this, need 1D BasisCache.
  // (not hard, but not yet done, and unclear whether this is important for present purposes.)
  ParametricFunctionPtr _underlyingFxn; // the original 0-to-1 function
  
  double _t0, _t1;
  
  double remap(double t) {
    // want to map (0,1) to (_t0,_t1)
    return _t0 + t * (_t1 - _t0);
  }
protected:
  ParametricFunction(ParametricFunctionPtr fxn, double t0, double t1) {
    _underlyingFxn = fxn;
    _t0 = t0;
    _t1 = t1;
  }
  
  ParametricFunctionPtr underlyingFunction() {
    return _underlyingFxn;
  }
public:
  ParametricFunction() {
    _t0 = 0;
    _t1 = 1;
  }
  // override one of these, according to the space dimension
  virtual void value(double t, double &x);
  virtual void value(double t, double &x, double &y);
  virtual void value(double t, double &x, double &y, double &z);
  
  static ParametricFunctionPtr line(double x0, double y0, double x1, double y1);
  
  static ParametricFunctionPtr remapParameter(ParametricFunctionPtr fxn, double t0, double t1);
};

#endif
