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
  
  double remap(double t);
protected:
  ParametricFunction(ParametricFunctionPtr fxn, double t0, double t1);
  
  ParametricFunctionPtr underlyingFunction();
public:
  ParametricFunction();
  // override one of these, according to the space dimension
  virtual void value(double t, double &x);
  virtual void value(double t, double &x, double &y);
  virtual void value(double t, double &x, double &y, double &z);
  
  static ParametricFunctionPtr line(double x0, double y0, double x1, double y1);
  
  static ParametricFunctionPtr circle(double r, double x0, double y0);
  static ParametricFunctionPtr circularArc(double r, double x0, double y0, double theta0, double theta1);
  
  static std::vector< ParametricFunctionPtr > referenceCellEdges(unsigned cellTopoKey);
  static std::vector< ParametricFunctionPtr > referenceQuadEdges();
  static std::vector< ParametricFunctionPtr > referenceTriangleEdges();
  
  static ParametricFunctionPtr remapParameter(ParametricFunctionPtr fxn, double t0, double t1);
};

#endif
