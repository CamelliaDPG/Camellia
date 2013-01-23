//
//  ParametricFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "ParametricFunction.h"

#include "Shards_CellTopology.hpp"

static const double PI  = 3.141592653589793238462;

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

class ParametricCircle : public ParametricFunction {
  double _x0, _y0; // center coords
  double _r; // radius
  
public:
  ParametricCircle(double r, double x0, double y0) {
    _r = r;
    _x0 = x0;
    _y0 = y0;
  }
  
  void value(double t, double &x, double &y) {
    double theta = t * 2.0 * PI;
    
    x = _r * cos(theta) + _x0;
    y = _r * sin(theta) + _y0;
  }
};

ParametricFunction::ParametricFunction(ParametricFunctionPtr fxn, double t0, double t1) {
  _underlyingFxn = fxn;
  _t0 = t0;
  _t1 = t1;
}

ParametricFunction::ParametricFunction() {
  _t0 = 0;
  _t1 = 1;
}

double ParametricFunction::remap(double t) {
  // want to map (0,1) to (_t0,_t1)
  return _t0 + t * (_t1 - _t0);
}

ParametricFunctionPtr ParametricFunction::underlyingFunction() {
  return _underlyingFxn;
}

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

ParametricFunctionPtr ParametricFunction::circle(double r, double x0, double y0) {
  return Teuchos::rcp( new ParametricCircle(r, x0, y0));
}

ParametricFunctionPtr ParametricFunction::circularArc(double r, double x0, double y0, double theta0, double theta1) {
  ParametricFunctionPtr circleFxn = circle(r, x0, y0);
  double t0 = theta0 / 2.0 * PI;
  double t1 = theta1 / 2.0 * PI;
  return remapParameter(circleFxn, t0, t1);
}

ParametricFunctionPtr ParametricFunction::line(double x0, double y0, double x1, double y1) {
  return Teuchos::rcp(new ParametricLine(x0,y0,x1,y1));
}

std::vector< ParametricFunctionPtr > ParametricFunction::referenceCellEdges(unsigned cellTopoKey) {
  if (cellTopoKey == shards::Quadrilateral<4>::key) {
    return referenceQuadEdges();
  } else if (cellTopoKey == shards::Triangle<3>::key) {
    return referenceTriangleEdges();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled cellTopoKey");
  }
}

std::vector< ParametricFunctionPtr > ParametricFunction::referenceQuadEdges() {
  std::vector< ParametricFunctionPtr > edges;
  edges.push_back(line(-1,-1,1,-1));
  edges.push_back(line(1,-1,1,1));
  edges.push_back(line(1,1,-1,1));
  edges.push_back(line(-1,1,-1,-1));
  return edges;
}

std::vector< ParametricFunctionPtr > ParametricFunction::referenceTriangleEdges() {
  std::vector< ParametricFunctionPtr > edges;
  edges.push_back(line(0,0,1,0));
  edges.push_back(line(1,0,0,1));
  edges.push_back(line(0,1,0,0));
  return edges;
}

ParametricFunctionPtr ParametricFunction::remapParameter(ParametricFunctionPtr fxn, double t0, double t1) {
  double t0_underlying = fxn->remap(t0);
  double t1_underlying = fxn->remap(t1);
  ParametricFunctionPtr underlyingFxn = (fxn->underlyingFunction().get()==NULL) ? fxn : fxn->underlyingFunction();
  return Teuchos::rcp( new ParametricFunction(underlyingFxn, t0_underlying, t1_underlying) );
}