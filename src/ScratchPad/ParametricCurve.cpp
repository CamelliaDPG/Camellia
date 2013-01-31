//
//  ParametricCurve.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "ParametricCurve.h"

#include "Shards_CellTopology.hpp"

static const double PI  = 3.141592653589793238462;

class ParametricUnion : public ParametricCurve {
  vector< ParametricCurvePtr > _curves;
  vector<double> _cutPoints;
  
  int matchingCurve(double t) {
    for (int i=0; i<_curves.size(); i++) {
      if ((t >= _cutPoints[i]) && (t<=_cutPoints[i+1])) {
        return i;
      }
    }
    return -1;
  }
public:
  ParametricUnion(const vector< ParametricCurvePtr > &curves, const vector<double> &weights) {
    _curves = curves;
    int numCurves = _curves.size();
    
    TEUCHOS_TEST_FOR_EXCEPTION(numCurves != weights.size(), std::invalid_argument, "must have same number of curves and weights");
    
    // make the weights add to 1.0
    double weightSum = 0;
    for (int i=0; i<numCurves; i++) {
      weightSum += weights[i];
    }
    _cutPoints.push_back(0);
//    cout << "_cutPoints: ";
//    cout << _cutPoints[0] << " ";
    for (int i=0; i<numCurves; i++) {
      _cutPoints.push_back(_cutPoints[i] + weights[i] / weightSum);
//      cout << _cutPoints[i+1] << " ";
    }
  }
  void value(double t, double &x, double &y) {
    int curveIndex = matchingCurve(t);
    // map t so that from curve's pov it spans (0,1)
    double curve_t0 = _cutPoints[curveIndex];
    double curve_t1 = _cutPoints[curveIndex+1];
    double curve_t = (t - curve_t0) / (curve_t1 - curve_t0);
    _curves[curveIndex]->value(curve_t, x,y);
  }
};

class ParametricLine : public ParametricCurve {
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

class ParametricCircle : public ParametricCurve {
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

ParametricCurve::ParametricCurve(ParametricCurvePtr fxn, double t0, double t1) {
  _underlyingFxn = fxn;
  _t0 = t0;
  _t1 = t1;
}

ParametricCurve::ParametricCurve() {
  _t0 = 0;
  _t1 = 1;
}

double ParametricCurve::remapForSubCurve(double t) {
  // want to map (0,1) to (_t0,_t1)
  return _t0 + t * (_t1 - _t0);
}

ParametricCurvePtr ParametricCurve::underlyingFunction() {
  return _underlyingFxn;
}

void ParametricCurve::value(double t, double &x) {
  if (_underlyingFxn.get()) {
    _underlyingFxn->value(remapForSubCurve(t), x);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unimplemented method");
  }
}

void ParametricCurve::value(double t, double &x, double &y) {
  if (_underlyingFxn.get()) {
    _underlyingFxn->value(remapForSubCurve(t), x, y);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unimplemented method");
  }
}

void ParametricCurve::value(double t, double &x, double &y, double &z) {
  if (_underlyingFxn.get()) {
    _underlyingFxn->value(remapForSubCurve(t), x, y, z);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unimplemented method");
  }
}

ParametricCurvePtr ParametricCurve::circle(double r, double x0, double y0) {
  return Teuchos::rcp( new ParametricCircle(r, x0, y0));
}

ParametricCurvePtr ParametricCurve::circularArc(double r, double x0, double y0, double theta0, double theta1) {
  ParametricCurvePtr circleFxn = circle(r, x0, y0);
  double t0 = theta0 / 2.0 * PI;
  double t1 = theta1 / 2.0 * PI;
  return subCurve(circleFxn, t0, t1);
}

ParametricCurvePtr ParametricCurve::curveUnion(vector< ParametricCurvePtr > curves, vector<double> weights) {
  int numCurves = curves.size();
  if (weights.size()==0) {
    // default to even weighting
    for (int i=1; i<=numCurves; i++) {
      weights.push_back(1.0);
    }
  }
  return Teuchos::rcp( new ParametricUnion(curves, weights) );
}

ParametricCurvePtr ParametricCurve::polygon(vector< pair<double,double> > vertices, vector<double> weights) {
  int numVertices = vertices.size();
  if (weights.size()==0) {
    // default to weighting by length
    double x_prev = vertices[0].first;
    double y_prev = vertices[0].second;
    for (int i=1; i<=numVertices; i++) {
      double x = vertices[i%numVertices].first;  // modulus to sweep back to first vertex
      double y = vertices[i%numVertices].second;
      double d = sqrt( (x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev));
      weights.push_back(d);
      x_prev = x;
      y_prev = y;
    }
  }
  vector< ParametricCurvePtr > lines;
  for (int i=0; i<numVertices; i++) {
    double x0 = vertices[i].first;
    double y0 = vertices[i].second;
    double x1 = vertices[(i+1)%numVertices].first;
    double y1 = vertices[(i+1)%numVertices].second;
    lines.push_back(line(x0, y0, x1, y1));
  }
  return curveUnion(lines,weights);
}


ParametricCurvePtr ParametricCurve::line(double x0, double y0, double x1, double y1) {
  return Teuchos::rcp(new ParametricLine(x0,y0,x1,y1));
}

std::vector< ParametricCurvePtr > ParametricCurve::referenceCellEdges(unsigned cellTopoKey) {
  if (cellTopoKey == shards::Quadrilateral<4>::key) {
    return referenceQuadEdges();
  } else if (cellTopoKey == shards::Triangle<3>::key) {
    return referenceTriangleEdges();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled cellTopoKey");
  }
}

std::vector< ParametricCurvePtr > ParametricCurve::referenceQuadEdges() {
  std::vector< ParametricCurvePtr > edges;
  edges.push_back(line(-1,-1,1,-1));
  edges.push_back(line(1,-1,1,1));
  edges.push_back(line(1,1,-1,1));
  edges.push_back(line(-1,1,-1,-1));
  return edges;
}

std::vector< ParametricCurvePtr > ParametricCurve::referenceTriangleEdges() {
  std::vector< ParametricCurvePtr > edges;
  edges.push_back(line(0,0,1,0));
  edges.push_back(line(1,0,0,1));
  edges.push_back(line(0,1,0,0));
  return edges;
}

ParametricCurvePtr ParametricCurve::reverse(ParametricCurvePtr fxn) {
  return subCurve(fxn, 1, 0);
}

ParametricCurvePtr ParametricCurve::subCurve(ParametricCurvePtr fxn, double t0, double t1) {
  double t0_underlying = fxn->remapForSubCurve(t0);
  double t1_underlying = fxn->remapForSubCurve(t1);
  ParametricCurvePtr underlyingFxn = (fxn->underlyingFunction().get()==NULL) ? fxn : fxn->underlyingFunction();
  return Teuchos::rcp( new ParametricCurve(underlyingFxn, t0_underlying, t1_underlying) );
}