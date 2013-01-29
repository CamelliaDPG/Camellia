//
//  ParametricSurface.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#include "ParametricSurface.h"

class ParametricBubble : public ParametricCurve {
  ParametricCurvePtr _edgeCurve;
  double _x0, _y0, _x1, _y1;
public:
  ParametricBubble(ParametricCurvePtr edgeCurve) {
    _edgeCurve = edgeCurve;
    _edgeCurve->value(0, _x0,_y0);
    _edgeCurve->value(1, _x1,_y1);
  }
  void value(double t, double &x, double &y) {
    _edgeCurve->value(t, x,y);
    x -= _x0*(1.0-t) + _x1*t;
    y -= _y0*(1.0-t) + _y1*t;
  }
  static ParametricCurvePtr bubble(ParametricCurvePtr edgeCurve) {
    return Teuchos::rcp( new ParametricBubble(edgeCurve) );
  }
};

class TransfiniteInterpolatingSurface : public ParametricSurface {
  vector< ParametricCurvePtr > _curves;
  vector< pair<double, double> > _vertices;
public:
  TransfiniteInterpolatingSurface(const vector< ParametricCurvePtr > &curves) {
    _curves = curves;
    for (int i=0; i<curves.size(); i++) {
      _vertices.push_back(make_pair(0,0));
      _curves[i]->value(0, _vertices[i].first, _vertices[i].second);
    }
    if (_curves.size() != 4) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
    }
    // we assume that the curves go CCW around the element; we flip the two opposite edges
    // so both sets of opposite edges run parallel to each other:
    _curves[2] = ParametricCurve::reverse(_curves[2]);
    _curves[3] = ParametricCurve::reverse(_curves[3]);
    
    // since we keep _vertices separately, can just store bubble functions in _curves
    _curves[0] = ParametricBubble::bubble(_curves[0]);
    _curves[1] = ParametricBubble::bubble(_curves[1]);
    _curves[2] = ParametricBubble::bubble(_curves[2]);
    _curves[3] = ParametricBubble::bubble(_curves[3]);
  }
  void value(double t1, double t2, double &x, double &y);
};

void TransfiniteInterpolatingSurface::value(double t1, double t2, double &x, double &y) {
  if (_curves.size() == 4) {
    // t1 indexes curves 0 and 2, t2 1 and 3
    double x0, y0, x2, y2;
    _curves[0]->value(t1, x0,y0);
    _curves[2]->value(t1, x2,y2);
    double x1, y1, x3, y3;
    _curves[1]->value(t2, x1,y1);
    _curves[3]->value(t2, x3,y3);
    x = _vertices[0].first*(1-t1)*(1-t2) + _vertices[1].first*   t1 *(1-t2)
      + _vertices[2].first*   t1*    t2  + _vertices[3].first*(1-t1)*   t2
      + x0*(1-t2) + x1 * t1 + x2*t2 + x3*(1-t1);
    
    y = _vertices[0].second*(1-t1)*(1-t2) + _vertices[1].second*   t1 *(1-t2)
      + _vertices[2].second*   t1*    t2  + _vertices[3].second*(1-t1)*   t2
      + y0*(1-t2) + y1 * t1 + y2*t2 + y3*(1-t1);
    
  } else if (_curves.size() == 3) {
    // TODO: implement this
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads and triangles supported for now...");
  }
  
}

ParametricSurfacePtr ParametricSurface::transfiniteInterpolant(const vector< ParametricCurvePtr > &curves) {
  return Teuchos::rcp( new TransfiniteInterpolatingSurface(curves) );
}