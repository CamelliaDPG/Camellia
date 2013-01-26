//
//  ParametricCurve.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_ParametricCurve_h
#define Camellia_debug_ParametricCurve_h

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

class ParametricCurve;
typedef Teuchos::RCP<ParametricCurve> ParametricCurvePtr;

class ParametricCurve {  
  // for now, we don't yet subclass Function.  In order to do this, need 1D BasisCache.
  // (not hard, but not yet done, and unclear whether this is important for present purposes.)
  ParametricCurvePtr _underlyingFxn; // the original 0-to-1 function
  
  double _t0, _t1;
  
  double remap(double t);
protected:
  ParametricCurve(ParametricCurvePtr fxn, double t0, double t1);
  
  ParametricCurvePtr underlyingFunction();
public:
  ParametricCurve();
  // override one of these, according to the space dimension
  virtual void value(double t, double &x);
  virtual void value(double t, double &x, double &y);
  virtual void value(double t, double &x, double &y, double &z);
  
  static ParametricCurvePtr line(double x0, double y0, double x1, double y1);
  
  static ParametricCurvePtr circle(double r, double x0, double y0);
  static ParametricCurvePtr circularArc(double r, double x0, double y0, double theta0, double theta1);
  
  static ParametricCurvePtr polygon(vector< pair<double,double> > vertices);
  static ParametricCurvePtr polygon(vector< pair<double,double> > vertices, vector<double> weights);
  
  static std::vector< ParametricCurvePtr > referenceCellEdges(unsigned cellTopoKey);
  static std::vector< ParametricCurvePtr > referenceQuadEdges();
  static std::vector< ParametricCurvePtr > referenceTriangleEdges();
  
  static ParametricCurvePtr subCurve(ParametricCurvePtr fxn, double t0, double t1);
};

#endif
