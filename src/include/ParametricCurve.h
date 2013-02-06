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
#include "Function.h"

using namespace std;

class ParametricCurve;
typedef Teuchos::RCP<ParametricCurve> ParametricCurvePtr;

class ParametricCurve : public Function {
  // for now, we don't yet subclass Function.  In order to do this, need 1D BasisCache.
  // (not hard, but not yet done, and unclear whether this is important for present purposes.)
  ParametricCurvePtr _underlyingFxn; // the original 0-to-1 function
  
  FunctionPtr _xFxn, _yFxn, _zFxn;
  
  double _t0, _t1;
  
  double remapForSubCurve(double t);
  
  FunctionPtr argumentMap();
  
//  void mapRefCellPointsToParameterSpace(FieldContainer<double> &refPoints);
protected:
  ParametricCurve(ParametricCurvePtr fxn, double t0, double t1);
  
  ParametricCurvePtr underlyingFunction();
public:
  ParametricCurve();
  ParametricCurve(FunctionPtr xFxn_x_as_t,
                  FunctionPtr yFxn_x_as_t = Function::null(),
                  FunctionPtr zFxn_x_as_t = Function::null());
  
  ParametricCurvePtr interpolatingLine();
  void projectionBasedInterpolant(FieldContainer<double> &basisCoefficients, BasisPtr basis1D, int component, bool useH1=true); // component 0 for x, 1 for y, 2 for z
  
  // override one of these, according to the space dimension
  virtual void value(double t, double &x);
  virtual void value(double t, double &x, double &y);
  virtual void value(double t, double &x, double &y, double &z);
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  
  virtual ParametricCurvePtr dt(); // the curve differentiated in t in each component.
  
  virtual FunctionPtr dx(); // same as dt() (overrides Function::dx())
  
  virtual FunctionPtr x();
  virtual FunctionPtr y();
  virtual FunctionPtr z();
  
  static ParametricCurvePtr bubble(ParametricCurvePtr edgeCurve);
  
  static ParametricCurvePtr line(double x0, double y0, double x1, double y1);
  
  static ParametricCurvePtr circle(double r, double x0, double y0);
  static ParametricCurvePtr circularArc(double r, double x0, double y0, double theta0, double theta1);
  
  static ParametricCurvePtr curve(FunctionPtr xFxn_x_as_t, FunctionPtr yFxn_x_as_t = Function::null(), FunctionPtr zFxn_x_as_t = Function::null());
  static ParametricCurvePtr curveUnion(vector< ParametricCurvePtr > curves, vector<double> weights = vector<double>());
  
  static ParametricCurvePtr polygon(vector< pair<double,double> > vertices, vector<double> weights = vector<double>());
  
  static vector< ParametricCurvePtr > referenceCellEdges(unsigned cellTopoKey);
  static vector< ParametricCurvePtr > referenceQuadEdges();
  static vector< ParametricCurvePtr > referenceTriangleEdges();
  
  static ParametricCurvePtr reverse(ParametricCurvePtr fxn);
  static ParametricCurvePtr subCurve(ParametricCurvePtr fxn, double t0, double t1); // t0: the start of the subcurve; t1: the end
};

#endif
