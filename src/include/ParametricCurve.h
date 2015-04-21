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

#include "Basis.h"

using namespace std;

namespace Camellia {
  class ParametricFunction : public TFunction<double> {
    typedef Teuchos::RCP<ParametricFunction> ParametricFunctionPtr;

    TFunctionPtr<double> _underlyingFxn; // the original 0-to-1 function
    TFunctionPtr<double> _argMap; // maps the t values from (0,1) on sub-curve into (t0,t1) on curve
    double _derivativeOrder;
    double _t0, _t1;

    void setArgumentMap();

    double remapForSubCurve(double t);
  protected:
    TFunctionPtr<double> underlyingFunction();

    ParametricFunction(TFunctionPtr<double> fxn, double t0, double t1, int derivativeOrder=0);
  public:
    ParametricFunction(TFunctionPtr<double> fxn);
    void value(double t, double &x);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);

    TFunctionPtr<double> dx(); // same function as dt_parametric()
    ParametricFunctionPtr dt_parametric(); // not named dt() to avoid conflict with Function, which means space-time by this...

    ParametricFunctionPtr subFunction(double t0, double t1);

    // parametric function: function on refCellPoints mapped to [0,1]
    static ParametricFunctionPtr parametricFunction(TFunctionPtr<double> fxn, double t0=0, double t1=1);
  };
  typedef Teuchos::RCP<ParametricFunction> ParamatricFunctionPtr;

  class ParametricCurve : public TFunction<double> {
  public:
    typedef Teuchos::RCP<ParametricCurve> ParametricCurvePtr;
  private:
    ParamatricFunctionPtr _xFxn, _yFxn, _zFxn; // parametric functions (defined on ref line mapped to [0,1])
    TFunctionPtr<double> argumentMap();

  //  void mapRefCellPointsToParameterSpace(Intrepid::FieldContainer<double> &refPoints);
  protected:
  //  ParametricCurve(ParametricCurvePtr fxn, double t0, double t1);
    public:
    ParametricCurve();
    ParametricCurve(ParamatricFunctionPtr xFxn_x_as_t,
                    ParamatricFunctionPtr yFxn_x_as_t = Teuchos::rcp((ParametricFunction*)NULL),
                    ParamatricFunctionPtr zFxn_x_as_t = Teuchos::rcp((ParametricFunction*)NULL));


    ParametricCurvePtr interpolatingLine();

    double linearLength();
    void projectionBasedInterpolant(Intrepid::FieldContainer<double> &basisCoefficients, BasisPtr basis1D, int component,
                                    double lengthScale, bool useH1); // component 0 for x, 1 for y, 2 for z

    // override one of these, according to the space dimension
    virtual void value(double t, double &x);
    virtual void value(double t, double &x, double &y);
    virtual void value(double t, double &x, double &y, double &z);

    virtual void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);

    virtual ParametricCurvePtr dt_parametric(); // the curve differentiated in t in each component.

    virtual TFunctionPtr<double> x();
    virtual TFunctionPtr<double> y();
    virtual TFunctionPtr<double> z();

    virtual ParamatricFunctionPtr xPart();
    virtual ParamatricFunctionPtr yPart();
    virtual ParamatricFunctionPtr zPart();

    static ParametricCurvePtr bubble(ParametricCurvePtr edgeCurve);

    static ParametricCurvePtr line(double x0, double y0, double x1, double y1);

    static ParametricCurvePtr circle(double r, double x0, double y0);
    static ParametricCurvePtr circularArc(double r, double x0, double y0, double theta0, double theta1);

    static ParametricCurvePtr curve(TFunctionPtr<double> xFxn_x_as_t, TFunctionPtr<double> yFxn_x_as_t = TFunction<double>::null(), TFunctionPtr<double> zFxn_x_as_t = TFunction<double>::null());
    static ParametricCurvePtr curveUnion(vector< ParametricCurvePtr > curves, vector<double> weights = vector<double>());

    static ParametricCurvePtr polygon(vector< pair<double,double> > vertices, vector<double> weights = vector<double>());

    static vector< ParametricCurvePtr > referenceCellEdges(Camellia::CellTopologyKey cellTopoKey);
    static vector< ParametricCurvePtr > referenceQuadEdges();
    static vector< ParametricCurvePtr > referenceTriangleEdges();

    static ParametricCurvePtr reverse(ParametricCurvePtr fxn);
    static ParametricCurvePtr subCurve(ParametricCurvePtr fxn, double t0, double t1); // t0: the start of the subcurve; t1: the end

    static TFunctionPtr<double> parametricGradientWrapper(TFunctionPtr<double> parametricGradient, bool convertBasisCacheToParametricSpace = false); // translates gradients from parametric to physical space.  If convertBasisCacheToParametricSpace is false, that's essentially an assertion that the Function passed in does its own conversion to parametric space for the points, and the only responsibility of the wrapper is the Piola transform.  If convertBasisCacheToParametricSpace is true, then the points will be remapped by the wrapper as well.  Right now, the convertBasisCacheToParametricSpace = false is the one in active use in the production code, because there the TFunctionPtr<double> is a ParametricSurface that does the spatial mapping, while we use convertBasisCacheToParametricSpace = true for a test where a Function is more manually constructed.
  };
  typedef Teuchos::RCP<ParametricCurve> ParametricCurvePtr;
}



#endif
