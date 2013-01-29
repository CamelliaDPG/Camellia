//
//  ParametricSurface.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#ifndef __Camellia_debug__ParametricSurface__
#define __Camellia_debug__ParametricSurface__

#include <iostream>

#include "ParametricCurve.h"

using namespace std;

class ParametricSurface;
typedef Teuchos::RCP<ParametricSurface> ParametricSurfacePtr;

class ParametricSurface {
public:
  virtual void value(double t1, double t2, double &x, double &y) = 0;
  
  static ParametricSurfacePtr transfiniteInterpolant(const vector< ParametricCurvePtr > &curves);
};

#endif /* defined(__Camellia_debug__ParametricSurface__) */
