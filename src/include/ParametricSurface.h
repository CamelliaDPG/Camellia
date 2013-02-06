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
#include "Function.h"
#include "Mesh.h"

using namespace std;

class ParametricSurface;
typedef Teuchos::RCP<ParametricSurface> ParametricSurfacePtr;

class ParametricSurface : public Function {
public:
  ParametricSurface() : Function(1) { // vector valued
    
  }
  virtual void value(double t1, double t2, double &x, double &y) = 0;
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  
  static FieldContainer<double> &parametricQuadNodes(); // for CellTools cellWorkset argument
  
  static void basisWeightsForL2ProjectedInterpolant(FieldContainer<double> &basisCoefficients,
                                                    VectorBasisPtr basis, MeshPtr mesh, int cellID);
  static ParametricSurfacePtr linearInterpolant(const vector< ParametricCurvePtr > &curves);
  static ParametricSurfacePtr transfiniteInterpolant(const vector< ParametricCurvePtr > &curves);
};

#endif /* defined(__Camellia_debug__ParametricSurface__) */
