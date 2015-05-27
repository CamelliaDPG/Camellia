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

#include "VectorizedBasis.h"

using namespace std;

namespace Camellia
{
class ParametricSurface;
typedef Teuchos::RCP<ParametricSurface> ParametricSurfacePtr;

class ParametricSurface : public TFunction<double>
{
public:
  ParametricSurface() : TFunction<double>(1)   // vector valued
  {

  }
  virtual TFunctionPtr<double> dt1();
  virtual TFunctionPtr<double> dt2();

  virtual void value(double t1, double t2, double &x, double &y) = 0;
  virtual void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);

  static Intrepid::FieldContainer<double> &parametricQuadNodes(); // for CellTools cellWorkset argument

  static void basisWeightsForEdgeInterpolant(Intrepid::FieldContainer<double> &basisCoefficients,
      Camellia::VectorBasisPtr basis, MeshPtr mesh, int cellID);
  static void basisWeightsForProjectedInterpolant(Intrepid::FieldContainer<double> &basisCoefficients,
      Camellia::VectorBasisPtr basis, MeshPtr mesh, int cellID);
  static ParametricSurfacePtr linearInterpolant(const vector< ParametricCurvePtr > &curves);
  static ParametricSurfacePtr transfiniteInterpolant(const vector< ParametricCurvePtr > &curves);
};
}

#endif /* defined(__Camellia_debug__ParametricSurface__) */
