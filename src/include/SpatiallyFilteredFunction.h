//
//  SpatiallyFilteredFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatiallyFilteredFunction_h
#define Camellia_SpatiallyFilteredFunction_h

#include "Function.h"
#include "SpatialFilter.h"

namespace Camellia
{
template <typename Scalar>
class SpatiallyFilteredFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f;
  SpatialFilterPtr _sf;

public:
  SpatiallyFilteredFunction(TFunctionPtr<Scalar> f, SpatialFilterPtr sf);
  virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();

  TFunctionPtr<Scalar> curl();
  TFunctionPtr<Scalar> div();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
};
}

#endif
