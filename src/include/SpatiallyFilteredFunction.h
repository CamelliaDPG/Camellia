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

namespace Camellia {
  template <typename Scalar>
  class SpatiallyFilteredFunction : public Function<Scalar> {
    FunctionPtr<Scalar> _f;
    SpatialFilterPtr _sf;

  public:
    SpatiallyFilteredFunction(FunctionPtr<Scalar> f, SpatialFilterPtr sf);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    FunctionPtr<Scalar> curl();
    FunctionPtr<Scalar> div();

    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();
  };
}

#endif
