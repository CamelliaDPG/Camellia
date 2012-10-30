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

class SpatiallyFilteredFunction : public Function {
  FunctionPtr _f;
  SpatialFilterPtr _sf;
  
public:
  SpatiallyFilteredFunction(FunctionPtr f, SpatialFilterPtr sf);
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

#endif
