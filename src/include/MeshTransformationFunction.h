//
//  MeshTransformationFunction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_MeshTransformationFunction_h
#define Camellia_debug_MeshTransformationFunction_h

#include "Function.h"

class MeshTransformationFunction : public Function {
  map< int, FunctionPtr > _curvedCells; // cellID --> cell transformation function
public:
  MeshTransformationFunction();
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

#endif
