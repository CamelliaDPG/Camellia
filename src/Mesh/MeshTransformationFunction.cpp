//
//  MeshTransformationFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "MeshTransformationFunction.h"
#include "ParametricFunction.h"

class CellTransformationFunction : public Function {
public:
  CellTransformationFunction(Mesh* mesh, int cellID, const vector< ParametricFunctionPtr > edgeFunctions) : Function(1) {
    
  }
};

MeshTransformationFunction::MeshTransformationFunction() : Function(1) { // vector-valued Function
  
}

void MeshTransformationFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  vector<int> cellIDs = basisCache->cellIDs();
  values = basisCache->getPhysicalCubaturePoints();
  // we'll do something different only where necessary (identity map is the right thing most of the time)
  for (int cellIndex=0; cellIndex < cellIDs.size(); cellIndex++) {
    
  }
}