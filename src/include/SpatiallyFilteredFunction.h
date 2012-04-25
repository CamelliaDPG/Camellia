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

class SpatiallyFilteredFunction : public Function {
  FunctionPtr _f;
  SpatialFilterPtr _sf;
  
public:
  SpatiallyFilteredFunction(FunctionPtr f, SpatialFilterPtr sf) : Function(f->rank()) {
    _f = f;
    _sf = sf;
  }
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    values.initialize(0.0);

    Teuchos::Array<int> dim;
    values.dimensions(dim);
    Teuchos::Array<int> fValuesDim = dim;
    int entriesPerPoint = 1;
    for (int d=2; d<values.rank(); d++) {
      entriesPerPoint *= dim[d];
      dim[d] = 0; // clear so that these indices point to the start of storage for (cellIndex,ptIndex)
    }
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    FieldContainer<bool> pointsMatch(numCells,numPoints);
    if (_sf->matchesPoints(pointsMatch,basisCache)) { // SOME point matches
//      cout << "pointsMatch:\n" << pointsMatch;
      FieldContainer<double> fValues(fValuesDim);
      _f->values(fValues,basisCache);
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        dim[0] = cellIndex;
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          dim[1] = ptIndex;
          if (pointsMatch(cellIndex,ptIndex)) {
            double* value = &values[values.getEnumeration(dim)];
            double* fValue = &fValues[fValues.getEnumeration(dim)];
            for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
              *value++ = *fValue++;
            }
          }
        }
      }
    }
  }
};

#endif
