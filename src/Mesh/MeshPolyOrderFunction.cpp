//
//  MeshPolyOrderFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "MeshPolyOrderFunction.h"

void MeshPolyOrderFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  vector<int> cellIDs = basisCache->cellIDs();
  int cellIndex = 0;
  int numPoints = values.dimension(1);
  for (vector<int>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++, cellIndex++) {
    int cellID = *cellIDIt;
    int polyOrder = _mesh->cellPolyOrder(cellID); // H1 order
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      values(cellIndex, ptIndex) = polyOrder-1;
    }
  }
}
