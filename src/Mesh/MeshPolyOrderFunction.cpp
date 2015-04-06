//
//  MeshPolyOrderFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Function.h"
#include "MeshPolyOrderFunction.h"
#include "BasisCache.h"

using namespace Intrepid;
using namespace Camellia;

void MeshPolyOrderFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  IndexType cellIndex = 0;
  int numPoints = values.dimension(1);
  for (vector<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++, cellIndex++) {
    GlobalIndexType cellID = *cellIDIt;
    int polyOrder = _mesh->cellPolyOrder(cellID); // H1 order
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      values(cellIndex, ptIndex) = polyOrder-1;
    }
  }
}
