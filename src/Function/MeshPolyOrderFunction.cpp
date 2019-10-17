//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Function.h"
#include "MeshPolyOrderFunction.h"
#include "BasisCache.h"

using namespace Camellia;

void MeshPolyOrderFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  IndexType cellIndex = 0;
  int numPoints = values.dimension(1);
  for (vector<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++, cellIndex++)
  {
    GlobalIndexType cellID = *cellIDIt;
    int polyOrder = _mesh->cellPolyOrder(cellID); // H1 order
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      values(cellIndex, ptIndex) = polyOrder-1;
    }
  }
}
