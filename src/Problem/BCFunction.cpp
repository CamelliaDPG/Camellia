#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"
#include "Projector.h"

#include "BCFunction.h"
#include "BC.h"

BCFunction::BCFunction(BCPtr bc, int varID) : Function(0) {
  _bc = bc;
  _varID = varID;
}

void BCFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  int numCells = basisCache->cellIDs().size();
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  _imposeHere.resize(numCells,numPoints);
  FieldContainer<double> unitNormals = basisCache->getSideNormals();
  _bc->imposeBC(values, _imposeHere, _varID, unitNormals, basisCache);
}

bool BCFunction::imposeOnCell(int cellIndex) {
  // returns true if at least one cubature point lies within the SpatialFilter or equivalent
  // MUST call values() with appropriate basisCache before calling this...
  int numPoints = _imposeHere.dimension(1);
  for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
    if (_imposeHere(cellIndex,ptIndex)) return true;
  }
  return false;
}

int BCFunction::varID() {
  return _varID;
}