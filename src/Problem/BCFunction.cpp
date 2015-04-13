#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"
#include "Projector.h"

#include "BCFunction.h"
#include "BC.h"

using namespace Intrepid;
using namespace Camellia;

Teuchos::RCP<BCFunction> BCFunction::bcFunction(BCPtr bc, int varID, bool isTrace) {
  FunctionPtr spatiallyFilteredFunction;
  int rank = 0;
  if (! bc->isLegacySubclass()) {
    spatiallyFilteredFunction = bc->getSpatiallyFilteredFunctionForDirichletBC(varID);
    rank = spatiallyFilteredFunction->rank();
  }

  return Teuchos::rcp( new BCFunction(bc, varID, isTrace, spatiallyFilteredFunction, rank) );
}

BCFunction::BCFunction(BCPtr bc, int varID, bool isTrace, FunctionPtr spatiallyFilteredFunction, int rank) : Function<double>(rank) {
  _bc = bc;
  _varID = varID;
  _isTrace = isTrace;
  _spatiallyFilteredFunction = spatiallyFilteredFunction;
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

bool BCFunction::isTrace() {
  return _isTrace;
}

int BCFunction::varID() {
  return _varID;
}

FunctionPtr BCFunction::curl() {
  return _spatiallyFilteredFunction->curl();
}

FunctionPtr BCFunction::div() {
  return _spatiallyFilteredFunction->div();
}

FunctionPtr BCFunction::dx() {
  return _spatiallyFilteredFunction->dx();
}

FunctionPtr BCFunction::dy() {
  return _spatiallyFilteredFunction->dy();
}

FunctionPtr BCFunction::dz() {
  return _spatiallyFilteredFunction->dz();
}
