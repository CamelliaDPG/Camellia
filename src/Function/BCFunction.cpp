#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"
#include "Projector.h"

#include "BCFunction.h"
#include "BC.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
Teuchos::RCP<BCFunction<Scalar>> BCFunction<Scalar>::bcFunction(BCPtr bc, int varID, bool isTrace) {
  TFunctionPtr<Scalar> spatiallyFilteredFunction;
  int rank = 0;
  if (! bc->isLegacySubclass()) {
    spatiallyFilteredFunction = bc->getSpatiallyFilteredFunctionForDirichletBC(varID);
    rank = spatiallyFilteredFunction->rank();
  }

  return Teuchos::rcp( new BCFunction<Scalar>(bc, varID, isTrace, spatiallyFilteredFunction, rank) );
}

template <typename Scalar>
BCFunction<Scalar>::BCFunction(BCPtr bc, int varID, bool isTrace, TFunctionPtr<Scalar> spatiallyFilteredFunction, int rank) : TFunction<Scalar>(rank) {
  _bc = bc;
  _varID = varID;
  _isTrace = isTrace;
  _spatiallyFilteredFunction = spatiallyFilteredFunction;
}

template <typename Scalar>
void BCFunction<Scalar>::values(FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
  int numCells = basisCache->cellIDs().size();
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  _imposeHere.resize(numCells,numPoints);
  FieldContainer<double> unitNormals = basisCache->getSideNormals();
  _bc->imposeBC(values, _imposeHere, _varID, unitNormals, basisCache);
}

template <typename Scalar>
bool BCFunction<Scalar>::imposeOnCell(int cellIndex) {
  // returns true if at least one cubature point lies within the SpatialFilter or equivalent
  // MUST call values() with appropriate basisCache before calling this...
  int numPoints = _imposeHere.dimension(1);
  for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
    if (_imposeHere(cellIndex,ptIndex)) return true;
  }
  return false;
}

template <typename Scalar>
bool BCFunction<Scalar>::isTrace() {
  return _isTrace;
}

template <typename Scalar>
int BCFunction<Scalar>::varID() {
  return _varID;
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::curl() {
  return _spatiallyFilteredFunction->curl();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::div() {
  return _spatiallyFilteredFunction->div();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::dx() {
  return _spatiallyFilteredFunction->dx();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::dy() {
  return _spatiallyFilteredFunction->dy();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::dz() {
  return _spatiallyFilteredFunction->dz();
}

namespace Camellia {
  template class BCFunction<double>;
}
