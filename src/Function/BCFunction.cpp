//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia_FieldContainer.hpp"
#include "BasisCache.h"

#include "BCFunction.h"
#include "BC.h"

#include <Intrepid2_TensorArgumentIterator.hpp>

using namespace Camellia;

template <typename Scalar>
Teuchos::RCP<BCFunction<Scalar>> BCFunction<Scalar>::bcFunction(BCPtr bc, int varID)
{
  TFunctionPtr<Scalar> spatiallyFilteredFunction;
  int rank = 0;
  if (! bc->isLegacySubclass())
  {
    spatiallyFilteredFunction = bc->getSpatiallyFilteredFunctionForDirichletBC(varID);
    rank = spatiallyFilteredFunction->rank();
  }

  return Teuchos::rcp( new BCFunction<Scalar>(bc, varID, spatiallyFilteredFunction, rank) );
}

template <typename Scalar>
BCFunction<Scalar>::BCFunction(BCPtr bc, int varID, TFunctionPtr<Scalar> spatiallyFilteredFunction, int rank) : TFunction<Scalar>("BCFunction", rank)
{
  _bc = bc;
  _varID = varID;
  _spatiallyFilteredFunction = spatiallyFilteredFunction;
}

template <typename Scalar>
void BCFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  int numCells = basisCache->cellIDs().size();
  int numPoints = basisCache->getPhysicalCubaturePointsFC().dimension(1);
  _imposeHere.resize(numCells,numPoints);
  Camellia::FieldContainer<double> unitNormals = basisCache->getSideNormals();
  _bc->imposeBC(values, _imposeHere, _varID, unitNormals, basisCache);
}

template <typename Scalar>
void BCFunction<Scalar>::values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  int numCells = basisCache->cellIDs().size();
  int numPoints = basisCache->getPhysicalCubaturePointsFC().dimension(1);
  _imposeHere.resize(numCells,numPoints);
  DeviceView<bool> imposeHereView = _imposeHere.checkoutDeviceView(false); // don't deep copy; _bc will overwrite
  
  auto unitNormals = basisCache->getSideNormals(); // copy; basisCache may change/recompute side normals container inside imposeBC() call
  DeviceViewConst<Scalar> unitNormalsView = unitNormals.checkoutDeviceView();
  
  if (modType == ASSIGN)
  {
    _bc->imposeBC(values, imposeHereView, _varID, unitNormalsView, basisCache);
  }
  else
  {
    DeviceView<Scalar> bcValues = SharedScratchStorage<Scalar>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(values);
    // bcValues comes to us uninitialized, and _bc will not necessarily overwrite the whole container, so we should clear first
    Kokkos::deep_copy(bcValues, 0.0);
    _bc->imposeBC(bcValues, imposeHereView, _varID, unitNormalsView, basisCache);
    
    Kokkos::fence(); // ensure that imposeBC completes before we assignValues
    assignValues<Scalar>(values, modType, bcValues);
    Kokkos::fence(); // ensure that assignValues completes before we return bcValues
    SharedScratchStorage<Scalar>::sharedScratchStorage().checkinScratchStorage(bcValues);
  }
  
  unitNormals.checkinDeviceView(false);
  _imposeHere.checkinDeviceView(true);
}

template <typename Scalar>
bool BCFunction<Scalar>::imposeOnCell(int cellOrdinal)
{
  // returns true if at least one cubature point lies within the SpatialFilter or equivalent
  // MUST call values() with appropriate basisCache before calling this...
  if (_imposeHere.size() > 0)
  {
    int numPoints = _imposeHere.dimension(1);
    for (int ptIndex=0; ptIndex < numPoints; ptIndex++)
    {
      if (_imposeHere(cellOrdinal,ptIndex)) return true;
    }
  }
  return false;
}

template <typename Scalar>
int BCFunction<Scalar>::varID()
{
  return _varID;
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::curl()
{
  return _spatiallyFilteredFunction->curl();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::div()
{
  return _spatiallyFilteredFunction->div();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::dx()
{
  return _spatiallyFilteredFunction->dx();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::dy()
{
  return _spatiallyFilteredFunction->dy();
}

template <typename Scalar>
TFunctionPtr<Scalar> BCFunction<Scalar>::dz()
{
  return _spatiallyFilteredFunction->dz();
}

template <typename Scalar>
std::vector< TFunctionPtr<Scalar> > BCFunction<Scalar>::memberFunctions()
{
  return {{_spatiallyFilteredFunction}};
}

namespace Camellia
{
template class BCFunction<double>;
}
