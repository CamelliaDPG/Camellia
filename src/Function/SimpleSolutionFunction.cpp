//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "SimpleSolutionFunction.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "GlobalDofAssignment.h"
#include "SharedScratchStorage.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
SimpleSolutionFunction<Scalar>::SimpleSolutionFunction(VarPtr var, TSolutionPtr<Scalar> soln,
                                                       bool weightFluxesBySideParity, SolutionComponent solutionComponent, int solutionStage,
                                                       const std::string &solutionIdentifierExponent)
:
TFunction<Scalar>("SimpleSolutionFunction", var->rank())
{
  _var = var;
  _soln = soln;
  _weightFluxesBySideParity = weightFluxesBySideParity;
  _solutionComponent = solutionComponent;
  _solutionStage = solutionStage;
  if (solutionIdentifierExponent != "")
    _solutionIdentifierExponent = solutionIdentifierExponent;
  else
    _solutionIdentifierExponent = soln->getIdentifier();
}

template <typename Scalar>
bool SimpleSolutionFunction<Scalar>::boundaryValueOnly()
{
  return (_var->varType() == FLUX) || (_var->varType() == TRACE);
}

template <typename Scalar>
string SimpleSolutionFunction<Scalar>::displayString()
{
  ostringstream str;
  if (_solutionIdentifierExponent == "")
  {
    str << "\\overline{" << _var->displayString() << "} ";
  }
  else
  {
    str << _var->displayString() << "^{" << _solutionIdentifierExponent << "} ";
  }
  return str.str();
}

template <typename Scalar>
size_t SimpleSolutionFunction<Scalar>::getCellDataSize(GlobalIndexType cellID)
{
  const int solutionOrdinal = _soln->solutionOrdinal(_solutionComponent,_solutionStage);
  bool warnAboutOffRankImports = true;
  auto & cellDofs = _soln->allCoefficientsForCellID(cellID, warnAboutOffRankImports, solutionOrdinal);
  return cellDofs.size() * sizeof(Scalar); // size in bytes
}

template <typename Scalar>
int SimpleSolutionFunction<Scalar>::numVectorComponents() const
{
  if (_var->rank() == 0)
  {
    return 0;
  }
  else
  {
    if (! _var->isSpaceTime() )
    {
      return _soln->mesh()->getDimension(); // this is an upper bound on the number of vector components; it may be that we can be a bit more granular, depending on _var and the mesh.
    }
    else
    {
      return _soln->mesh()->getDimension() - 1;
    }
  }
}

template <typename Scalar>
void SimpleSolutionFunction<Scalar>::packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength)
{
  const int solutionOrdinal = _soln->solutionOrdinal(_solutionComponent,_solutionStage);
  size_t requiredLength = getCellDataSize(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(requiredLength > bufferLength, std::invalid_argument, "Buffer length too small");
  bool warnAboutOffRankImports = true;
  auto & cellDofs = _soln->allCoefficientsForCellID(cellID, warnAboutOffRankImports, solutionOrdinal);
  size_t objSize = sizeof(Scalar);
  const Scalar* copyFromLocation = &cellDofs[0];
  memcpy(cellData, copyFromLocation, objSize * cellDofs.size());
}

template <typename Scalar>
size_t SimpleSolutionFunction<Scalar>::unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength)
{
  const int solutionOrdinal = _soln->solutionOrdinal(_solutionComponent,_solutionStage);
  int numDofs = _soln->mesh()->getElementType(cellID)->trialOrderPtr->totalDofs();
  size_t numBytes = numDofs * sizeof(Scalar);
  TEUCHOS_TEST_FOR_EXCEPTION(numBytes > bufferLength, std::invalid_argument, "buffer is too short");
  Camellia::FieldContainer<Scalar> cellDofs(numDofs);
  Scalar* copyToLocation = &cellDofs[0];
  memcpy(copyToLocation, cellData, numBytes);
  _soln->setSolnCoeffsForCellID(cellDofs,cellID,solutionOrdinal);
  return numBytes;
}

template <typename Scalar>
void SimpleSolutionFunction<Scalar>::importCellData(std::vector<GlobalIndexType> cells)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  set<GlobalIndexType> offRankCells;
  const set<GlobalIndexType>* rankLocalCells = &_soln->mesh()->globalDofAssignment()->cellsInPartition(rank);
  for (int cellOrdinal=0; cellOrdinal < cells.size(); cellOrdinal++)
  {
    if (rankLocalCells->find(cells[cellOrdinal]) == rankLocalCells->end())
    {
      offRankCells.insert(cells[cellOrdinal]);
    }
  }
  _soln->importSolutionForOffRankCells(offRankCells);
}

template <typename Scalar>
bool SimpleSolutionFunction<Scalar>::isDefined(BasisCachePtr basisCache) const
{
  bool volumeCache = !basisCache->isSideCache();
  if (volumeCache)
  {
    return _var->isDefinedOnVolume();
  }
  else // sideCache
  {
    int sideIndex = basisCache->getSideIndex();
    
    bool temporalSide = _var->isSpaceTime() && !basisCache->cellTopology()->sideIsSpatial(sideIndex);
    
    if (temporalSide)
    {
      return _var->isDefinedOnTemporalInterface();
    }
    else
    {
      return true;
    }
  }
}

template <typename Scalar>
void SimpleSolutionFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("SimpleSolutionFunction values() - host path");
  // if the device-path values() supports this basisCache, then call that and return.
  // otherwise, use the legacy "host-path" implementation below
  if (basisCache->mesh() != Teuchos::null)
  {
    this->CHECK_VALUES_RANK(values);
    auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
    this->values(dv, ASSIGN, basisCache);
    values.checkinDeviceView(true);
    return;
  }
  
  // legacy, host implementation follows
  bool dontWeightForCubature = false;
  if (basisCache->mesh() != Teuchos::null)   // then we assume that the BasisCache is appropriate for solution's mesh...
  {
    using DeviceView = typename Camellia::FieldContainer<Scalar>::DeviceView;
    DeviceView dvValues = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
    _soln->solutionValues(dvValues, _var->ID(), basisCache, _solutionComponent, _solutionStage, dontWeightForCubature, _var->op());
    values.checkinDeviceView(true);
  }
  else
  {
    // the following adapted from PreviousSolutionFunction.  Probably would do well to consolidate
    // that class with this one at some point...
    LinearTermPtr solnExpression = 1.0 * _var;
    // get the physicalPoints, and make a basisCache for each...
    Camellia::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePointsFC();
    Camellia::FieldContainer<Scalar> value(1,1); // assumes scalar-valued solution function.
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    physicalPoints.resize(numCells*numPoints,spaceDim);
    vector< ElementPtr > elements = _soln->mesh()->elementsForPoints(physicalPoints, false); // false: don't make elements null just because they're off-rank.
    Camellia::FieldContainer<double> point(1,1,spaceDim);
    Camellia::FieldContainer<double> refPoint(1,spaceDim);
    int combinedIndex = 0;
    vector<GlobalIndexType> cellID;
    cellID.push_back(-1);
    BasisCachePtr basisCacheOnePoint;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++, combinedIndex++)
      {
        if (elements[combinedIndex].get()==NULL) continue; // no element found for point; skip it…
        ElementTypePtr elemType = elements[combinedIndex]->elementType();
        for (int d=0; d<spaceDim; d++)
        {
          point(0,0,d) = physicalPoints(combinedIndex,d);
        }
        if (elements[combinedIndex]->cellID() != cellID[0])
        {
          cellID[0] = elements[combinedIndex]->cellID();
          basisCacheOnePoint = Teuchos::rcp( new BasisCache(elemType, _soln->mesh()) );
          basisCacheOnePoint->setPhysicalCellNodes(_soln->mesh()->physicalCellNodesForCell(cellID[0]),cellID,false); // false: don't createSideCacheToo
        }
        refPoint.resize(1,1,spaceDim); // CamelliaCellTools<Scalar>::mapToReferenceFrame wants a numCells dimension...  (perhaps it shouldn't, though!)
        // compute the refPoint:
        CamelliaCellTools::mapToReferenceFrame(refPoint,point,_soln->mesh()->getTopology(), cellID[0],
                                               _soln->mesh()->globalDofAssignment()->getCubatureDegree(cellID[0]));
        refPoint.resize(1,spaceDim);
        basisCacheOnePoint->setRefCellPointsFC(refPoint);
        //          cout << "refCellPoints:\n " << refPoint;
        //          cout << "physicalCubaturePoints:\n " << basisCacheOnePoint->getPhysicalCubaturePointsFC();
        const bool applyCubatureWeights = false;
        solnExpression->evaluate(value, _soln, basisCacheOnePoint, applyCubatureWeights, _solutionComponent, _solutionStage);
        //          cout << "value at point (" << point(0,0) << ", " << point(0,1) << ") = " << value(0,0) << endl;
        values(cellIndex,ptIndex) = value(0,0);
      }
    }
  }
  if (_weightFluxesBySideParity) // makes for non-uniquely-valued Functions.
  {
    if (_var->varType()==FLUX)
    {
      Function::sideParity()->scalarMultiplyFunctionValues(values, basisCache);
    }
  }
}

template <typename Scalar>
void SimpleSolutionFunction<Scalar>::values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("SimpleSolutionFunction values()");
  bool dontWeightForCubature = false;
  if (basisCache->mesh() != Teuchos::null)   // then we assume that the BasisCache is appropriate for solution's mesh...
  {
    using DeviceView = typename Camellia::FieldContainer<Scalar>::DeviceView;
    DeviceView valuesToAssign = values;
    if (modType != ASSIGN)
    {
      // need to place solution values in a separate container prior to combining
      valuesToAssign = basisCache->checkoutScratchStorage<Scalar>(values);
    }
    _soln->solutionValues(valuesToAssign, _var->ID(), basisCache,  _solutionComponent, _solutionStage, dontWeightForCubature, _var->op());
    if (_weightFluxesBySideParity) // makes for non-uniquely-valued Functions.
    {
      if (_var->varType()==FLUX)
      {
        Kokkos::fence();
        Function::sideParity()->values(valuesToAssign, ValueModificationType::MULTIPLY, basisCache);
      }
    }
    if (modType != ASSIGN)
    {
      Kokkos::fence();
      assignValues<Scalar>(values, modType, valuesToAssign);
      basisCache->checkinScratchStorage<Scalar>(valuesToAssign);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "SimpleSolutionFunction<Scalar> device-path values() does not support evaluation on BasisCaches that don't have meshes defined.");
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::curl()
{
  if (_var->op() == Camellia::OP_VALUE)
  {
    const int spaceDim = _soln->mesh()->getDimension();
    return TFunction<Scalar>::solution(_var->curl(spaceDim), _soln, _weightFluxesBySideParity);
  }
  else
  {
    return this->TFunction<Scalar>::curl();
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::div()
{
  if (_var->op() == Camellia::OP_VALUE)
  {
    return TFunction<Scalar>::solution(_var->div(), _soln, _weightFluxesBySideParity);
  }
  else
  {
    return this->TFunction<Scalar>::div();
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::dx()
{
  if (_var->op() == Camellia::OP_GRAD)
  {
    const int spaceDim = _soln->mesh()->getDimension();
    std::vector< TFunctionPtr<Scalar> > gradDxComponents;
    for (int d=0; d<spaceDim; d++)
    {
      gradDxComponents.push_back(this->spatialComponent(d+1)->dx());
    }
    return Function::vectorize(gradDxComponents);
  }
  else
  {
    return TFunction<Scalar>::solution(_var->dx(), _soln, _weightFluxesBySideParity);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::dy()
{
  if (_var->op() == Camellia::OP_GRAD)
  {
    const int spaceDim = _soln->mesh()->getDimension();
    std::vector< TFunctionPtr<Scalar> > gradDyComponents;
    for (int d=0; d<spaceDim; d++)
    {
      gradDyComponents.push_back(this->spatialComponent(d+1)->dy());
    }
    return Function::vectorize(gradDyComponents);
  }
  else
  {
    return TFunction<Scalar>::solution(_var->dy(), _soln, _weightFluxesBySideParity);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::dz()
{
  if (_var->op() == Camellia::OP_GRAD)
  {
    const int spaceDim = _soln->mesh()->getDimension();
    std::vector< TFunctionPtr<Scalar> > gradDzComponents;
    for (int d=0; d<spaceDim; d++)
    {
      gradDzComponents.push_back(this->spatialComponent(d+1)->dz());
    }
    return Function::vectorize(gradDzComponents);
  }
  else
  {
    return TFunction<Scalar>::solution(_var->dz(), _soln, _weightFluxesBySideParity);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::di(int i)
{
  if (_var->op() == Camellia::OP_GRAD)
  {
    const int spaceDim = _soln->mesh()->getDimension();
    std::vector< TFunctionPtr<Scalar> > gradDxComponents;
    for (int d=0; d<spaceDim; d++)
    {
      gradDxComponents.push_back(this->spatialComponent(d+1)->di(i));
    }
    return Function::vectorize(gradDxComponents);
  }
  else
  {
    return TFunction<Scalar>::solution(_var->di(i), _soln, _weightFluxesBySideParity);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::grad(int spaceDim)
{
  if (_var->op() == Camellia::OP_VALUE)
  {
    if (spaceDim == -1) spaceDim = _soln->mesh()->getDimension();
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != _soln->mesh()->getDimension(), std::invalid_argument, "spaceDim argument to grad() must match dimension of the mesh.");
    return TFunction<Scalar>::solution(_var->grad(), _soln, _weightFluxesBySideParity);
  }
  else
  {
    return this->TFunction<Scalar>::grad(spaceDim);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::x()
{
  return TFunction<Scalar>::solution(_var->x(), _soln, _weightFluxesBySideParity, _solutionComponent, _solutionStage, _solutionIdentifierExponent);
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->x(), _soln, _weightFluxesBySideParity, _solutionComponent, _solutionStage, _solutionIdentifierExponent);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::y()
{
  return TFunction<Scalar>::solution(_var->y(), _soln, _weightFluxesBySideParity, _solutionComponent, _solutionStage, _solutionIdentifierExponent);

//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->y(), _soln, _weightFluxesBySideParity, _solutionComponent, _solutionStage, _solutionIdentifierExponent);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::z()
{
  return TFunction<Scalar>::solution(_var->z(), _soln, _weightFluxesBySideParity, _solutionComponent, _solutionStage, _solutionIdentifierExponent);
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->z(), _soln, _weightFluxesBySideParity, _solutionComponent, _solutionStage, _solutionIdentifierExponent);
//  }
}

namespace Camellia
{
template class SimpleSolutionFunction<double>;
}

