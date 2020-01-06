//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "SimpleSolutionFunction.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "GlobalDofAssignment.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
SimpleSolutionFunction<Scalar>::SimpleSolutionFunction(VarPtr var, TSolutionPtr<Scalar> soln,
                                                       bool weightFluxesBySideParity, int solutionOrdinal, const std::string &solutionIdentifierExponent) : TFunction<Scalar>(var->rank())
{
  _var = var;
  _soln = soln;
  _weightFluxesBySideParity = weightFluxesBySideParity;
  _solutionOrdinal = solutionOrdinal;
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
  bool warnAboutOffRankImports = true;
  auto & cellDofs = _soln->allCoefficientsForCellID(cellID, warnAboutOffRankImports, _solutionOrdinal);
  return cellDofs.size() * sizeof(Scalar); // size in bytes
}

template <typename Scalar>
void SimpleSolutionFunction<Scalar>::packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength)
{
  size_t requiredLength = getCellDataSize(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(requiredLength > bufferLength, std::invalid_argument, "Buffer length too small");
  bool warnAboutOffRankImports = true;
  auto & cellDofs = _soln->allCoefficientsForCellID(cellID, warnAboutOffRankImports, _solutionOrdinal);
  size_t objSize = sizeof(Scalar);
  const Scalar* copyFromLocation = &cellDofs[0];
  memcpy(cellData, copyFromLocation, objSize * cellDofs.size());
}

template <typename Scalar>
size_t SimpleSolutionFunction<Scalar>::unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength)
{
//  Epetra_CommPtr Comm = _soln->mesh()->Comm();
//  int rank = Comm->MyPID();
  int numDofs = _soln->mesh()->getElementType(cellID)->trialOrderPtr->totalDofs();
  size_t numBytes = numDofs * sizeof(Scalar);
  TEUCHOS_TEST_FOR_EXCEPTION(numBytes > bufferLength, std::invalid_argument, "buffer is too short");
  Camellia::FieldContainer<Scalar> cellDofs(numDofs);
  Scalar* copyToLocation = &cellDofs[0];
  memcpy(copyToLocation, cellData, numBytes);
  _soln->setSolnCoeffsForCellID(cellDofs,cellID,_solutionOrdinal);
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
void SimpleSolutionFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  bool dontWeightForCubature = false;
  if (basisCache->mesh() != Teuchos::null)   // then we assume that the BasisCache is appropriate for solution's mesh...
  {
    _soln->solutionValues(values, _var->ID(), basisCache, dontWeightForCubature, _var->op(), _solutionOrdinal);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_solutionOrdinal != 0, std::invalid_argument, "SimpleSolutionFunction only supports non-zero solutionOrdinals for BasisCaches with meshes defined.");
    // the following adapted from PreviousSolutionFunction.  Probably would do well to consolidate
    // that class with this one at some point...
    LinearTermPtr solnExpression = 1.0 * _var;
    // get the physicalPoints, and make a basisCache for each...
    Camellia::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
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
        basisCacheOnePoint->setRefCellPoints(refPoint);
        //          cout << "refCellPoints:\n " << refPoint;
        //          cout << "physicalCubaturePoints:\n " << basisCacheOnePoint->getPhysicalCubaturePoints();
        solnExpression->evaluate(value, _soln, basisCacheOnePoint);
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
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::dx()
{
  return TFunction<Scalar>::solution(_var->dx(), _soln, _weightFluxesBySideParity);
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->dx(), _soln, _weightFluxesBySideParity);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::dy()
{
  return TFunction<Scalar>::solution(_var->dy(), _soln, _weightFluxesBySideParity);
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->dy(), _soln, _weightFluxesBySideParity);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::dz()
{
  return TFunction<Scalar>::solution(_var->dz(), _soln, _weightFluxesBySideParity);
  
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->dz(), _soln, _weightFluxesBySideParity);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::x()
{
  return TFunction<Scalar>::solution(_var->x(), _soln, _weightFluxesBySideParity);
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->x(), _soln, _weightFluxesBySideParity);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::y()
{
  return TFunction<Scalar>::solution(_var->y(), _soln, _weightFluxesBySideParity);

//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->y(), _soln, _weightFluxesBySideParity);
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SimpleSolutionFunction<Scalar>::z()
{
  return TFunction<Scalar>::solution(_var->z(), _soln, _weightFluxesBySideParity);
//  if (_var->op() != Camellia::OP_VALUE)
//  {
//    return TFunction<Scalar>::null();
//  }
//  else
//  {
//    return TFunction<Scalar>::solution(_var->z(), _soln, _weightFluxesBySideParity);
//  }
}

namespace Camellia
{
template class SimpleSolutionFunction<double>;
}

