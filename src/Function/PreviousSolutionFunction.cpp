//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include <Teuchos_GlobalMPISession.hpp>

#include "PreviousSolutionFunction.h"

#include "CamelliaCellTools.h"
#include "Element.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "InnerProductScratchPad.h"
#include "Solution.h"

using namespace Camellia;

template <typename Scalar>
PreviousSolutionFunction<Scalar>::PreviousSolutionFunction(TSolutionPtr<Scalar> soln, LinearTermPtr solnExpression,
                                                           bool multiplyFluxesByCellParity, SolutionComponent solutionComponent, int stageNumber)
: TFunction<Scalar>("PreviousSolutionFunction", solnExpression->rank()),
_overrideMeshCheck(false), _soln(soln), _solnExpression(solnExpression), _solnComponent(solutionComponent), _stageNumber(stageNumber)
{
  if (solnExpression->rank() == -1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid solnExpression");
  }
  if ((solnExpression->termType() == FLUX) && multiplyFluxesByCellParity)
  {
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    _solnExpression = parity * solnExpression;
  }
}
template <typename Scalar>
PreviousSolutionFunction<Scalar>::PreviousSolutionFunction(TSolutionPtr<Scalar> soln, VarPtr var,
                                                           bool multiplyFluxesByCellParity, SolutionComponent solutionComponent, int stageNumber)
: TFunction<Scalar>("PreviousSolutionFunction", var->rank()),
_overrideMeshCheck(false), _soln(soln), _solnExpression(1.0 * var), _solnComponent(solutionComponent), _stageNumber(stageNumber)
{
  if ((var->varType() == FLUX) && multiplyFluxesByCellParity)
  {
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    _solnExpression = parity * var;
  }
}
template <typename Scalar>
bool PreviousSolutionFunction<Scalar>::boundaryValueOnly()   // fluxes and traces are only defined on element boundaries
{
  return _solnExpression->isPureBoundaryTerm();
//  return (_solnExpression->termType() == FLUX) || (_solnExpression->termType() == TRACE);
}

template <typename Scalar>
size_t PreviousSolutionFunction<Scalar>::getCellDataSize(GlobalIndexType cellID)
{
  const int solutionOrdinal = _soln->solutionOrdinal(_solnComponent,_stageNumber);
  bool warnAboutOffRankImports = true;
  auto & cellDofs = _soln->allCoefficientsForCellID(cellID, warnAboutOffRankImports, solutionOrdinal);
  return cellDofs.size() * sizeof(Scalar); // size in bytes
}

template <typename Scalar>
void PreviousSolutionFunction<Scalar>::importCellData(std::vector<GlobalIndexType> cells)
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
void PreviousSolutionFunction<Scalar>::packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength)
{
  const int solutionOrdinal = _soln->solutionOrdinal(_solnComponent,_stageNumber);
  size_t requiredLength = getCellDataSize(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(requiredLength > bufferLength, std::invalid_argument, "Buffer length too small");
  bool warnAboutOffRankImports = true;
  auto & cellDofs = _soln->allCoefficientsForCellID(cellID, warnAboutOffRankImports, solutionOrdinal);
  size_t objSize = sizeof(Scalar);
  const Scalar* copyFromLocation = &cellDofs[0];
  memcpy(cellData, copyFromLocation, objSize * cellDofs.size());
}

template <typename Scalar>
void PreviousSolutionFunction<Scalar>::setOverrideMeshCheck(bool value, bool dontWarn)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0)
  {
    if (value==true)
    {
      if (!dontWarn)
      {
        cout << "Overriding mesh check in PreviousSolutionFunction.  This is intended as an optimization for cases where two distinct meshes have IDENTICAL geometry for each of their cellIDs.  If this is not your situation, the override will produce unpredictable results and should NOT be used.\n";
      }
    }
  }
  _overrideMeshCheck = value;
}

template <typename Scalar>
size_t PreviousSolutionFunction<Scalar>::unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength)
{
  const int solutionOrdinal = _soln->solutionOrdinal(_solnComponent,_stageNumber);
  //  Epetra_CommPtr Comm = _soln->mesh()->Comm();
  //  int rank = Comm->MyPID();
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
void PreviousSolutionFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  const bool applyCubatureWeights = false;
  
  if (_overrideMeshCheck)
  {
    _solnExpression->evaluate(values, _soln, basisCache, applyCubatureWeights, _solnComponent, _stageNumber);
    return;
  }
  if (!basisCache.get()) cout << "basisCache is nil!\n";
  if (!_soln.get()) cout << "_soln is nil!\n";
  // values are stored in (C,P,D) order
  if (basisCache->mesh().get() == _soln->mesh().get())
  {
    _solnExpression->evaluate(values, _soln, basisCache, applyCubatureWeights, _solnComponent, _stageNumber);
  }
  else
  {
    static bool warningIssued = false;
    if (!warningIssued)
    {
      if (rank==0)
        cout << "NOTE: In PreviousSolutionFunction, basisCache's mesh doesn't match solution's.  If this is not what you intended, it would be a good idea to make sure that the mesh is passed in on BasisCache construction; the evaluation will be a lot slower without it...\n";
      warningIssued = true;
    }
    // get the physicalPoints, and make a basisCache for each...
    Camellia::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePointsFC();
    Camellia::FieldContainer<double> value(1,1); // assumes scalar-valued solution function.
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
        refPoint.resize(1,1,spaceDim); // CamelliaCellTools::mapToReferenceFrame wants a numCells dimension...  (perhaps it shouldn't, though!)
        // compute the refPoint:
        CamelliaCellTools::mapToReferenceFrame(refPoint,point,_soln->mesh()->getTopology(), cellID[0],
                                               _soln->mesh()->globalDofAssignment()->getCubatureDegree(cellID[0]));
        refPoint.resize(1,spaceDim);
        basisCacheOnePoint->setRefCellPointsFC(refPoint);
        //          cout << "refCellPoints:\n " << refPoint;
        //          cout << "physicalCubaturePoints:\n " << basisCacheOnePoint->getPhysicalCubaturePointsFC();
        _solnExpression->evaluate(value, _soln, basisCacheOnePoint, applyCubatureWeights, _solnComponent, _stageNumber);
        //          cout << "value at point (" << point(0,0) << ", " << point(0,1) << ") = " << value(0,0) << endl;
        values(cellIndex,ptIndex) = value(0,0);
      }
    }
  }
}

template <typename Scalar>
map<int, TFunctionPtr<Scalar> > PreviousSolutionFunction<Scalar>::functionMap( vector< VarPtr > varPtrs, TSolutionPtr<Scalar> soln,
                                                                              SolutionComponent solutionComponent, int stageNumber,
                                                                              bool weightFluxesByParity)
{
  // TODO: move this somewhere else.  Solution? Function?  Somewhere else; the PreviousSolutionFunction class should be eliminated soon.
  map<int, TFunctionPtr<Scalar> > functionMap;
  for (VarPtr var : varPtrs)
  {
    functionMap[var->ID()] = Function::solution(var, soln, weightFluxesByParity, solutionComponent, stageNumber);
  }
  return functionMap;
}

template <typename Scalar>
string PreviousSolutionFunction<Scalar>::displayString()
{
  ostringstream str;
  str << "\\overline{" << _solnExpression->displayString() << "} ";
  return str.str();
}

namespace Camellia
{
template class PreviousSolutionFunction<double>;
}
