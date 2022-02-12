//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "FixedCountRefinementStrategy.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "RefinementStrategy.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

FixedCountRefinementStrategy::FixedCountRefinementStrategy( ErrorIndicatorPtr errorIndicator, int numElementsToRefine, double min_h, int max_p, bool preferPRefinements)
:
_targetRefinementCount(numElementsToRefine),
TRefinementStrategy<double>(errorIndicator, 1.0, min_h, max_p, preferPRefinements)
{
  
}

void FixedCountRefinementStrategy::refineAndUnrefine(bool unrefine, bool printToConsole)
{
  TEUCHOS_TEST_FOR_EXCEPTION(unrefine, std::invalid_argument, "FixedCountRefinementStrategy doesn't support unrefinements");
  
  // fixed count refinement - mark cells for refinement
  MeshPtr mesh = this->mesh();
  auto meshTopo = mesh->getTopology();
  
  auto errorIndicator = this->errorIndicator();
  errorIndicator->measureError();
  const double totalError = errorIndicator->totalError();

  std::set<GlobalIndexType> localCellsToExcludeFromRefinement = this->localCellsIneligibleForRefinement(); // exclude cells that can't be refined further from the max error computation
  
  // record results prior to refinement
  RefinementResults results = TRefinementStrategy<double>::setResults(mesh->numActiveElements(), mesh->numGlobalDofs(), totalError);
  this->_results.push_back(results);
  
  auto globalMaxErrorCells = errorIndicator->globalMaximumErrorCells(_targetRefinementCount, localCellsToExcludeFromRefinement);

  map<GlobalIndexType, double> cellMeasuresLocal;
  std::set<GlobalIndexType> localCellsToRefine;
  for (const auto & entry : globalMaxErrorCells)
  {
    const GlobalIndexType & cellID = entry.first;
    if (mesh->myCellsInclude(cellID))
    {
      localCellsToRefine.insert(cellID);
      cellMeasuresLocal[cellID] = mesh->getCellMeasure(cellID);
    }
  }
  
  const double meshDim = mesh->getDimension();
  
  vector<GlobalIndexType> myCellsToHRefine;
  vector<GlobalIndexType> myCellsToPRefine;
  
  for (GlobalIndexType cellID : localCellsToRefine)
  {
    double h = pow(cellMeasuresLocal[cellID], 1.0 / meshDim); // dth root of volume measure = h
    int p = mesh->cellPolyOrder(cellID);

    const int cellLevel = meshTopo->getCell(cellID)->level();
    
    const bool canHRefine = ((h > this->_min_h) && (cellLevel < this->_maxRefinementLevel));
    const bool canPRefine = (p < this->_max_p);
    
    if (!this->_preferPRefinements)
    {
      // prefer h refinements
      if (canHRefine)
      {
        myCellsToHRefine.push_back(cellID);
      }
      else if (canPRefine)
      {
        myCellsToPRefine.push_back(cellID);
      }
    }
    else
    {
      if (canPRefine)
      {
        myCellsToPRefine.push_back(cellID);
      }
      else if (canHRefine)
      {
        myCellsToHRefine.push_back(cellID);
      }
    }
  }
  
  GlobalIndexTypeToCast numCellsToHRefine = 0;
  GlobalIndexTypeToCast myNumCellsToHRefine = myCellsToHRefine.size();
  mesh->Comm()->SumAll(&myNumCellsToHRefine, &numCellsToHRefine, 1);
  GlobalIndexTypeToCast myCellOrdinalOffset = 0;
  mesh->Comm()->ScanSum(&myNumCellsToHRefine, &myCellOrdinalOffset, 1);
  myCellOrdinalOffset -= myNumCellsToHRefine;
  
  vector<GlobalIndexTypeToCast> globalCellsToRefineVector(numCellsToHRefine, 0);
  for (int i=0; i<myNumCellsToHRefine; i++)
  {
    globalCellsToRefineVector[myCellOrdinalOffset + i] = myCellsToHRefine[i];
  }
  vector<GlobalIndexTypeToCast> gatheredCellsToRefineVector(numCellsToHRefine, 0);
  mesh->Comm()->SumAll(&globalCellsToRefineVector[0], &gatheredCellsToRefineVector[0], numCellsToHRefine);

  vector<GlobalIndexType> cellsToRefine(gatheredCellsToRefineVector.begin(),gatheredCellsToRefineVector.end());;
  
  GlobalIndexTypeToCast numCellsToPRefine = 0;
  GlobalIndexTypeToCast myNumCellsToPRefine = myCellsToPRefine.size();
  mesh->Comm()->SumAll(&myNumCellsToPRefine, &numCellsToPRefine, 1);
  
  vector<GlobalIndexType> cellsToPRefine;
  if (numCellsToPRefine > 0)
  {
    myCellOrdinalOffset = 0;
    mesh->Comm()->ScanSum(&myNumCellsToPRefine, &myCellOrdinalOffset, 1);
    myCellOrdinalOffset -= myNumCellsToPRefine;

    vector<GlobalIndexTypeToCast> globalCellsToPRefineVector(numCellsToPRefine, 0);
    for (int i=0; i<myNumCellsToPRefine; i++)
    {
      globalCellsToPRefineVector[myCellOrdinalOffset + i] = myCellsToPRefine[i];
    }
    vector<GlobalIndexTypeToCast> gatheredCellsToPRefineVector(numCellsToPRefine, 0);
    mesh->Comm()->SumAll(&globalCellsToPRefineVector[0], &gatheredCellsToPRefineVector[0], numCellsToPRefine);

    cellsToPRefine = vector<GlobalIndexType>(gatheredCellsToPRefineVector.begin(),gatheredCellsToPRefineVector.end());
  }
  
  std::sort(cellsToRefine.begin(), cellsToRefine.end());
  std::sort(cellsToPRefine.begin(), cellsToPRefine.end());
  
  if (printToConsole)
  {
    if (cellsToRefine.size() > 0)         Camellia::print("cells for h-refinement", cellsToRefine);
    if (cellsToPRefine.size() > 0)        Camellia::print("cells for p-refinement", cellsToPRefine);
  }
  
  this->refineCells(cellsToRefine);
  this->pRefineCells(mesh, cellsToPRefine);

  bool repartitionAndRebuild = false;
  if (this->_enforceOneIrregularity)
  {
    mesh->enforceOneIrregularity(repartitionAndRebuild);
  }

  // now, repartition and rebuild:
  mesh->repartitionAndRebuild();
  
  GlobalIndexType activeElementCount = mesh->numActiveElements();
  GlobalIndexType globalDofCount = mesh->numGlobalDofs();
  
  if (printToConsole && (mesh->Comm()->MyPID() == 0))
  {
    cout << "Prior to refinement, total error: " << totalError << endl;
    cout << "After refinement, mesh has " << activeElementCount << " elements and " << globalDofCount << " global dofs" << endl;
  }
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::getCellsAboveErrorThreshhold(vector<GlobalIndexType> &cellsToRefine, std::set<GlobalIndexType> localCellsToExclude)
{
  // measure error
  _errorIndicator->measureError();
  
  double maxError = _errorIndicator->maxError(localCellsToExclude);
  
  double errorThreshold = maxError * _relativeErrorThreshold;
  _errorIndicator->localCellsAboveErrorThreshold(errorThreshold, cellsToRefine);
}

// defaults to h-refinement
template <typename Scalar>
void TRefinementStrategy<Scalar>::refineCells(vector<GlobalIndexType> &cellIDs)
{
  MeshPtr mesh = this->mesh();
  hRefineCells(mesh, cellIDs);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::pRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs)
{
  bool repartitionAndRebuild = false;
  int pToAdd = 1;
  set<GlobalIndexType> cellIDSet(cellIDs.begin(),cellIDs.end());
  mesh->pRefine(cellIDSet, pToAdd, repartitionAndRebuild);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineCells(MeshPtr mesh, const vector<GlobalIndexType> &cellIDs)
{
  bool repartitionAndRebuild = false;
  mesh->hRefine(cellIDs, repartitionAndRebuild);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineUniformly()
{
  // record results prior to refinement
  _errorIndicator->measureError();
  double totalError = _errorIndicator->totalError();

  MeshPtr mesh = this->mesh();
  RefinementResults results = setResults(mesh->numElements(), mesh->numGlobalDofs(), totalError);
  _results.push_back(results);
  TRefinementStrategy<Scalar>::hRefineUniformly(mesh);
}

// ! static method
template <typename Scalar>
void TRefinementStrategy<Scalar>::hRefineUniformly(MeshPtr mesh)
{
  set<GlobalIndexType> cellsToRefine = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> cellsToRefineVector(cellsToRefine.begin(),cellsToRefine.end());
  hRefineCells(mesh, cellsToRefineVector);
  mesh->repartitionAndRebuild();
}

template <typename Scalar>
std::set<GlobalIndexType> TRefinementStrategy<Scalar>::localCellsIneligibleForRefinement()
{
  auto mesh = this->mesh();
  const set<GlobalIndexType> & myCellIDs = mesh->cellIDsInPartition();
  map<GlobalIndexType, double> cellMeasuresLocal;
  for (GlobalIndexType cellID : myCellIDs)
  {
    cellMeasuresLocal[cellID] = mesh->getCellMeasure(cellID);
  }
  
  std::set<GlobalIndexType> localCellsToExcludeFromRefinement;
  double meshDim = mesh->getDimension();
  for (GlobalIndexType cellID : myCellIDs)
  {
    double h = pow(cellMeasuresLocal[cellID], 1.0 / meshDim); // dth root of volume measure = h
    int p = mesh->cellPolyOrder(cellID);

    const int cellLevel = mesh->getTopology()->getCell(cellID)->level();
    
    const bool canHRefine = ((h > _min_h) && (cellLevel < _maxRefinementLevel));
    const bool canPRefine = (p < _max_p);
    
    if (!canHRefine && !canPRefine)
    {
      localCellsToExcludeFromRefinement.insert(cellID);
    }
  }
  return localCellsToExcludeFromRefinement;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMaxElements(int maxElementCount)
{
  _maxElementCount = maxElementCount;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMaxP(int max_p)
{
  _max_p = max_p;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMaximumRefinementLevel(int maxLevel)
{
  _maxRefinementLevel = maxLevel;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setRelativeErrorThreshold(double value)
{
  _relativeErrorThreshold = value;
}

template <typename Scalar>
RefinementResults TRefinementStrategy<Scalar>::setResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError)
{
  RefinementResults solnResults;
  solnResults.numElements = numElements;
  solnResults.numDofs = numDofs;
  solnResults.totalEnergyError = totalEnergyError;
  return solnResults;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setUnrefinementErrorIndicator(ErrorIndicatorPtr unrefinementErrorIndicator)
{
  _unrefinementErrorIndicator = unrefinementErrorIndicator;
}

template <typename Scalar>
double TRefinementStrategy<Scalar>::getEnergyError(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].totalEnergyError;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumElements(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].numElements;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumDofs(int refinementNumber)
{
  if (refinementNumber < _results.size())
  {
    return _results[refinementNumber].numDofs;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refinementNumber out of bounds!");
  }
}

template <typename Scalar>
GlobalIndexType TRefinementStrategy<Scalar>::getNumRefinements() const
{
  return _results.size();
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::unrefineCells(const std::set<GlobalIndexType> &cellIDs)
{
  const bool repartitionAndRebuild = false;
  this->mesh()->hUnrefine(cellIDs, repartitionAndRebuild);
}

namespace Camellia
{
template class TRefinementStrategy<double>;
}
