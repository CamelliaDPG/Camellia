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

void FixedCountRefinementStrategy::setTargetRefinementCount(const int targetRefinementCount)
{
  _targetRefinementCount = targetRefinementCount;
}
