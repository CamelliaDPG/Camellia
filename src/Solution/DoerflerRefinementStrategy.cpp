//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "DoerflerRefinementStrategy.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "RefinementStrategy.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

DoerflerRefinementStrategy::DoerflerRefinementStrategy( ErrorIndicatorPtr errorIndicator, double doerflerTarget, double min_h, int max_p, bool preferPRefinements)
:
_doerflerTarget(doerflerTarget),
TRefinementStrategy<double>(errorIndicator, 1.0, min_h, max_p, preferPRefinements)
{
  
}

void DoerflerRefinementStrategy::refineAndUnrefine(bool unrefine, bool printToConsole)
{
  TEUCHOS_TEST_FOR_EXCEPTION(unrefine, std::invalid_argument, "DoerflerRefinementStrategy doesn't support unrefinements");
  
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
  
  // get globalMaximumErrorCells() to basically do an all-gather for us.
  auto globalMaxErrorCells = errorIndicator->globalMaximumErrorCells(mesh->numActiveElements(), localCellsToExcludeFromRefinement);

  // invert the map:
  std::map<double, std::set<GlobalIndexType>, std::greater<double> > sortedMaxErrors; // use set as key due to possibility of collisions in error value; order from greatest to least error
  for (const auto &entry : globalMaxErrorCells)
  {
    sortedMaxErrors[entry.second].insert(entry.first);
  }
  
  std::set<GlobalIndexType> globalCellsToRefine;
  double errorSquaredSoFar = 0.0;
  const double errorSquaredTarget = _doerflerTarget * _doerflerTarget * totalError * totalError;
  for (auto entry : sortedMaxErrors)
  {
    int entryMultiplicity = entry.second.size();
    
    globalCellsToRefine.insert(entry.second.begin(),entry.second.end());
    const double entryErrorSquared = entry.first * entry.first;
    errorSquaredSoFar += entryErrorSquared * entryMultiplicity;
    if (errorSquaredSoFar >= errorSquaredTarget)
    {
      break;
    }
  }
  
  map<GlobalIndexType, double> cellMeasuresLocal;
  std::set<GlobalIndexType> localCellsToRefine;
  for (const GlobalIndexType & cellID : globalCellsToRefine)
  {
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
  
  GlobalIndexTypeToCast myNumCellsToHRefine = myCellsToHRefine.size();
  GlobalIndexTypeToCast numCellsToHRefine = MPIWrapper::sum(mesh->Comm(), myNumCellsToHRefine);
  
  GlobalIndexTypeToCast myCellOrdinalOffset = 0;
  Teuchos::scan(*mesh->Comm(), Teuchos::REDUCE_SUM, 1, &myNumCellsToHRefine, &myCellOrdinalOffset);
  myCellOrdinalOffset -= myNumCellsToHRefine;
  
  vector<GlobalIndexTypeToCast> globalCellsToRefineVector(numCellsToHRefine, 0);
  for (int i=0; i<myNumCellsToHRefine; i++)
  {
    globalCellsToRefineVector[myCellOrdinalOffset + i] = myCellsToHRefine[i];
  }
  vector<GlobalIndexTypeToCast> gatheredCellsToRefineVector(numCellsToHRefine, 0);
  Teuchos::reduceAll(*mesh->Comm(), Teuchos::REDUCE_SUM, numCellsToHRefine, &globalCellsToRefineVector[0], &gatheredCellsToRefineVector[0]);

  vector<GlobalIndexType> cellsToRefine(gatheredCellsToRefineVector.begin(),gatheredCellsToRefineVector.end());;
  
  GlobalIndexTypeToCast myNumCellsToPRefine = myCellsToPRefine.size();
  GlobalIndexTypeToCast numCellsToPRefine = MPIWrapper::sum(mesh->Comm(), myNumCellsToPRefine);
  
  vector<GlobalIndexType> cellsToPRefine;
  if (numCellsToPRefine > 0)
  {
    myCellOrdinalOffset = MPIWrapper::scanSum(mesh->Comm(), myNumCellsToPRefine);
    myCellOrdinalOffset -= myNumCellsToPRefine;

    vector<GlobalIndexTypeToCast> gatheredCellsToPRefineVector(numCellsToPRefine, 0);
    for (int i=0; i<myNumCellsToPRefine; i++)
    {
      gatheredCellsToPRefineVector[myCellOrdinalOffset + i] = myCellsToPRefine[i];
    }
    MPIWrapper::entryWiseSum(mesh->Comm(), gatheredCellsToPRefineVector);

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
  
  if (printToConsole && (mesh->Comm()->getRank() == 0))
  {
    cout << "Prior to refinement, total error: " << totalError << endl;
    cout << "After refinement, mesh has " << activeElementCount << " elements and " << globalDofCount << " global dofs" << endl;
  }
}
