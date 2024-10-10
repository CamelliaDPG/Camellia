//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "RefinementStrategy.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( ErrorIndicatorPtr errorIndicator, double relativeErrorThreshold,
                                                 double min_h, int max_p , bool preferPRefinements)
{
  _errorIndicator = errorIndicator;
  _relativeErrorThreshold = relativeErrorThreshold;
  _min_h = min_h;
  _max_p = max_p;
  _preferPRefinements = preferPRefinements;
  _enforceOneIrregularity = true;
}

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( TSolutionPtr<Scalar> solution,
                                                 double relativeEnergyThreshold, double min_h,
                                                 int max_p, bool preferPRefinements) :
TRefinementStrategy<Scalar>::TRefinementStrategy(ErrorIndicator::energyErrorIndicator(solution),
                                                 relativeEnergyThreshold, min_h, max_p, preferPRefinements)
{}

template <typename Scalar>
TRefinementStrategy<Scalar>::TRefinementStrategy( MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip,
                                                 double relativeEnergyThreshold, double min_h,
                                                 int max_p, bool preferPRefinements, int cubatureEnrichment) :
TRefinementStrategy<Scalar>::TRefinementStrategy(ErrorIndicator::energyErrorIndicator(mesh, residual, ip, cubatureEnrichment),
                                                 relativeEnergyThreshold, min_h, max_p, preferPRefinements)

{}

template <typename Scalar>
double TRefinementStrategy<Scalar>::computeTotalEnergyError()
{
  _errorIndicator->measureError();
  return _errorIndicator->totalError();
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::energyErrorRefinementStrategy(SolutionPtr soln, double relativeEnergyThreshold)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(soln);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::energyErrorRefinementStrategy(MeshPtr mesh, TLinearTermPtr<Scalar> residual,
                                                                                 TIPPtr<Scalar> ip,
                                                                                 double relativeEnergyThreshold,
                                                                                 int cubatureEnrichmentDegree)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(mesh, residual, ip, cubatureEnrichmentDegree);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
ErrorIndicatorPtr TRefinementStrategy<Scalar>::errorIndicator() const
{
  return _errorIndicator;
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::gradientRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::gradientErrorIndicator(soln, scalarVar);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
RefinementStrategyPtr TRefinementStrategy<Scalar>::hessianRefinementStrategy(SolutionPtr soln, VarPtr scalarVar, double relativeEnergyThreshold)
{
  ErrorIndicatorPtr errorIndicator = ErrorIndicator::hessianErrorIndicator(soln, scalarVar);
  return Teuchos::rcp( new TRefinementStrategy<Scalar>(errorIndicator, relativeEnergyThreshold) );
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setMinH(double value)
{
  _min_h = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setEnforceOneIrregularity(bool value)
{
  _enforceOneIrregularity = value;
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::setReportPerCellErrors(bool value)
{
  _reportPerCellErrors = value;
}

template <typename Scalar>
MeshPtr TRefinementStrategy<Scalar>::mesh()
{
  return _errorIndicator->mesh();
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::recordResults(GlobalIndexType numElements, GlobalIndexType numDofs, double totalEnergyError)
{
  auto resultsRecord = setResults(numElements, numDofs, totalEnergyError);
  _results.push_back(resultsRecord);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::refine(bool printToConsole)
{
  const bool unrefine = (_maxElementCount != std::numeric_limits<int>::max());
  this->refineAndUnrefine(unrefine, printToConsole);
}

template <typename Scalar>
void TRefinementStrategy<Scalar>::refineAndUnrefine(bool unrefine, bool printToConsole)
{
  // greedy refinement algorithm - mark cells for refinement
  MeshPtr mesh = this->mesh();
  auto meshTopo = mesh->getTopology();

  const set<GlobalIndexType> & myCellIDs = mesh->cellIDsInPartition();
  // these cell measure determinations are redundant with localCellsIneligibleForRefinement().  Consider adding a state variable to track these, either in Mesh or in RefinementStrategy.  (An optimization of what is likely not a large cost.)
  map<GlobalIndexType, double> cellMeasuresLocal;
  for (GlobalIndexType cellID : myCellIDs)
  {
    cellMeasuresLocal[cellID] = mesh->getCellMeasure(cellID);
  }
  const double meshDim = mesh->getDimension();
  
  std::set<GlobalIndexType> localCellsToExcludeFromRefinement = localCellsIneligibleForRefinement(); // exclude cells that can't be refined further from the max error computation
  
  vector<GlobalIndexType> localCellsToRefine;
  getCellsAboveErrorThreshhold(localCellsToRefine, localCellsToExcludeFromRefinement); // calls _errorIndicator->measureError()
  double totalError = _errorIndicator->totalError();
  
  // record results prior to refinement
  RefinementResults results = setResults(mesh->numActiveElements(), mesh->numGlobalDofs(), totalError);
  _results.push_back(results);
  
  vector<GlobalIndexType> myCellsToHRefine;
  vector<GlobalIndexType> myCellsToPRefine;
  
  for (GlobalIndexType cellID : localCellsToRefine)
  {
    double h = pow(cellMeasuresLocal[cellID], 1.0 / meshDim); // dth root of volume measure = h
    int p = mesh->cellPolyOrder(cellID);

    const int cellLevel = mesh->getTopology()->getCell(cellID)->level();
    
    const bool canHRefine = ((h > _min_h) && (cellLevel < _maxRefinementLevel));
    const bool canPRefine = (p < _max_p);
    
    if (!_preferPRefinements)
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
  
  std::set<GlobalIndexType> parentCellsToUnrefine;
  
  if (unrefine)
  {
    int numCellsToBeAdded = 0;
    int localNumCellsToBeAdded = 0;
    std::set<GlobalIndexType> cellsToRefineSet(cellsToRefine.begin(),cellsToRefine.end());
    for (const auto & cellID : cellsToRefine)
    {
      if (myCellIDs.find(cellID) != myCellIDs.end())
      {
        CellPtr cell = meshTopo->getCell(cellID);
        auto cellTopoKey = cell->topology()->getKey();
        RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopoKey);
        
        localNumCellsToBeAdded += refPattern->numChildren() - 1; // -1 because the parent cell will no longer be active
        
        if (_enforceOneIrregularity)
        {
          // if any of cell's neighbors are at a lower level of refinement than cell, then they too will be refined during 1-irregularity enforcement
          // (there can be a cascade, but for present purposes -- estimating the number of new cells that will be added -- we don't worry about that)
          auto neighbors = cell->getNeighbors(meshTopo);
          for (auto neighbor : neighbors)
          {
            if (cellsToRefineSet.find(neighbor->cellIndex()) != cellsToRefineSet.end())
            {
              // the neighbor will be refined anyway prior to 1-irregularity enforcement; will be counted by its owner
              continue;
            }
            // we are making an assumption here that we're doing isotropic refinements; if not, then level comparison isn't the right thing to check
            // (anisotropic refinements are not a heavily exercised code path…)
            if (neighbor->level() < cell->level())
            {
              localNumCellsToBeAdded += refPattern->numChildren() - 1;  // -1 because the parent cell will no longer be active
            }
          }
        }
        
      }
    }
    mesh->Comm()->SumAll(&localNumCellsToBeAdded, &numCellsToBeAdded, 1);

    const int numActiveCellsAfterRefinement = mesh->numActiveElements() + numCellsToBeAdded;
    const int cellReductionTarget = std::max(numActiveCellsAfterRefinement - _maxElementCount, 0);
    
    // if a child has been marked for refinement, we're not allowed to unrefine its parent:
    std::set<GlobalIndexType> childCellsToExclude(cellsToRefine.begin(),cellsToRefine.end());
    childCellsToExclude.insert(cellsToPRefine.begin(), cellsToPRefine.end());
    
    if (_enforceOneIrregularity)
    {
      std::set<GlobalIndexType> myAdditionalChildCellsToExclude;
      // peer neighbors of cellsToRefine should also be excluded from having their parents unrefined
      for (const auto & myCellID : myCellIDs)
      {
        if (childCellsToExclude.find(myCellID) != childCellsToExclude.end())
        {
          // this cell is already excluded; no need to check it further
          continue;
        }
        
        CellPtr myCell = meshTopo->getCell(myCellID);
        const unsigned int sideCount = myCell->getSideCount();
        for (unsigned int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
        {
          auto neighborInfo = myCell->getNeighborInfo(sideOrdinal, meshTopo);
          auto neighborCellID = neighborInfo.first;
          if (childCellsToExclude.find(neighborCellID) != childCellsToExclude.end())
          {
            // neighbor is marked for refinement; myCell's parent should not be unrefined if myCell is a peer of neighbor
            CellPtr neighborCell = meshTopo->getCell(neighborCellID);
            const unsigned & mySideOrdinalInNeighbor = neighborInfo.second;
            auto neighborNeighborInfo = neighborCell->getNeighborInfo(mySideOrdinalInNeighbor, meshTopo);
            bool isPeer = neighborNeighborInfo.first == myCellID;
            if (isPeer)
            {
              myAdditionalChildCellsToExclude.insert(myCellID);
              break; // no need to check further neighbors
            }
          }
          else if (neighborCellID != -1)
          {
            CellPtr neighbor = meshTopo->getCell(neighborCellID);
            if (neighbor->isParent(meshTopo))
            {
              // if neighbor is refined along side, then we should not unrefine myCell's parent
              const auto sideOrdinalInNeighbor = neighborInfo.second;
              const bool neighborSideIsRefined = (neighbor->refinementPattern()->childrenForSides()[sideOrdinalInNeighbor].size() > 1);
              if (neighborSideIsRefined)
              {
                myAdditionalChildCellsToExclude.insert(myCellID);
                break; // already excluded; no need to check further neighbors…
              }
            }
          }
        }
      }
      
      std::vector<GlobalIndexType> myAdditionalChildCellsToExcludeVector(myAdditionalChildCellsToExclude.begin(),myAdditionalChildCellsToExclude.end());
      std::vector<GlobalIndexType> globalAdditionalChildCellsToExclude;
      
      std::vector<int> offsets;
      MPIWrapper::allGatherVariable(*mesh->Comm(), globalAdditionalChildCellsToExclude, myAdditionalChildCellsToExcludeVector, offsets);
      
      childCellsToExclude.insert(globalAdditionalChildCellsToExclude.begin(),globalAdditionalChildCellsToExclude.end());
    }
    
    ErrorIndicatorPtr unrefinementErrorIndicator = (_unrefinementErrorIndicator == Teuchos::null) ? _errorIndicator : _unrefinementErrorIndicator;
    
    std::vector< std::pair<GlobalIndexType,double> > parentEntries = unrefinementErrorIndicator->globalMinimumErrorParentCells(cellReductionTarget, childCellsToExclude);
    
    parentCellsToUnrefine = std::set<GlobalIndexType>();
    for (const auto &entry : parentEntries)
    {
      parentCellsToUnrefine.insert(entry.first);
    }
  }
  
  if (printToConsole)
  {
    if (cellsToRefine.size() > 0)         Camellia::print("cells for h-refinement", cellsToRefine);
    if (cellsToPRefine.size() > 0)        Camellia::print("cells for p-refinement", cellsToPRefine);
    if (parentCellsToUnrefine.size() > 0) Camellia::print("cells for unrefinement", parentCellsToUnrefine);
  }
  
  refineCells(cellsToRefine);
  pRefineCells(mesh, cellsToPRefine);
  unrefineCells(parentCellsToUnrefine);

  bool repartitionAndRebuild = false;
  if (_enforceOneIrregularity)
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
  
  double maxError = _errorIndicator->maxError();
  double errorThreshold = maxError * _relativeErrorThreshold;
  
  vector<GlobalIndexType> cellsAboveThreshold;
  _errorIndicator->localCellsAboveErrorThreshold(errorThreshold, cellsAboveThreshold);
  
  for (auto cell : cellsAboveThreshold)
  {
    if (localCellsToExclude.find(cell) == localCellsToExclude.end())
    {
      cellsToRefine.push_back(cell);
    }
  }
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
double TRefinementStrategy<Scalar>::relativeErrorThreshold() const
{
  return _relativeErrorThreshold;
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
int TRefinementStrategy<Scalar>::greedyUnrefinement(double relativeErrorThreshold, // relative to maximum element error in the mesh
                                                     bool printToConsole)
{
  ErrorIndicatorPtr unrefinementErrorIndicator = (_unrefinementErrorIndicator == Teuchos::null) ? _errorIndicator : _unrefinementErrorIndicator;
  unrefinementErrorIndicator->measureError();
  const double maxElementError = unrefinementErrorIndicator->maxError();
  const double absoluteErrorThreshold = relativeErrorThreshold * maxElementError;
  
  std::set<GlobalIndexType> childCellsToExclude;
  std::vector< std::pair<GlobalIndexType,double> > parentEntries = unrefinementErrorIndicator->globalParentCellsBelowThreshold(absoluteErrorThreshold, childCellsToExclude);
  
  auto parentCellsBelowThreshold = std::set<GlobalIndexType>();
  for (auto entry : parentEntries)
  {
    parentCellsBelowThreshold.insert(entry.first);
  }
  
  MeshPtr mesh = this->mesh();
  auto meshTopo = mesh->getTopology();
  
  std::set<GlobalIndexType> compatibleUnrefinements;   // not more than 1-irregularity that results
  
  auto unrefinementPermitted = [&](const GlobalIndexType &cellID) -> bool // return of true indicates that unrefinement is permissible
  {
    CellPtr cell = meshTopo->getCell(cellID);
    const unsigned int sideCount = cell->getSideCount();
    for (unsigned int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      auto neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
      auto neighborCellID = neighborInfo.first;
      if (neighborCellID != -1)
      {
        CellPtr neighbor = meshTopo->getCell(neighborCellID);
        if (neighbor->isParent(meshTopo))
        {
          // if neighbor is refined along side, then we are only allowed to unrefine this cell if neighbor's children along the side are active -- most precisely, they should be unrefined peers of this cell's children along the shared side.
          const auto sideOrdinalInNeighbor = neighborInfo.second;
          auto neighborChildEntries = neighbor->refinementPattern()->childrenForSides()[sideOrdinalInNeighbor];
          const bool neighborSideIsRefined = (neighborChildEntries.size() > 1);
          if (neighborSideIsRefined)
          {
            auto cellChildEntries = cell->refinementPattern()->childrenForSides()[sideOrdinal];
            for (const auto & cellChildEntry : cellChildEntries)
            {
              const unsigned & cellChildOrdinal     = cellChildEntry.first;
              const unsigned & cellChildSideOrdinal = cellChildEntry.second;
              
              CellPtr cellChild = cell->children()[cellChildOrdinal];
              auto cellChildNeighborInfo = cellChild->getNeighborInfo(cellChildSideOrdinal, meshTopo);
              auto cellChildNeighborInfoCellID = cellChildNeighborInfo.first;
              
              if (meshTopo->isParent(cellChildNeighborInfoCellID))
              {
                // neighbor's child on side is further refined => we can't unrefine cell
                return false;
              }
            }
          }
        }
      }
    }
    
    // if we get here, no neighbors prevent this cell from being unrefined
    return true;
  };
  
  for (auto parentCellID : parentCellsBelowThreshold)
  {
    if (unrefinementPermitted(parentCellID))
    {
      compatibleUnrefinements.insert(parentCellID);
    }
  }
  
  if (compatibleUnrefinements.size() > 0)
  {
    if (printToConsole) Camellia::print("cells for unrefinement", compatibleUnrefinements);
    
    unrefineCells(compatibleUnrefinements);
    mesh->repartitionAndRebuild();
    unrefinementErrorIndicator->measureError(); // now that we have unrefined, measure the error again
  }
  
  return int(compatibleUnrefinements.size());
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
