//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  MeshTopologyView.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/23/15.
//
//

#include "MeshTopologyView.h"

#include "CamelliaDebugUtility.h"
#include "CellDataMigration.h"
#include "GlobalDofAssignment.h"
#include "MeshTopology.h"
#include "MPIWrapper.h"
#include "TimeLogger.h"

using namespace Camellia;
using namespace std;

template<typename A>
long long approximateSetSizeLLVM(const set<A> &someSet)   // in bytes
{
  // 48 bytes for the set itself; nodes are 32 bytes + sizeof(pair<A,B>) each
  // if A and B are containers, this won't count their contents...
  
  set<int> emptySet;
  int SET_OVERHEAD = sizeof(emptySet);
  
  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal
  
  return SET_OVERHEAD + (MAP_NODE_OVERHEAD + sizeof(A)) * someSet.size();
}

// ! Constructor for use by MeshTopology and any other subclasses
MeshTopologyView::MeshTopologyView()
{
  _globalCellCount = -1;
  _globalActiveCellCount = -1;
}

// ! Constructor that defines a view in terms of an existing MeshTopology and a set of cells selected to be active.
MeshTopologyView::MeshTopologyView(ConstMeshTopologyPtr meshTopoPtr, const std::set<IndexType> &activeCellIDs)
{
  _meshTopo = meshTopoPtr;
  _activeCells = activeCellIDs;
  _globalActiveCellCount = -1;
  _globalCellCount = -1;
  buildLookups();
}

IndexType MeshTopologyView::activeCellCount() const
{
  return _globalActiveCellCount;
}

// ! This method only gets within a factor of 2 or so, but can give a rough estimate
long long MeshTopologyView::approximateMemoryFootprint() const
{
  // size of pointers plus size of sets:
  long long footprint = sizeof(_meshTopo);
  footprint += approximateSetSizeLLVM(_activeCells);
  footprint += approximateSetSizeLLVM(_rootCells);
  footprint += approximateSetSizeLLVM(_allKnownCells);
  footprint += sizeof(_gda);
  return  footprint;
}

const MeshTopology* MeshTopologyView::baseMeshTopology() const
{
  return _meshTopo.get();
}

IndexType MeshTopologyView::cellCount() const
{
  return _globalCellCount;
}


template<typename GlobalIndexContainer>
void MeshTopologyView::cellHalo(GlobalIndexContainer &haloCellIndices, const set<GlobalIndexType> &cellIndices,
                                unsigned dimForNeighborRelation) const
{
  // the cells passed in are the ones the user wants to include -- e.g. those owned by the MPI rank.
  // we keep more than that; we keep all ancestors and siblings of the cells, as well as all cells that share
  // dimForNeighborRelation-dimensional entities with the cells or their ancestors.
  
  int timerHandle = TimeLogger::sharedInstance()->startTimer("cellHalo");
  
  ConstMeshTopologyViewPtr thisPtr = Teuchos::rcp(this,false);
  
  // for all the original cells, also add the peer neighbors of their ancestors
  // (these need to be included here so that, for example, geometric multigrid can work properly.)
  set<GlobalIndexType> cellsThatMatch = cellIndices;

  vector<set<IndexType>> entitiesToMatchSetForLevel;

  for (GlobalIndexType cellID : cellIndices)
  {
    CellPtr cell = getCell(cellID);
    
    while (cell->getParent() != Teuchos::null)
    {
      cell = cell->getParent();
      bool peersOnly = true;
      vector<IndexType> entitiesToMatch = cell->entitiesOnNeighborInterfaces(dimForNeighborRelation, peersOnly, thisPtr);
      if (entitiesToMatchSetForLevel.size() <= cell->level())
      {
        entitiesToMatchSetForLevel.resize(cell->level() + 1);
      }
      entitiesToMatchSetForLevel[cell->level()].insert(entitiesToMatch.begin(),entitiesToMatch.end());
    }
  }
  
  int sideDim = this->getDimension() - 1;
  
  for (int level=0; level<entitiesToMatchSetForLevel.size(); level++)
  {
    set<IndexType>* entitiesToMatchSet = &entitiesToMatchSetForLevel[level];
    vector<IndexType> entitiesToMatch(entitiesToMatchSet->begin(),entitiesToMatchSet->end());
    vector<IndexType> sidesThatMatch = this->getSidesContainingEntities(dimForNeighborRelation, entitiesToMatch);
    set<pair<IndexType,unsigned>> cellPairs = this->getCellsContainingSides(sidesThatMatch);
    // add just the ones that are actually peers (i.e. are on this level)
    for (auto cellPair : cellPairs)
    {
      CellPtr cell = getCell(cellPair.first);
      if (cell->level() == level)
      {
        cellsThatMatch.insert(cellPair.first);
      }
      else if (cell->level() > level)
      {
        int levelDiff = cell->level() - level;
        for (int i=0; i<levelDiff; i++)
        {
          cell = cell->getParent();
        }
        IndexType cellIndex = cell->cellIndex();
        if (cellsThatMatch.find(cellIndex) != cellsThatMatch.end()) continue; // we've already matched this cell
        
        // otherwise, before we add, check to make sure this cell has a side among sidesThatMatch
        int sideCount = cell->getSideCount();
        for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
        {
          IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
          if (std::find(sidesThatMatch.begin(), sidesThatMatch.end(), sideEntityIndex) != sidesThatMatch.end())
          {
            cellsThatMatch.insert(cellIndex);
            break;
          }
        }
      }
    }
  }
  
  // now, let's find all the entities that these cells touch
  set<IndexType> entitiesToMatchSet;
  bool peersOnly = false; // also get descendants
  for (GlobalIndexType cellID : cellsThatMatch)
  {
    CellPtr cell = getCell(cellID);
    vector<IndexType> cellEntities = cell->entitiesOnNeighborInterfaces(dimForNeighborRelation, peersOnly, thisPtr);
    entitiesToMatchSet.insert(cellEntities.begin(),cellEntities.end());
  }
  
  vector<IndexType> entitiesToMatch(entitiesToMatchSet.begin(),entitiesToMatchSet.end());
  // now, determine which cells match those entities
  set< pair<IndexType, unsigned> > cellPairs = getCellsContainingEntities(dimForNeighborRelation, entitiesToMatch);
  for (pair<IndexType, unsigned> cellPair : cellPairs)
  {
    IndexType cellID = cellPair.first;
    cellsThatMatch.insert(cellID);
  }

  // now, for each cell, ascend the refinement hierarchy, adding ancestors
  for (GlobalIndexType cellID : cellsThatMatch)
  {
    CellPtr cell = getCell(cellID);
    haloCellIndices.insert(cellID);
    while (cell->getParent() != Teuchos::null)
    {
      cell = cell->getParent();
      haloCellIndices.insert(cell->cellIndex());
    }
  }
  
  TimeLogger::sharedInstance()->stopTimer(timerHandle);
}

template void MeshTopologyView::cellHalo<set<GlobalIndexType>>(set<GlobalIndexType> &haloCellIndices, const set<GlobalIndexType> &cellIndices,
                                                               unsigned dimForNeighborRelation) const;
template void MeshTopologyView::cellHalo<RangeList<GlobalIndexType>>(RangeList<GlobalIndexType> &haloCellIndices, const set<GlobalIndexType> &cellIndices,
                                                                     unsigned dimForNeighborRelation) const;

std::vector<IndexType> MeshTopologyView::cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const
{
  std::vector<IndexType> descendentIDs = _meshTopo->cellIDsForPoints(physicalPoints);
  vector<IndexType> myIDs;
  for (IndexType descendentCellID : descendentIDs)
  {
    while ((descendentCellID != -1) && (_activeCells.find(descendentCellID) == _activeCells.end()))
    {
      CellPtr descendentCell = _meshTopo->getCell(descendentCellID);
      CellPtr parentCell = descendentCell->getParent();
      if (parentCell == Teuchos::null)
        descendentCellID = -1;
      else
        descendentCellID = parentCell->cellIndex();
    }
    myIDs.push_back(descendentCellID);
  }
  return myIDs;
}

void MeshTopologyView::buildLookups()
{
  set<IndexType> visitedIndices;
  set<IndexType> invalidActiveCells;
  for (IndexType cellID : _activeCells)
  {
    if (!_meshTopo->isValidCellIndex(cellID))
    {
      // this can happen for distributed MeshTopology
      // simply ignore invalid cells, and remove them from the _activeCells container
      invalidActiveCells.insert(cellID);
      continue;
    }
    CellPtr cell = _meshTopo->getCell(cellID);
    while ((cell->getParent() != Teuchos::null) && (visitedIndices.find(cellID) == visitedIndices.end()))
    {
      visitedIndices.insert(cellID);
      cell = cell->getParent();
      cellID = cell->cellIndex();
    }
    if (cell->getParent() == Teuchos::null)
    {
      _rootCells.insert(cellID);
      visitedIndices.insert(cellID);
    }
  }
  for (IndexType invalidCellID : invalidActiveCells)
  {
    _activeCells.erase(invalidCellID);
  }
  _allKnownCells.insert(_rootCells.begin(),_rootCells.end());
  _allKnownCells.insert(visitedIndices.begin(),visitedIndices.end());
  
  /*
   If _meshTopo is not distributed, then we can assume that our view of _activeCells and _allKnownCells
   is complete.  Otherwise, we need to do a little extra work to determine _globalActiveCellCount and
   _globalCellCount.
   */
  
  if (isDistributed())
  {
    // we're not too concerned, really, with _globalCellCount.  For MeshTopologyView, this is pretty much
    // only used in tests.  But in keeping with what MeshTopology does, we can simply define this as the
    // greatest cell index seen by any rank, plus 1.
    GlobalIndexTypeToCast myGreatestID;
    if ( visitedIndices.size() > 0)
      myGreatestID = *visitedIndices.rbegin();
    else
      myGreatestID = -1;
    GlobalIndexTypeToCast globalMaxID = 0;
//    _meshTopo->Comm()->MaxAll(&myGreatestID, &globalMaxID, 1);
    Teuchos::reduceAll(*_meshTopo->Comm(), Teuchos::REDUCE_MAX, 1, &myGreatestID, &globalMaxID);
    _globalCellCount = globalMaxID + 1;
    
    // determine which cells are locally owned
    set<IndexType> myActiveCellIndices = getMyActiveCellIndices();
    int myActiveCellCount = myActiveCellIndices.size();
    int globalActiveCellCount = 0;
//    _meshTopo->Comm()->SumAll(&myActiveCellCount, &globalActiveCellCount, 1);
    Teuchos::reduceAll(*_meshTopo->Comm(), Teuchos::REDUCE_SUM, 1, &myActiveCellCount, &globalActiveCellCount);
    _globalActiveCellCount = globalActiveCellCount;
  }
  else
  {
    _globalActiveCellCount = _activeCells.size();
    _globalCellCount = _allKnownCells.size();
  }
}

vector<IndexType> MeshTopologyView::cellIDsWithCentroids(const std::vector<std::vector<double>> &centroids, double tol) const
{
  // returns a vector of an active element per point, or null if there is no locally known element including that point
  vector<GlobalIndexType> cellIDs;
  //  cout << "entered elementsForPoints: \n" << physicalPoints;
  int numPoints = centroids.size();
  int spaceDim = this->getDimension();
  
  // define lambda for checking if we have a match:
  auto cellHasCentroid = [this, &centroids, tol] (CellPtr cell, int centroidIndex) -> bool
  {
    vector<double> centroid = getCellCentroid(cell->cellIndex());
    for (int d=0; d<centroid.size(); d++)
    {
      double diff = abs(centroid[d]-centroids[centroidIndex][d]);
      if (diff > tol) return false;
    }
    return true;
  };
  
  set<GlobalIndexType> rootCellIndices = this->getRootCellIndicesLocal();
  for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
  {
    vector<double> point = centroids[pointIndex];
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != point.size(), std::invalid_argument, "Each point must have size equal to spaceDim");
    
    // find the element from the original mesh that contains this point
    CellPtr cell;
    bool foundMatch = false;
    for (GlobalIndexType cellID : rootCellIndices)
    {
      int cubatureDegreeForCell = 1;
      if ( _gda != NULL)
      {
        cubatureDegreeForCell = _gda->getCubatureDegree(cellID);
      }
      if (baseMeshTopology()->cellContainsPoint(cellID,point,cubatureDegreeForCell))
      {
        cell = getCell(cellID);
        if (cellHasCentroid(cell,pointIndex)) foundMatch = true;
        break;
      }
    }
    if ((cell.get() != NULL) && (!foundMatch))
    {
      ConstMeshTopologyViewPtr thisPtr = Teuchos::rcp(this,false);
      while ( cell->isParent(thisPtr) && !foundMatch )
      {
        int numChildren = cell->numChildren();
        for (int childOrdinal = 0; childOrdinal < numChildren; childOrdinal++)
        {
          CellPtr child = cell->children()[childOrdinal];
          int cubatureDegreeForCell = 1;
          if (_gda != NULL)
          {
            cubatureDegreeForCell = _gda->getCubatureDegree(child->cellIndex());
          }
          if (baseMeshTopology()->cellContainsPoint(child->cellIndex(),point,cubatureDegreeForCell) )
          {
            cell = child;
            if (cellHasCentroid(cell,pointIndex))
            {
              foundMatch = true;
            }
            break;
          }
        }
      }
    }
    GlobalIndexType cellID = -1;
    if (foundMatch)
    {
      cellID = cell->cellIndex();
    }
    cellIDs.push_back(cellID);
  }
  return cellIDs;
}

Teuchos_CommPtr MeshTopologyView::Comm() const
{
  return _meshTopo->Comm();
}

// ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
Teuchos::RCP<MeshTopology> MeshTopologyView::deepCopy() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "deepCopy() not supported by MeshTopologyView; this method is defined for potential subclass support.");
}

bool MeshTopologyView::entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent) const
{
  return _meshTopo->entityIsAncestor(d, ancestor, descendent);
}

bool MeshTopologyView::entityIsGeneralizedAncestor(unsigned int ancestorDimension, IndexType ancestor, unsigned int descendentDimension, IndexType descendent) const
{
  return _meshTopo->entityIsGeneralizedAncestor(ancestorDimension, ancestor, descendentDimension, descendent);
}

bool MeshTopologyView::entityMatchesTag(unsigned int d, IndexType entityIndex, const std::pair<std::string, int> &tag) const
{
  return _meshTopo->entityMatchesTag(d, entityIndex, tag);
}

const set<IndexType> &MeshTopologyView::getLocallyKnownActiveCellIndices() const
{
  return _activeCells;
}

IndexType MeshTopologyView::getActiveCellCount(unsigned d, IndexType entityIndex) const
{
  // first entry in pair is the cellIndex, the second is the ordinal of the entity in that cell (the subcord).
  set<IndexType> activeCellIndices;
  
  vector<IndexType> sideIndices = this->getSidesContainingEntity(d, entityIndex);
  for (IndexType sideEntityIndex : sideIndices)
  {
    vector<IndexType> cells = this->getActiveCellsForSide(sideEntityIndex);
    for (IndexType cellIndex : cells)
    {
      activeCellIndices.insert(cellIndex);
    }
  }
  return activeCellIndices.size();
}

vector< pair<IndexType,unsigned> > MeshTopologyView::getActiveCellIndices(unsigned d, IndexType entityIndex) const
{
  // first entry in pair is the cellIndex, the second is the ordinal of the entity in that cell (the subcord).
  set<pair<IndexType,unsigned>> activeCellIndicesSet;
  
  vector<IndexType> sideIndices = this->getSidesContainingEntity(d, entityIndex);
  for (IndexType sideEntityIndex : sideIndices)
  {
    vector<IndexType> cells = this->getActiveCellsForSide(sideEntityIndex);
    for (IndexType cellIndex : cells)
    {
      // one of our active cells contains the entity
      CellPtr cell = _meshTopo->getCell(cellIndex);
      unsigned subcord = cell->findSubcellOrdinal(d, entityIndex);
      activeCellIndicesSet.insert({cellIndex,subcord});
    }
  }
  vector<pair<IndexType,unsigned>> activeCellIndicesVector(activeCellIndicesSet.begin(),activeCellIndicesSet.end());
  return activeCellIndicesVector;
}

std::set<IndexType> MeshTopologyView::getActiveCellIndicesForAncestorsOfMyCellsInBaseMeshTopology() const
{
  if (_meshTopo->Comm() == Teuchos::null)
  {
    // not distributed, so we don't know which ones belong to us.  Best we can do is take all active cells.
    return _activeCells;
  }
  const set<IndexType>* baseMyCellIndices = &_meshTopo->getMyActiveCellIndices();
  set<IndexType> ancestralActiveCellIndices;
  for (IndexType myBaseCellIndex : *baseMyCellIndices)
  {
    CellPtr cell = _meshTopo->getCell(myBaseCellIndex);
    bool foundAncestor = false;
    while (cell != Teuchos::null)
    {
      if (_activeCells.find(cell->cellIndex()) != _activeCells.end())
      {
        ancestralActiveCellIndices.insert(cell->cellIndex());
        foundAncestor = true;
        break;
      }
      cell = cell->getParent();
    }
    if (!foundAncestor)
    {
      cout << "No active ancestor found for cellIndex " << myBaseCellIndex << " in MeshTopologyView.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No active ancestor found");
    }
  }
  return ancestralActiveCellIndices;
}

vector<IndexType> MeshTopologyView::getActiveCellsForSide(IndexType sideEntityIndex) const
{
  vector<IndexType> cellsForSide = getCellsForSide(sideEntityIndex);
  
  vector<IndexType> activeCells;
  for (IndexType cellIndex : cellsForSide)
  {
    if (_activeCells.find(cellIndex) != _activeCells.end())
    {
      activeCells.push_back(cellIndex);
    }
  }
  return activeCells;
}

CellPtr MeshTopologyView::getCell(IndexType cellIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "Invalid cellIndex!");
  return _meshTopo->getCell(cellIndex);
}

vector<double> MeshTopologyView::getCellCentroid(IndexType cellIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "Invalid cellIndex!");
  return _meshTopo->getCellCentroid(cellIndex);
}

// ! pairs are (cellIndex, sideOrdinal) where the sideOrdinal is a side that contains the entity
set< pair<IndexType, unsigned> > MeshTopologyView::getCellsContainingEntity(unsigned d, IndexType entityIndex,
                                                                            std::vector< std::pair<std::string, int> > tags) const  // not *all* cells, but within any refinement branch, the most refined cell that contains the entity will be present in this set.  The unsigned value is the ordinal of a *side* in the cell containing this entity.  There may be multiple sides in a cell that contain the entity; this method will return just one entry per cell.  New 6-22-16: guaranteed to return the side in the cell with least ordinal.  This allows the returned set to be independent of the side indexing in MeshTopology (potentially important for distributed MeshTopology).
{
  if (d==getDimension())
  {
    // entityIndex is a cell; the side then is contained within the cell; we'll flag this fact by setting the side ordinal to -1.
    return {{entityIndex,-1}};
  }
  if ((d == 0) && (this->getPeriodicBCs().size() > 0))
  {
    // then let's ensure that entityIndex points to the canonical vertex index (which is what getSidesContainingEntity() expects)
    std::vector<IndexType> vertexIndexVector { entityIndex };
    auto canonicalVertexIndexVector = baseMeshTopology()->getCanonicalEntityNodesViaPeriodicBCs(d, vertexIndexVector);
    entityIndex = canonicalVertexIndexVector[0];
  }
  vector<IndexType> sidesForEntity = getSidesContainingEntity(d, entityIndex, tags);
  typedef pair<IndexType,unsigned> CellPair;
  map< IndexType, unsigned > cellSides;
  int sideDim = getDimension() - 1;
  
  const MeshTopology* meshTopo = this->baseMeshTopology();
  for (IndexType sideEntityIndex : sidesForEntity)
  {
    vector<IndexType> cellsForSide = getCellsForSide(sideEntityIndex);
    
    // we now allow cellsForSide to be empty -- this can happen after an unrefinement removes all cells containing a side
    // but cellsForSide should not have more than two cells
    TEUCHOS_TEST_FOR_EXCEPTION(cellsForSide.size() > 2, std::invalid_argument, "Unexpected cell count for side.");
    
    for (IndexType cellIndex : cellsForSide)
    {
      CellPtr cell = meshTopo->getCell(cellIndex);
      unsigned sideSubcord = cell->findSubcellOrdinal(sideDim, sideEntityIndex);
      
      if (cellSides.find(cellIndex) != cellSides.end())
      {
        if (cellSides[cellIndex] > sideSubcord)
        {
          cellSides[cellIndex] = sideSubcord;
        }
      }
      else
      {
        cellSides[cellIndex] = sideSubcord;
      }
    }
  }
  set< CellPair > cells;
  cells.insert(cellSides.begin(),cellSides.end());
  return cells;
}

set< pair<IndexType, unsigned> >
MeshTopologyView::getCellsContainingEntities(unsigned d, const vector<IndexType> &entities,
                                             std::vector< std::pair<std::string, int> > tags) const
{
  vector<IndexType> sidesForEntities = getSidesContainingEntities(d,entities,tags);
  return getCellsContainingSides(sidesForEntities);
}

set< pair<IndexType, unsigned> > MeshTopologyView::getCellsContainingSides(const vector<IndexType> &sideEntityIndices) const
{
  set<pair<IndexType, unsigned>> cellEntries;
  const MeshTopology* baseMeshTopo = baseMeshTopology();
  for (IndexType sideEntityIndex : sideEntityIndices)
  {
    for (int whichCell=0; whichCell<2; whichCell++)
    {
      IndexType cellIndex;
      if (whichCell == 0)
        cellIndex = baseMeshTopo->getFirstCellForSide(sideEntityIndex).first;
      else
        cellIndex =  baseMeshTopo->getSecondCellForSide(sideEntityIndex).first;
      unsigned sideOrdinalInCell = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
      int sideDim = baseMeshTopo->getDimension() - 1;
      while ((cellIndex != -1) && !isValidCellIndex(cellIndex))
      {
        // we should check whether there is a valid ancestor of the cell that also contains the side
        // (this can happen both in 1D and with anisotropic refinements)
        CellPtr cell = baseMeshTopo->getCell(cellIndex);
        sideOrdinalInCell = cell->findSubcellOrdinal(sideDim, sideEntityIndex);
        CellPtr parent = cell->getParent();
        if (parent != Teuchos::null)
        {
          // check whether the parent contains the side:
          unsigned sideOrdinal = parent->findSubcellOrdinal(sideDim, sideEntityIndex);
          if (sideOrdinal != INVALID_CELLTOPOLOGY_SIDE_ORDINAL) cellIndex = parent->cellIndex();
          else cellIndex = -1;
        }
        else
        {
          cellIndex = -1;
        }
      }
      if (isValidCellIndex(cellIndex)) cellEntries.insert({cellIndex,sideOrdinalInCell});
    }
  }
  return cellEntries;
}

vector<IndexType> MeshTopologyView::getSidesContainingEntities(unsigned d, const vector<IndexType> &entities,
                                                               std::vector< std::pair<std::string, int> > tags) const
{
  vector<IndexType> viewSides;
  vector<IndexType> meshTopoSides = _meshTopo->getSidesContainingEntities(d,entities,tags);
  
  for (IndexType topoSideIndex : meshTopoSides)
  {
    if (this->getCellsForSide(topoSideIndex).size() > 0)
    {
      viewSides.push_back(topoSideIndex);
    }
  }
  return viewSides;
}

vector<IndexType> MeshTopologyView::getCellsForSide(IndexType sideEntityIndex) const
{
  vector<IndexType> cells;
  for (int whichCell=0; whichCell<2; whichCell++)
  {
    IndexType cellIndex;
    if (whichCell == 0)
      cellIndex = _meshTopo->getFirstCellForSide(sideEntityIndex).first;
    else
      cellIndex =  _meshTopo->getSecondCellForSide(sideEntityIndex).first;
    while ((cellIndex != -1) && !isValidCellIndex(cellIndex))
    {
      // we should check whether there is a valid ancestor of the cell that also contains the side
      // (this can happen both in 1D and with anisotropic refinements)
      CellPtr cell = _meshTopo->getCell(cellIndex);
      CellPtr parent = cell->getParent();
      if (parent != Teuchos::null)
      {
        // check whether the parent contains the side:
        int sideDim = _meshTopo->getDimension() - 1;
        unsigned sideOrdinal = parent->findSubcellOrdinal(sideDim, sideEntityIndex);
        if (sideOrdinal != INVALID_CELLTOPOLOGY_SIDE_ORDINAL) cellIndex = parent->cellIndex();
        else cellIndex = -1;
      }
      else
      {
        cellIndex = -1;
      }
    }
    if (isValidCellIndex(cellIndex)) cells.push_back(cellIndex);
  }
  return cells;
}

pair<IndexType, unsigned> MeshTopologyView::getConstrainingEntity(unsigned d, IndexType entityIndex) const
{
  // copying from MeshTopology's implementation:
  unsigned sideDim = getDimension() - 1;
  
  pair<IndexType, unsigned> constrainingEntity; // we store the highest-dimensional constraint.  (This will be the maximal constraint.)
  constrainingEntity.first = entityIndex;
  constrainingEntity.second = d;
  
  IndexType generalizedAncestorEntityIndex = entityIndex;
  for (unsigned generalizedAncestorDim=d; generalizedAncestorDim <= sideDim; )
  {
    IndexType possibleConstrainingEntityIndex = getConstrainingEntityIndexOfLikeDimension(generalizedAncestorDim, generalizedAncestorEntityIndex);
    if (possibleConstrainingEntityIndex != generalizedAncestorEntityIndex)
    {
      constrainingEntity.second = generalizedAncestorDim;
      constrainingEntity.first = possibleConstrainingEntityIndex;
    }
    else
    {
      // if the generalized parent has no constraint of like dimension, then either the generalized parent is the constraint, or there is no constraint of this dimension
      // basic rule: if there exists a side belonging to an active cell that contains the putative constraining entity, then we constrain
      // I am a bit vague on whether this will work correctly in the context of anisotropic refinements.  (It might, but I'm not sure.)  But first we are targeting isotropic.
      vector<IndexType> sidesForEntity;
      if (generalizedAncestorDim==sideDim)
      {
        sidesForEntity.push_back(generalizedAncestorEntityIndex);
      }
      else
      {
        sidesForEntity = getSidesContainingEntity(generalizedAncestorDim, generalizedAncestorEntityIndex);
      }
      for (vector<IndexType>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++)
      {
        IndexType sideEntityIndex = *sideEntityIt;
        if (getActiveCellsForSide(sideEntityIndex).size() > 0)
        {
          constrainingEntity.second = generalizedAncestorDim;
          constrainingEntity.first = possibleConstrainingEntityIndex;
          break;
        }
      }
    }
    while (_meshTopo->entityHasParent(generalizedAncestorDim, generalizedAncestorEntityIndex))   // parent of like dimension
    {
      generalizedAncestorEntityIndex = _meshTopo->getEntityParent(generalizedAncestorDim, generalizedAncestorEntityIndex);
    }
    if (_meshTopo->entityHasGeneralizedParent(generalizedAncestorDim, generalizedAncestorEntityIndex))
    {
      pair< IndexType, unsigned > generalizedParent = _meshTopo->getEntityGeneralizedParent(generalizedAncestorDim, generalizedAncestorEntityIndex);
      generalizedAncestorEntityIndex = generalizedParent.first;
      generalizedAncestorDim = generalizedParent.second;
    }
    else     // at top of refinement tree -- break out of for loop
    {
      break;
    }
  }
  return constrainingEntity;
}

// copied from MeshTopology; once that's a subclass of MeshTopologyView, could possibly eliminate it in MeshTopology
IndexType MeshTopologyView::getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex) const
{
  IndexType constrainingEntityIndex = entityIndex;
  
  if (d==0)   // one vertex can't constrain another...
  {
    return entityIndex;
  }
  
  // 3-9-16: I've found an example in which the below fails in a 2-irregular mesh
  // I think the following, simpler thing will work just fine.  (It does pass tests!)
  IndexType ancestorEntityIndex = entityIndex;
  while (_meshTopo->entityHasParent(d, ancestorEntityIndex))
  {
    ancestorEntityIndex = _meshTopo->getEntityParent(d, ancestorEntityIndex);
    if (getActiveCellCount(d, ancestorEntityIndex) > 0)
    {
      constrainingEntityIndex = ancestorEntityIndex;
    }
  }
  
  return constrainingEntityIndex;
}

// getConstrainingSideAncestry() copied from MeshTopology; once that's a subclass of MeshTopologyView, could possibly eliminate it in MeshTopology
// pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)
vector< pair<IndexType,unsigned> > MeshTopologyView::getConstrainingSideAncestry(IndexType sideEntityIndex) const
{
  // three possibilities: 1) compatible side, 2) side is parent, 3) side is child
  // 1) and 2) mean unconstrained.  3) means constrained (by parent)
  unsigned sideDim = getDimension() - 1;
  vector< pair<IndexType, unsigned> > ancestry;
  if (_meshTopo->isBoundarySide(sideEntityIndex))
  {
    return ancestry; // sides on boundary are unconstrained...
  }
  
  vector< pair<IndexType,unsigned> > sideCellEntries = getActiveCellIndices(sideDim, sideEntityIndex); //_activeCellsForEntities[sideDim][sideEntityIndex];
  int activeCellCountForSide = sideCellEntries.size();
  if (activeCellCountForSide == 2)
  {
    // compatible side
    return ancestry; // will be empty
  }
  else if ((activeCellCountForSide == 0) || (activeCellCountForSide == 1))
  {
    // then we're either parent or child of an active side
    // if we are a parent, then *this* sideEntityIndex is unconstrained, and we can return an empty ancestry.
    // if we are a child, then we should find and return an ancestral path that ends in an active side
    IndexType ancestorIndex = sideEntityIndex;
    // the possibility of multiple parents is there for the sake of anisotropic refinements.  We don't fully support
    // these yet, but may in the future.
    while (_meshTopo->entityHasParent(sideDim, ancestorIndex))
    {
      int entityParentCount = _meshTopo->getEntityParentCount(sideDim, ancestorIndex);
      IndexType entityParentIndex = -1;
      for (int entityParentOrdinal=0; entityParentOrdinal<entityParentCount; entityParentOrdinal++)
      {
        entityParentIndex = _meshTopo->getEntityParent(sideDim, ancestorIndex, entityParentOrdinal);
        if (getActiveCellIndices(sideDim, entityParentIndex).size() > 0)
        {
          // active cell; we've found our final ancestor
          ancestry.push_back({entityParentIndex, entityParentOrdinal});
          return ancestry;
        }
      }
      // if we get here, then (parentEntityIndex, entityParentCount-1) refers to the last of the possible parents, which by convention must be a regular refinement (more precisely, one whose subentities are at least as fine as all previous possible parents)
      // this is therefore an acceptable entry in our ancestry path.
      ancestry.push_back({entityParentIndex, entityParentCount-1});
      ancestorIndex = entityParentIndex;
    }
    // if no such ancestral path exists, then we are a parent, and are unconstrained (return empty ancestry)
    ancestry.clear();
    return ancestry;
  }
  else
  {
    cout << "MeshTopologyView internal error: # active cells for side is not 0, 1, or 2\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTopologyView internal error: # active cells for side is not 0, 1, or 2\n");
  }
}

unsigned MeshTopologyView::getDimension() const
{
  return _meshTopo->getDimension();
}

std::vector<IndexType> MeshTopologyView::getEntityVertexIndices(unsigned d, IndexType entityIndex) const
{
  return _meshTopo->getEntityVertexIndices(d,entityIndex);
}


set<IndexType> MeshTopologyView::getGatheredActiveCellsForTime(double t) const
{
  set<IndexType> localSidesForTime = getLocallyKnownSidesForTime(t);
  set<IndexType> localActiveCellsForTime;
  for (IndexType side : localSidesForTime)
  {
    vector<IndexType> cellsForSide = getCellsForSide(side);
    for (IndexType cellIndex : cellsForSide)
    {
      if (_activeCells.find(cellIndex) != _activeCells.end())
      {
        localActiveCellsForTime.insert(cellIndex);
      }
    }
  }
  if (!isDistributed())
  {
    return localActiveCellsForTime;
  }
  const set<IndexType>* myCells = &getMyActiveCellIndices();
  vector<GlobalIndexTypeToCast> myCellsForTime;
  for (IndexType localActiveCellIndex : localActiveCellsForTime)
  {
    if (myCells->find(localActiveCellIndex) != myCells->end())
    {
      myCellsForTime.push_back(localActiveCellIndex);
    }
  }
  vector<GlobalIndexTypeToCast> gatheredCellsForTime;
  vector<int> offsets;
  MPIWrapper::allGatherVariable(Comm(), gatheredCellsForTime, myCellsForTime, offsets);
  set<GlobalIndexType> gatheredCellSet(gatheredCellsForTime.begin(),gatheredCellsForTime.end());
  return gatheredCellSet;
}

MeshTopologyPtr MeshTopologyView::getGatheredCopy() const
{
  MeshGeometryInfo baseMeshGeometry;
  CellDataMigration::getGeometry(this, baseMeshGeometry);
  auto periodicBCs = this->getPeriodicBCs();
  const bool isDistributed = (Comm() != Teuchos::null) && (Comm()->getSize() > 1);
  if (isDistributed)
  {
    int myGeometrySize = CellDataMigration::getGeometryDataSize(baseMeshGeometry);
    vector<char> myGeometryData(myGeometrySize);
    char *myWriteLocation = &myGeometryData[0];
    CellDataMigration::writeGeometryData(baseMeshGeometry, myWriteLocation, myGeometrySize);
    
    vector<char> gatheredGeometryData;
    vector<int> offsets;
    MPIWrapper::allGatherVariable<char>(Comm(),gatheredGeometryData,myGeometryData,offsets);
    
    MeshGeometryInfo gatheredBaseMeshGeometry;
    const char* gatheredDataLocation = &gatheredGeometryData[0];
    CellDataMigration::readGeometryData(gatheredDataLocation, gatheredGeometryData.size(), gatheredBaseMeshGeometry);
    
    MeshTopologyPtr gatheredMeshTopo = Teuchos::rcp( new MeshTopology(Teuchos::null, gatheredBaseMeshGeometry, periodicBCs) );
    return gatheredMeshTopo;
  }
  else
  {
    // no gather necessary; just copy
    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(Teuchos::null, baseMeshGeometry, periodicBCs) );
    return meshTopo;
  }
  
//  // if this is a pure view, we apply that to the gatheredMeshTopo
//  bool isView = (dynamic_cast<const MeshTopology*>(this) == NULL);
//  if (!isView)
//  {
//    // if this is a MeshTopology, return gatheredMeshTopo
//    return gatheredMeshTopo;
//  }
//  else
//  {
//    set<IndexType> myActiveIndices = getMyActiveCellIndices();
//    vector<int> myActiveIndicesVector(myActiveIndices.begin(),myActiveIndices.end());
//    
//    vector<int> gatheredActiveIndicesVector;
//    vector<int> offsets;
//    MPIWrapper::allGatherVariable(*Comm(), gatheredActiveIndicesVector, myActiveIndicesVector, offsets);
//
//    set<IndexType> gatheredActiveIndices(gatheredActiveIndicesVector.begin(),gatheredActiveIndicesVector.end());
//    
//    return gatheredMeshTopo->getView(gatheredActiveIndices);
//  }
}

MeshTopologyPtr MeshTopologyView::getGatheredCopy(const std::set<IndexType> &cellsToInclude) const
{
  // first, create a (still-distributed) view for the indicated cells
  MeshTopologyViewPtr viewForCells = getView(cellsToInclude);
  return viewForCells->getGatheredCopy();
}


const set<IndexType> &MeshTopologyView::getMyActiveCellIndices() const
{
  if (_ownedCellIndicesPruningOrdinal != _meshTopo->pruningOrdinal())
  {
    // rebuild _ownedCellIndices
    _ownedCellIndices.clear();
    
    /* A View owns a cell if:
       (a) that cell is owned by its MeshTopology,
       (b) the first descendant (first child of the first child of the first child, etc.) is owned
           by its MeshTopology
     
       (a) is actually just a special case of (b).
     */
    const set<IndexType>* meshTopoOwnedCellIndices = &_meshTopo->getMyActiveCellIndices();
    for (IndexType leafCellIndex : *meshTopoOwnedCellIndices)
    {
      IndexType ancestorCellIndex = leafCellIndex;
      CellPtr ancestorCell = _meshTopo->getCell(ancestorCellIndex);
      
      while (!isValidCellIndex(ancestorCellIndex) && (ancestorCell->getParent() != Teuchos::null))
      {
        int childOrdinal = ancestorCell->getParent()->findChildOrdinal(ancestorCellIndex);
        if (childOrdinal != 0)
        {
          // then we definitely do not own
          ancestorCellIndex = -1;
          ancestorCell = Teuchos::null;
          break;
        }
        else
        {
          ancestorCell = ancestorCell->getParent();
          ancestorCellIndex = ancestorCell->cellIndex();
        }
      }
      if (isValidCellIndex(ancestorCellIndex))
      {
        // then we own
        _ownedCellIndices.insert(ancestorCellIndex);
      }
    }
    
    _ownedCellIndicesPruningOrdinal = _meshTopo->pruningOrdinal();
  }
  return _ownedCellIndices;
}

const std::vector<PeriodicBCPtr> & MeshTopologyView::getPeriodicBCs() const
{
  return _meshTopo->getPeriodicBCs();
}

const set<IndexType> & MeshTopologyView::getRootCellIndicesLocal() const
{
  return _rootCells;
}

vector<IndexType> MeshTopologyView::getSidesContainingEntity(unsigned d, IndexType entityIndex,
                                                             vector< pair<string, int> > tags) const
{
  unsigned sideDim = getDimension() - 1;
  vector<IndexType> meshTopoSides;
  if (d == sideDim) meshTopoSides = {entityIndex};
  else meshTopoSides = _meshTopo->getSidesContainingEntity(d, entityIndex, tags);
  
  vector<IndexType> viewSides; // meshTopoSides, filtered for sides that belong to valid cells
  for (IndexType topoSideIndex : meshTopoSides)
  {
    if (this->getCellsForSide(topoSideIndex).size() > 0)
    {
      viewSides.push_back(topoSideIndex);
    }
  }
  return viewSides;
}

std::set<IndexType> MeshTopologyView::getLocallyKnownSidesForTime(double t) const
{
  // find all sides all of whose vertices match time t
  const MeshTopology* meshTopo = dynamic_cast<const MeshTopology*>(this);
  if (meshTopo == NULL)
  {
    meshTopo = _meshTopo.get();
  }
  
  // find vertices that match t
  vector<IndexType> matchingVertexIndices = meshTopo->getVertexIndicesForTime(t);
  // matchingVertexIndices is sorted; we use that fact below

  set<IndexType> sidesThatMatch;
  unsigned vertexDim = 0;
  unsigned sideDim = getDimension() - 1;
  for (IndexType vertexIndex : matchingVertexIndices)
  {
    vector<IndexType> sidesForVertex = getSidesContainingEntity(vertexDim, vertexIndex);
    for (IndexType sideForVertex : sidesForVertex)
    {
      vector<IndexType> verticesForSide = meshTopo->getEntityVertexIndices(sideDim, sideForVertex);
      bool nonMatchFound = false;
      for (IndexType vertexForSide : verticesForSide)
      {
        if (std::find(matchingVertexIndices.begin(), matchingVertexIndices.end(), vertexForSide) == matchingVertexIndices.end())
        {
          nonMatchFound = true;
          break;
        }
      }
      if (!nonMatchFound) sidesThatMatch.insert(sideForVertex);
    }
  }
  return sidesThatMatch;
}

const std::vector<double>& MeshTopologyView::getVertex(IndexType vertexIndex) const
{
  return _meshTopo->getVertex(vertexIndex);
}

bool MeshTopologyView::getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol) const
{
  return _meshTopo->getVertexIndex(vertex, vertexIndex, tol);
}

vector<IndexType> MeshTopologyView::getVertexIndicesMatching(const vector<double> &vertexInitialCoordinates, double tol) const
{
  return _meshTopo->getVertexIndicesMatching(vertexInitialCoordinates, tol);
}

bool MeshTopologyView::isDistributed() const
{
  return (_meshTopo->Comm() != Teuchos::null) && (_meshTopo->Comm()->getSize() > 1);
}

bool MeshTopologyView::isParent(IndexType cellIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "cellIndex is invalid!");
  return _activeCells.find(cellIndex) == _activeCells.end();
}

bool MeshTopologyView::isValidCellIndex(IndexType cellIndex) const
{
  return _allKnownCells.find(cellIndex) != _allKnownCells.end();
}

Camellia::FieldContainer<double> MeshTopologyView::physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension) const
{
  return _meshTopo->physicalCellNodesForCell(cellIndex,includeCellDimension);
}

void MeshTopologyView::printActiveCellAncestors() const
{
  for (IndexType cellID : getLocallyKnownActiveCellIndices())
  {
    printCellAncestors(cellID);
  }
}

void MeshTopologyView::printAllEntitiesInBaseMeshTopology() const
{
  if (_meshTopo != Teuchos::null)
  {
    _meshTopo->printAllEntities();
  }
  else
  {
    // this must be a MeshTopology object
    const MeshTopology* meshTopo = dynamic_cast<const MeshTopology*>(this);
    meshTopo->printAllEntities();
  }
}

void MeshTopologyView::printCellAncestors(IndexType cellID) const
{
  vector<IndexType> cellAncestors;
  CellPtr cell = getCell(cellID);
  while (cell->getParent() != Teuchos::null) {
    cell = cell->getParent();
    cellAncestors.push_back(cell->cellIndex());
  }
  ostringstream cellLabel;
  cellLabel << cellID;
  print(cellLabel.str(),cellAncestors);
}

double MeshTopologyView::totalTimeComputingCellHalos() const
{
  return TimeLogger::sharedInstance()->totalTime("cellHalo");
}

Teuchos::RCP<MeshTransformationFunction> MeshTopologyView::transformationFunction() const
{
  return Teuchos::null; // pure MeshTopologyViews are defined to have straight-edge geometry only, for now.  (Unclear whether this is actually the best thing...)
}

void MeshTopologyView::setGlobalDofAssignment(GlobalDofAssignment* gda)
{ // for cubature degree lookups
  _gda = gda;
}

void MeshTopologyView::verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const
{
  _meshTopo->verticesForCell(vertices, cellID);
}

MeshTopologyViewPtr MeshTopologyView::getView(const set<IndexType> &activeCells) const
{
  return Teuchos::rcp( new MeshTopologyView(_meshTopo, activeCells) );
}

MeshTopologyViewPtr MeshTopologyView::readFromFile(Epetra_CommPtr Comm, string filename)
{
#ifdef HAVE_EPETRAEXT_HDF5
  EpetraExt::HDF5 hdf5(*Comm);
  hdf5.Open(filename);
  MeshTopologyViewPtr meshTopoView = readFromHDF5(Comm, hdf5);
  hdf5.Close();
  return meshTopoView;
#else
  // no file reading if we don't have HDF5 support…
  return Teuchos::null;
#endif
}

void MeshTopologyView::writeToFile(string filename) const
{
#ifdef HAVE_EPETRAEXT_HDF5
  Epetra_CommPtr comm = MPIWrapper::EpetraComm(Comm());
  
  EpetraExt::HDF5 hdf5(*comm);
  hdf5.Create(filename);
  
  writeToHDF5(comm, hdf5);
  hdf5.Close();
#endif
}

#ifdef HAVE_EPETRAEXT_HDF5
MeshTopologyViewPtr MeshTopologyView::readFromHDF5(Epetra_CommPtr Comm, EpetraExt::HDF5 &hdf5)
{
  int numChunks;
  hdf5.Read("MeshTopology", "num chunks", numChunks);
  vector<int> chunkSizes(numChunks);
  hdf5.Read("MeshTopology", "chunk sizes", H5T_NATIVE_INT, numChunks, &chunkSizes[0]);
  
  int myRank = Comm->MyPID();
  int numProcs = Comm->NumProc();
  vector<int> myChunkRanks; // ranks that were part of the write, now assigned to me
  if (numProcs < numChunks)
  {
    int chunksPerRank = numChunks / numProcs;
    int extraChunks = numChunks % numProcs;
    int myChunkCount;
    int myChunkStart;
    if (myRank < extraChunks)
    {
      myChunkCount = chunksPerRank + 1;
      myChunkStart = (chunksPerRank + 1) * myRank;
    }
    else
    {
      myChunkCount = chunksPerRank;
      myChunkStart = extraChunks + chunksPerRank * myRank;
    }
    for (int i=0; i<myChunkCount; i++)
    {
      myChunkRanks.push_back(myChunkStart + i);
    }
  }
  else
  {
    if (myRank < numChunks)
    {
      myChunkRanks.push_back(myRank);
    }
  }
  int myGeometrySize = 0;
  for (int myChunkRank : myChunkRanks)
  {
    myGeometrySize += chunkSizes[myChunkRank];
  }
  int globalGeometrySize;
  Comm->SumAll(&myGeometrySize, &globalGeometrySize, 1);
  vector<char> myGeometryData(myGeometrySize);
  void* myGeometryLocation = (myGeometrySize > 0) ? &myGeometryData[0] : NULL;
  hdf5.Read("MeshTopology", "geometry chunks", myGeometrySize, globalGeometrySize, H5T_NATIVE_CHAR, myGeometryLocation);
  MeshGeometryInfo geometryInfo;
  
  const char* geometryDataLocation = (myGeometrySize > 0) ? &myGeometryData[0] : NULL;
  CellDataMigration::readGeometryData(geometryDataLocation, myGeometrySize, geometryInfo);
  
  Teuchos_CommPtr teuchosComm = MPIWrapper::TeuchosComm(Comm);
  
  MeshTopologyPtr baseMeshTopo = Teuchos::rcp( new MeshTopology(teuchosComm, geometryInfo) );
  
  // if the base mesh topology has a pure view, record that, too
  bool isDistributedView = hdf5.IsContained("MeshTopologyViewDistributed");
  bool isSerialView = hdf5.IsContained("MeshTopologyView");
  bool isView = (isDistributedView || isSerialView);
  MeshTopologyViewPtr meshTopoView;
  if (!isView)
  {
    meshTopoView = baseMeshTopo;
  }
  else
  {
    set<IndexType> viewCells;
    if (isDistributedView)
    {
      int numWritingProcs;
      hdf5.Read("MeshTopologyViewDistributed", "numProcs", numWritingProcs);
      vector<int> knownCellCounts(numWritingProcs);
      void* knownCellCountsLocation = (numWritingProcs > 0) ? &knownCellCounts[0] : NULL;
      hdf5.Write("MeshTopologyViewDistributed", "known cell counts", H5T_NATIVE_INT, numWritingProcs, knownCellCountsLocation);
      TEUCHOS_TEST_FOR_EXCEPTION(numWritingProcs != numChunks, std::invalid_argument, "numWritingProcs != numChunks");
      
      // we use the same assignments as above
      int myKnownCellCount = 0;
      for (int myChunkRank : myChunkRanks)
      {
        myKnownCellCount += knownCellCounts[myChunkRank];
      }
      int globalKnownCellCount;
      Comm()->SumAll(&myKnownCellCount, &globalKnownCellCount, 1);
      
      vector<int> myKnownCellsVector(myKnownCellCount); // may contain duplicates
      void* myKnownCellsLocation = (myKnownCellCount > 0) ? &myKnownCellsVector[0] : NULL;
      hdf5.Read("MeshTopologyViewDistributed", "known cell IDs", myKnownCellCount, globalKnownCellCount,
                H5T_NATIVE_INT, myKnownCellsLocation);
      viewCells.insert(myKnownCellsVector.begin(),myKnownCellsVector.end());
    }
    else
    {
      int knownCellCount;
      hdf5.Read("MeshTopologyView", "known cell count", knownCellCount);
      vector<int> cellIDVector(knownCellCount);
      void* cellIDLocation = (cellIDVector.size() > 0) ? &cellIDVector[0] : NULL;
      hdf5.Write("MeshTopologyView", "known cell IDs", H5T_NATIVE_INT, knownCellCount, cellIDLocation);
      viewCells.insert(cellIDVector.begin(),cellIDVector.end());
    }
    meshTopoView = baseMeshTopo->getView(viewCells);
  }
  return meshTopoView;
}

void MeshTopologyView::writeToHDF5(Epetra_CommPtr Comm, EpetraExt::HDF5 &hdf5) const
{
  // get my view of the base mesh topology; write it to a byte array
  const MeshTopology* baseMeshTopology = this->baseMeshTopology();
  MeshGeometryInfo baseMeshGeometry;
  CellDataMigration::getGeometry(baseMeshTopology, baseMeshGeometry);
  int myGeometrySize = CellDataMigration::getGeometryDataSize(baseMeshGeometry);
  vector<char> myGeometryData(myGeometrySize);
  char *myWriteLocation = &myGeometryData[0];
  CellDataMigration::writeGeometryData(baseMeshGeometry, myWriteLocation, myGeometrySize);
  
  // gather all the rank sizes so that we can write a global "header" indicating where the boundaries lie
  // among other things, this will allow safe reading when the number of reading processors is
  // different from the number of writing processors
  int numProcs = Comm->NumProc();
  vector<int> chunkSizes(numProcs);
  Comm->GatherAll(&myGeometrySize, &chunkSizes[0], 1);
  
  int globalGeometrySize;
  Comm->SumAll(&myGeometrySize, &globalGeometrySize, 1);
  
  hdf5.Write("MeshTopology", "num chunks", numProcs);
  hdf5.Write("MeshTopology", "chunk sizes", H5T_NATIVE_INT, numProcs, &chunkSizes[0]);
  hdf5.Write("MeshTopology", "geometry chunks", myGeometrySize, globalGeometrySize, H5T_NATIVE_CHAR, &myGeometryData[0]);
  
  // if the base mesh topology has a pure view, record that, too
  bool isView = (dynamic_cast<const MeshTopology*>(this) == NULL);
  if (isView)
  {
    if (isDistributed())
    {
      hdf5.Write("MeshTopologyViewDistributed", "numProcs", numProcs);
      const set<IndexType>* myViewCellIDs = &getLocallyKnownActiveCellIndices();
      vector<int> cellIDVector(myViewCellIDs->begin(),myViewCellIDs->end());
      int myKnownCellCount = myViewCellIDs->size();
      vector<int> knownCellCounts(numProcs);
      Comm->GatherAll(&myKnownCellCount, &knownCellCounts[0], 1);
      hdf5.Write("MeshTopologyViewDistributed", "known cell counts", H5T_NATIVE_INT, numProcs, &knownCellCounts[0]);
      
      int globalKnownActiveCellCount;
      Comm->SumAll(&myKnownCellCount, &globalKnownActiveCellCount, 1);
      hdf5.Write("MeshTopologyViewDistributed", "known cell IDs", myKnownCellCount, globalKnownActiveCellCount,
                 H5T_NATIVE_INT, &cellIDVector[0]);
    }
    else
    {
      // if topology is *not* distributed, everyone sees the same cells, and we can just write those out
      // (reader can detect the difference based on whether there is an entry named "known cell sizes")
      const set<IndexType>* viewCellIDs = &getLocallyKnownActiveCellIndices();
      vector<int> cellIDVector(viewCellIDs->begin(),viewCellIDs->end());
      int knownCellCount = viewCellIDs->size();
      
      hdf5.Write("MeshTopologyView", "known cell count", knownCellCount);
      hdf5.Write("MeshTopologyView", "known cell IDs", H5T_NATIVE_INT, knownCellCount, &cellIDVector[0]);
    }
  }
}
#endif // ifdef HAVE_EPETRAEXT_HDF5
