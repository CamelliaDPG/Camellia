//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//

#include "Cell.h"
#include "RefinementPattern.h"
#include "CamelliaCellTools.h"

#include "GnuPlotUtil.h"

using namespace Camellia;
using namespace std;

vector< pair< unsigned, unsigned > > Cell::boundarySubcells()
{
  if (! _haveComputedBoundarySubcells)
  {
    // strategy is:
    // for each subcell sc
    //   - determine descendants (including sc itself)
    //   - for each descendant, get all sides in the MeshTopo that contain that descendant
    //   - check whether any of these sides are on the mesh boundary
    //   - if all descendants belong to some side on the mesh boundary, then the sc belongs to mesh boundary
    //   - if it does, add it to _boundarySubcells
        
    int spaceDim = _cellTopo->getDimension();
    int sideDim = spaceDim - 1;
    
    set< pair<unsigned, unsigned> > boundarySubcellSet; // use set for uniquing/lookup
    
    for (int d=sideDim; d>=0; d--)
    {
      int scCount = _cellTopo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++)
      {
        if (boundarySubcellSet.find({d,scord}) != boundarySubcellSet.end())
        {
          // we've already identified that this subcell is on the boundary; continue
          continue;
        }
        IndexType scEntityIndex = entityIndex(d, scord);
        
        vector<IndexType> entitiesToCheck = _meshTopo->getChildEntities(d,scEntityIndex);
        entitiesToCheck.push_back(scEntityIndex);
        
        bool eachEntityHasSideOnBoundary = true;
        for (auto entityIndex : entitiesToCheck)
        {
          vector<IndexType> sideIndices = _meshTopo->getSidesContainingEntity(d, entityIndex);
          bool foundSideOnBoundary = false;
          for (IndexType sideIndex : sideIndices)
          {
            if (_meshTopo->isBoundarySide(sideIndex))
            {
              foundSideOnBoundary = true;
              break;
            }
          }
          if (!foundSideOnBoundary)
          {
            eachEntityHasSideOnBoundary = false;
          }
        }
        if (eachEntityHasSideOnBoundary)
        {
          // this subcell and all its subcells lie on the boundary
          boundarySubcellSet.insert({d,scord});
          CellTopoPtr subcell = _cellTopo->getSubcell(d, scord);
          for (unsigned subsubcellDim=0; subsubcellDim < d; subsubcellDim++ )
          {
            int subsubcellCount = subcell->getSubcellCount(subsubcellDim);
            for (unsigned subsubcellOrd=0; subsubcellOrd<subsubcellCount; subsubcellOrd++)
            {
              unsigned subsubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(_cellTopo, d, scord, subsubcellDim, subsubcellOrd);
              boundarySubcellSet.insert({subsubcellDim,subsubcellOrdinalInCell});
            }
          }
        }
      }
    }
    _boundarySubcells = vector< pair<unsigned,unsigned> > (boundarySubcellSet.begin(), boundarySubcellSet.end());
    
    _haveComputedBoundarySubcells = true;
  }
  return _boundarySubcells;
}

vector< pair<GlobalIndexType, unsigned> > Cell::childrenForSide(unsigned sideIndex)
{
  vector< pair<GlobalIndexType, unsigned> > childIndicesForSide;

  if (_refPattern.get() != NULL)
  {
    vector< pair<unsigned, unsigned> > refinementChildIndicesForSide = _refPattern->childrenForSides()[sideIndex];

    for( vector< pair<unsigned, unsigned> >::iterator entryIt = refinementChildIndicesForSide.begin();
         entryIt != refinementChildIndicesForSide.end(); entryIt++)
    {
      unsigned childIndex = _children[entryIt->first]->cellIndex();
      unsigned childSide = entryIt->second;
      childIndicesForSide.push_back(make_pair(childIndex,childSide));
    }
  }

  return childIndicesForSide;
}

set<GlobalIndexType> Cell::getActiveNeighborIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  int sideCount = _cellTopo->getSideCount();
  set<GlobalIndexType> neighborIndices;
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    pair<GlobalIndexType, unsigned> neighborInfo = getNeighborInfo(sideOrdinal,meshTopoViewForCellValidity);
    GlobalIndexType neighborID  = neighborInfo.first;
    if (neighborID == -1) continue; // no neighbor on this side
    unsigned mySideOrdinalInNeighbor = neighborInfo.second;
    CellPtr neighbor = meshTopoViewForCellValidity->getCell(neighborID);
    vector< pair< GlobalIndexType, unsigned> > neighborDescendants = neighbor->getDescendantsForSide(mySideOrdinalInNeighbor, meshTopoViewForCellValidity); // descendants will be the active ones
    for (auto entry : neighborDescendants)
    {
      neighborIndices.insert(entry.first);
    }
  }
  return neighborIndices;
}

vector<IndexType> Cell::entitiesOnNeighborInterfaces(unsigned dimensionOfInterest,
                                                     bool peersOnly,
                                                     ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // strategy:
  /*
      Find the entities of dimension dimensionOfInterest that belong to our sides and, if
      peersOnly is false, their descendants.
   */
  int spaceDim = _cellTopo->getDimension();
  int sideDim = spaceDim - 1;
  
  vector<IndexType> sidesOfInterest;
  int sideCount = _cellTopo->getSideCount();
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    if ( !isBoundary(sideOrdinal) )
    {
      IndexType sideEntityIndex = entityIndex(sideDim, sideOrdinal);
      sidesOfInterest.push_back(sideEntityIndex);
    }
  }
  
  if (!peersOnly)
  {
    vector<IndexType> sidesToProcess = sidesOfInterest;
    while (sidesToProcess.size() > 0)
    {
      vector<IndexType> nextSidesToProcess;
      for (IndexType sideEntityIndex : sidesToProcess)
      {
        if (_meshTopo->entityHasChildren(sideDim, sideEntityIndex))
        {
          vector<IndexType> childSides = _meshTopo->getChildEntities(sideDim, sideEntityIndex);
          for (IndexType childEntityIndex : childSides)
          {
            if (childEntityIndex != -1)
            {
              nextSidesToProcess.push_back(childEntityIndex);
            }
          }
        }
      }
      sidesOfInterest.insert(sidesOfInterest.begin(),nextSidesToProcess.begin(),nextSidesToProcess.end());
      sidesToProcess = nextSidesToProcess;
    }
  }
  
  set<IndexType> entitiesToMatchSet; // of dimension dimensionOfInterest
  for (IndexType sideEntityIndex : sidesOfInterest)
  {
    vector<IndexType> subEntityIndices;
    _meshTopo->getSubEntityIndices(sideDim, sideEntityIndex, dimensionOfInterest, subEntityIndices);
    entitiesToMatchSet.insert(subEntityIndices.begin(),subEntityIndices.end());
  }
  
  return vector<IndexType>(entitiesToMatchSet.begin(),entitiesToMatchSet.end());
}

set<GlobalIndexType> Cell::getActiveNeighborIndices(unsigned dimensionForNeighborRelation, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  int sideDim = _cellTopo->getDimension() - 1;
  if (sideDim == dimensionForNeighborRelation)
  {
    return getActiveNeighborIndices(meshTopoViewForCellValidity);
  }
  else
  {
    bool peersOnly = false;
    vector<IndexType> entitiesToMatch = entitiesOnNeighborInterfaces(dimensionForNeighborRelation, peersOnly, meshTopoViewForCellValidity);
    set< pair<IndexType, unsigned> > cellPairs = meshTopoViewForCellValidity->getCellsContainingEntities(dimensionForNeighborRelation,
                                                                                                         entitiesToMatch);
    
    set<GlobalIndexType> neighborIndices;
    for (pair<IndexType, unsigned> cellPair : cellPairs)
    {
      IndexType cellID = cellPair.first;
      CellPtr neighborCell = meshTopoViewForCellValidity->getCell(cellID);
      if (! meshTopoViewForCellValidity->isParent(cellID)) // if it is a parent, then it isn't active...
      {
        neighborIndices.insert(cellID);
      }
    }

    // don't include this cell as a neighbor:
    neighborIndices.erase(_cellIndex);
    return neighborIndices;
  }
}

set<GlobalIndexType> Cell::getPeerNeighborIndices(unsigned dimensionForNeighborRelation, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  bool peersOnly = true;
  vector<IndexType> entitiesToMatch = entitiesOnNeighborInterfaces(dimensionForNeighborRelation, peersOnly, meshTopoViewForCellValidity);
  vector<IndexType> sidesForEntities = meshTopoViewForCellValidity->getSidesContainingEntities(dimensionForNeighborRelation,entitiesToMatch);
  set< pair<IndexType, unsigned> > cellPairs = meshTopoViewForCellValidity->getCellsContainingSides(sidesForEntities);
  
  int sideDim = _cellTopo->getDimension() - 1;
  // We find all the most-refined cells containing those entities.
  set<GlobalIndexType> neighborIndices;
  for (pair<IndexType, unsigned> cellPair : cellPairs)
  {
    IndexType neighborCellID = cellPair.first;
    CellPtr neighborCell = meshTopoViewForCellValidity->getCell(neighborCellID);
    if (neighborCell->level() == _level)
    {
      neighborIndices.insert(neighborCell->cellIndex());
    }
    else if (neighborCell->level() >= _level) // if not, there's no peer with this entity
    {
      int levelDiff = neighborCell->level() - _level;
      for (int i=0; i<levelDiff; i++)
      {
        neighborCell = neighborCell->getParent();
      }
      // check to make sure the peer is actually a neighbor (contains one of the sides that match our entities)
      int neighborSideCount = neighborCell->getSideCount();
      for (int neighborSideOrdinal=0; neighborSideOrdinal < neighborSideCount; neighborSideOrdinal++)
      {
        IndexType sideEntityIndex = neighborCell->entityIndex(sideDim, neighborSideOrdinal);
        if (std::find(sidesForEntities.begin(), sidesForEntities.end(), sideEntityIndex) != sidesForEntities.end())
        {
          neighborIndices.insert(neighborCell->cellIndex());
          break;
        }
      }
    }
  }
  
  // don't include this cell among its neighbors...
  if (neighborIndices.find(_cellIndex) != neighborIndices.end())
  {
    neighborIndices.erase(_cellIndex);
  }
  return neighborIndices;
}

set<IndexType> Cell::getDescendants(ConstMeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopoViewForCellValidity->isValidCellIndex(_cellIndex), std::invalid_argument, "_cellIndex is not valid");
  
  set<IndexType> descendants;
  if (!meshTopoViewForCellValidity->isParent(_cellIndex))
  {
    // no descendants save the present cell, which is a leaf node in any case
    descendants.insert(_cellIndex);
  }
  else
  {
    vector< CellPtr > parentCells;
    CellPtr thisPtr = Teuchos::rcp(this, false);
    parentCells.push_back(thisPtr);

    while (parentCells.size() > 0)
    {
      CellPtr parentCell = parentCells[parentCells.size()-1];
      if (!leafNodesOnly)
      {
        // then include this parent cell in the list
        descendants.insert(parentCell->cellIndex());
      }
      parentCells.pop_back(); // delete last element
      vector< CellPtr > children = parentCell->children();
      for (int childOrdinal=0; childOrdinal < children.size(); childOrdinal++)
      {
        CellPtr child = children[childOrdinal];

        if (meshTopoViewForCellValidity->isParent(child->cellIndex()))
        {
          parentCells.push_back(child);
        }
        else
        {
          descendants.insert(child->cellIndex());
        }
      }
    }
  }
  return descendants;
}

vector< pair< GlobalIndexType, unsigned> > Cell::getDescendantsForSide(int sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly)
{
  // if leafNodesOnly == true,  returns a flat list of leaf nodes (descendants that are not themselves parents)
  // if leafNodesOnly == false, returns a list in descending order: immediate children, then their children, and so on.

  // guarantee is that if a child and its parent are both in the list, the parent will come first
  
  // for distributed MeshTopology, list is only guaranteed to be complete for cells that are owned by the MeshTopology.

  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopoViewForCellValidity->isValidCellIndex(_cellIndex), std::invalid_argument, "_cellIndex is not valid");
  TEUCHOS_TEST_FOR_EXCEPTION(sideOrdinal >= _cellTopo->getSideCount(), std::invalid_argument, "sideOrdinal is out of bounds!");
  
  // pair (descendantCellIndex, descendantSideIndex)
  vector< pair< GlobalIndexType, unsigned> > descendantsForSide;
  if ( ! meshTopoViewForCellValidity->isParent(_cellIndex) )
  {
    descendantsForSide.push_back( {_cellIndex, sideOrdinal} );
    return descendantsForSide;
  }

  vector< pair<unsigned,unsigned> > childIndices = _refPattern->childrenForSides()[sideOrdinal];

  for (pair<unsigned,unsigned> entry : childIndices)
  {
    unsigned childOrdinal = entry.first;
    unsigned childSideOrdinal = entry.second;
    TEUCHOS_TEST_FOR_EXCEPTION(childOrdinal >= _childIndices.size(), std::invalid_argument, "Internal error: childOrdinal is out of bounds!");
    IndexType childCellIndex = _childIndices[childOrdinal];
    if ( (meshTopoViewForCellValidity->isValidCellIndex(childCellIndex) && !meshTopoViewForCellValidity->isParent(childCellIndex)) || (! leafNodesOnly ) )
    {
      // (            leaf node              ) || ...
      descendantsForSide.push_back( {_childIndices[childOrdinal], childSideOrdinal} );
    }
    if ( (meshTopoViewForCellValidity->isValidCellIndex(_childIndices[childOrdinal])) && _children[childOrdinal]->isParent(meshTopoViewForCellValidity) )
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_children[childOrdinal] == Teuchos::null, std::invalid_argument, "Internal error: although MeshTopologyView claims child has a valid cell index, its parent doesn't have a Cell object for it.");
      vector< pair<GlobalIndexType,unsigned> > childDescendants = _children[childOrdinal]->getDescendantsForSide(childSideOrdinal,meshTopoViewForCellValidity,leafNodesOnly);
      for (pair<GlobalIndexType,unsigned> childEntry : childDescendants)
      {
        descendantsForSide.push_back(childEntry);
      }
    }
  }
  return descendantsForSide;
}

Cell::Cell(CellTopoPtr cellTopo, const vector<IndexType> &vertices, const vector< vector< unsigned > > &subcellPermutations,
           GlobalIndexType cellIndex, MeshTopology* meshTopo)
{
  _cellTopo = cellTopo;
  _vertices = vertices;
  _subcellPermutations = subcellPermutations;
  _cellIndex = cellIndex;
  _meshTopo = meshTopo;
  int sideCount = cellTopo->getSideCount();
  _level = 0;
  _neighbors = vector< pair<GlobalIndexType, IndexType> >(sideCount,{-1,-1});
  _haveComputedBoundarySubcells = false;
}

map<string, long long> Cell::approximateMemoryCosts()
{
  map<string, long long> variableCosts;

  // calibrate by computing some sizes
//  map<int, int> emptyMap;
  vector<int> emptyVector;

//  int MAP_OVERHEAD = sizeof(emptyMap);
  int VECTOR_OVERHEAD = sizeof(emptyVector);

//  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal

  variableCosts["_cellIndex"] = sizeof(_cellIndex);
  variableCosts["_cellTopo"] = sizeof(_cellTopo);
  variableCosts["_vertices"] = VECTOR_OVERHEAD + sizeof(unsigned) * _vertices.capacity();

  variableCosts["_subcellPermutations"] = VECTOR_OVERHEAD;
  for (vector< vector< unsigned > >::iterator entryIt = _subcellPermutations.begin(); entryIt != _subcellPermutations.end(); entryIt++)
  {
    variableCosts["_subcellPermutations"] += VECTOR_OVERHEAD + entryIt->capacity() * sizeof(unsigned);
  }
  variableCosts["_subcellPermutations"] += VECTOR_OVERHEAD * (_subcellPermutations.capacity() - _subcellPermutations.size());

  variableCosts["_meshTopo"] += sizeof(MeshTopology*);

  variableCosts["_children"] = VECTOR_OVERHEAD + _children.size() * sizeof(CellPtr);

  variableCosts["_refPattern"] = sizeof(RefinementPatternPtr);

  variableCosts["_parent"] = sizeof(_parent);

  variableCosts["_neighbors"] = VECTOR_OVERHEAD + _neighbors.size() * sizeof(pair<GlobalIndexType, unsigned>);
  
  variableCosts["_boundarySubcells"] = VECTOR_OVERHEAD + _boundarySubcells.size() * sizeof(pair<unsigned,unsigned>);

  variableCosts["_haveComputedBoundarySubcells"] = sizeof(_haveComputedBoundarySubcells);
  
  return variableCosts;
}

long long Cell::approximateMemoryFootprint()
{
  long long memSize = 0; // in bytes

  map<string, long long> variableCost = approximateMemoryCosts();
  for (map<string, long long>::iterator entryIt = variableCost.begin(); entryIt != variableCost.end(); entryIt++)
  {
    memSize += entryIt->second;
  }
  return memSize;
}

GlobalIndexType Cell::cellIndex()
{
  return _cellIndex;
}

unsigned Cell::childOrdinal(IndexType childIndex)
{
  for (unsigned childOrdinal=0; childOrdinal<_childIndices.size(); childOrdinal++)
  {
    if (_childIndices[childOrdinal] == childIndex)
    {
      return childOrdinal;
    }
  }
  cout << "ERROR: child with ID childIndex not found in parent.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child with ID childIndex not found in parent");
//  return -1; // NOT FOUND
}

const vector< CellPtr > &Cell::children()
{
  return _children;
}

void Cell::setChildren(const vector<GlobalIndexType> &childIndices)
{
  _childIndices = childIndices;
  Teuchos::RCP< Cell > thisPtr = Teuchos::rcp( this, false ); // doesn't own memory
  _children.resize(childIndices.size(),Teuchos::null);
  for (int childOrdinal=0; childOrdinal < childIndices.size(); childOrdinal++)
  {
    GlobalIndexType childCellIndex = childIndices[childOrdinal];
    
    if (_meshTopo->isValidCellIndex(childCellIndex))
    {
      CellPtr childCell = _meshTopo->getCell(childCellIndex);
      childCell->setParent(thisPtr);
      _children[childOrdinal] = childCell;
    }
  }
}

vector<IndexType> Cell::getChildIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if the mesh topo doesn't think we're a parent, then no children
  if (! isParent(meshTopoViewForCellValidity))
    return vector<IndexType>();
  
  return _childIndices;
}

IndexType Cell::entityIndex(unsigned subcdim, unsigned subcord)
{
  int spaceDim = _cellTopo->getDimension();
  if ((subcdim == spaceDim) && (subcord == 0))
  {
    return _cellIndex;
  }
  
  if (_entityIndices.size() == 0)
  {
    _entityIndices = vector< vector<IndexType> >(spaceDim);
  }

  if (_entityIndices[subcdim].size() == 0)
  {
    _entityIndices[subcdim] = vector<IndexType>(_cellTopo->getSubcellCount(subcdim), -2); // -2 indicates not yet looked for; -1 indicates looked for and not found.
  }
  
  if (_entityIndices[subcdim][subcord] != -2)
  {
    return _entityIndices[subcdim][subcord];
  }
  
  set< IndexType > nodes;
  if (subcdim != 0)
  {
    int entityNodeCount = _cellTopo->getNodeCount(subcdim, subcord);
    for (int node=0; node<entityNodeCount; node++)
    {
      unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, subcord, node);
      nodes.insert(_vertices[nodeIndexInCell]);
    }
  }
  else
  {
    nodes.insert(_vertices[subcord]);
  }
  _entityIndices[subcdim][subcord] = _meshTopo->getEntityIndex(subcdim, nodes);
  
  return _entityIndices[subcdim][subcord];
}

vector<IndexType> Cell::getEntityVertexIndices(unsigned int subcdim, unsigned int subcord)
{
  vector< IndexType > nodes;
  if (subcdim != 0)
  {
    int entityNodeCount = _cellTopo->getNodeCount(subcdim, subcord);
    for (int node=0; node<entityNodeCount; node++)
    {
      unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, subcord, node);
      nodes.push_back(_vertices[nodeIndexInCell]);
    }
  }
  else
  {
    nodes.push_back(_vertices[subcord]);
  }
  return nodes;
}

vector<IndexType> Cell::getEntityIndices(unsigned subcdim)
{
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  vector<IndexType> cellEntityIndices(entityCount);
  for (int j=0; j<entityCount; j++)
  {
    cellEntityIndices[j] = this->entityIndex(subcdim, j);
  }
  return cellEntityIndices;
}

int Cell::findChildOrdinal(IndexType cellIndex)
{
  for (int childOrdinal=0; childOrdinal < _childIndices.size(); childOrdinal++)
  {
    if (_childIndices[childOrdinal] == cellIndex) return childOrdinal;
  }
  return -1;
}

unsigned Cell::findSubcellOrdinal(unsigned subcdim, IndexType subcEntityIndex)
{
  // this is pretty brute force right now
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  for (int scord=0; scord<entityCount; scord++)
  {
    IndexType scEntityIndex = entityIndex(subcdim, scord);
    if (scEntityIndex == subcEntityIndex)
    {
      return scord;
    }
  }
  return INVALID_CELLTOPOLOGY_SIDE_ORDINAL; // NOT FOUND
}

unsigned Cell::findSubcellOrdinalInSide(unsigned int subcdim, IndexType subcEntityIndex, unsigned sideOrdinal)
{
  unsigned subcOrdinalInCell = findSubcellOrdinal(subcdim, subcEntityIndex);
  int sideDim = _cellTopo->getDimension() - 1;
  if (subcOrdinalInCell == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL) return INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL;
  bool assertContainment = false; // don't throw exception if not found (return -1)
  return CamelliaCellTools::subcellReverseOrdinalMap(topology(), sideDim, sideOrdinal, subcdim, subcOrdinalInCell, assertContainment);
}

Teuchos::RCP<Cell> Cell::getParent()
{
  return _parent;
}

int Cell::level() const
{
  return _level;
}

void Cell::setParent(Teuchos::RCP<Cell> parent)
{
  _parent = Teuchos::rcp(parent.get(),false); // make weak reference
  if (_parent != Teuchos::null)
  {
    _level = _parent->level() + 1;
  }
}

bool Cell::isDescendant(IndexType ancestralCellIndex) // true if the provided cell index is one of this cell's ancestors
{
  if (ancestralCellIndex == _cellIndex) return true;
  CellPtr ancestor = this->getParent();
  while (ancestor != Teuchos::null)
  {
    if (ancestor->cellIndex() == ancestralCellIndex)
    {
      return true;
    }
    ancestor = ancestor->getParent();
  }
  return false;
}

bool Cell::isParent(ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  return meshTopoViewForCellValidity->isParent(_cellIndex);
//  if (_childIndices.size() == 0) return false;
//  
//  /*
//   With distributed MeshTopology, isValidCellIndex() returns false for 
//   Cells that are not locally known.  It can happen that a parent is
//   known but not all of its children are.  But this should only happen
//   if there is *some* child that is locally known.  Therefore, we need
//   to check all children.  If some child is valid, then this is a parent.
//   */
//  
//  for (GlobalIndexType childCellIndex : _childIndices)
//  {
//    if (meshTopoViewForCellValidity->isValidCellIndex(childCellIndex))
//      return true;
//  }
//  // no valid child found: return false
//  return false;
}

RefinementBranch Cell::refinementBranch(IndexType ancestralCellID)
{
  RefinementBranch refBranch;
  if ((_parent == Teuchos::null) || (ancestralCellID == _cellIndex))
  {
    refBranch.push_back({RefinementPattern::noRefinementPattern(_cellTopo).get(),0});
  }
  else
  {
    CellPtr parent = _parent;
    GlobalIndexType cellIndex = _cellIndex;
    while ((parent != Teuchos::null) && (cellIndex != ancestralCellID)) {
      RefinementPatternPtr refPattern = parent->refinementPattern();
      unsigned childOrdinal = parent->findChildOrdinal(cellIndex);
      refBranch.insert(refBranch.begin(), {refPattern.get(),childOrdinal});
      
      cellIndex = parent->cellIndex();
      parent = parent->getParent();
    }
    if (ancestralCellID != -1)
    {
      // then check that we did find the specified cell
      TEUCHOS_TEST_FOR_EXCEPTION(ancestralCellID != cellIndex, std::invalid_argument, "specified ancestralCellID not found among cell's ancestors");
    }
  }
  return refBranch;
}

RefinementBranch Cell::refinementBranchForSide(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if this cell (on this side) is the finer side of a hanging node, returns the RefinementBranch starting
  // with the coarse neighbor's neighbor (this cell's ancestor).  (Otherwise, the RefinementBranch will be empty.)
  RefinementBranch refBranch;
  pair<GlobalIndexType, unsigned> neighborInfo = this->getNeighborInfo(sideOrdinal, meshTopoViewForCellValidity);
  GlobalIndexType neighborCellIndex = neighborInfo.first;
  unsigned sideIndexInNeighbor = neighborInfo.second;
  if (neighborCellIndex == -1)
  {
    return refBranch; // no refinements
  }
  CellPtr neighbor = _meshTopo->getCell(neighborCellIndex);
  if (neighbor->getNeighborInfo(sideIndexInNeighbor, meshTopoViewForCellValidity).first == this->_cellIndex)   // peers!
  {
    return refBranch; // no refinements
  }
  else
  {
    GlobalIndexType ancestorCellIndex = neighbor->getNeighborInfo(sideIndexInNeighbor, meshTopoViewForCellValidity).first;
    vector< CellPtr > ancestors;
    vector< unsigned > childOrdinals;
    CellPtr currentAncestor = _meshTopo->getCell(_cellIndex);
    while (currentAncestor->cellIndex() != ancestorCellIndex)
    {
      GlobalIndexType childCellIndex = currentAncestor->cellIndex();
      currentAncestor = currentAncestor->getParent();
      ancestors.push_back(currentAncestor);
      vector< CellPtr > children = currentAncestor->children();
      for (int i=0; i<children.size(); i++)
      {
        if (children[i]->cellIndex() == childCellIndex)
        {
          childOrdinals.push_back(i);
          break;
        }
      }
      if (childOrdinals.size() != ancestors.size())
      {
        cout << "ERROR: child not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: child not found.");
      }
    }
    // now the ancestors and childOrdinals containers have the RefinementBranch info, in reverse order
    unsigned ancestorCount = ancestors.size();
    for (int i=ancestorCount-1; i >= 0; i--)
    {
      refBranch.push_back(make_pair(ancestors[i]->refinementPattern().get(), childOrdinals[i]));
    }
  }
  return refBranch;
}

RefinementBranch Cell::refinementBranchForSubcell(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if the given subcell is constrained by another cell, this method will return a RefinementBranch which has as its root
  // this cell's ancestor that is compatible with the constraining cell, and as its leaf this cell.
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);

  pair<IndexType, unsigned> constrainingEntity = meshTopoViewForCellValidity->getConstrainingEntity(subcdim, subcellEntityIndex);
  IndexType constrainingEntityIndex = constrainingEntity.first;
  unsigned constrainingEntityDim = constrainingEntity.second;

  CellPtr currentAncestor = meshTopoViewForCellValidity->getCell(_cellIndex);
  vector< CellPtr > ancestors;
  vector< unsigned > childOrdinals;

  unsigned subcellEntityDimension = subcdim;

  while ((subcellEntityIndex != constrainingEntityIndex) || (subcellEntityDimension != constrainingEntityDim))
  {
    GlobalIndexType childCellIndex = currentAncestor->cellIndex();
    currentAncestor = currentAncestor->getParent();
    ancestors.push_back(currentAncestor);

    vector< CellPtr > children = currentAncestor->children();
    for (int i=0; i<children.size(); i++)
    {
      if (children[i]->cellIndex() == childCellIndex)
      {
        childOrdinals.push_back(i);

        subcord = currentAncestor->refinementPattern()->mapSubcellOrdinalFromChildToParent(i, subcellEntityDimension, subcord);
        if (subcord == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL)
        {
          // then it should be the case that the subcell entity has as generalized parent a higher-dimension subcell of currentAncestor
          pair<IndexType, unsigned> generalizedParent = _meshTopo->getEntityGeneralizedParent(subcellEntityDimension, subcellEntityIndex);
          if (generalizedParent.second <= subcellEntityDimension)
          {
            cout << "Cell detected MeshTopology Internal Error: did not find higher-dimensional generalized parent for entity of dimension " << subcellEntityDimension << " with entity index " << subcellEntityIndex << endl;
            cout << "MeshTopology entity report:\n";
            _meshTopo->printAllEntities();
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Cell detected MeshTopology Internal Error: did not find higher-dimensional generalized parent for entity");
          }
          else
          {
            subcellEntityDimension = generalizedParent.second;
            subcellEntityIndex = generalizedParent.first;
            subcord = currentAncestor->findSubcellOrdinal(subcellEntityDimension, subcellEntityIndex);
          }
        }
        if (subcord == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL)
        {
          cout << "Error: corresponding subcell not found in parent, even though the subcell is constrained...\n";
          cout << "Subcell entity:\n";
          _meshTopo->printEntityVertices(subcellEntityDimension, subcellEntityIndex);
          cout << "Constraining entity:\n";
          _meshTopo->printEntityVertices(constrainingEntityDim, constrainingEntityIndex);
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "corresponding subcell not found in parent, even though the subcell is constrained...");
        }
        subcellEntityIndex = currentAncestor->entityIndex(subcellEntityDimension, subcord);
        break;
      }
    }
    if (childOrdinals.size() != ancestors.size())
    {
      cout << "ERROR: child not found.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: child not found.");
    }
  }

  RefinementBranch refBranch;
  // now the ancestors and childOrdinals containers have the RefinementBranch info, in reverse order
  unsigned ancestorCount = ancestors.size();
  for (int i=ancestorCount-1; i >= 0; i--)
  {
    refBranch.push_back(make_pair(ancestors[i]->refinementPattern().get(), childOrdinals[i]));
  }
  return refBranch;
}

GlobalIndexType Cell::rootCellIndex()
{
  RefinementBranch refBranch;
  if (_parent == Teuchos::null)
  {
    return _cellIndex;
  }
  else
  {
    CellPtr parent = _parent;
    GlobalIndexType cellIndex = _cellIndex;
    while (parent != Teuchos::null) {
      cellIndex = parent->cellIndex();
      parent = parent->getParent();
    }
    return cellIndex;
  }
}

pair<unsigned, unsigned> Cell::ancestralSubcellOrdinalAndDimension(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  pair<IndexType,unsigned> constrainingEntity = meshTopoViewForCellValidity->getConstrainingEntity(subcdim, subcellEntityIndex);

  CellPtr ancestralCell = this->ancestralCellForSubcell(subcdim, subcord, meshTopoViewForCellValidity);

  unsigned constrainingSubcellOrdinal = ancestralCell->findSubcellOrdinal(constrainingEntity.second, constrainingEntity.first);
  return make_pair(constrainingSubcellOrdinal, constrainingEntity.second);
}

unsigned Cell::ancestralPermutationForSubcell(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if the given subcell is constrained by another cell, this method will return the subcell permutation of
  // this cell's nearest ancestor that is compatible with the constraining cell.
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  pair<IndexType,unsigned> constrainingEntity = meshTopoViewForCellValidity->getConstrainingEntity(subcdim, subcellEntityIndex);

  CellPtr ancestralCell = this->ancestralCellForSubcell(subcdim, subcord, meshTopoViewForCellValidity);

  unsigned constrainingSubcellOrdinal = ancestralCell->findSubcellOrdinal(constrainingEntity.second, constrainingEntity.first);

  if (constrainingSubcellOrdinal == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL)
  {
    cout << "constraining subcell ordinal not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraining subcell ordinal not found.");
  }

  return ancestralCell->subcellPermutation(constrainingEntity.second, constrainingSubcellOrdinal);
}

CellPtr Cell::ancestralCellForSubcell(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if the given subcell is constrained by another cell, this method will return this cell's nearest ancestor that is compatible with the constraining cell.
  RefinementBranch refBranch = refinementBranchForSubcell(subcdim, subcord, meshTopoViewForCellValidity);

  CellPtr currentAncestor = _meshTopo->getCell(_cellIndex);

  for (int refOrdinal=0; refOrdinal < refBranch.size(); refOrdinal++)
  {
    currentAncestor = currentAncestor->getParent();
  }

  return currentAncestor;
}

vector<unsigned> Cell::boundarySides()
{
  int sideCount = _cellTopo->getSideCount();
  vector<unsigned> sides;
  for (unsigned sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
  {
    if (_neighbors[sideOrdinal].first == -1) sides.push_back(sideOrdinal);
  }
  return sides;
}

bool Cell::isBoundary(unsigned int sideOrdinal)
{
  return _neighbors[sideOrdinal].first == -1;
}

bool Cell::isInteriorChild()
{
  if (_parent == Teuchos::null) return false;
  unsigned myOrdinalInParent = _parent->findChildOrdinal(_cellIndex);
  return _parent->refinementPattern()->childIsInterior(myOrdinalInParent);
}

MeshTopology* Cell::meshTopology()
{
  return _meshTopo;
}

int Cell::numChildren()
{
  return _children.size();
}

bool Cell::ownsSide(unsigned int sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  bool ownsSide;

  pair<GlobalIndexType,unsigned> neighborInfo = getNeighborInfo(sideOrdinal, meshTopoViewForCellValidity);
  GlobalIndexType neighborCellID = neighborInfo.first;
  unsigned neighborSideOrdinal = neighborInfo.second;
  if (neighborCellID == -1)   // boundary side
  {
    ownsSide = true;
  }
  else
  {
    CellPtr neighborCell = _meshTopo->getCell(neighborCellID);
    bool isPeer = neighborCell->getNeighborInfo(neighborSideOrdinal, meshTopoViewForCellValidity).first == _cellIndex;

    if (isPeer && !neighborCell->isParent(meshTopoViewForCellValidity))   // then the lower cellID owns
    {
      ownsSide = (_cellIndex < neighborCellID);
    }
    else if (isPeer)
    {
      // neighbor is parent (inactive), but we are peers: we own the side
      ownsSide = true;
    }
    else if (!neighborCell->isParent(meshTopoViewForCellValidity))
    {
      // neighbor is unbroken, and we are not peers: neighbor owns
      ownsSide = false;
    }
    else
    {
      // neighbor is parent, and we are a descendant of neighbor's neighbor (i.e. there is an anisotropic refinement)
      // in this case, we decide based on which of the ancestral cell IDs is lower
      GlobalIndexType ancestralCellID = neighborCell->getNeighborInfo(neighborSideOrdinal, meshTopoViewForCellValidity).first;
      ownsSide = (ancestralCellID < neighborCellID);
    }
  }
  return ownsSide;
}

RefinementPatternPtr Cell::refinementPattern()
{
  return _refPattern;
}

void Cell::setRefinementPattern(RefinementPatternPtr refPattern)
{
  _refPattern = refPattern;
}

void Cell::setVertices(const vector<IndexType> &vertexIndices)
{
  _vertices = vertexIndices;
  _entityIndices.clear();
}

unsigned Cell::sideSubcellPermutation(unsigned int sideOrdinal, unsigned int sideSubcdim, unsigned int sideSubcord)
{
  if (sideSubcdim==0) return 0; // no permutations / identity permutation for vertices
  vector< IndexType > subcellVertexIndices; //
  unsigned sideDim = _cellTopo->getDimension() - 1;
  CellTopoPtr sideTopo = _cellTopo->getSubcell(sideDim, sideOrdinal);
  unsigned subcellNodeCount = sideTopo->getNodeCount(sideSubcdim, sideSubcord);
  for (int nodeOrdinal=0; nodeOrdinal<subcellNodeCount; nodeOrdinal++)
  {
    unsigned nodeInSide = sideTopo->getNodeMap(sideSubcdim, sideSubcord, nodeOrdinal);
    unsigned nodeInCell = _cellTopo->getNodeMap(sideDim, sideOrdinal, nodeInSide);
    subcellVertexIndices.push_back(_vertices[nodeInCell]);
  }
  unsigned subcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(_cellTopo, sideDim, sideOrdinal, sideSubcdim, sideSubcord);
  IndexType subcellEntityIndex = entityIndex(sideSubcdim, subcellOrdinalInCell);
  vector< IndexType > canonicalOrdering = _meshTopo->getEntityVertexIndices(sideSubcdim, subcellEntityIndex);
  CellTopoPtr subEntityTopo = _meshTopo->getEntityTopology(sideSubcdim, subcellEntityIndex);
  subcellVertexIndices = _meshTopo->getCanonicalEntityNodesViaPeriodicBCs(sideSubcdim, subcellVertexIndices);

  return CamelliaCellTools::permutationMatchingOrder(subEntityTopo, canonicalOrdering, subcellVertexIndices);
}

unsigned Cell::subcellPermutation(unsigned d, unsigned scord)
{
  if (d==0) return 0;
  if ((d==_cellTopo->getDimension()) && (scord==0)) return 0;

  if (d >= _subcellPermutations.size())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Dimension d is out of bounds.");
  }
  if (_subcellPermutations[d].size() <= scord)
  {
    cout << "ERROR: scord out of bounds (maybe because _subcellPermutations is unset?).\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: scord out of bounds (maybe because _subcellPermutations is unset?).");
  }

  return _subcellPermutations[d][scord];
}

CellTopoPtr Cell::topology()
{
  return _cellTopo;
}

CellPtr Cell::getNeighbor(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  GlobalIndexType neighborCellIndex = this->getNeighborInfo(sideOrdinal, meshTopoViewForCellValidity).first;
  if (neighborCellIndex == -1) return Teuchos::null;
  else return _meshTopo->getCell(neighborCellIndex);
}

pair<GlobalIndexType, unsigned> Cell::getNeighborInfo(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  int sideCount = _cellTopo->getSideCount();
  if (sideOrdinal >= sideCount)
  {
    cout << "sideOrdinal " << sideOrdinal << " >= sideCount " << sideCount << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal must be less than sideCount!");
  }
  if (meshTopoViewForCellValidity->isValidCellIndex(_neighbors[sideOrdinal].first))
    return _neighbors[sideOrdinal];
  else if (_meshTopo->isValidCellIndex(_neighbors[sideOrdinal].first))
  {
    // check whether _neighbors[sideOrdinal] might have an ancestor that *is* valid, and also a neighbor
    pair<GlobalIndexType, unsigned> ancestralNeighborInfo = _neighbors[sideOrdinal];
    while (_meshTopo->isValidCellIndex(ancestralNeighborInfo.first)) {
      CellPtr neighbor = _meshTopo->getCell(ancestralNeighborInfo.first);
      CellPtr neighborParent = neighbor->getParent();
      if (neighborParent == Teuchos::null)
        return {-1,-1};
      ancestralNeighborInfo.first = neighborParent->cellIndex();
      unsigned childOrdinal = neighborParent->findChildOrdinal(neighbor->cellIndex());
      map< unsigned, unsigned > parentSideLookup = neighborParent->refinementPattern()->parentSideLookupForChild(childOrdinal);
      if (parentSideLookup.find(ancestralNeighborInfo.second) == parentSideLookup.end())
      {
        // child side internal to parent
        return {-1,-1};
      }
      ancestralNeighborInfo.second = parentSideLookup[ancestralNeighborInfo.second];
      
      MeshTopologyViewPtr meshTopoPtr = Teuchos::rcp(_meshTopo,false);
      // check that the parent is still a neighbor
      if (neighborParent->getNeighborInfo(ancestralNeighborInfo.second, meshTopoPtr).first != _cellIndex)
      {
        return {-1,-1};
      }
      if (meshTopoViewForCellValidity->isValidCellIndex(ancestralNeighborInfo.first))
      {
        return ancestralNeighborInfo;
      }
    }
  }
  
  return {-1,-1};
}

vector<CellPtr> Cell::getNeighbors(ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  vector<CellPtr> neighbors;
  for (int sideOrdinal=0; sideOrdinal<getSideCount(); sideOrdinal++)
  {
    CellPtr neighbor = getNeighbor(sideOrdinal, meshTopoViewForCellValidity);
    if (! Teuchos::is_null(neighbor))
    {
      neighbors.push_back(neighbor);
    }
  }
  return neighbors;
}

void Cell::setNeighbor(unsigned sideOrdinal, GlobalIndexType neighborCellIndex, unsigned neighborSideOrdinal, bool allowSameCellIndex)
{
  if ((neighborCellIndex == _cellIndex) && !allowSameCellIndex)
  {
    cout << "ERROR: neighborCellIndex == _cellIndex.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: neighborCellIndex == _cellIndex.\n");
  }
  int sideCount = _cellTopo->getSideCount();
  if (sideOrdinal >= sideCount)
  {
    cout << "sideOrdinal " << sideOrdinal << " >= sideCount " << sideCount << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal must be less than sideCount!");
  }
  _neighbors[sideOrdinal] = make_pair(neighborCellIndex, neighborSideOrdinal);
}

unsigned Cell::getSideCount()
{
  if (_cellTopo->getDimension() == 1) return _vertices.size();
  else return _cellTopo->getSideCount();
}

void Cell::printApproximateMemoryReport()
{
  cout << "**** Cell Memory Report ****\n";
  cout << "Memory sizes are in bytes.\n";

  long long memSize = 0;

  map<string, long long> variableCost = approximateMemoryCosts();

  map<long long, string> variableOrderedByCost;
  for (map<string, long long>::iterator entryIt = variableCost.begin(); entryIt != variableCost.end(); entryIt++)
  {
    variableOrderedByCost[entryIt->second] = entryIt->first;
  }

  for (map<long long, string>::iterator entryIt = variableOrderedByCost.begin(); entryIt != variableOrderedByCost.end(); entryIt++)
  {
    cout << setw(30) << entryIt->second << setw(30) << entryIt->first << endl;
    memSize += entryIt->first;
  }
  cout << "Total: " << memSize << " bytes.\n";
}

void Cell::printNeighborInfo(ConstMeshTopologyViewPtr meshTopoView)
{
  cout << "**** Cell " << _cellIndex << " Neighbors ****\n";
  int sideCount = getSideCount();
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    pair<GlobalIndexType, unsigned> neighborInfo = getNeighborInfo(sideOrdinal, meshTopoView);
    auto neighborID = neighborInfo.first;
    cout << "Side " << sideOrdinal << ": ";
    if (neighborID >= 0)
    {
      cout << neighborID << std::endl;
    }
    else
    {
      cout << "[NONE]\n";
    }
  }
}

const vector< vector< unsigned > > &Cell::subcellPermutations()
{
  return _subcellPermutations;
}

const vector< IndexType > & Cell::vertices()
{
  return _vertices;
}
