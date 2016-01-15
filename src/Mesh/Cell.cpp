//
//  Cell.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/22/14.
//
//

#include "Cell.h"
#include "RefinementPattern.h"
#include "CamelliaCellTools.h"

#include "GnuPlotUtil.h"

using namespace Camellia;

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

set<GlobalIndexType> Cell::getActiveNeighborIndices(MeshTopologyViewPtr meshTopoViewForCellValidity)
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

set<IndexType> Cell::getDescendants(MeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly)
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

vector< pair< GlobalIndexType, unsigned> > Cell::getDescendantsForSide(int sideIndex, MeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly)
{
  // if leafNodesOnly == true,  returns a flat list of leaf nodes (descendants that are not themselves parents)
  // if leafNodesOnly == false, returns a list in descending order: immediate children, then their children, and so on.

  // guarantee is that if a child and its parent are both in the list, the parent will come first

  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopoViewForCellValidity->isValidCellIndex(_cellIndex), std::invalid_argument, "_cellIndex is not valid");
  
  // pair (descendantCellIndex, descendantSideIndex)
  vector< pair< GlobalIndexType, unsigned> > descendantsForSide;
  if ( ! meshTopoViewForCellValidity->isParent(_cellIndex) )
  {
    descendantsForSide.push_back( {_cellIndex, sideIndex} );
    return descendantsForSide;
  }

  vector< pair<unsigned,unsigned> > childIndices = _refPattern->childrenForSides()[sideIndex];
  vector< pair<unsigned,unsigned> >::iterator entryIt;

  for (entryIt=childIndices.begin(); entryIt != childIndices.end(); entryIt++)
  {
    unsigned childOrdinal = (*entryIt).first;
    unsigned childSideOrdinal = (*entryIt).second;
    IndexType childCellIndex = _children[childOrdinal]->cellIndex();
    if ( ( !meshTopoViewForCellValidity->isParent(childCellIndex)) || (! leafNodesOnly ) )
    {
      // (            leaf node              ) || ...
      descendantsForSide.push_back( {_children[childOrdinal]->cellIndex(), childSideOrdinal} );
    }
    if ( _children[childOrdinal]->isParent(meshTopoViewForCellValidity) )
    {
      vector< pair<GlobalIndexType,unsigned> > childDescendants = _children[childOrdinal]->getDescendantsForSide(childSideOrdinal,meshTopoViewForCellValidity,leafNodesOnly);
//      descendantsForSide.insert(descendantsForSide.end(), childDescendants.begin(), childDescendants.end());
      vector< pair<GlobalIndexType,unsigned> >::iterator childEntryIt;
      for (childEntryIt=childDescendants.begin(); childEntryIt != childDescendants.end(); childEntryIt++)
      {
        descendantsForSide.push_back(*childEntryIt);
      }
    }
  }
  return descendantsForSide;
}

Cell::Cell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< vector< unsigned > > &subcellPermutations,
           unsigned cellIndex, MeshTopology* meshTopo)
{
  _cellTopo = cellTopo;
  _vertices = vertices;
  _subcellPermutations = subcellPermutations;
  _cellIndex = cellIndex;
  _meshTopo = meshTopo;
  int sideCount = cellTopo->getSideCount();
  _neighbors = vector< pair<GlobalIndexType, unsigned> >(sideCount,{-1,-1});
}

map<string, long long> Cell::approximateMemoryCosts()
{
  map<string, long long> variableCosts;

  // calibrate by computing some sizes
  map<int, int> emptyMap;
  vector<int> emptyVector;

  int MAP_OVERHEAD = sizeof(emptyMap);
  int VECTOR_OVERHEAD = sizeof(emptyVector);

  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal

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

  variableCosts["_children"] = VECTOR_OVERHEAD + _children.capacity() * sizeof(CellPtr);

  variableCosts["_refPattern"] = sizeof(RefinementPatternPtr);

  variableCosts["_parent"] = sizeof(_parent);

  variableCosts["_neighbors"] = VECTOR_OVERHEAD + _neighbors.capacity() * sizeof(pair<GlobalIndexType, unsigned>);

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

unsigned Cell::cellIndex()
{
  return _cellIndex;
}

unsigned Cell::childOrdinal(IndexType childIndex)
{
  for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++)
  {
    if (_children[childOrdinal]->cellIndex() == childIndex)
    {
      return childOrdinal;
    }
  }
  cout << "ERROR: child with ID childIndex not found in parent.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child with ID childIndex not found in parent");
  return -1; // NOT FOUND
}

const vector< Teuchos::RCP< Cell > > &Cell::children()
{
  return _children;
}

void Cell::setChildren(vector< Teuchos::RCP< Cell > > children)
{
  _children = children;
  Teuchos::RCP< Cell > thisPtr = Teuchos::rcp( this, false ); // doesn't own memory
  for (vector< Teuchos::RCP< Cell > >::iterator childIt = children.begin(); childIt != children.end(); childIt++)
  {
    (*childIt)->setParent(thisPtr);
  }
}

vector<unsigned> Cell::getChildIndices(MeshTopologyViewPtr meshTopoViewForCellValidity)
{
  if (! isParent(meshTopoViewForCellValidity))
    return vector<unsigned>();
  // if we are a parent, then assumption is *all* children are valid.
  vector<unsigned> indices(_children.size());
  for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++)
  {
    indices[childOrdinal] = _children[childOrdinal]->cellIndex();
  }
  return indices;
}

unsigned Cell::entityIndex(unsigned subcdim, unsigned subcord)
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
  
  set< unsigned > nodes;
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

vector<unsigned> Cell::getEntityVertexIndices(unsigned int subcdim, unsigned int subcord)
{
  vector< unsigned > nodes;
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

vector<unsigned> Cell::getEntityIndices(unsigned subcdim)
{
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  vector<unsigned> cellEntityIndices(entityCount);
  for (int j=0; j<entityCount; j++)
  {
    unsigned entityIndex;
    set< unsigned > nodes;
    if (subcdim != 0)
    {
      int entityNodeCount = _cellTopo->getNodeCount(subcdim, j);
      for (int node=0; node<entityNodeCount; node++)
      {
        unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, j, node);
        nodes.insert(_vertices[nodeIndexInCell]);
      }
    }
    else
    {
      nodes.insert(_vertices[j]);
    }

    entityIndex = _meshTopo->getEntityIndex(subcdim, nodes);
    cellEntityIndices[j] = entityIndex;
  }
  return cellEntityIndices;
}

int Cell::findChildOrdinal(IndexType cellIndex)
{
  for (int childOrdinal=0; childOrdinal < _children.size(); childOrdinal++)
  {
    if (_children[childOrdinal]->cellIndex() == cellIndex) return childOrdinal;
  }
  return -1;
}

unsigned Cell::findSubcellOrdinal(unsigned subcdim, IndexType subcEntityIndex)
{
  // this is pretty brute force right now
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  for (int scord=0; scord<entityCount; scord++)
  {
    unsigned scEntityIndex = entityIndex(subcdim, scord);
    if (scEntityIndex == subcEntityIndex)
    {
      return scord;
    }
  }
  return -1; // NOT FOUND
}

unsigned Cell::findSubcellOrdinalInSide(unsigned int subcdim, IndexType subcEntityIndex, unsigned sideOrdinal)
{
  unsigned subcOrdinalInCell = findSubcellOrdinal(subcdim, subcEntityIndex);
  int sideDim = _cellTopo->getDimension() - 1;
  if (subcOrdinalInCell == -1) return -1;
  bool assertContainment = false; // don't throw exception if not found (return -1)
  return CamelliaCellTools::subcellReverseOrdinalMap(topology(), sideDim, sideOrdinal, subcdim, subcOrdinalInCell, assertContainment);
}

Teuchos::RCP<Cell> Cell::getParent()
{
  return _parent;
}

void Cell::setParent(Teuchos::RCP<Cell> parent)
{
  _parent = Teuchos::rcp(parent.get(),false); // make weak reference
}

bool Cell::isParent(MeshTopologyViewPtr meshTopoViewForCellValidity)
{
  if (_children.size() == 0) return false;
  return meshTopoViewForCellValidity->isValidCellIndex(_children[0]->cellIndex()); // if first child is not valid, then presumably not a parent from this MeshTopo's point of view
//  return _children.size() > 0;
}

RefinementBranch Cell::refinementBranchForSide(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity)
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

RefinementBranch Cell::refinementBranchForSubcell(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if the given subcell is constrained by another cell, this method will return a RefinementBranch which has as its root
  // this cell's ancestor that is compatible with the constraining cell, and as its leaf this cell.
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);

  pair<IndexType, unsigned> constrainingEntity = meshTopoViewForCellValidity->getConstrainingEntity(subcdim, subcellEntityIndex);
//  IndexType constrainingEntityIndex = _meshTopo->getConstrainingEntityIndex(subcdim, subcellEntityIndex);
  IndexType constrainingEntityIndex = constrainingEntity.first;
  unsigned constrainingEntityDim = constrainingEntity.second;

  CellPtr currentAncestor = _meshTopo->getCell(_cellIndex);
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
        if (subcord == -1)
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
        if (subcord==-1)
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

pair<unsigned, unsigned> Cell::ancestralSubcellOrdinalAndDimension(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity)
{
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  pair<IndexType,unsigned> constrainingEntity = meshTopoViewForCellValidity->getConstrainingEntity(subcdim, subcellEntityIndex);

  CellPtr ancestralCell = this->ancestralCellForSubcell(subcdim, subcord, meshTopoViewForCellValidity);

  unsigned constrainingSubcellOrdinal = ancestralCell->findSubcellOrdinal(constrainingEntity.second, constrainingEntity.first);
  return make_pair(constrainingSubcellOrdinal, constrainingEntity.second);
}

unsigned Cell::ancestralPermutationForSubcell(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity)
{
  // if the given subcell is constrained by another cell, this method will return the subcell permutation of
  // this cell's nearest ancestor that is compatible with the constraining cell.
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  pair<IndexType,unsigned> constrainingEntity = meshTopoViewForCellValidity->getConstrainingEntity(subcdim, subcellEntityIndex);

  CellPtr ancestralCell = this->ancestralCellForSubcell(subcdim, subcord, meshTopoViewForCellValidity);

  unsigned constrainingSubcellOrdinal = ancestralCell->findSubcellOrdinal(constrainingEntity.second, constrainingEntity.first);

  if (constrainingSubcellOrdinal == -1)
  {
    cout << "constraining subcell ordinal not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraining subcell ordinal not found.");
  }

  return ancestralCell->subcellPermutation(constrainingEntity.second, constrainingSubcellOrdinal);
}

CellPtr Cell::ancestralCellForSubcell(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity)
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

MeshTopology* Cell::meshTopology()
{
  return _meshTopo;
}

int Cell::numChildren()
{
  return _children.size();
}

bool Cell::ownsSide(unsigned int sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity)
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

CellPtr Cell::getNeighbor(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity)
{
  GlobalIndexType neighborCellIndex = this->getNeighborInfo(sideOrdinal, meshTopoViewForCellValidity).first;
  if (neighborCellIndex == -1) return Teuchos::null;
  else return _meshTopo->getCell(neighborCellIndex);
}

pair<GlobalIndexType, unsigned> Cell::getNeighborInfo(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity)
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

vector<CellPtr> Cell::getNeighbors(MeshTopologyViewPtr meshTopoViewForCellValidity)
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

void Cell::setNeighbor(unsigned sideOrdinal, GlobalIndexType neighborCellIndex, unsigned neighborSideOrdinal)
{
  if (neighborCellIndex == _cellIndex)
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

const vector< vector< unsigned > > &Cell::subcellPermutations()
{
  return _subcellPermutations;
}

const vector< unsigned > & Cell::vertices()
{
  return _vertices;
}
