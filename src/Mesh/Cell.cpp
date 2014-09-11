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

vector< pair<GlobalIndexType, unsigned> > Cell::childrenForSide(unsigned sideIndex) {
  vector< pair<GlobalIndexType, unsigned> > childIndicesForSide;
  
  if (_refPattern.get() != NULL) {
    vector< pair<unsigned, unsigned> > refinementChildIndicesForSide = _refPattern->childrenForSides()[sideIndex];
    
    for( vector< pair<unsigned, unsigned> >::iterator entryIt = refinementChildIndicesForSide.begin();
        entryIt != refinementChildIndicesForSide.end(); entryIt++) {
      unsigned childIndex = _children[entryIt->first]->cellIndex();
      unsigned childSide = entryIt->second;
      childIndicesForSide.push_back(make_pair(childIndex,childSide));
    }
  }
  
  return childIndicesForSide;
}

vector< pair< GlobalIndexType, unsigned> > Cell::getDescendantsForSide(int sideIndex, bool leafNodesOnly) {
  // if leafNodesOnly == true,  returns a flat list of leaf nodes (descendants that are not themselves parents)
  // if leafNodesOnly == false, returns a list in descending order: immediate children, then their children, and so on.
  
  // guarantee is that if a child and its parent are both in the list, the parent will come first
  
  // pair (descendantCellIndex, descendantSideIndex)
  vector< pair< GlobalIndexType, unsigned> > descendantsForSide;
  if ( ! isParent() ) {
    descendantsForSide.push_back( make_pair( _cellIndex, sideIndex) );
    return descendantsForSide;
  }
  
  vector< pair<unsigned,unsigned> > childIndices = _refPattern->childrenForSides()[sideIndex];
  vector< pair<unsigned,unsigned> >::iterator entryIt;
  
  for (entryIt=childIndices.begin(); entryIt != childIndices.end(); entryIt++) {
    unsigned childIndex = (*entryIt).first;
    unsigned childSideIndex = (*entryIt).second;
    if ( (! _children[childIndex]->isParent()) || (! leafNodesOnly ) ) {
      // (            leaf node              ) || ...
      descendantsForSide.push_back( make_pair( _children[childIndex]->cellIndex(), childSideIndex) );
    }
    if ( _children[childIndex]->isParent() ) {
      vector< pair<GlobalIndexType,unsigned> > childDescendants = _children[childIndex]->getDescendantsForSide(childSideIndex,leafNodesOnly);
//      descendantsForSide.insert(descendantsForSide.end(), childDescendants.begin(), childDescendants.end());
      vector< pair<GlobalIndexType,unsigned> >::iterator childEntryIt;
      for (childEntryIt=childDescendants.begin(); childEntryIt != childDescendants.end(); childEntryIt++) {
        descendantsForSide.push_back(*childEntryIt);
      }
    }
  }
  return descendantsForSide;
}

Cell::Cell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< map< unsigned, unsigned > > &subcellPermutations,
     unsigned cellIndex, MeshTopology* meshTopo) {
  _cellTopo = cellTopo;
  _vertices = vertices;
  _subcellPermutations = subcellPermutations;
  _cellIndex = cellIndex;
  _meshTopo = meshTopo;
  int sideCount = CamelliaCellTools::getSideCount(*_cellTopo);
  _neighbors = vector< pair<GlobalIndexType, unsigned> >(sideCount,make_pair(-1,-1));
}
unsigned Cell::cellIndex() {
  return _cellIndex;
}

unsigned Cell::childOrdinal(IndexType childIndex) {
  for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++) {
    if (_children[childOrdinal]->cellIndex() == childIndex) {
      return childOrdinal;
    }
  }
  cout << "ERROR: child with ID childIndex not found in parent.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child with ID childIndex not found in parent");
  return -1; // NOT FOUND
}

const vector< Teuchos::RCP< Cell > > &Cell::children() {
  return _children;
}

void Cell::setChildren(vector< Teuchos::RCP< Cell > > children) {
  _children = children;
  Teuchos::RCP< Cell > thisPtr = Teuchos::rcp( this, false ); // doesn't own memory
  for (vector< Teuchos::RCP< Cell > >::iterator childIt = children.begin(); childIt != children.end(); childIt++) {
    (*childIt)->setParent(thisPtr);
  }
}

vector<unsigned> Cell::getChildIndices() {
  vector<unsigned> indices(_children.size());
  for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++) {
    indices[childOrdinal] = _children[childOrdinal]->cellIndex();
  }
  return indices;
}

unsigned Cell::entityIndex(unsigned subcdim, unsigned subcord) {
  set< unsigned > nodes;
  if (subcdim != 0) {
    int entityNodeCount = _cellTopo->getNodeCount(subcdim, subcord);
    for (int node=0; node<entityNodeCount; node++) {
      unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, subcord, node);
      nodes.insert(_vertices[nodeIndexInCell]);
    }
  } else {
    nodes.insert(_vertices[subcord]);
  }
  return _meshTopo->getEntityIndex(subcdim, nodes);
}

vector<unsigned> Cell::getEntityVertexIndices(unsigned int subcdim, unsigned int subcord) {
  vector< unsigned > nodes;
  if (subcdim != 0) {
    int entityNodeCount = _cellTopo->getNodeCount(subcdim, subcord);
    for (int node=0; node<entityNodeCount; node++) {
      unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, subcord, node);
      nodes.push_back(_vertices[nodeIndexInCell]);
    }
  } else {
    nodes.push_back(_vertices[subcord]);
  }
  return nodes;
}

vector<unsigned> Cell::getEntityIndices(unsigned subcdim) {
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  vector<unsigned> cellEntityIndices(entityCount);
  for (int j=0; j<entityCount; j++) {
    unsigned entityIndex;
    set< unsigned > nodes;
    if (subcdim != 0) {
      int entityNodeCount = _cellTopo->getNodeCount(subcdim, j);
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = _cellTopo->getNodeMap(subcdim, j, node);
        nodes.insert(_vertices[nodeIndexInCell]);
      }
    } else {
      nodes.insert(_vertices[j]);
    }
    
    entityIndex = _meshTopo->getEntityIndex(subcdim, nodes);
    cellEntityIndices[j] = entityIndex;
  }
  return cellEntityIndices;
}

unsigned Cell::findSubcellOrdinal(unsigned subcdim, IndexType subcEntityIndex) {
  // this is pretty brute force right now
  int entityCount = _cellTopo->getSubcellCount(subcdim);
  for (int scord=0; scord<entityCount; scord++) {
    unsigned scEntityIndex = entityIndex(subcdim, scord);
    if (scEntityIndex == subcEntityIndex) {
      return scord;
    }
  }
  return -1; // NOT FOUND
}

unsigned Cell::findSubcellOrdinalInSide(unsigned int subcdim, IndexType subcEntityIndex, unsigned sideOrdinal) {
  int sideDim = _cellTopo->getDimension() - 1;
  IndexType sideEntityIndex = entityIndex(sideDim, sideOrdinal);
  int scCount = _meshTopo->getSubEntityCount(sideDim, sideEntityIndex, subcdim);
  for (unsigned scordInSide=0; scordInSide < scCount; scordInSide++) {
    if (subcEntityIndex == _meshTopo->getSubEntityIndex(sideDim, sideEntityIndex, subcdim, scordInSide)) {
      return scordInSide;
    }
  }
  return -1; // NOT FOUND
}

Teuchos::RCP<Cell> Cell::getParent() {
  return _parent;
}

void Cell::setParent(Teuchos::RCP<Cell> parent) {
  _parent = parent;
}

bool Cell::isParent() {
  return _children.size() > 0;
}

RefinementBranch Cell::refinementBranchForSide(unsigned sideOrdinal) {
  // if this cell (on this side) is the finer side of a hanging node, returns the RefinementBranch starting
  // with the coarse neighbor's neighbor (this cell's ancestor).  (Otherwise, the RefinementBranch will be empty.)
  RefinementBranch refBranch;
  pair<GlobalIndexType, unsigned> neighborInfo = this->getNeighbor(sideOrdinal);
  GlobalIndexType neighborCellIndex = neighborInfo.first;
  unsigned sideIndexInNeighbor = neighborInfo.second;
  if (neighborCellIndex == -1) {
    return refBranch; // no refinements
  }
  CellPtr neighbor = _meshTopo->getCell(neighborCellIndex);
  if (neighbor->getNeighbor(sideIndexInNeighbor).first == this->_cellIndex) { // peers!
    return refBranch; // no refinements
  } else {
    GlobalIndexType ancestorCellIndex = neighbor->getNeighbor(sideIndexInNeighbor).first;
    vector< CellPtr > ancestors;
    vector< unsigned > childOrdinals;
    CellPtr currentAncestor = _meshTopo->getCell(_cellIndex);
    while (currentAncestor->cellIndex() != ancestorCellIndex) {
      GlobalIndexType childCellIndex = currentAncestor->cellIndex();
      currentAncestor = currentAncestor->getParent();
      ancestors.push_back(currentAncestor);
      vector< CellPtr > children = currentAncestor->children();
      for (int i=0; i<children.size(); i++) {
        if (children[i]->cellIndex() == childCellIndex) {
          childOrdinals.push_back(i);
          break;
        }
      }
      if (childOrdinals.size() != ancestors.size()) {
        cout << "ERROR: child not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: child not found.");
      }
    }
    // now the ancestors and childOrdinals containers have the RefinementBranch info, in reverse order
    unsigned ancestorCount = ancestors.size();
    for (int i=ancestorCount-1; i >= 0; i--) {
      refBranch.push_back(make_pair(ancestors[i]->refinementPattern().get(), childOrdinals[i]));
    }
  }
  return refBranch;
}

RefinementBranch Cell::refinementBranchForSubcell(unsigned subcdim, unsigned subcord) {
  // if the given subcell is constrained by another cell, this method will return a RefinementBranch which has as its root
  // this cell's ancestor that is compatible with the constraining cell, and as its leaf this cell.
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  
  pair<IndexType, unsigned> constrainingEntity = _meshTopo->getConstrainingEntity(subcdim, subcellEntityIndex);
//  IndexType constrainingEntityIndex = _meshTopo->getConstrainingEntityIndex(subcdim, subcellEntityIndex);
  IndexType constrainingEntityIndex = constrainingEntity.first;
  unsigned constrainingEntityDim = constrainingEntity.second;

  CellPtr currentAncestor = _meshTopo->getCell(_cellIndex);
  vector< CellPtr > ancestors;
  vector< unsigned > childOrdinals;
  
  unsigned subcellEntityDimension = subcdim;
  
  while ((subcellEntityIndex != constrainingEntityIndex) || (subcellEntityDimension != constrainingEntityDim)) {
    GlobalIndexType childCellIndex = currentAncestor->cellIndex();
    currentAncestor = currentAncestor->getParent();
    ancestors.push_back(currentAncestor);
    
    vector< CellPtr > children = currentAncestor->children();
    for (int i=0; i<children.size(); i++) {
      if (children[i]->cellIndex() == childCellIndex) {
        childOrdinals.push_back(i);
        
        subcord = currentAncestor->refinementPattern()->mapSubcellOrdinalFromChildToParent(i, subcellEntityDimension, subcord);
        if (subcord == -1) {
          // then it should be the case that the subcell entity has as generalized parent a higher-dimension subcell of currentAncestor
          pair<IndexType, unsigned> generalizedParent = _meshTopo->getEntityGeneralizedParent(subcellEntityDimension, subcellEntityIndex);
          if (generalizedParent.second <= subcellEntityDimension) {
            cout << "Cell detected MeshTopology Internal Error: did not find higher-dimensional generalized parent for entity of dimension " << subcellEntityDimension << " with entity index " << subcellEntityIndex << endl;
            cout << "MeshTopology entity report:\n";
            _meshTopo->printAllEntities();
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Cell detected MeshTopology Internal Error: did not find higher-dimensional generalized parent for entity");
          } else {
            subcellEntityDimension = generalizedParent.second;
            subcellEntityIndex = generalizedParent.first;
            subcord = currentAncestor->findSubcellOrdinal(subcellEntityDimension, subcellEntityIndex);
          }
        }
        if (subcord==-1) {
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
    if (childOrdinals.size() != ancestors.size()) {
      cout << "ERROR: child not found.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: child not found.");
    }
  }
  
  RefinementBranch refBranch;
  // now the ancestors and childOrdinals containers have the RefinementBranch info, in reverse order
  unsigned ancestorCount = ancestors.size();
  for (int i=ancestorCount-1; i >= 0; i--) {
    refBranch.push_back(make_pair(ancestors[i]->refinementPattern().get(), childOrdinals[i]));
  }
  return refBranch;
}

pair<unsigned, unsigned> Cell::ancestralSubcellOrdinalAndDimension(unsigned subcdim, unsigned subcord) {
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  pair<IndexType,unsigned> constrainingEntity = _meshTopo->getConstrainingEntity(subcdim, subcellEntityIndex);
  
  CellPtr ancestralCell = this->ancestralCellForSubcell(subcdim, subcord);
  
  unsigned constrainingSubcellOrdinal = ancestralCell->findSubcellOrdinal(constrainingEntity.second, constrainingEntity.first);
  return make_pair(constrainingSubcellOrdinal, constrainingEntity.second);
}

unsigned Cell::ancestralPermutationForSubcell(unsigned subcdim, unsigned subcord) {
  // if the given subcell is constrained by another cell, this method will return the subcell permutation of
  // this cell's nearest ancestor that is compatible with the constraining cell.
  IndexType subcellEntityIndex = entityIndex(subcdim, subcord);
  pair<IndexType,unsigned> constrainingEntity = _meshTopo->getConstrainingEntity(subcdim, subcellEntityIndex);
  
  CellPtr ancestralCell = this->ancestralCellForSubcell(subcdim, subcord);
  
  unsigned constrainingSubcellOrdinal = ancestralCell->findSubcellOrdinal(constrainingEntity.second, constrainingEntity.first);
  
  if (constrainingSubcellOrdinal == -1) {
    cout << "constraining subcell ordinal not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraining subcell ordinal not found.");
  }
  
  return ancestralCell->subcellPermutation(constrainingEntity.second, constrainingSubcellOrdinal);
}

CellPtr Cell::ancestralCellForSubcell(unsigned subcdim, unsigned subcord) {
  // if the given subcell is constrained by another cell, this method will return this cell's nearest ancestor that is compatible with the constraining cell.
  RefinementBranch refBranch = refinementBranchForSubcell(subcdim, subcord);
  
  CellPtr currentAncestor = _meshTopo->getCell(_cellIndex);
  
  for (int refOrdinal=0; refOrdinal < refBranch.size(); refOrdinal++) {
    currentAncestor = currentAncestor->getParent();
  }
  
  return currentAncestor;
}

MeshTopology* Cell::meshTopology() {
  return _meshTopo;
}

bool Cell::ownsSide(unsigned int sideOrdinal) {
  bool ownsSide;
  
  pair<GlobalIndexType,unsigned> neighborInfo = getNeighbor(sideOrdinal);
  GlobalIndexType neighborCellID = neighborInfo.first;
  unsigned neighborSideOrdinal = neighborInfo.second;
  if (neighborCellID == -1) { // boundary side
    ownsSide = true;
  } else {
    CellPtr neighborCell = _meshTopo->getCell(neighborCellID);
    bool isPeer = neighborCell->getNeighbor(neighborSideOrdinal).first == _cellIndex;
    
    if (isPeer && !neighborCell->isParent()) { // then the lower cellID owns
      ownsSide = (_cellIndex < neighborCellID);
    } else if (isPeer) {
      // neighbor is parent (inactive), but we are peers: we own the side
      ownsSide = true;
    } else if (!neighborCell->isParent()) {
      // neighbor is unbroken, and we are not peers: neighbor owns
      ownsSide = false;
    } else {
      // neighbor is parent, and we are a descendant of neighbor's neighbor (i.e. there is an anisotropic refinement)
      // in this case, we decide based on which of the ancestral cell IDs is lower
      GlobalIndexType ancestralCellID = neighborCell->getNeighbor(neighborSideOrdinal).first;
      ownsSide = (ancestralCellID < neighborCellID);
    }
  }
  return ownsSide;
}

RefinementPatternPtr Cell::refinementPattern() {
  return _refPattern;
}

void Cell::setRefinementPattern(RefinementPatternPtr refPattern) {
  _refPattern = refPattern;
}

unsigned Cell::sideSubcellPermutation(unsigned int sideOrdinal, unsigned int sideSubcdim, unsigned int sideSubcord) {
  if (sideSubcdim==0) return 0; // no permutations / identity permutation for vertices
  vector< IndexType > subcellVertexIndices; //
  unsigned sideDim = _cellTopo->getDimension() - 1;
  shards::CellTopology sideTopo = _cellTopo->getCellTopologyData(sideDim, sideOrdinal);
  unsigned subcellNodeCount = sideTopo.getNodeCount(sideSubcdim, sideSubcord);
  for (int nodeOrdinal=0; nodeOrdinal<subcellNodeCount; nodeOrdinal++) {
    unsigned nodeInSide = sideTopo.getNodeMap(sideSubcdim, sideSubcord, nodeOrdinal);
    unsigned nodeInCell = _cellTopo->getNodeMap(sideDim, sideOrdinal, nodeInSide);
    subcellVertexIndices.push_back(_vertices[nodeInCell]);
  }
  unsigned subcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(*_cellTopo, sideDim, sideOrdinal, sideSubcdim, sideSubcord);
  IndexType subcellEntityIndex = entityIndex(sideSubcdim, subcellOrdinalInCell);
  vector< IndexType > canonicalOrdering = _meshTopo->getEntityVertexIndices(sideSubcdim, subcellEntityIndex);
  shards::CellTopology subEntityTopo = _meshTopo->getEntityTopology(sideSubcdim, subcellEntityIndex);
  subcellVertexIndices = _meshTopo->getCanonicalEntityNodesViaPeriodicBCs(sideSubcdim, subcellVertexIndices);
  
  return CamelliaCellTools::permutationMatchingOrder(subEntityTopo, canonicalOrdering, subcellVertexIndices);
}

unsigned Cell::subcellPermutation(unsigned d, unsigned scord) {
  if (_subcellPermutations[d].find(scord) == _subcellPermutations[d].end()) {
    cout << "ERROR: subcell permutations appear to be unset.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: subcell permutations appear to be unset.");
  }
  return _subcellPermutations[d][scord];
}

CellTopoPtr Cell::topology() {
  return _cellTopo;
}

pair<GlobalIndexType, unsigned> Cell::getNeighbor(unsigned sideOrdinal) {
  int sideCount = CamelliaCellTools::getSideCount(*_cellTopo);
  if (sideOrdinal >= sideCount) {
    cout << "sideOrdinal " << sideOrdinal << " >= sideCount " << sideCount << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal must be less than sideCount!");
  }
  return _neighbors[sideOrdinal];
}

void Cell::setNeighbor(unsigned sideOrdinal, GlobalIndexType neighborCellIndex, unsigned neighborSideOrdinal) {
  if (neighborCellIndex == _cellIndex) {
    cout << "ERROR: neighborCellIndex == _cellIndex.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: neighborCellIndex == _cellIndex.\n");
  }
  int sideCount = CamelliaCellTools::getSideCount(*_cellTopo);
  if (sideOrdinal >= sideCount) {
    cout << "sideOrdinal " << sideOrdinal << " >= sideCount " << sideCount << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal must be less than sideCount!");
  }
  _neighbors[sideOrdinal] = make_pair(neighborCellIndex, neighborSideOrdinal);
}

const vector< unsigned > & Cell::vertices() {return _vertices;}
