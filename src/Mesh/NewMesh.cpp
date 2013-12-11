//
//  NewMesh.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#include "NewMesh.h"

#include "CamelliaCellTools.h"

void NewMesh::init(unsigned spaceDim) {
  RefinementPattern::initializeAnisotropicRelationships(); // not sure this is the optimal place for this call
  
  _spaceDim = spaceDim;
  _entities = vector< vector< set< unsigned > > >(_spaceDim);
  _knownEntities = vector< map< set<unsigned>, unsigned > >(_spaceDim); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< map< unsigned, vector<unsigned> > >(_spaceDim);
  _activeCellsForEntities = vector< map< unsigned, set< pair<unsigned, unsigned> > > >(_spaceDim); // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _activeEntities = vector< set<unsigned > >(_spaceDim);
  _constrainingEntities = vector< map< unsigned, unsigned > >(_spaceDim); // map from broken entity to the whole (constraining) one.
  _constrainedEntities = vector< map< unsigned, set< unsigned > > >(_spaceDim); // map from constraining entity to all broken ones constrained by it.
  _parentEntities = vector< map< unsigned, vector< pair<unsigned, unsigned> > > >(_spaceDim); // map to possible parents
  _childEntities = vector< map< unsigned, vector< pair<RefinementPatternPtr, vector<unsigned> > > > >(_spaceDim);
  _entityCellTopologyKeys = vector< map< unsigned, unsigned > >(_spaceDim);
}

NewMesh::NewMesh(unsigned spaceDim) {
  init(spaceDim);
}

NewMesh::NewMesh(NewMeshGeometryPtr meshGeometry) {
  unsigned spaceDim = meshGeometry->vertices()[0].size();

  init(spaceDim);
  _vertices = meshGeometry->vertices();
  
  TEUCHOS_TEST_FOR_EXCEPTION(meshGeometry->cellTopos().size() != meshGeometry->elementVertices().size(), std::invalid_argument,
                             "length of cellTopos != length of elementVertices");
  
  int numElements = meshGeometry->cellTopos().size();
  
  for (int i=0; i<numElements; i++) {
    CellTopoPtr cellTopo = meshGeometry->cellTopos()[i];
    vector< unsigned > cellVertices = meshGeometry->elementVertices()[i];
    addCell(cellTopo, cellVertices);
  }
}

unsigned NewMesh::activeCellCount() {
  return _activeCells.size();
}

set<unsigned> NewMesh::activeDescendants(unsigned d, unsigned entityIndex) {
  set<unsigned> allDescendants = descendants(d,entityIndex);
  set<unsigned> filteredDescendants;
  for (set<unsigned>::iterator descIt=allDescendants.begin(); descIt!=allDescendants.end(); descIt++) {
    if (_activeEntities[d].find(*descIt) != _activeEntities[d].end()) {
      filteredDescendants.insert(*descIt);
    }
  }
  return filteredDescendants;
}

set<unsigned> NewMesh::activeDescendantsNotInSet(unsigned d, unsigned entityIndex, const set<unsigned> &excludedSet) {
  set<unsigned> filteredDescendants;
  
  set<unsigned> activeDesc = activeDescendants(d, entityIndex);
  for (set<unsigned>::iterator myDesc=activeDesc.begin(); myDesc !=activeDesc.end(); myDesc++) {
    if (excludedSet.find(*myDesc)==excludedSet.end()) {
      filteredDescendants.insert(*myDesc);
    }
  }
  return filteredDescendants;
}

unsigned NewMesh::eldestActiveAncestor(unsigned d, unsigned entityIndex) {
  bool hasActiveParent = false;
  while (hasActiveParent) {
    map< unsigned, vector< pair<unsigned,unsigned> > >::iterator parentEntityIt = _parentEntities[d].find(entityIndex);
    if (parentEntityIt != _parentEntities[d].end()) {
      vector< pair<unsigned,unsigned> > possibleParents = parentEntityIt->second;
      for (vector< pair<unsigned,unsigned> >::iterator possibleParentIt = possibleParents.begin(); possibleParentIt != possibleParents.end(); possibleParentIt++) {
        unsigned possibleParentIndex = possibleParentIt->first;
        if (_activeEntities[d].find(possibleParentIndex) != _activeEntities[d].end()) {
          hasActiveParent = true;
          entityIndex = possibleParentIndex; // the new entityIndex will be the last found active parent (which is the coarsest/eldest one)
        }
      }
    }
  }
  return entityIndex;
}

NewMeshCellPtr NewMesh::addCell(CellTopoPtr cellTopo, const vector<vector<double> > &cellVertices) {
  vector<unsigned> vertexIndices = getVertexIndices(cellVertices);
  unsigned cellIndex = addCell(cellTopo, vertexIndices);
  return _cells[cellIndex];
}

unsigned NewMesh::addCell(CellTopoPtr cellTopo, const vector<unsigned> &cellVertices) {
  vector< map< unsigned, unsigned > > cellEntityPermutations;
  unsigned cellIndex = _cells.size();
  vector< vector<unsigned> > cellEntityIndices(_spaceDim); // subcdim, subcord
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    cellEntityPermutations.push_back(map<unsigned, unsigned>());
    int entityCount = cellTopo->getSubcellCount(d);
    cellEntityIndices[d] = vector<unsigned>(entityCount);
    for (int j=0; j<entityCount; j++) {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      unsigned entityIndex, entityPermutation;
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      vector< unsigned > nodes;
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
        nodes.push_back(cellVertices[nodeIndexInCell]);
      }
      
      entityIndex = addEntity(cellTopo->getCellTopologyData(d, j), nodes, entityPermutation);
      cellEntityIndices[d][j] = entityIndex;

      cellEntityPermutations[d][entityIndex] = entityPermutation;
      _activeCellsForEntities[d][entityIndex].insert(make_pair(cellIndex,j));
      _activeEntities[d].insert(entityIndex);
    }
  }
  
  NewMeshCellPtr cell = Teuchos::rcp( new NewMeshCell(cellTopo, cellVertices, cellEntityPermutations, cellIndex, cellEntityIndices) );
  _cells.push_back(cell);
  _activeCells.insert(cellIndex);
  return cellIndex;
}

unsigned NewMesh::addEntity(const shards::CellTopology &entityTopo, const vector<unsigned> &entityVertices, unsigned &entityPermutation) {
  set< unsigned > nodeSet;
  nodeSet.insert(entityVertices.begin(),entityVertices.end());
  if (nodeSet.size() != entityVertices.size()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Entities may not have repeated vertices");
  }
  unsigned d  = entityTopo.getDimension();
  unsigned entityIndex;
  if (_knownEntities[d].find(nodeSet) == _knownEntities[d].end()) {
    // new entity
    entityIndex = _entities[d].size();
    _entities[d].push_back(nodeSet);
    _knownEntities[d][nodeSet] = entityIndex;
    _canonicalEntityOrdering[d][entityIndex] = entityVertices;
    entityPermutation = 0;
    if (_knownTopologies.find(entityTopo.getKey()) == _knownTopologies.end()) {
      _knownTopologies[entityTopo.getKey()] = entityTopo;
    }
    _entityCellTopologyKeys[d][entityIndex] = entityTopo.getKey();
//    cout << "NewMesh::addEntity: added entity of dimension " << d << " with index " << entityIndex << " and vertices:" << endl;
//    printVertices(nodeSet);
  } else {
    // existing entity
    entityIndex = _knownEntities[d][nodeSet];
    entityPermutation = CamelliaCellTools::permutationMatchingOrder(entityTopo, _canonicalEntityOrdering[d][entityIndex], entityVertices);
  }
  return entityIndex;
}

void NewMesh::addChildren(NewMeshCellPtr parentCell, const vector< CellTopoPtr > &childTopos, const vector< vector<unsigned> > &childVertices) {
  int numChildren = childTopos.size();
  TEUCHOS_TEST_FOR_EXCEPTION(numChildren != childVertices.size(), std::invalid_argument, "childTopos and childVertices must be the same size");
  vector< NewMeshCellPtr > children;
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    unsigned cellIndex = addCell(childTopos[childIndex], childVertices[childIndex]);
    children.push_back(_cells[cellIndex]);
  }
  parentCell->setChildren(children);
}

unsigned NewMesh::cellCount() {
  return _cells.size();
}

void NewMesh::deactivateCell(NewMeshCellPtr cell) {
  CellTopoPtr cellTopo = cell->topology();
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    int entityCount = cellTopo->getSubcellCount(d);
    for (int j=0; j<entityCount; j++) {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      set< unsigned > nodeSet;
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
        nodeSet.insert(cell->vertices()[nodeIndexInCell]);
      }
      
      map< set<unsigned>, unsigned >::iterator knownEntry = _knownEntities[d].find(nodeSet);
      if (knownEntry == _knownEntities[d].end()) {
        // entity not found: an error
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cell entity not found!");
      }
    
      // delete from the _activeCellsForEntities store
      unsigned entityIndex = knownEntry->second;
      unsigned eraseCount = _activeCellsForEntities[d][entityIndex].erase(make_pair(cell->cellIndex(),j));
      if (eraseCount==0) {
        cout << "WARNING: attempt was made to deactivate a non-active subcell topology...\n";
      }
      if (_activeCellsForEntities[d][entityIndex].size() == 0) {
        // the entity itself has been deactivated, and we should check whether we can relax any constraints imposed by it
        // the rule is: if there is an active parent, then we cannot deactivate.  Otherwise, we can.
        
        // That's what's implemented below, but I don't think that's quite the right rule.  Is it?

        bool hasActiveParent = false;
        map< unsigned, vector< pair<unsigned,unsigned> > >::iterator parentEntityIt = _parentEntities[d].find(entityIndex);
        if (parentEntityIt != _parentEntities[d].end()) {
          vector< pair<unsigned,unsigned> > possibleParents = parentEntityIt->second;
          for (vector< pair<unsigned,unsigned> >::iterator possibleParentIt = possibleParents.begin(); possibleParentIt != possibleParents.end(); possibleParentIt++) {
            unsigned possibleParentIndex = possibleParentIt->first;
            if (_activeEntities[d].find(possibleParentIndex) != _activeEntities[d].end()) {
              hasActiveParent = true;
            }
          }
        }
        
        if (! hasActiveParent) {
          // then we can deactivate, provided that there are no children that properly have this as a constraining entity.
          // it's that "properly" that's the challenge.
          _activeEntities[d].erase(entityIndex);
        }
        
      }
    }
  }
  _activeCells.erase(cell->cellIndex());
}

set<unsigned> NewMesh::descendants(unsigned d, unsigned entityIndex) {
  set<unsigned> allDescendants;

  allDescendants.insert(entityIndex);
  if (_childEntities[d].find(entityIndex) != _childEntities[d].end()) {
    set<unsigned> unfollowedDescendants;
    for (unsigned i=0; i<_childEntities[d][entityIndex].size(); i++) {
      vector<unsigned> immediateChildren = _childEntities[d][entityIndex][i].second;
      unfollowedDescendants.insert(immediateChildren.begin(), immediateChildren.end());
    }
    for (set<unsigned>::iterator descIt=unfollowedDescendants.begin(); descIt!=unfollowedDescendants.end(); descIt++) {
      set<unsigned> myDescendants = descendants(d,*descIt);
      allDescendants.insert(myDescendants.begin(),myDescendants.end());
    }
  }
  
  return allDescendants;
}

pair< unsigned, set<unsigned> > NewMesh::determineEntityConstraints(unsigned d, unsigned entityIndex) {
  // pair.first:  index of the constraining entity.
  // pair.second: the set of all constrained entities (does not include the constraining entity).
  pair< unsigned, set<unsigned> > constraints;
  if (d==0) { // no constraints for vertices
    constraints.first = entityIndex;
    constraints.second = set<unsigned>();
    return constraints;
  }
  
  unsigned ancestor = eldestActiveAncestor(d,entityIndex);
  
  set<unsigned> markedEntities; // entities we already know about
  set<unsigned> elderDescendants = activeDescendants(d, ancestor);
  markedEntities.insert(elderDescendants.begin(), elderDescendants.end());
  
  unsigned constrainingIndex = ancestor;
  
  bool parentExists = true;
  while (parentExists) {
    map< unsigned, vector< pair<unsigned, unsigned> > >::iterator parentsEntry = _parentEntities[d].find(ancestor);
    if (parentsEntry != _parentEntities[d].end()) {
      vector< pair<unsigned,unsigned> > parents = parentsEntry->second;
      for (vector< pair<unsigned,unsigned> >::iterator parentIt = parents.begin(); parentIt != parents.end(); parentIt++) {
        unsigned parentEntityIndex = parentIt->first;
        unsigned refinementIndex = parentIt->second;
        vector< pair< RefinementPatternPtr, vector<unsigned> > > refinements = _childEntities[d][parentEntityIndex];
        bool parentHasAlternateChildren = false;
        for (unsigned refIndex=0; refIndex<refinements.size(); refIndex++) {
          if (refIndex != refinementIndex) {
            vector<unsigned> children = refinements[refIndex].second;
            for (unsigned childOrdinal=0; childOrdinal<children.size(); childOrdinal++) {
              unsigned childIndex = children[childOrdinal];
              set<unsigned> additionalConstrainedEntities = activeDescendantsNotInSet(d, childIndex, markedEntities);
              if (additionalConstrainedEntities.size() > 0) {
                parentHasAlternateChildren = true;
                markedEntities.insert(additionalConstrainedEntities.begin(),additionalConstrainedEntities.end());
              }
            }
          }
        }
        if (parentHasAlternateChildren) {
          // parent is the new constrainingIndex
          constrainingIndex = parentEntityIndex;
        }
        ancestor = parentEntityIndex; // this will end up meaning that by the time we look up in _parentEntities again, we're pointing to the coarsest of the possible parents.  I believe that's correct...
      }
    }
    else {
      parentExists = false;
    }
  }
  constraints.first = constrainingIndex;
  // if the constraingIndex is in the constrained set, remove it:
  markedEntities.erase(constrainingIndex);
  constraints.second = markedEntities;
  return constraints;
}

bool NewMesh::entityHasParent(unsigned d, unsigned entityIndex) {
  if (_parentEntities[d].find(entityIndex) == _parentEntities[d].end()) return false;
  return _parentEntities[d][entityIndex].size() > 0;
}

unsigned NewMesh::getActiveCellCount(unsigned int d, unsigned int entityIndex) {
  return _activeCellsForEntities[d][entityIndex].size();
}

const set< pair<unsigned,unsigned> > & NewMesh::getActiveCellIndices(unsigned d, unsigned entityIndex) {
  return _activeCellsForEntities[d][entityIndex];
}

NewMeshCellPtr NewMesh::getCell(unsigned cellIndex) {
  return _cells[cellIndex];
}

unsigned NewMesh::getEntityCount(unsigned int d) {
  return _entities[d].size();
}

unsigned NewMesh::getEntityParent(unsigned d, unsigned entityIndex, unsigned parentOrdinal) {
  TEUCHOS_TEST_FOR_EXCEPTION(! entityHasParent(d, entityIndex), std::invalid_argument, "entity does not have parent");
  return _parentEntities[d][entityIndex][parentOrdinal].first;
}

unsigned NewMesh::getFaceEdgeIndex(unsigned int faceIndex, unsigned int edgeOrdinalInFace) {
  return getSubEntityIndex(2, faceIndex, 1, edgeOrdinalInFace);
}

unsigned NewMesh::getSpaceDim() {
  return _spaceDim;
}
unsigned NewMesh::getSubEntityIndex(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim, unsigned int subEntityOrdinal) {
  shards::CellTopology *entityTopo = &_knownTopologies[_entityCellTopologyKeys[d][entityIndex]];
  set<unsigned> subEntityNodes;
  unsigned nodeCount = entityTopo->getNodeCount(subEntityDim, subEntityOrdinal);
  vector<unsigned> entityNodes = _canonicalEntityOrdering[d][entityIndex];
  for (unsigned nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++) {
    unsigned nodeIndexInEntity = entityTopo->getNodeMap(subEntityDim, subEntityOrdinal, nodeOrdinal);
    unsigned nodeIndexInMesh = entityNodes[nodeIndexInEntity];
    subEntityNodes.insert(nodeIndexInMesh);
  }
  if (_knownEntities[subEntityDim].find(subEntityNodes) == _knownEntities[d].end()) {
    cout << "sub-entity not found with vertices:\n";
    printVertices(subEntityNodes);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-entity not found");
  }
  return _knownEntities[subEntityDim][subEntityNodes];
}

bool NewMesh::getVertexIndex(const vector<double> &vertex, unsigned &vertexIndex, double tol) {
  vector<double> vertexForLowerBound;
  for (int d=0; d<_spaceDim; d++) {
    vertexForLowerBound.push_back(vertex[d]-tol);
  }
  
  map< vector<double>, unsigned >::iterator lowerBoundIt = _vertexMap.lower_bound(vertexForLowerBound);
  long bestMatchIndex = -1;
  double bestMatchDistance = tol;
  double xDist = 0; // xDist because vector<double> sorts according to the first entry: so we'll end up looking at
  // all the vertices that are near (x,...) in x...
  while ((lowerBoundIt != _vertexMap.end()) && (xDist < tol)) {
    double dist = 0;
    for (int d=0; d<_spaceDim; d++) {
      double ddist = (lowerBoundIt->first[d] - vertex[d]);
      dist += ddist * ddist;
    }
    dist = sqrt( dist );
    if (dist < bestMatchDistance) {
      bestMatchDistance = dist;
      bestMatchIndex = lowerBoundIt->second;
    }
    xDist = abs(lowerBoundIt->first[0] - vertex[0]);
    lowerBoundIt++;
  }
  if (bestMatchIndex == -1) {
    return false;
  } else {
    vertexIndex = bestMatchIndex;
    return true;
  }
}

unsigned NewMesh::getVertexIndexAdding(const vector<double> &vertex, double tol) {
  unsigned vertexIndex;
  if (getVertexIndex(vertex, vertexIndex, tol)) {
    return vertexIndex;
  }
  // if we get here, then we should add
  vertexIndex = _vertices.size();
  _vertices.push_back(vertex);
  
  if (_vertexMap.find(vertex) != _vertexMap.end() ) {
    cout << "Mesh error: attempting to add existing vertex.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Mesh error: attempting to add existing vertex");
  }
  _vertexMap[vertex] = vertexIndex;
  return vertexIndex;
}

// key: index in vertices; value: index in _vertices
vector<unsigned> NewMesh::getVertexIndices(const FieldContainer<double> &vertices) {
  double tol = 1e-14; // tolerance for vertex equality
  
  int numVertices = vertices.dimension(0);
  vector<unsigned> localToGlobalVertexIndex(numVertices);
  for (int i=0; i<numVertices; i++) {
    vector<double> vertex;
    for (int d=0; d<_spaceDim; d++) {
      vertex.push_back(vertices(i,d));
    }
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertex,tol);
  }
  return localToGlobalVertexIndex;
}

// key: index in vertices; value: index in _vertices
map<unsigned, unsigned> NewMesh::getVertexIndicesMap(const FieldContainer<double> &vertices) {
  map<unsigned, unsigned> vertexMap;
  vector<unsigned> vertexVector = getVertexIndices(vertices);
  unsigned numVertices = vertexVector.size();
  for (unsigned i=0; i<numVertices; i++) {
    vertexMap[i] = vertexVector[i];
  }
  return vertexMap;
}

vector<unsigned> NewMesh::getVertexIndices(const vector< vector<double> > &vertices) {
  double tol = 1e-14; // tolerance for vertex equality
  
  int numVertices = vertices.size();
  vector<unsigned> localToGlobalVertexIndex(numVertices);
  for (int i=0; i<numVertices; i++) {
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertices[i],tol);
  }
  return localToGlobalVertexIndex;
}

set<unsigned> NewMesh::getChildEntities(unsigned int d, unsigned int entityIndex) {
  set<unsigned> childIndices;
  if (d==0) return childIndices;
  if (_childEntities[d].find(entityIndex) == _childEntities[d].end()) return childIndices;
  vector< pair< RefinementPatternPtr, vector<unsigned> > > childEntries = _childEntities[d][entityIndex];
  for (vector< pair< RefinementPatternPtr, vector<unsigned> > >::iterator entryIt = childEntries.begin();
       entryIt != childEntries.end(); entryIt++) {
    childIndices.insert(entryIt->second.begin(),entryIt->second.end());
  }
  return childIndices;
}

unsigned NewMesh::getConstrainingEntityIndex(unsigned int d, unsigned int entityIndex) {
  if (_constrainingEntities[d].find(entityIndex) != _constrainingEntities[d].end()) {
    return _constrainingEntities[d][entityIndex];
  } else {
    return entityIndex;
  }
}

void NewMesh::printVertex(unsigned int vertexIndex) {
  cout << "vertex " << vertexIndex << ": (";
  for (unsigned d=0; d<_spaceDim; d++) {
    cout << _vertices[vertexIndex][d];
    if (d != _spaceDim-1) cout << ",";
  }
  cout << ")\n";
}

void NewMesh::printVertices(set<unsigned int> vertexIndices) {
  for (set<unsigned>::iterator indexIt=vertexIndices.begin(); indexIt!=vertexIndices.end(); indexIt++) {
    unsigned vertexIndex = *indexIt;
    printVertex(vertexIndex);
  }
}

void NewMesh::printEntityVertices(unsigned int d, unsigned int entityIndex) {
  vector<unsigned> entityVertices = _canonicalEntityOrdering[d][entityIndex];
  for (vector<unsigned>::iterator vertexIt=entityVertices.begin(); vertexIt !=entityVertices.end(); vertexIt++) {
    printVertex(*vertexIt);
  }
}

void NewMesh::refineCell(unsigned cellIndex, RefinementPatternPtr refPattern) {
  // TODO: worry about the case (currently unsupported in RefinementPattern) of children that do not share topology with the parent.  E.g. quad broken into triangles.  (3D has better examples.)
  
  NewMeshCellPtr cell = _cells[cellIndex];
  FieldContainer<double> cellNodes(cell->vertices().size(), _spaceDim);
  
  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(0); vertexIndex++) {
    for (int d=0; d<_spaceDim; d++) {
      cellNodes(vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }
  
  FieldContainer<double> vertices = refPattern->verticesForRefinement(cellNodes);
  map<unsigned, unsigned> localToGlobalVertexIndex = getVertexIndicesMap(vertices); // key: index in vertices; value: index in _vertices
  
  // get the children, as vectors of vertex indices:
  vector< vector<unsigned> > childVertices = refPattern->children(localToGlobalVertexIndex);
  
  int numChildren = childVertices.size();
  // this is where we assume all the children have same topology as parent:
  vector< CellTopoPtr > childTopos(numChildren,cell->topology());
  
  refineCellEntities(cell, refPattern);
  cell->setRefinementPattern(refPattern);
  
  addChildren(cell, childTopos, childVertices);
  deactivateCell(cell);
  
  set<unsigned> cellsAffected;
  cellsAffected.insert(cellIndex);
  vector<unsigned> childIndices = cell->getChildIndices();
  cellsAffected.insert(childIndices.begin(), childIndices.end());
  updateConstraintsForCells(cellsAffected);
}

void NewMesh::refineCellEntities(NewMeshCellPtr cell, RefinementPatternPtr refPattern) {
  // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities
  
  FieldContainer<double> cellNodes(1,cell->vertices().size(), _spaceDim);
  
  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(1); vertexIndex++) {
    for (int d=0; d<_spaceDim; d++) {
      cellNodes(0,vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }
  
  vector< RefinementPatternRecipe > relatedRecipes = refPattern->relatedRecipes();
  if (relatedRecipes.size()==0) {
    RefinementPatternRecipe recipe;
    vector<unsigned> initialCell;
    recipe.push_back(make_pair(refPattern,vector<unsigned>()));
    relatedRecipes.push_back(recipe);
  }
  
  // TODO generalize the below code to apply recipes instead of just the refPattern...
  
  CellTopoPtr cellTopo = cell->topology();
  for (unsigned d=1; d<_spaceDim; d++) {
    unsigned subcellCount = cellTopo->getSubcellCount(d);
    for (unsigned subcord = 0; subcord < subcellCount; subcord++) {
      RefinementPatternPtr subcellRefPattern = refPattern->patternForSubcell(d, subcord);
      FieldContainer<double> refinedNodes = subcellRefPattern->refinedNodes(); // refinedNodes implicitly assumes that all child topos are the same
      unsigned childCount = refinedNodes.dimension(0);
      if (childCount==1) continue; // we already have the appropriate entities and parent relationships defined...
      
//      cout << "Refined nodes:\n" << refinedNodes;
      
      vector<unsigned> parentIndices;
      unsigned parentIndex = cell->entityIndex(d, subcord);
      // if we ever allow multiple parentage, then we'll need to record things differently in both _childEntities and _parentEntities
      // (and the if statement just below will need to change in a corresponding way, indexed by the particular refPattern in question maybe
      if (_childEntities[d].find(parentIndex) == _childEntities[d].end()) {
        vector<unsigned> childEntityIndices(childCount);
        for (unsigned childIndex=0; childIndex<childCount; childIndex++) {
          unsigned nodeCount = refinedNodes.dimension(1);
          FieldContainer<double> nodesOnSubcell(nodeCount,d);
          for (int nodeIndex=0; nodeIndex<nodeCount; nodeIndex++) {
            for (int dimIndex=0; dimIndex<d; dimIndex++) {
              nodesOnSubcell(nodeIndex,dimIndex) = refinedNodes(childIndex,nodeIndex,dimIndex);
            }
          }
//          cout << "nodesOnSubcell:\n" << nodesOnSubcell;
          FieldContainer<double> nodesOnRefCell(nodeCount,_spaceDim);
          CellTools<double>::mapToReferenceSubcell(nodesOnRefCell, nodesOnSubcell, d, subcord, *cellTopo);
//          cout << "nodesOnRefCell:\n" << nodesOnRefCell;
          FieldContainer<double> physicalNodes(1,nodeCount,_spaceDim);
          // map to physical space:
          CellTools<double>::mapToPhysicalFrame(physicalNodes, nodesOnRefCell, cellNodes, *cellTopo);
//          cout << "physicalNodes:\n" << physicalNodes;
          
//          cout << "cellNodes:\n" << cellNodes;
          
          // add vertices as necessary and get their indices
          physicalNodes.resize(nodeCount,_spaceDim);
          vector<unsigned> childEntityVertices = getVertexIndices(physicalNodes); // key: index in physicalNodes; value: index in _vertices
          
          unsigned entityPermutation;
          shards::CellTopology childTopo = cellTopo->getCellTopologyData(d, subcord);
          unsigned childEntityIndex = addEntity(childTopo, childEntityVertices, entityPermutation);
//          cout << "for d=" << d << ", entity index " << childEntityIndex << " is child of " << parentIndex << endl;
          _parentEntities[d][childEntityIndex] = vector< pair<unsigned,unsigned> >(1, make_pair(parentIndex,0)); // TODO: this is where we want to fill in a proper list of possible parents once we work through recipes
          childEntityIndices[childIndex] = childEntityIndex;
          set< pair<unsigned, unsigned> > parentActiveCells = _activeCellsForEntities[d][parentIndex];
        }
        _childEntities[d][parentIndex] = vector< pair<RefinementPatternPtr,vector<unsigned> > >(1, make_pair(subcellRefPattern, childEntityIndices) ); // TODO: this also needs to change when we work through recipes.  Note that the correct parent will vary here...  i.e. in the anisotropic case, the child we're ultimately interested in will have an anisotropic parent, and *its* parent would be the bigger guy referred to here.
      }
    }
  }
}

void NewMesh::updateConstraintsForCells(const set<unsigned> &cellIndices) {
  for (unsigned d=1; d<_spaceDim; d++) {
    set<unsigned> entitiesEncountered;
    map< unsigned, set<unsigned > > constraints;
    for (set<unsigned>::iterator cellIt=cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
      NewMeshCellPtr cell = _cells[*cellIt];
      unsigned numEntities = cell->topology()->getSubcellCount(d);
      for (unsigned entityOrdinal=0; entityOrdinal < numEntities; entityOrdinal++) {
        unsigned entityIndex = cell->entityIndex(d, entityOrdinal);
        if (getActiveCellCount(d, entityIndex) > 0) {
          if (entitiesEncountered.find(entityIndex) == entitiesEncountered.end()) {
            pair<unsigned, set<unsigned> > constraint = determineEntityConstraints(d, entityIndex);
            constraints.insert(constraint);
            entitiesEncountered.insert(constraint.second.begin(), constraint.second.end());
            entitiesEncountered.insert(constraint.first);
          }
        }
      }
    }
    for (map< unsigned, set<unsigned > >::iterator constraintIt = constraints.begin();
         constraintIt != constraints.end(); constraintIt++) {
      unsigned constrainingEntity = constraintIt->first;
      set<unsigned> *constrainedEntities = &(constraintIt->second);
      _constrainingEntities[d].erase(constrainingEntity); // if prior to update, this entity was constrained, clear that out
      _constrainedEntities[d][constrainingEntity] = *constrainedEntities;
      for (set<unsigned>::iterator constrainedIt=constrainedEntities->begin(); constrainedIt != constrainedEntities->end(); constrainedIt++) {
        unsigned constrainedEntityIndex = *constrainedIt;
        // if prior to update, this entity functioned as a constraint, clear that out
        _constrainedEntities[d].erase(constrainedEntityIndex);
        _constrainingEntities[d][constrainedEntityIndex] = constrainingEntity;
//        cout << "for d=" << d << ", entity " << constrainedEntityIndex << " is constrained by " << constrainingEntity << endl;
      }
    }
  }
}