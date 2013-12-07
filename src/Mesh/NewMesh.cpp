//
//  NewMesh.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#include "NewMesh.h"

#include "CamelliaCellTools.h"

NewMesh::NewMesh(NewMeshGeometryPtr meshGeometry) {
  RefinementPattern::initializeAnisotropicRelationships(); // not sure this is the optimal place for this call
  
  _vertices = meshGeometry->vertices();
  _spaceDim = _vertices[0].size();
  
  _entities = vector< vector< set< unsigned > > >(_spaceDim);
  _knownEntities = vector< map< set<unsigned>, unsigned > >(_spaceDim); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< map< unsigned, vector<unsigned> > >(_spaceDim);
  _activeCellsForEntities = vector< map< unsigned, set< pair<unsigned, unsigned> > > >(_spaceDim); // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _activeEntities = vector< set<unsigned > >(_spaceDim);
  _constrainingEntities = vector< map< unsigned, unsigned > >(_spaceDim); // map from broken entity to the whole (constraining) one.
  _parentEntities = vector< map< unsigned, vector<unsigned> > >(_spaceDim); // map to possible parents
  _childEntities = vector< map< unsigned, vector< pair<RefinementPatternPtr, vector<unsigned> > > > >(_spaceDim);
  
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

set<unsigned> NewMesh::activeAncestors(unsigned d, unsigned entityIndex) {
  set<unsigned> ancestors;
  
  bool hasActiveParent = false;
  while (hasActiveParent) {
    map< unsigned, vector<unsigned> >::iterator parentEntityIt = _parentEntities[d].find(entityIndex);
    if (parentEntityIt != _parentEntities[d].end()) {
      vector<unsigned> possibleParents = parentEntityIt->second;
      for (vector<unsigned>::iterator possibleParentIt = possibleParents.begin(); possibleParentIt != possibleParents.end(); possibleParentIt++) {
        unsigned possibleParentIndex = *possibleParentIt;
        if (_activeEntities[d].find(possibleParentIndex) != _activeEntities[d].end()) {
          hasActiveParent = true;
          ancestors.insert(possibleParentIndex);
          entityIndex = possibleParentIndex; // the new entityIndex will be the last found active parent (which is the finest one)
        }
      }
    }
  }
  return ancestors;
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
  for (int nodeIndex=0; nodeIndex<entityVertices.size(); nodeIndex++) {
    nodeSet.insert(entityVertices[nodeIndex]);
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
        map< unsigned, vector<unsigned> >::iterator parentEntityIt = _parentEntities[d].find(entityIndex);
        if (parentEntityIt != _parentEntities[d].end()) {
          vector<unsigned> possibleParents = parentEntityIt->second;
          for (vector<unsigned>::iterator possibleParentIt = possibleParents.begin(); possibleParentIt != possibleParents.end(); possibleParentIt++) {
            unsigned possibleParentIndex = *possibleParentIt;
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

unsigned NewMesh::findConstrainingEntity(unsigned d, unsigned entityIndex) {
  if (d==0) return entityIndex; // no constraint for vertices
  
  // TODO implement this...
}

unsigned NewMesh::getVertexIndexAdding(const vector<double> &vertex, double tol) {
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
  // if we get here and bestMatchIndex == -1, then we should add
  if (bestMatchIndex == -1) {
    bestMatchIndex = _vertices.size();
    _vertices.push_back(vertex);
    
    if (_vertexMap.find(vertex) != _vertexMap.end() ) {
      cout << "Mesh error: attempting to add existing vertex.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Mesh error: attempting to add existing vertex");
    }
    _vertexMap[vertex] = bestMatchIndex;
  }
  return bestMatchIndex;
}

// key: index in vertices; value: index in _vertices
map<unsigned, unsigned> NewMesh::getVertexIndices(const FieldContainer<double> &vertices) {
  double tol = 1e-14; // tolerance for vertex equality
  
  map<unsigned, unsigned> localToGlobalVertexIndex;
  int numVertices = vertices.dimension(0);
  for (int i=0; i<numVertices; i++) {
    vector<double> vertex;
    for (int d=0; d<_spaceDim; d++) {
      vertex.push_back(vertices(i,d));
    }
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertex,tol);
  }
  return localToGlobalVertexIndex;
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
  map<unsigned, unsigned> localToGlobalVertexIndex = getVertexIndices(vertices); // key: index in vertices; value: index in _vertices
  
  // get the children, as vectors of vertex indices:
  vector< vector<unsigned> > childVertices = refPattern->children(localToGlobalVertexIndex);
  
  int numChildren = childVertices.size();
  // this is where we assume all the children have same topology as parent:
  vector< CellTopoPtr > childTopos(numChildren,cell->topology());
  
  refineCellEntities(cell, refPattern);
  cell->setRefinementPattern(refPattern);
  
  addChildren(cell, childTopos, childVertices);
  deactivateCell(cell);
}

void NewMesh::refineCellEntities(NewMeshCellPtr cell, RefinementPatternPtr refPattern) {
  // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities
  
  FieldContainer<double> cellNodes(1,cell->vertices().size(), _spaceDim);
  
  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(0); vertexIndex++) {
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
          FieldContainer<double> nodesOnRefCell(nodeCount,_spaceDim);
          CellTools<double>::mapToReferenceSubcell(nodesOnRefCell, nodesOnSubcell, d, subcord, *cellTopo);
          FieldContainer<double> physicalNodes(1,nodeCount,_spaceDim);
          // map to physical space:
          CellTools<double>::mapToPhysicalFrame(physicalNodes, nodesOnRefCell, cellNodes, *cellTopo);
          
          // add vertices as necessary and get their indices
          physicalNodes.resize(nodeCount,_spaceDim);
          map<unsigned, unsigned> localToGlobalVertexIndex = getVertexIndices(physicalNodes); // key: index in physicalNodes; value: index in _vertices
          // could save ourselves the following few lines if getVertexIndices return a vector, which would be sensible
          vector<unsigned> childEntityVertices(nodeCount);
          for (unsigned node=0; node<nodeCount; node++) {
            childEntityVertices[node] = localToGlobalVertexIndex[node];
          }
          
          unsigned entityPermutation;
          shards::CellTopology childTopo = cellTopo->getCellTopologyData(d, subcord);
          unsigned childEntityIndex = addEntity(childTopo, childEntityVertices, entityPermutation);
          _parentEntities[d][childEntityIndex] = vector<unsigned>(1, parentIndex); // TODO: this is where we want to fill in a proper list of possible parents once we work through recipes
          childEntityIndices[childIndex] = childEntityIndex;
          set< pair<unsigned, unsigned> > parentActiveCells = _activeCellsForEntities[d][parentIndex];
        }
        _childEntities[d][parentIndex] = vector< pair<RefinementPatternPtr,vector<unsigned> > >(1, make_pair(subcellRefPattern, childEntityIndices) ); // TODO: this also needs to change when we work through recipes.  Note that the correct parent will vary here...  i.e. in the anisotropic case, the child we're ultimately interested in will have an anisotropic parent, and *its* parent would be the bigger guy referred to here.

      }
    }
  }
  
}
