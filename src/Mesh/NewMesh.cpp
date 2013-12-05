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
  _vertices = meshGeometry->vertices();
  _spaceDim = _vertices[0].size();
  
  _entities = vector< vector< set< unsigned > > >(_spaceDim);
  _knownEntities = vector< map< set<unsigned>, unsigned > >(_spaceDim); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< map< unsigned, vector<unsigned> > >(_spaceDim);
  _activeCellsForEntities = vector< map< unsigned, set< pair<unsigned, unsigned> > > >(_spaceDim); // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _constrainingEntities = vector< map< unsigned, unsigned > >(_spaceDim); // map from broken entity to the whole (constraining) one.
  
  TEUCHOS_TEST_FOR_EXCEPTION(meshGeometry->cellTopos().size() != meshGeometry->elementVertices().size(), std::invalid_argument,
                             "length of cellTopos != length of elementVertices");
  
  int numElements = meshGeometry->cellTopos().size();
  
  for (int i=0; i<numElements; i++) {
    CellTopoPtr cellTopo = meshGeometry->cellTopos()[i];
    vector< unsigned > cellVertices = meshGeometry->elementVertices()[i];
    addCell(cellTopo, cellVertices);
  }
}

unsigned NewMesh::addCell(CellTopoPtr cellTopo, const vector<unsigned> &cellVertices) {
  vector< map< unsigned, unsigned > > cellEntityPermutations;
  unsigned cellIndex = _cells.size();
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    cellEntityPermutations.push_back(map<unsigned, unsigned>());
    int entityCount = cellTopo->getSubcellCount(d);
    for (int j=0; j<entityCount; j++) {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      unsigned entityIndex, entityPermutation;
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      vector< unsigned > nodes;
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
        nodes.push_back(cellVertices[nodeIndexInCell]);
      }

      addEntity(cellTopo->getCellTopologyData(d, j), nodes, entityPermutation);

      cellEntityPermutations[d][entityIndex] = entityPermutation;
      _activeCellsForEntities[d][entityIndex].insert(make_pair(cellIndex,j));
    }
  }
  
  NewMeshCellPtr cell = Teuchos::rcp( new NewMeshCell(cellTopo, cellVertices, cellEntityPermutations, cellIndex) );
  _cells.push_back(cell);
  return _cells.size() - 1;
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
        // how do we know that?  what is the rule?
        
        // vertices are never constraining entities -- i.e. we can skip here
        // edges are constraining entities if and only if they remain active but are broken in some cells
        // faces are constraining entities if
        
        
        // edges are constrained by their last (most distant) active ancestor
        //
      }
    }
  }
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
  
  CellTopoPtr cellTopo = cell->topology();
  
  vector< vector< RefinementPatternPtr > > refPatternForSubcell(_spaceDim+1); // outer vector: dimension; inner: subcord
  refPatternForSubcell[_spaceDim] = vector< RefinementPatternPtr >(1,refPattern);
  
  for (unsigned d=_spaceDim-1; d>0; d--) {
    unsigned subcellCount = cellTopo->getSubcellCount(d);
    refPatternForSubcell[d] = vector< RefinementPatternPtr >(subcellCount);
    for (unsigned subcord=0; subcord<subcellCount; subcord++) {
      // TODO: work out how to implement filling in the RefinementPattern for each subcell.
      // idea: look at previous iterate's refPattern->sideRefinementPatterns().  Trick is to figure out the map from sideOrdinal in subcell to subcord in cell
    }
  }

  // let's start by implementing for sides.  We'll generalize from there
  unsigned d = _spaceDim - 1;
  unsigned sideCount = cellTopo->getSubcellCount(d);
  for (unsigned sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++) {
    RefinementPatternPtr sideRefPattern = refPattern->sideRefinementPatterns()[sideOrdinal];
    FieldContainer<double> refinedNodes = sideRefPattern->refinedNodes(); // refinedNodes implicitly assumes that all child topos are the same
    unsigned childCount = refinedNodes.dimension(0);
    if (childCount==1) continue; // we already have the appropriate entities and parent relationships defined...
    
    for (unsigned childIndex=0; childIndex<childCount; childIndex++) {
      unsigned nodeCount = refinedNodes.dimension(1);
      FieldContainer<double> nodesOnSide(nodeCount,d);
      for (int nodeIndex=0; nodeIndex<nodeCount; nodeIndex++) {
        for (int dimIndex=0; dimIndex<d; dimIndex++) {
          nodesOnSide(nodeIndex,dimIndex) = refinedNodes(childIndex,nodeIndex,dimIndex);
        }
      }
      FieldContainer<double> nodesOnRefCell(nodeCount,_spaceDim);
      CellTools<double>::mapToReferenceSubcell(nodesOnRefCell, refinedNodes, d, sideOrdinal, *cellTopo);
      FieldContainer<double> physicalNodes(1,nodeCount,_spaceDim);
      // map to physical space:
      CellTools<double>::mapToPhysicalFrame(physicalNodes, nodesOnRefCell, cellNodes, *cellTopo);
    }
  }
  
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    int entityCount = cellTopo->getSubcellCount(d);
    for (int j=0; j<entityCount; j++) {
      unsigned entityIndex, entityPermutation;
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      vector< unsigned > nodes;
      set< unsigned > nodeSet;
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
        nodes.push_back(cell->vertices()[nodeIndexInCell]);
      }
      shards::CellTopology entityTopo = cellTopo->getCellTopologyData(d, j);
      entityIndex = addEntity(entityTopo, nodes, entityPermutation); // this is actually just a lookup
      
    }
  }

}
