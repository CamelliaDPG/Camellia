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
  _activeEntities = vector< map< unsigned, ActiveEntityListEntry* > >(_spaceDim);
//  _cellsForEntities = vector< map< unsigned, set< pair<unsigned, unsigned> > > >(_spaceDim); // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
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
  vector< vector< ActiveEntityListEntry* > > entries(_spaceDim);
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    cellEntityPermutations.push_back(map<unsigned, unsigned>());
    int entityCount = cellTopo->getSubcellCount(d);
    entries[d] = vector<ActiveEntityListEntry *>(entityCount,NULL);
    for (int j=0; j<entityCount; j++) {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      unsigned entityIndex, entityPermutation;
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      vector< unsigned > nodes;
      set< unsigned > nodeSet;
      for (int node=0; node<entityNodeCount; node++) {
        unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
        nodes.push_back(cellVertices[nodeIndexInCell]);
        nodeSet.insert(cellVertices[nodeIndexInCell]);
      }
      
      if (_knownEntities[d].find(nodeSet) == _knownEntities[d].end()) {
        // new entity
        entityIndex = _entities[d].size();
        _entities[d].push_back(nodeSet);
        _knownEntities[d][nodeSet] = entityIndex;
        _canonicalEntityOrdering[d][entityIndex] = nodes;
        entityPermutation = 0;
      } else {
        // existing entity
        entityIndex = _knownEntities[d][nodeSet];
        entityPermutation = CamelliaCellTools::permutationMatchingOrder(cellTopo->getCellTopologyData(d, j), _canonicalEntityOrdering[d][entityIndex], nodes);
      }
      cellEntityPermutations[d][entityIndex] = entityPermutation;
      ActiveEntityListEntry* newListEntry = new ActiveEntityListEntry(); // TODO: add destructor with delete()s to mirror this new() -- also delete() on cell deactivation
      ActiveEntityListEntry *firstEntry, *lastEntry;
      newListEntry->cellIndex = cellIndex;
      newListEntry->subcellOrdinal = j;
      entries[d][j] = newListEntry;
      if ( _activeEntities[d].find(entityIndex) == _activeEntities[d].end() ) {
        _activeEntities[d][entityIndex] = newListEntry;
        firstEntry = newListEntry;
      } else {
        firstEntry = _activeEntities[d][entityIndex];
      }
      lastEntry = firstEntry;
      while (lastEntry->nextEntry != lastEntry) {
        lastEntry = lastEntry->nextEntry;
      }
      lastEntry->nextEntry = newListEntry;
      newListEntry->nextEntry = _activeEntities[d][entityIndex];
    }
  }
  
  NewMeshCellPtr cell = Teuchos::rcp( new NewMeshCell(cellTopo, cellVertices, cellEntityPermutations) );
  _cells.push_back(cell);
  return _cells.size() - 1;
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
    
    //    cout << "Added vertex " << bestMatchIndex << " (" << x << "," << y << ")\n";
  }
  return bestMatchIndex;
}

map<unsigned, unsigned> NewMesh::getVertexIndices(const FieldContainer<double> &vertices) {
  double tol = 1e-12; // tolerance for vertex equality
  
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
}