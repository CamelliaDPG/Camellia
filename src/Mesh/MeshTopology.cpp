//
//  MeshTopology.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#include "MeshTopology.h"

#include "MeshTransformationFunction.h"

#include "CamelliaCellTools.h"

void MeshTopology::init(unsigned spaceDim) {
  RefinementPattern::initializeAnisotropicRelationships(); // not sure this is the optimal place for this call
  
  _spaceDim = spaceDim;
  _entities = vector< vector< set< unsigned > > >(_spaceDim);
  _knownEntities = vector< map< set<unsigned>, unsigned > >(_spaceDim); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< map< unsigned, vector<unsigned> > >(_spaceDim);
  _activeCellsForEntities = vector< map< unsigned, set< pair<unsigned, unsigned> > > >(_spaceDim); // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _activeSidesForEntities = vector< map< unsigned, set< unsigned > > >(_spaceDim);
  _parentEntities = vector< map< unsigned, vector< pair<unsigned, unsigned> > > >(_spaceDim); // map to possible parents
  _childEntities = vector< map< unsigned, vector< pair<RefinementPatternPtr, vector<unsigned> > > > >(_spaceDim);
  _entityCellTopologyKeys = vector< map< unsigned, unsigned > >(_spaceDim);
}

MeshTopology::MeshTopology(unsigned spaceDim) {
  init(spaceDim);
}

MeshTopology::MeshTopology(MeshGeometryPtr meshGeometry) {
  unsigned spaceDim = meshGeometry->vertices()[0].size();

  init(spaceDim);
  _vertices = meshGeometry->vertices();
  
  for (int vertexIndex=0; vertexIndex<_vertices.size(); vertexIndex++) {
    _vertexMap[_vertices[vertexIndex]] = vertexIndex;
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(meshGeometry->cellTopos().size() != meshGeometry->elementVertices().size(), std::invalid_argument,
                             "length of cellTopos != length of elementVertices");
  
  int numElements = meshGeometry->cellTopos().size();
  
  for (int i=0; i<numElements; i++) {
    CellTopoPtr cellTopo = meshGeometry->cellTopos()[i];
    vector< unsigned > cellVertices = meshGeometry->elementVertices()[i];
    addCell(cellTopo, cellVertices);
  }
}

unsigned MeshTopology::activeCellCount() {
  return _activeCells.size();
}

const set<unsigned> & MeshTopology::getActiveCellIndices() {
  return _activeCells;
}

CellPtr MeshTopology::addCell(CellTopoPtr cellTopo, const vector<vector<double> > &cellVertices) {
  vector<unsigned> vertexIndices = getVertexIndices(cellVertices);
  unsigned cellIndex = addCell(cellTopo, vertexIndices);
  return _cells[cellIndex];
}

unsigned MeshTopology::addCell(CellTopoPtr cellTopo, const vector<unsigned> &cellVertices, unsigned parentCellIndex) {
  vector< map< unsigned, unsigned > > cellEntityPermutations;
  unsigned cellIndex = _cells.size();
  
  set<unsigned> ancestralCellIndices; // includes the cell itself
  ancestralCellIndices.insert(cellIndex);
  if (parentCellIndex != -1) {
    ancestralCellIndices.insert(parentCellIndex);
    CellPtr parent = getCell(parentCellIndex);
    while (parent->getParent().get() != NULL) {
      parent = parent->getParent();
      ancestralCellIndices.insert(parent->cellIndex());
    }
  }
  
  vector< vector<unsigned> > cellEntityIndices(_spaceDim); // subcdim, subcord
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    cellEntityPermutations.push_back(map<unsigned, unsigned>());
    int entityCount = cellTopo->getSubcellCount(d);
    cellEntityIndices[d] = vector<unsigned>(entityCount);
    for (int j=0; j<entityCount; j++) {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      unsigned entityIndex, entityPermutation;
      vector< unsigned > nodes;
      if (d != 0) {
        int entityNodeCount = cellTopo->getNodeCount(d, j);
        for (int node=0; node<entityNodeCount; node++) {
          unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
          nodes.push_back(cellVertices[nodeIndexInCell]);
        }
      } else {
        nodes.push_back(cellVertices[j]);
      }
      
      entityIndex = addEntity(cellTopo->getCellTopologyData(d, j), nodes, entityPermutation);
      cellEntityIndices[d][j] = entityIndex;

      cellEntityPermutations[d][entityIndex] = entityPermutation;
      _activeCellsForEntities[d][entityIndex].insert(make_pair(cellIndex,j));
    }
  }
  CellPtr cell = Teuchos::rcp( new Cell(cellTopo, cellVertices, cellEntityPermutations, cellIndex, cellEntityIndices) );
  _cells.push_back(cell);
  _activeCells.insert(cellIndex);
  _rootCells.insert(cellIndex); // will remove if a parent relationship is established

  // set neighbors:
  unsigned sideDim = _spaceDim - 1;
  unsigned sideCount = cellTopo->getSideCount();
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
    unsigned activeCellCountForSide = getActiveCellCount(sideDim, sideEntityIndex);
    if (activeCellCountForSide == 2) { // compatible neighbors
      set< pair<unsigned,unsigned> >::iterator neighborIt = _activeCellsForEntities[sideDim][sideEntityIndex].begin();
      pair<unsigned,unsigned> firstNeighbor  = *neighborIt++;
      pair<unsigned,unsigned> secondNeighbor = *neighborIt;
      CellPtr firstCell = _cells[firstNeighbor.first];
      CellPtr secondCell = _cells[secondNeighbor.first];
      firstCell->setNeighbor(firstNeighbor.second, secondNeighbor.first, secondNeighbor.second);
      secondCell->setNeighbor(secondNeighbor.second, firstNeighbor.first, firstNeighbor.second);
      if (_boundarySides.find(sideEntityIndex) != _boundarySides.end()) {
        if (_childEntities[sideDim].find(sideEntityIndex) != _childEntities[sideDim].end()) {
          cout << "Unhandled case: boundary side acquired neighbor after being refined.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled case: boundary side acquired neighbor after being refined");
        }
        _boundarySides.erase(sideEntityIndex);
      }
    } else if (activeCellCountForSide > 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Too many cells for side");
    } else if (_activeCellsForEntities[sideDim][sideEntityIndex].size() == 1) { // just this side
      if (parentCellIndex == -1) { // for now anyway, we are on the boundary...
        _boundarySides.insert(sideEntityIndex);
      } else {
        vector< pair<unsigned, unsigned> > sideAncestry = getConstrainingSideAncestry(sideEntityIndex);
        // the last entry, if any, should refer to an active cell's side...
        if (sideAncestry.size() > 0) {
          unsigned sideAncestorIndex = sideAncestry[sideAncestry.size()-1].first;
          set< pair<unsigned, unsigned> > activeCellEntrySet = _activeCellsForEntities[sideDim][sideAncestorIndex];
          if (activeCellEntrySet.size() != 1) {
            cout << "Internal error: activeCellEntrySet does not have the expected size.\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Internal error: activeCellEntrySet does not have the expected size.\n");
          }
          pair<unsigned,unsigned> activeCellEntry = *(activeCellEntrySet.begin());
          unsigned neighborCellIndex = activeCellEntry.first;
          unsigned sideIndexInNeighbor = activeCellEntry.second;
          cell->setNeighbor(sideOrdinal, neighborCellIndex, sideIndexInNeighbor);
        }
      }
    }
    
    for (int d=0; d<sideDim; d++) {
      set<unsigned> sideSubcellIndices = getEntitiesForSide(sideEntityIndex, d);
      for (set<unsigned>::iterator subcellIt = sideSubcellIndices.begin(); subcellIt != sideSubcellIndices.end(); subcellIt++) {
        unsigned subcellEntityIndex = *subcellIt;
        _activeSidesForEntities[d][subcellEntityIndex].insert(sideEntityIndex);
      }
    }
  }
  
  return cellIndex;
}

void MeshTopology::addEdgeCurve(pair<unsigned,unsigned> edge, ParametricCurvePtr curve) {
  // note: does NOT update the MeshTransformationFunction.  That's caller's responsibility,
  // because we don't know whether there are more curves coming for the affected elements.
  
  unsigned edgeDim = 1;
  set<unsigned> edgeNodes;
  edgeNodes.insert(edge.first);
  edgeNodes.insert(edge.second);
  
  if (_knownEntities[edgeDim].find(edgeNodes) == _knownEntities[edgeDim].end() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "edge not found.");
  }
  unsigned edgeIndex = _knownEntities[edgeDim][edgeNodes];
  if (getChildEntities(edgeDim, edgeIndex).size() > 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setting curves along broken edges not supported.  Should set for each piece separately.");
  }
  
  // check that the curve agrees with the vertices in the mesh:
  vector<double> v0 = getVertex(edge.first);
  vector<double> v1 = getVertex(edge.second);
  
  int spaceDim = v0.size();
  FieldContainer<double> curve0(spaceDim);
  FieldContainer<double> curve1(spaceDim);
  curve->value(0, curve0(0), curve0(1));
  curve->value(1, curve1(0), curve1(1));
  double maxDiff = 0;
  double tol = 1e-14;
  for (int d=0; d<spaceDim; d++) {
    maxDiff = max(maxDiff, abs(curve0(d)-v0[d]));
    maxDiff = max(maxDiff, abs(curve1(d)-v1[d]));
  }
  if (maxDiff > tol) {
    cout << "Error: curve's endpoints do not match edge vertices (maxDiff in coordinates " << maxDiff << ")" << endl;
    cout << "curve0:\n" << curve0;
    cout << "v0: (" << v0[0] << ", " << v0[1] << ")\n";
    cout << "curve1:\n" << curve1;
    cout << "v1: (" << v1[0] << ", " << v1[1] << ")\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Curve does not match vertices");
  }
  
  _edgeToCurveMap[edge] = curve;
  
  set< pair<unsigned, unsigned> > cellIDsForEdge = _activeCellsForEntities[edgeDim][edgeIndex];
//  (cellIndex, entityIndexInCell)
  unsigned cellID;
  if (cellIDsForEdge.size() != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "right now, only edges belonging to exactly one cell are supported by curvilinear geometry...");
  }
  for (set< pair<unsigned, unsigned> >::iterator edgeCellIt = cellIDsForEdge.begin(); edgeCellIt != cellIDsForEdge.end(); edgeCellIt++) {
    cellID = edgeCellIt->first;
  }
  _cellIDsWithCurves.insert(cellID);
}

unsigned MeshTopology::addEntity(const shards::CellTopology &entityTopo, const vector<unsigned> &entityVertices, unsigned &entityPermutation) {
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
//    cout << "MeshTopology::addEntity: added entity of dimension " << d << " with index " << entityIndex << " and vertices:" << endl;
//    printVertices(nodeSet);
  } else {
    // existing entity
    entityIndex = _knownEntities[d][nodeSet];
    entityPermutation = CamelliaCellTools::permutationMatchingOrder(entityTopo, _canonicalEntityOrdering[d][entityIndex], entityVertices);
  }
  return entityIndex;
}

void MeshTopology::addChildren(CellPtr parentCell, const vector< CellTopoPtr > &childTopos, const vector< vector<unsigned> > &childVertices) {
  int numChildren = childTopos.size();
  TEUCHOS_TEST_FOR_EXCEPTION(numChildren != childVertices.size(), std::invalid_argument, "childTopos and childVertices must be the same size");
  vector< CellPtr > children;
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    unsigned cellIndex = addCell(childTopos[childIndex], childVertices[childIndex],parentCell->cellIndex());
    children.push_back(_cells[cellIndex]);
    _rootCells.erase(cellIndex);
  }
  parentCell->setChildren(children);
  
  // now, set children's neighbors to agree with parent, if the children don't have their own peer neighbors
  
//  unsigned numSides = parentCell->topology()->getSideCount();
////  unsigned sideDim = _spaceDim - 1;
//  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
//    vector< pair<unsigned, unsigned> > childrenForSide = parentCell->childrenForSide(sideOrdinal);
//    for (vector< pair<unsigned, unsigned> >::iterator childIt = childrenForSide.begin(); childIt != childrenForSide.end(); childIt++) {
//      unsigned childIndex = childIt->first;
//      unsigned childSideOrdinal = childIt->second;
//      CellPtr child = _cells[childIndex];
////      unsigned childSideEntityIndex = child->entityIndex(sideDim, childSideOrdinal);
////      if (_activeCellsForEntities[sideDim][childSideEntityIndex].size()==1) {
//      if (child->getNeighbor(childSideOrdinal).first == -1) { // i.e. child hasn't yet had a neighbor set here...
//        // then child should inherit neighbor relationship from parent
//        pair< unsigned, unsigned > neighborInfo = parentCell->getNeighbor(sideOrdinal);
//        child->setNeighbor(childSideOrdinal, neighborInfo.first, neighborInfo.second);
//      }
//    }
//  }
}

unsigned MeshTopology::cellCount() {
  return _cells.size();
}

bool MeshTopology::cellHasCurvedEdges(unsigned cellIndex) {
  CellPtr cell = getCell(cellIndex);
  unsigned edgeCount = cell->topology()->getEdgeCount();
  unsigned edgeDim = 1;
  for (int edgeOrdinal=0; edgeOrdinal<edgeCount; edgeOrdinal++) {
    unsigned edgeIndex = cell->entityIndex(edgeDim, edgeOrdinal);
    unsigned v0 = _canonicalEntityOrdering[edgeDim][edgeIndex][0];
    unsigned v1 = _canonicalEntityOrdering[edgeDim][edgeIndex][1];
    pair<unsigned, unsigned> edge = make_pair(v0, v1);
    pair<unsigned, unsigned> edgeReversed = make_pair(v1, v0);
    if (_edgeToCurveMap.find(edge) != _edgeToCurveMap.end()) {
      return true;
    }
    if (_edgeToCurveMap.find(edgeReversed) != _edgeToCurveMap.end()) {
      return true;
    }
  }
  return false;
}

vector<double> MeshTopology::getCellCentroid(unsigned cellIndex) {
  // average of the cell vertices
  vector<double> centroid(_spaceDim);
  CellPtr cell = getCell(cellIndex);
  unsigned vertexCount = cell->vertices().size();
  for (unsigned vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++) {
    unsigned vertexIndex = cell->vertices()[vertexOrdinal];
    for (unsigned d=0; d<_spaceDim; d++) {
      centroid[d] += _vertices[vertexIndex][d];
    }
  }
  for (unsigned d=0; d<_spaceDim; d++) {
    centroid[d] /= vertexCount;
  }
  return centroid;
}

void MeshTopology::deactivateCell(CellPtr cell) {
//  cout << "deactivating cell " << cell->cellIndex() << endl;
  CellTopoPtr cellTopo = cell->topology();
  for (int d=0; d<_spaceDim; d++) { // start with vertices, and go up to sides
    int entityCount = cellTopo->getSubcellCount(d);
    for (int j=0; j<entityCount; j++) {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      set< unsigned > nodeSet;
      if (d != 0) {
        for (int node=0; node<entityNodeCount; node++) {
          unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
          nodeSet.insert(cell->vertices()[nodeIndexInCell]);
        }
      } else {
        nodeSet.insert(cell->vertices()[j]);
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
      } else {
//        cout << "Erased _activeCellsForEntities[" << d << "][" << entityIndex << "] entry for (";
//        cout << cell->cellIndex() << "," << j << ").  Remaining entries: ";
//        set< pair<unsigned,unsigned> > remainingEntries = _activeCellsForEntities[d][entityIndex];
//        for (set< pair<unsigned,unsigned> >::iterator entryIt = remainingEntries.begin(); entryIt != remainingEntries.end(); entryIt++) {
//          cout << "(" << entryIt->first << "," << entryIt->second << ") ";
//        }
//        cout << endl;
      }
    }
  }
  _activeCells.erase(cell->cellIndex());
}

set<unsigned> MeshTopology::descendants(unsigned d, unsigned entityIndex) {
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

bool MeshTopology::entityHasParent(unsigned d, unsigned entityIndex) {
  if (_parentEntities[d].find(entityIndex) == _parentEntities[d].end()) return false;
  return _parentEntities[d][entityIndex].size() > 0;
}

bool MeshTopology::entityIsAncestor(unsigned d, unsigned ancestor, unsigned descendent) {
  map< unsigned, vector< pair<unsigned, unsigned> > >::iterator parentIt = _parentEntities[d].find(descendent);
  while (parentIt != _parentEntities[d].end()) {
    vector< pair<unsigned, unsigned> > parents = parentIt->second;
    unsigned parentEntityIndex = -1;
    for (vector< pair<unsigned, unsigned> >::iterator entryIt = parents.begin(); entryIt != parents.end(); entryIt++) {
      parentEntityIndex = entryIt->first;
      if (parentEntityIndex==ancestor) {
        return true;
      }
    }
    parentIt = _parentEntities[d].find(parentEntityIndex);
  }
  return false;
}

unsigned MeshTopology::getActiveCellCount(unsigned int d, unsigned int entityIndex) {
  map<unsigned, set<pair<unsigned,unsigned> > >::iterator activeCellsForEntityIt = _activeCellsForEntities[d].find(entityIndex);
  if (activeCellsForEntityIt == _activeCellsForEntities[d].end()) {
    return 0;
  } else {
    return activeCellsForEntityIt->second.size();
  }
}

const set< pair<unsigned,unsigned> > & MeshTopology::getActiveCellIndices(unsigned d, unsigned entityIndex) {
  return _activeCellsForEntities[d][entityIndex];
}

CellPtr MeshTopology::getCell(unsigned cellIndex) {
  return _cells[cellIndex];
}

unsigned MeshTopology::getEntityCount(unsigned int d) {
  return _entities[d].size();
}

unsigned MeshTopology::getEntityParent(unsigned d, unsigned entityIndex, unsigned parentOrdinal) {
  TEUCHOS_TEST_FOR_EXCEPTION(! entityHasParent(d, entityIndex), std::invalid_argument, "entity does not have parent");
  return _parentEntities[d][entityIndex][parentOrdinal].first;
}

const vector<unsigned> & MeshTopology::getEntityVertexIndices(unsigned d, unsigned entityIndex) {
  return _canonicalEntityOrdering[d][entityIndex];
}

set<unsigned> MeshTopology::getEntitiesForSide(unsigned sideEntityIndex, unsigned d) {
  unsigned sideDim = _spaceDim - 1;
  unsigned subEntityCount = getSubEntityCount(sideDim, sideEntityIndex, d);
  set<unsigned> subEntities;
  for (int subEntityOrdinal=0; subEntityOrdinal<subEntityCount; subEntityOrdinal++) {
    subEntities.insert(getSubEntityIndex(sideDim, sideEntityIndex, d, subEntityOrdinal));
  }
  return subEntities;
}

unsigned MeshTopology::getFaceEdgeIndex(unsigned int faceIndex, unsigned int edgeOrdinalInFace) {
  return getSubEntityIndex(2, faceIndex, 1, edgeOrdinalInFace);
}

unsigned MeshTopology::getSpaceDim() {
  return _spaceDim;
}

unsigned MeshTopology::getSubEntityCount(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim) {
  shards::CellTopology *entityTopo = &_knownTopologies[_entityCellTopologyKeys[d][entityIndex]];
  return entityTopo->getSubcellCount(subEntityDim);
}

unsigned MeshTopology::getSubEntityIndex(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim, unsigned int subEntityOrdinal) {
  shards::CellTopology *entityTopo = &_knownTopologies[_entityCellTopologyKeys[d][entityIndex]];
  set<unsigned> subEntityNodes;
  unsigned nodeCount = (d > 0) ? entityTopo->getNodeCount(subEntityDim, subEntityOrdinal) : entityTopo->getNodeCount();
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

const vector<double>& MeshTopology::getVertex(unsigned vertexIndex) {
  return _vertices[vertexIndex];
}

bool MeshTopology::getVertexIndex(const vector<double> &vertex, unsigned &vertexIndex, double tol) {
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

unsigned MeshTopology::getVertexIndexAdding(const vector<double> &vertex, double tol) {
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
vector<unsigned> MeshTopology::getVertexIndices(const FieldContainer<double> &vertices) {
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
map<unsigned, unsigned> MeshTopology::getVertexIndicesMap(const FieldContainer<double> &vertices) {
  map<unsigned, unsigned> vertexMap;
  vector<unsigned> vertexVector = getVertexIndices(vertices);
  unsigned numVertices = vertexVector.size();
  for (unsigned i=0; i<numVertices; i++) {
    vertexMap[i] = vertexVector[i];
  }
  return vertexMap;
}

vector<unsigned> MeshTopology::getVertexIndices(const vector< vector<double> > &vertices) {
  double tol = 1e-14; // tolerance for vertex equality
  
  int numVertices = vertices.size();
  vector<unsigned> localToGlobalVertexIndex(numVertices);
  for (int i=0; i<numVertices; i++) {
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertices[i],tol);
  }
  return localToGlobalVertexIndex;
}

vector<unsigned> MeshTopology::getChildEntities(unsigned int d, unsigned int entityIndex) {
  vector<unsigned> childIndices;
  if (d==0) return childIndices;
  if (_childEntities[d].find(entityIndex) == _childEntities[d].end()) return childIndices;
  vector< pair< RefinementPatternPtr, vector<unsigned> > > childEntries = _childEntities[d][entityIndex];
  for (vector< pair< RefinementPatternPtr, vector<unsigned> > >::iterator entryIt = childEntries.begin();
       entryIt != childEntries.end(); entryIt++) {
    childIndices.insert(childIndices.end(),entryIt->second.begin(),entryIt->second.end());
  }
  return childIndices;
}

set<unsigned> MeshTopology::getChildEntitiesSet(unsigned int d, unsigned int entityIndex) {
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

unsigned MeshTopology::getConstrainingEntityIndex(unsigned int d, unsigned int entityIndex) {
  unsigned constrainingEntityIndex = entityIndex;
  
  set<unsigned> sidesForEntity;
  unsigned sideDim = _spaceDim - 1;
  if (d==sideDim) {
    sidesForEntity.insert(entityIndex);
  } else {
    sidesForEntity = _activeSidesForEntities[d][entityIndex];
  }
  for (set<unsigned>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++) {
    unsigned sideEntityIndex = *sideEntityIt;
    vector< pair<unsigned,unsigned> > sideAncestry = getConstrainingSideAncestry(sideEntityIndex);
    unsigned constrainingEntityIndexForSide = entityIndex;
    if (sideAncestry.size() > 0) {
      // need to find the subcellEntity for the constraining side that overlaps with the one on our present side
      for (vector< pair<unsigned,unsigned> >::iterator entryIt=sideAncestry.begin(); entryIt != sideAncestry.end(); entryIt++) {
        // need to map constrained entity index from the current side to its parent in sideAncestry
        unsigned parentSideEntityIndex = entryIt->first;
        if (_parentEntities[d].find(constrainingEntityIndexForSide) == _parentEntities[d].end()) {
          // no parent for this entity (may be that it was a refinement-interior edge, e.g.)
          break;
        }
        constrainingEntityIndexForSide = getEntityParentForSide(d,constrainingEntityIndexForSide,parentSideEntityIndex);
        sideEntityIndex = parentSideEntityIndex;
      }
    }
    constrainingEntityIndex = maxConstraint(d, constrainingEntityIndex, constrainingEntityIndexForSide);
  }
  return constrainingEntityIndex;
}

// pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)
vector< pair<unsigned,unsigned> > MeshTopology::getConstrainingSideAncestry(unsigned int sideEntityIndex) {
  // three possibilities: 1) compatible side, 2) side is parent, 3) side is child
  // 1) and 2) mean unconstrained.  3) means constrained (by parent)
  unsigned sideDim = _spaceDim - 1;
  vector< pair<unsigned, unsigned> > ancestry;
  if (_boundarySides.find(sideEntityIndex) != _boundarySides.end()) {
    return ancestry; // sides on boundary are unconstrained...
  }
  
  set< pair<unsigned,unsigned> > sideCellEntries = _activeCellsForEntities[sideDim][sideEntityIndex];
  int activeCellCountForSide = sideCellEntries.size();
  if (activeCellCountForSide == 2) {
    // compatible side
    return ancestry; // will be empty
  } else if ((activeCellCountForSide == 0) || (activeCellCountForSide == 1)) {
    // then we're either parent or child of an active side
    // if we are a child, then we should find and return an ancestral path that ends in an active side
    map< unsigned, vector< pair<unsigned, unsigned> > >::iterator parentIt = _parentEntities[sideDim].find(sideEntityIndex);
    while (parentIt != _parentEntities[sideDim].end()) {
      vector< pair<unsigned, unsigned> > parents = parentIt->second;
      unsigned parentEntityIndex, refinementIndex;
      for (vector< pair<unsigned, unsigned> >::iterator entryIt = parents.begin(); entryIt != parents.end(); entryIt++) {
        parentEntityIndex = entryIt->first;
        refinementIndex = entryIt->second;
        if (getActiveCellCount(sideDim, parentEntityIndex) > 0) {
          // active cell; we've found our final ancestor
          ancestry.push_back(*entryIt);
          return ancestry;
        }
      }
      // if we get here, then (parentEntityIndex, refinementIndex) points to the last of the possible parents, which by convention must be a regular refinement (more precisely, one whose subentities are at least as fine as all previous possible parents)
      // this is therefore an acceptable entry in our ancestry path.
      ancestry.push_back(make_pair(parentEntityIndex, refinementIndex));
      parentIt = _parentEntities[sideDim].find(parentEntityIndex);
    }
    // if no such ancestral path exists, then we are a parent, and are unconstrained (return empty ancestry)
    ancestry.clear();
    return ancestry;
  } else {
    cout << "MeshTopology internal error: # active cells for side is not 0, 1, or 2\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTopology internal error: # active cells for side is not 0, 1, or 2\n");
    return ancestry; // unreachable, but most compilers don't seem to know that.
  }
}

unsigned MeshTopology::getEntityParentForSide(unsigned d, unsigned entityIndex,
                                              unsigned parentSideEntityIndex) {
  // returns the entity index for the parent (which might be the entity itself) of entity (d,entityIndex) that is
  // a subcell of side parentSideEntityIndex
  
  // assuming valid input, three possibilities:
  // 1) parent side has entity as a subcell
  // 2) parent side has exactly one of entity's immediate parents as a subcell
  
  set<unsigned> entitiesForParentSide = getEntitiesForSide(parentSideEntityIndex, d);
//  cout << "entitiesForParentSide with sideEntityIndex " << parentSideEntityIndex << ": ";
//  for (set<unsigned>::iterator entityIt = entitiesForParentSide.begin(); entityIt != entitiesForParentSide.end(); entityIt++) {
//    cout << *entityIt << " ";
//  }
//  cout << endl;
//  for (set<unsigned>::iterator entityIt = entitiesForParentSide.begin(); entityIt != entitiesForParentSide.end(); entityIt++) {
//    cout << "entity " << *entityIt << ":\n";
//    printEntityVertices(d, *entityIt);
//  }
//  cout << "parentSide vertices:\n";
//  printEntityVertices(_spaceDim-1, parentSideEntityIndex);
  
  if (entitiesForParentSide.find(entityIndex) != entitiesForParentSide.end()) {
    return entityIndex;
  }
  vector< pair<unsigned, unsigned> > entityParents = _parentEntities[d][entityIndex];
//  cout << "parent entities of entity " << entityIndex << ": ";
  for (vector< pair<unsigned, unsigned> >::iterator parentIt = entityParents.begin(); parentIt != entityParents.end(); parentIt++) {
    unsigned parentEntityIndex = parentIt->first;
//    cout << parentEntityIndex << " ";
    if (entitiesForParentSide.find(parentEntityIndex) != entitiesForParentSide.end()) {
//      cout << endl;
      return parentEntityIndex;
    }
  }
  cout << endl << "entity " << entityIndex << " vertices:\n";
  printEntityVertices(d, entityIndex);
  
  cout << "parent entity not found in parent side.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "parent entity not found in parent side.\n");
  return -1;
}

unsigned MeshTopology::maxConstraint(unsigned d, unsigned entityIndex1, unsigned entityIndex2) {
  // if one of the entities is the ancestor of the other, returns that one.  Otherwise returns (unsigned) -1.
  
  if (entityIndex1==entityIndex2) return entityIndex1;
  
  // a good guess is that the entity with lower index is the ancestor
  unsigned smallerEntityIndex = min(entityIndex1, entityIndex2);
  unsigned largerEntityIndex = max(entityIndex1, entityIndex2);
  if (entityIsAncestor(d,smallerEntityIndex,largerEntityIndex)) {
    return smallerEntityIndex;
  } else if (entityIsAncestor(d,largerEntityIndex,smallerEntityIndex)) {
    return largerEntityIndex;
  }
  return -1;
}

vector< ParametricCurvePtr > MeshTopology::parametricEdgesForCell(unsigned cellIndex, bool neglectCurves) {
  vector< ParametricCurvePtr > edges;
  CellPtr cell = getCell(cellIndex);
  int numNodes = cell->vertices().size();
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim != 2, std::invalid_argument, "Only 2D supported right now.");
  vector<unsigned> vertices = cell->vertices();
  for (int nodeIndex=0; nodeIndex<numNodes; nodeIndex++) {
    int v0_index = vertices[nodeIndex];
    int v1_index = vertices[(nodeIndex+1)%numNodes];
    vector<double> v0 = getVertex(v0_index);
    vector<double> v1 = getVertex(v1_index);
    
    pair<int, int> edge = make_pair(v0_index, v1_index);
    pair<int, int> reverse_edge = make_pair(v1_index, v0_index);
    ParametricCurvePtr edgeFxn;
    
    double x0 = v0[0], y0 = v0[1];
    double x1 = v1[0], y1 = v1[1];
    
    ParametricCurvePtr straightEdgeFxn = ParametricCurve::line(x0, y0, x1, y1);
    
    if (neglectCurves) {
      edgeFxn = straightEdgeFxn;
    } if ( _edgeToCurveMap.find(edge) != _edgeToCurveMap.end() ) {
      edgeFxn = _edgeToCurveMap[edge];
    } else if ( _edgeToCurveMap.find(reverse_edge) != _edgeToCurveMap.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No support yet for curved edges outside mesh boundary.");
      // TODO: make ParametricCurves reversible (swap t=0 and t=1)
    } else {
      edgeFxn = straightEdgeFxn;
    }
    edges.push_back(edgeFxn);
  }
  return edges;
}

void MeshTopology::printVertex(unsigned int vertexIndex) {
  cout << "vertex " << vertexIndex << ": (";
  for (unsigned d=0; d<_spaceDim; d++) {
    cout << _vertices[vertexIndex][d];
    if (d != _spaceDim-1) cout << ",";
  }
  cout << ")\n";
}

void MeshTopology::printVertices(set<unsigned int> vertexIndices) {
  for (set<unsigned>::iterator indexIt=vertexIndices.begin(); indexIt!=vertexIndices.end(); indexIt++) {
    unsigned vertexIndex = *indexIt;
    printVertex(vertexIndex);
  }
}

void MeshTopology::printEntityVertices(unsigned int d, unsigned int entityIndex) {
  vector<unsigned> entityVertices = _canonicalEntityOrdering[d][entityIndex];
  for (vector<unsigned>::iterator vertexIt=entityVertices.begin(); vertexIt !=entityVertices.end(); vertexIt++) {
    printVertex(*vertexIt);
  }
}

FieldContainer<double> MeshTopology::physicalCellNodesForCell(unsigned int cellIndex) {
  CellPtr cell = getCell(cellIndex);
  unsigned vertexCount = cell->vertices().size();
  FieldContainer<double> nodes(vertexCount, _spaceDim);
  for (unsigned vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++) {
    unsigned vertexIndex = cell->vertices()[vertexOrdinal];
    for (unsigned d=0; d<_spaceDim; d++) {
      nodes(vertexOrdinal,d) = _vertices[vertexIndex][d];
    }
  }
  return nodes;
}

void MeshTopology::refineCell(unsigned cellIndex, RefinementPatternPtr refPattern) {
  // TODO: worry about the case (currently unsupported in RefinementPattern) of children that do not share topology with the parent.  E.g. quad broken into triangles.  (3D has better examples.)
  
  CellPtr cell = _cells[cellIndex];
  FieldContainer<double> cellNodes(cell->vertices().size(), _spaceDim);
  
  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(0); vertexIndex++) {
    for (int d=0; d<_spaceDim; d++) {
      cellNodes(vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }
  
  FieldContainer<double> vertices = refPattern->verticesForRefinement(cellNodes);
  if (_transformationFunction.get()) {
    bool changedVertices = _transformationFunction->mapRefCellPointsUsingExactGeometry(vertices, refPattern->verticesOnReferenceCell(), cellIndex);
//    cout << "transformed vertices:\n" << vertices;
  }
  map<unsigned, unsigned> localToGlobalVertexIndex = getVertexIndicesMap(vertices); // key: index in vertices; value: index in _vertices
  
  // get the children, as vectors of vertex indices:
  vector< vector<unsigned> > childVertices = refPattern->children(localToGlobalVertexIndex);
  
  int numChildren = childVertices.size();
  // this is where we assume all the children have same topology as parent:
  vector< CellTopoPtr > childTopos(numChildren,cell->topology());
  
  refineCellEntities(cell, refPattern);
  cell->setRefinementPattern(refPattern);
  
  deactivateCell(cell);
  addChildren(cell, childTopos, childVertices);
  
  if (_edgeToCurveMap.size() > 0) {
    vector< vector< pair< unsigned, unsigned> > > childrenForSides = refPattern->childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
    // handle any broken curved edges
//    set<int> childrenWithCurvedEdges;
    vector<unsigned> parentVertices = cell->vertices();
    int numVertices = parentVertices.size();
    for (int edgeIndex=0; edgeIndex < numVertices; edgeIndex++) {
      int numChildrenForSide = childrenForSides[edgeIndex].size();
      if (numChildrenForSide==1) continue; // unbroken edge: no treatment necessary
      int v0 = parentVertices[edgeIndex];
      int v1 = parentVertices[ (edgeIndex+1) % numVertices];
      pair<int,int> edge = make_pair(v0, v1);
      if (_edgeToCurveMap.find(edge) != _edgeToCurveMap.end()) {
        // then define the new curves
        double child_t0 = 0.0;
        double increment = 1.0 / numChildrenForSide;
        for (int i=0; i<numChildrenForSide; i++) {
          int childIndex = childrenForSides[edgeIndex][i].first;
          int childSideIndex = childrenForSides[edgeIndex][i].second;
          int childCellIndex = cell->getChildIndices()[childIndex];
          CellPtr child = getCell(childCellIndex);
          // here, we rely on the fact that childrenForSides[sideIndex] goes in order from parent's v0 to parent's v1
          ParametricCurvePtr parentCurve = _edgeToCurveMap[edge];
          ParametricCurvePtr childCurve = ParametricCurve::subCurve(parentCurve, child_t0, child_t0 + increment);
          vector<unsigned> childVertices = child->vertices();
          pair<unsigned, unsigned> childEdge = make_pair( childVertices[childSideIndex], childVertices[(childSideIndex+1)% childVertices.size()] );
          addEdgeCurve(childEdge, childCurve);
//          childrenWithCurvedEdges.insert(childCellIndex);
          child_t0 += increment;
        }
      }
    }
//    if (_transformationFunction.get()) {
//      _transformationFunction->updateCells(childrenWithCurvedEdges);
//    }
  }
}

void MeshTopology::refineCellEntities(CellPtr cell, RefinementPatternPtr refPattern) {
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
    recipe.push_back(make_pair(refPattern.get(),vector<unsigned>()));
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
          
          if (_transformationFunction.get()) {
            physicalNodes.resize(nodeCount,_spaceDim);
            bool changedVertices = _transformationFunction->mapRefCellPointsUsingExactGeometry(physicalNodes, nodesOnRefCell, cell->cellIndex());
//            cout << "physicalNodes after transformation:\n" << physicalNodes;
          }
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
          // TODO: ?? do something with parentActiveCells?  Seems like we just trailed off here...
        }
        _childEntities[d][parentIndex] = vector< pair<RefinementPatternPtr,vector<unsigned> > >(1, make_pair(subcellRefPattern, childEntityIndices) ); // TODO: this also needs to change when we work through recipes.  Note that the correct parent will vary here...  i.e. in the anisotropic case, the child we're ultimately interested in will have an anisotropic parent, and *its* parent would be the bigger guy referred to here.
        if (d==_spaceDim-1) { // side
          if (_boundarySides.find(parentIndex) != _boundarySides.end()) { // parent is a boundary side, so children are, too
            _boundarySides.insert(childEntityIndices.begin(),childEntityIndices.end());
          }
        }
      }
    }
  }
}

const set<unsigned> &MeshTopology::getRootCellIndices() {
  return _rootCells;
}

void MeshTopology::setEdgeToCurveMap(const map< pair<int, int>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh) {
  _edgeToCurveMap.clear();
  map< pair<int, int>, ParametricCurvePtr >::const_iterator edgeIt;
  _cellIDsWithCurves.clear();
  
  for (edgeIt = edgeToCurveMap.begin(); edgeIt != edgeToCurveMap.end(); edgeIt++) {
    addEdgeCurve(edgeIt->first, edgeIt->second);
  }
  _transformationFunction = Teuchos::rcp(new MeshTransformationFunction(mesh, _cellIDsWithCurves));
}

Teuchos::RCP<MeshTransformationFunction> MeshTopology::transformationFunction() {
  return _transformationFunction;
}
