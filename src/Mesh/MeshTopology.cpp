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

#include "Intrepid_CellTools.hpp"

void MeshTopology::init(unsigned spaceDim) {
  RefinementPattern::initializeAnisotropicRelationships(); // not sure this is the optimal place for this call
  
  _spaceDim = spaceDim;
  _entities = vector< vector< set< unsigned > > >(_spaceDim);
  _knownEntities = vector< map< set<unsigned>, unsigned > >(_spaceDim); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< map< unsigned, vector<unsigned> > >(_spaceDim);
  _activeCellsForEntities = vector< map< unsigned, set< pair<unsigned, unsigned> > > >(_spaceDim); // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _sidesForEntities = vector< map< unsigned, set< unsigned > > >(_spaceDim);
  _parentEntities = vector< map< unsigned, vector< pair<unsigned, unsigned> > > >(_spaceDim); // map to possible parents
  _generalizedParentEntities = vector< map<unsigned, pair<unsigned,unsigned> > >(_spaceDim);
  _childEntities = vector< map< unsigned, vector< pair<RefinementPatternPtr, vector<unsigned> > > > >(_spaceDim);
  _entityCellTopologyKeys = vector< map< unsigned, unsigned > >(_spaceDim);
}

MeshTopology::MeshTopology(unsigned spaceDim, vector<PeriodicBCPtr> periodicBCs) {
  init(spaceDim);
  _periodicBCs = periodicBCs;
}

MeshTopology::MeshTopology(MeshGeometryPtr meshGeometry, vector<PeriodicBCPtr> periodicBCs) {
  unsigned spaceDim = meshGeometry->vertices()[0].size();

  init(spaceDim);
  _periodicBCs = periodicBCs;
  
  vector< vector<double> > vertices = meshGeometry->vertices();
  
  vector<int> myVertexIndexForMeshGeometryIndex(vertices.size());
  for (int i=0; i<vertices.size(); i++) {
    myVertexIndexForMeshGeometryIndex[i] = getVertexIndexAdding(vertices[i], 1e-14);
  }
//  _vertices = meshGeometry->vertices();
  
//  for (int vertexIndex=0; vertexIndex<_vertices.size(); vertexIndex++) {
//    _vertexMap[_vertices[vertexIndex]] = vertexIndex;
//  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(meshGeometry->cellTopos().size() != meshGeometry->elementVertices().size(), std::invalid_argument,
                             "length of cellTopos != length of elementVertices");
  
  int numElements = meshGeometry->cellTopos().size();
  
  for (int i=0; i<numElements; i++) {
    CellTopoPtr cellTopo = meshGeometry->cellTopos()[i];
    vector< unsigned > cellVerticesInMeshGeometry = meshGeometry->elementVertices()[i];
    vector<unsigned> cellVertices;
    for (int j=0; j<cellVerticesInMeshGeometry.size(); j++) {
      cellVertices.push_back(myVertexIndexForMeshGeometryIndex[cellVerticesInMeshGeometry[j]]);
    }
    
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

      cellEntityPermutations[d][j] = entityPermutation;
      if (_activeCellsForEntities[d].find(entityIndex) == _activeCellsForEntities[d].end()) {
        _activeCellsForEntities[d].insert(make_pair(entityIndex,set< pair<IndexType, unsigned> >()));
      }
      _activeCellsForEntities[d][entityIndex].insert(make_pair(cellIndex,j));
      
      if (d == 0) { // vertex --> should set parent relationships for any vertices that are equivalent via periodic BCs
        if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end()) {
          for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[entityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[entityIndex].end(); bcIt++) {
            IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, *bcIt)];
            _activeCellsForEntities[d][equivalentNode].insert(make_pair(cellIndex, j));
          }
        }
      }

    }
  }
  CellPtr cell = Teuchos::rcp( new Cell(cellTopo, cellVertices, cellEntityPermutations, cellIndex, this) );
  _cells.push_back(cell);
  _activeCells.insert(cellIndex);
  _rootCells.insert(cellIndex); // will remove if a parent relationship is established
  if (parentCellIndex != -1) {
    cell->setParent(getCell(parentCellIndex));
  }
  
  // set neighbors:
  unsigned sideDim = _spaceDim - 1;
  unsigned sideCount = CamelliaCellTools::getSideCount(*cellTopo);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
    addCellForSide(cellIndex,sideOrdinal,sideEntityIndex);
  }
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
    unsigned cellCountForSide = getCellCountForSide(sideEntityIndex);
    if (cellCountForSide == 2) { // compatible neighbors
      pair<unsigned,unsigned> firstNeighbor  = _cellsForSideEntities[sideEntityIndex].first;
      pair<unsigned,unsigned> secondNeighbor = _cellsForSideEntities[sideEntityIndex].second;
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
      // if the pre-existing neighbor is refined, set its descendants to have the appropriate neighbor.
      if (firstCell->isParent()) {
        vector< pair< GlobalIndexType, unsigned> > firstCellDescendants = firstCell->getDescendantsForSide(firstNeighbor.second);
        for (vector< pair< GlobalIndexType, unsigned> >::iterator descIt = firstCellDescendants.begin(); descIt != firstCellDescendants.end(); descIt++) {
          unsigned childCellIndex = descIt->first;
          unsigned childSideIndex = descIt->second;
          getCell(childCellIndex)->setNeighbor(childSideIndex, secondNeighbor.first, secondNeighbor.second);
        }
      }
      if (secondCell->isParent()) { // I don't think we should ever get here
        vector< pair< GlobalIndexType, unsigned> > secondCellDescendants = secondCell->getDescendantsForSide(secondNeighbor.first);
        for (vector< pair< GlobalIndexType, unsigned> >::iterator descIt = secondCellDescendants.begin(); descIt != secondCellDescendants.end(); descIt++) {
          GlobalIndexType childCellIndex = descIt->first;
          unsigned childSideOrdinal = descIt->second;
          getCell(childCellIndex)->setNeighbor(childSideOrdinal, firstNeighbor.first, firstNeighbor.second);
        }
      }
    } else if (cellCountForSide == 1) { // just this side
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
        _sidesForEntities[d][subcellEntityIndex].insert(sideEntityIndex);
        if (d==0) {
          if (_periodicBCIndicesMatchingNode.find(subcellEntityIndex) != _periodicBCIndicesMatchingNode.end()) {
            for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[subcellEntityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[subcellEntityIndex].end(); bcIt++) {
              IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(subcellEntityIndex, *bcIt)];
              _sidesForEntities[d][equivalentNode].insert(sideEntityIndex);
            }
          }
        }
      }
    }
    // for convenience, include the side itself in the _sidesForEntities lookup:
    set<unsigned> thisSideSet;
    thisSideSet.insert(sideEntityIndex);
    _sidesForEntities[sideDim][sideEntityIndex] = thisSideSet;
  }
  
  return cellIndex;
}

void MeshTopology::addCellForSide(unsigned int cellIndex, unsigned int sideOrdinal, unsigned int sideEntityIndex) {
  if (_cellsForSideEntities.find(sideEntityIndex) == _cellsForSideEntities.end()) {
    pair< unsigned, unsigned > cell1 = make_pair(cellIndex, sideOrdinal);
    pair< unsigned, unsigned > cell2 = make_pair(-1, -1);
    _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
  } else {
    pair< unsigned, unsigned > cell1 = _cellsForSideEntities[sideEntityIndex].first;
    pair< unsigned, unsigned > cell2 = _cellsForSideEntities[sideEntityIndex].second;
    
    CellPtr cellToAdd = getCell(cellIndex);
    unsigned parentCellIndex;
    if ( cellToAdd->getParent().get() == NULL) {
      parentCellIndex = -1;
    } else {
      parentCellIndex = cellToAdd->getParent()->cellIndex();
    }
    if (parentCellIndex == cell1.first) {
      // then replace cell1's entry with the new one
      cell1.first = cellIndex;
      cell1.second = sideOrdinal;
    } else if ((cell2.first == -1) || (parentCellIndex == cell2.first)) {
      cell2.first = cellIndex;
      cell2.second = sideOrdinal;
    } else {
      cout << "Internal error: attempt to add 3rd cell for side with entity index " << sideEntityIndex << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: attempt to add 3rd cell for side");
    }
    _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
  }
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
  unsigned entityIndex = getEntityIndex(d, nodeSet);
  
  if ( entityIndex == -1 ) {
    // new entity
    entityIndex = _entities[d].size();
    _entities[d].push_back(nodeSet);
    _knownEntities[d].insert(make_pair(nodeSet, entityIndex));
    _canonicalEntityOrdering[d].insert(make_pair(entityIndex, entityVertices));
    entityPermutation = 0;
    if (_knownTopologies.find(entityTopo.getKey()) == _knownTopologies.end()) {
      _knownTopologies[entityTopo.getKey()] = entityTopo;
    }
    _entityCellTopologyKeys[d].insert(make_pair(entityIndex, entityTopo.getKey()) );
  } else {
    // existing entity
    vector<IndexType> canonicalVertices = getCanonicalEntityNodesViaPeriodicBCs(d, entityVertices);
//    
//    Camellia::print("canonicalEntityOrdering",_canonicalEntityOrdering[d][entityIndex]);
    entityPermutation = CamelliaCellTools::permutationMatchingOrder(entityTopo, _canonicalEntityOrdering[d][entityIndex], canonicalVertices);
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
  
//  unsigned numSides = CamelliaCellTools::getSideCount(*parentCell->topology());
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

vector<IndexType> MeshTopology::getCanonicalEntityNodesViaPeriodicBCs(unsigned d, const vector<IndexType> &myEntityNodes) {
  set<IndexType> myNodeSet(myEntityNodes.begin(),myEntityNodes.end());
  if (_knownEntities[d].find(myNodeSet) != _knownEntities[d].end()) {
    return myEntityNodes;
  } else {
    // compute the intersection of the periodic BCs that match each node in nodeSet
    set< pair<int, int> > matchingPeriodicBCsIntersection;
    bool firstNode = true;
    for (vector<IndexType>::const_iterator nodeIt=myEntityNodes.begin(); nodeIt!=myEntityNodes.end(); nodeIt++) {
      if (_periodicBCIndicesMatchingNode.find(*nodeIt) == _periodicBCIndicesMatchingNode.end()) {
        matchingPeriodicBCsIntersection.clear();
        break;
      }
      if (firstNode) {
        matchingPeriodicBCsIntersection = _periodicBCIndicesMatchingNode[*nodeIt];
        firstNode = false;
      } else {
        set< pair<int, int> > newSet;
        set< pair<int, int> > matchesForThisNode = _periodicBCIndicesMatchingNode[*nodeIt];
        for (set< pair<int, int> >::iterator prevMatchIt=matchingPeriodicBCsIntersection.begin();
             prevMatchIt != matchingPeriodicBCsIntersection.end(); prevMatchIt++) {
          if (matchesForThisNode.find(*prevMatchIt) != matchesForThisNode.end()) {
            newSet.insert(*prevMatchIt);
          }
        }
        matchingPeriodicBCsIntersection = newSet;
      }
    }
    // for each periodic BC that remains, convert the nodeSet using that periodic BC
    for (set< pair<int, int> >::iterator bcIt=matchingPeriodicBCsIntersection.begin();
         bcIt != matchingPeriodicBCsIntersection.end(); bcIt++) {
      pair<int,int> matchingBC = *bcIt;
      vector<IndexType> equivalentNodeVector;
      for (vector<IndexType>::const_iterator nodeIt=myEntityNodes.begin(); nodeIt!=myEntityNodes.end(); nodeIt++) {
        equivalentNodeVector.push_back(_equivalentNodeViaPeriodicBC[make_pair(*nodeIt, matchingBC)]);
      }
      set<IndexType> equivalentNodeSet(equivalentNodeVector.begin(),equivalentNodeVector.end());
      if (_knownEntities[d].find(equivalentNodeSet) != _knownEntities[d].end()) {
        return equivalentNodeVector;
      }
    }
  }
  return vector<IndexType>(); // empty result meant to indicate not found...
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

CellPtr MeshTopology::findCellWithVertices(const vector< vector<double> > &cellVertices) {
  CellPtr cell;
  vector<IndexType> vertexIndices;
  bool firstVertex = true;
  unsigned vertexDim = 0;
  set<IndexType> matchingCells;
  for (vector< vector<double> >::const_iterator vertexIt = cellVertices.begin(); vertexIt != cellVertices.end(); vertexIt++) {
    vector<double> vertex = *vertexIt;
    IndexType vertexIndex;
    if (! getVertexIndex(vertex, vertexIndex) ) {
      cout << "vertex not found. returning NULL.\n";
      return cell;
    }
    // otherwise, vertexIndex has been populated
    vertexIndices.push_back(vertexIndex);
   
    set< pair<IndexType, unsigned> > matchingCellPairs = getCellsContainingEntity(vertexDim, vertexIndex);
    set<IndexType> matchingCellsIntersection;
    for (set< pair<IndexType, unsigned> >::iterator cellPairIt = matchingCellPairs.begin(); cellPairIt != matchingCellPairs.end(); cellPairIt++) {
      IndexType cellID = cellPairIt->first;
      if (firstVertex) {
        matchingCellsIntersection.insert(cellID);
      } else {
        if (matchingCells.find(cellID) != matchingCells.end()) {
          matchingCellsIntersection.insert(cellID);
        }
      }
    }
    matchingCells = matchingCellsIntersection;
    firstVertex = false;
  }
  if (matchingCells.size() == 0) {
    return cell; // null
  }
  if (matchingCells.size() > 1) {
    cout << "WARNING: multiple matching cells found.  Returning first one that matches.\n";
  }
  cell = getCell(*matchingCells.begin());
  return cell;
}

set< pair<IndexType, unsigned> > MeshTopology::getActiveBoundaryCells() { // (cellIndex, sideOrdinal)
  set< pair<IndexType, unsigned> > boundaryCells;
  for (set<IndexType>::iterator boundarySideIt = _boundarySides.begin(); boundarySideIt != _boundarySides.end(); boundarySideIt++) {
    IndexType sideEntityIndex = *boundarySideIt;
    int cellCount = getCellCountForSide(sideEntityIndex);
    if (cellCount == 1) {
      pair<IndexType, unsigned> cellInfo = _cellsForSideEntities[sideEntityIndex].first;
      if (cellInfo.first == -1) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid cellIndex for side boundary.");
      }
      if (_activeCells.find(cellInfo.first) != _activeCells.end()) {
        boundaryCells.insert(cellInfo);
        // DEBUGGING:
//        if (getCell(cellInfo.first)->isParent()) {
//          cout << "ERROR: cell is parent, but is stored as an active cell in the mesh...\n";
//          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cell is parent, but is stored as an active cell in the mesh...");
//        }
      }
    } else if (cellCount > 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "boundary side has more than 1 cell!");
    } // cellCount = 0 just means that the side has been refined; that's acceptable
  }
  return boundaryCells;
}

vector<double> MeshTopology::getCellCentroid(IndexType cellIndex) {
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

unsigned MeshTopology::getCellCountForSide(IndexType sideEntityIndex) {
  if (_cellsForSideEntities.find(sideEntityIndex) == _cellsForSideEntities.end()) {
    return 0;
  } else {
    pair<IndexType, unsigned> cell1 = _cellsForSideEntities[sideEntityIndex].first;
    pair<IndexType, unsigned> cell2 = _cellsForSideEntities[sideEntityIndex].second;
    if (cell2.first == -1) {
      return 1;
    } else {
      return 2;
    }
  }
}

pair<IndexType, unsigned> MeshTopology::getFirstCellForSide(IndexType sideEntityIndex) {
  return _cellsForSideEntities[sideEntityIndex].first;
}

pair<IndexType, unsigned> MeshTopology::getSecondCellForSide(IndexType sideEntityIndex) {
  return _cellsForSideEntities[sideEntityIndex].second;
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
      
      unsigned entityIndex = getEntityIndex(d, nodeSet);
      if (entityIndex == -1) {
        // entity not found: an error
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cell entity not found!");
      }
    
      // delete from the _activeCellsForEntities store
      if (_activeCellsForEntities[d].find(entityIndex) == _activeCellsForEntities[d].end()) {
        cout << "WARNING: No entry found for _activeCellsForEntities[" << d << "][" << entityIndex << "]\n";
      } else {
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
      if (d == 0) { // vertex --> should delete entries for any that are equivalent via periodic BCs
        if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end()) {
          for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[entityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[entityIndex].end(); bcIt++) {
            IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, *bcIt)];
            if (_activeCellsForEntities[d].find(equivalentNode) == _activeCellsForEntities[d].end()) {
              cout << "WARNING: No entry found for _activeCellsForEntities[" << d << "][" << equivalentNode << "]\n";
            } else {
              unsigned eraseCount = _activeCellsForEntities[d][equivalentNode].erase(make_pair(cell->cellIndex(),j));
              if (eraseCount==0) {
                cout << "WARNING: attempt was made to deactivate a non-active subcell topology...\n";
              }
            }
          }
        }
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

bool MeshTopology::entityHasChildren(unsigned int d, IndexType entityIndex) {
  if (_childEntities[d].find(entityIndex) == _childEntities[d].end()) return false;
  return _childEntities[d][entityIndex].size() > 0;
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
  if (cellIndex > _cells.size()) {
    cout << "MeshTopology::getCell: cellIndex " << cellIndex << " out of bounds (0, " << _cells.size() - 1 << ").\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex out of bounds.\n");
  }
  return _cells[cellIndex];
}

unsigned MeshTopology::getEntityCount(unsigned int d) {
  if (d==0) return _vertices.size();
  return _entities[d].size();
}

pair<IndexType, unsigned> MeshTopology::getEntityGeneralizedParent(unsigned int d, IndexType entityIndex) {
  if (_generalizedParentEntities[d].find(entityIndex) == _generalizedParentEntities[d].end()) return make_pair(-1,-1);
  return _generalizedParentEntities[d][entityIndex];
}

unsigned MeshTopology::getEntityIndex(unsigned d, const set<unsigned> &nodeSet) {
  if (d==0) {
    if (nodeSet.size()==1) {
      return *nodeSet.begin();
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "node set for vertex should not have more than one entry!");
    }
  }
  if (_knownEntities[d].find(nodeSet) != _knownEntities[d].end()) {
    return _knownEntities[d][nodeSet];
  } else {
    // look for alternative, equivalent nodeSets, arrived at via periodic BCs
    vector<IndexType> nodeVector(nodeSet.begin(),nodeSet.end());
    vector<IndexType> equivalentNodeVector = getCanonicalEntityNodesViaPeriodicBCs(d, nodeVector);
    
    if (equivalentNodeVector.size() > 0) {
      set<IndexType> equivalentNodeSet(equivalentNodeVector.begin(),equivalentNodeVector.end());
      if (_knownEntities[d].find(equivalentNodeSet) != _knownEntities[d].end()) {
        return _knownEntities[d][equivalentNodeSet];
      }
    }
  }
  return -1;
}

unsigned MeshTopology::getEntityParent(unsigned d, unsigned entityIndex, unsigned parentOrdinal) {
  TEUCHOS_TEST_FOR_EXCEPTION(! entityHasParent(d, entityIndex), std::invalid_argument, "entity does not have parent");
  return _parentEntities[d][entityIndex][parentOrdinal].first;
}

const shards::CellTopology &MeshTopology::getEntityTopology(unsigned d, IndexType entityIndex) {
  unsigned cellKey = _entityCellTopologyKeys[d][entityIndex];
  return _knownTopologies[cellKey];
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
  if (d==0) {
    if (subEntityDim==0) {
      return 1; // the vertex is its own sub-entity then
    } else {
      return 0;
    }
  }
  shards::CellTopology *entityTopo = &_knownTopologies[_entityCellTopologyKeys[d][entityIndex]];
  return entityTopo->getSubcellCount(subEntityDim);
}

unsigned MeshTopology::getSubEntityIndex(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim, unsigned int subEntityOrdinal) {
  if (d==0) {
    if ((subEntityDim==0) && (subEntityOrdinal==0))  {
      return entityIndex; // the vertex is its own sub-entity then
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-entity not found for vertex");
    }
  }
  
  shards::CellTopology *entityTopo = &_knownTopologies[_entityCellTopologyKeys[d][entityIndex]];
  set<unsigned> subEntityNodes;
  unsigned subEntityNodeCount = (subEntityDim > 0) ? entityTopo->getNodeCount(subEntityDim, subEntityOrdinal) : 1; // vertices are by definition just one node
  vector<unsigned> entityNodes = _canonicalEntityOrdering[d][entityIndex];
  
  for (unsigned nodeOrdinal=0; nodeOrdinal<subEntityNodeCount; nodeOrdinal++) {
    unsigned nodeOrdinalInEntity = entityTopo->getNodeMap(subEntityDim, subEntityOrdinal, nodeOrdinal);
    unsigned nodeIndexInMesh = entityNodes[nodeOrdinalInEntity];
    if (subEntityDim == 0) {
      return nodeIndexInMesh;
    }
    subEntityNodes.insert(nodeIndexInMesh);
  }
  unsigned subEntityIndex = getEntityIndex(subEntityDim, subEntityNodes);
  if (subEntityIndex == -1) {
    cout << "sub-entity not found with vertices:\n";
    printVertices(subEntityNodes);
    cout << "entity vertices:\n";
    set<unsigned> entityNodeSet(entityNodes.begin(),entityNodes.end());
    printVertices(entityNodeSet);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-entity not found");
  }
  return subEntityIndex;
}

const vector<double>& MeshTopology::getVertex(unsigned vertexIndex) {
  return _vertices[vertexIndex];
}

bool MeshTopology::getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol) {
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

  { // update the various entity containers
    int vertexDim = 0;
    set<IndexType> nodeSet;
    nodeSet.insert(vertexIndex);
    _entities[vertexDim].push_back(nodeSet);
    _knownEntities[vertexDim][nodeSet] = vertexIndex;
    vector<IndexType> entityVertices;
    entityVertices.push_back(vertexIndex);
    _canonicalEntityOrdering[vertexDim][vertexIndex] = entityVertices;
    shards::CellTopology nodeTopo = shards::getCellTopologyData< shards::Node >();
    if (_knownTopologies.find(nodeTopo.getKey()) == _knownTopologies.end()) {
      _knownTopologies[nodeTopo.getKey()] = nodeTopo;
    }
    _entityCellTopologyKeys[vertexDim][vertexIndex] = nodeTopo.getKey();
  }
  
  set< pair<int,int> > matchingPeriodicBCs;

  for (int i=0; i<_periodicBCs.size(); i++) {
    vector<int> matchingSides = _periodicBCs[i]->getMatchingSides(vertex);
    for (int j=0; j<matchingSides.size(); j++) {
      int matchingSide = matchingSides[j];
      pair<int,int> matchingBC = make_pair(i, matchingSide);
      matchingPeriodicBCs.insert(matchingBC);
      vector<double> matchingPoint = _periodicBCs[i]->getMatchingPoint(vertex, matchingSide);
      unsigned equivalentVertexIndex = getVertexIndexAdding(matchingPoint, tol);
      _equivalentNodeViaPeriodicBC[make_pair(vertexIndex, matchingBC)] = equivalentVertexIndex;
      
//      cout << "vertex " << vertexIndex << " is equivalent to " << equivalentVertexIndex << endl;
//      printVertex(vertexIndex);
//      printVertex(equivalentVertexIndex);
    }
  }
  
  _periodicBCIndicesMatchingNode[vertexIndex] = matchingPeriodicBCs;
  
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
map<unsigned, IndexType> MeshTopology::getVertexIndicesMap(const FieldContainer<double> &vertices) {
  map<unsigned, IndexType> vertexMap;
  vector<IndexType> vertexVector = getVertexIndices(vertices);
  unsigned numVertices = vertexVector.size();
  for (unsigned i=0; i<numVertices; i++) {
    vertexMap[i] = vertexVector[i];
  }
  return vertexMap;
}

vector<IndexType> MeshTopology::getVertexIndices(const vector< vector<double> > &vertices) {
  double tol = 1e-14; // tolerance for vertex equality
  
  int numVertices = vertices.size();
  vector<IndexType> localToGlobalVertexIndex(numVertices);
  for (int i=0; i<numVertices; i++) {
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertices[i],tol);
  }
  return localToGlobalVertexIndex;
}

vector<unsigned> MeshTopology::getChildEntities(unsigned int d, IndexType entityIndex) {
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

pair<IndexType, unsigned> MeshTopology::getConstrainingEntity(unsigned d, IndexType entityIndex) {
  unsigned sideDim = _spaceDim - 1;
  
  pair<IndexType, unsigned> constrainingEntity; // we store the highest-dimensional constraint.  (This will be the maximal constraint.)
  constrainingEntity.first = entityIndex;
  constrainingEntity.second = d;
  
  IndexType generalizedAncestorEntityIndex = entityIndex;
  for (unsigned generalizedAncestorDim=d; generalizedAncestorDim <= sideDim; ) {
    IndexType possibleConstrainingEntityIndex = getConstrainingEntityIndexOfLikeDimension(generalizedAncestorDim, generalizedAncestorEntityIndex);
    if (possibleConstrainingEntityIndex != generalizedAncestorEntityIndex) {
      constrainingEntity.second = generalizedAncestorDim;
      constrainingEntity.first = possibleConstrainingEntityIndex;
    } else {
      // if the generalized parent has no constraint of like dimension, then either the generalized parent is the constraint, or there is no constraint of this dimension
      // basic rule: if there exists a side belonging to an active cell that contains the putative constraining entity, then we constrain
      // I am a bit vague on whether this will work correctly in the context of anisotropic refinements.  (It might, but I'm not sure.)  But first we are targeting isotropic.
      set<unsigned> sidesForEntity;
      if (generalizedAncestorDim==sideDim) {
        sidesForEntity.insert(generalizedAncestorEntityIndex);
      } else {
        sidesForEntity = _sidesForEntities[generalizedAncestorDim][generalizedAncestorEntityIndex];
      }
      for (set<unsigned>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++) {
        unsigned sideEntityIndex = *sideEntityIt;
        if (getActiveCellCount(sideDim, sideEntityIndex) > 0) {
          constrainingEntity.second = generalizedAncestorDim;
          constrainingEntity.first = possibleConstrainingEntityIndex;
          break;
        }
      }
    }
    while (entityHasParent(generalizedAncestorDim, generalizedAncestorEntityIndex)) { // parent of like dimension
      generalizedAncestorEntityIndex = getEntityParent(generalizedAncestorDim, generalizedAncestorEntityIndex);
    }
    if (_generalizedParentEntities[generalizedAncestorDim].find(generalizedAncestorEntityIndex)
        != _generalizedParentEntities[generalizedAncestorDim].end()) {
      pair< IndexType, unsigned > generalizedParent = _generalizedParentEntities[generalizedAncestorDim][generalizedAncestorEntityIndex];
      generalizedAncestorEntityIndex = generalizedParent.first;
      generalizedAncestorDim = generalizedParent.second;
    } else { // at top of refinement tree -- break out of for loop
      break;
    }
  }
  return constrainingEntity;
}

unsigned MeshTopology::getConstrainingEntityIndexOfLikeDimension(unsigned int d, unsigned int entityIndex) {
  unsigned constrainingEntityIndex = entityIndex;
  
  if (d==0) { // one vertex can't constrain another...
    return entityIndex;
  }
  
  set<unsigned> sidesForEntity;
  unsigned sideDim = _spaceDim - 1;
  if (d==sideDim) {
    sidesForEntity.insert(entityIndex);
  } else {
    sidesForEntity = _sidesForEntities[d][entityIndex];
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

RefinementBranch MeshTopology::getSideConstraintRefinementBranch(IndexType sideEntityIndex) {
  // Returns a RefinementBranch that goes from the constraining side to the side indicated.
  vector< pair<IndexType,unsigned> > constrainingSideAncestry = getConstrainingSideAncestry(sideEntityIndex);
  pair< RefinementPattern*, unsigned > branchEntry;
  unsigned sideDim = _spaceDim - 1;
  IndexType previousChild = sideEntityIndex;
  RefinementBranch refBranch;
  for (vector< pair<IndexType,unsigned> >::iterator ancestorIt = constrainingSideAncestry.begin();
       ancestorIt != constrainingSideAncestry.end(); ancestorIt++) {
    IndexType ancestorSideEntityIndex = ancestorIt->first;
    unsigned refinementIndex = ancestorIt->second;
    pair<RefinementPatternPtr, vector<IndexType> > children = _childEntities[sideDim][ancestorSideEntityIndex][refinementIndex];
    branchEntry.first = children.first.get();
    for (int i=0; i<children.second.size(); i++) {
      if (children.second[i]==previousChild) {
        branchEntry.second = i;
        break;
      }
    }
    refBranch.insert(refBranch.begin(), branchEntry);
    previousChild = ancestorSideEntityIndex;
  }
  return refBranch;
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

set< pair<IndexType, unsigned> > MeshTopology::getCellsContainingEntity(unsigned d, unsigned entityIndex) { // not *all* cells, but within any refinement branch, the most refined cell that contains the entity will be present in this set.  The unsigned value is the ordinal of a *side* in the cell containing this entity.  There may be multiple sides in a cell that contain the entity; this method will return just one entry per cell.
  set<IndexType> sidesForEntity = _sidesForEntities[d][entityIndex];
  typedef pair<IndexType,unsigned> CellPair;
  set< CellPair > cells;
  set< IndexType > cellIndices;  // container to keep track of which cells we've already counted -- we only return one (cell, side) pair per cell that contains the entity...
  for (set<IndexType>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++) {
    IndexType sideEntityIndex = *sideEntityIt;
    int numCellsForSide = getCellCountForSide(sideEntityIndex);
    if (numCellsForSide == 2) {
      CellPair cell1 = getFirstCellForSide(sideEntityIndex);
      if (cellIndices.find(cell1.first) == cellIndices.end()) {
        cells.insert(cell1);
        cellIndices.insert(cell1.first);
      }
      CellPair cell2 = getSecondCellForSide(sideEntityIndex);
      if (cellIndices.find(cell2.first) == cellIndices.end()) {
        cells.insert(cell2);
        cellIndices.insert(cell2.first);
      }
    } else if (numCellsForSide == 1) {
      CellPair cell1 = getFirstCellForSide(sideEntityIndex);
      if (cellIndices.find(cell1.first) == cellIndices.end()) {
        cells.insert(cell1);
        cellIndices.insert(cell1.first);
      }
    } else {
      cout << "Unexpected cell count for side.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unexpected cell count for side.");
    }
  }
  return cells;
}

set< IndexType > MeshTopology::getSidesContainingEntity(unsigned d, unsigned entityIndex) {
  return _sidesForEntities[d][entityIndex];
}

unsigned MeshTopology::getSubEntityPermutation(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal) {
  vector<unsigned> entityNodes = getEntityVertexIndices(d,entityIndex);
  shards::CellTopology topo = getEntityTopology(d, entityIndex);
  vector<unsigned> subEntityNodes;
  int subEntityNodeCount = topo.getNodeCount(subEntityDim, subEntityOrdinal);
  for (int seNodeOrdinal = 0; seNodeOrdinal<subEntityNodeCount; seNodeOrdinal++) {
    unsigned entityNodeOrdinal = topo.getNodeMap(subEntityDim, subEntityOrdinal, seNodeOrdinal);
    subEntityNodes.push_back(entityNodes[entityNodeOrdinal]);
  }
  subEntityNodes = getCanonicalEntityNodesViaPeriodicBCs(subEntityDim, subEntityNodes);
  unsigned subEntityIndex = getSubEntityIndex(d, entityIndex, subEntityDim, subEntityOrdinal);
  shards::CellTopology subEntityTopo = getEntityTopology(subEntityDim, subEntityIndex);
  return CamelliaCellTools::permutationMatchingOrder(subEntityTopo, _canonicalEntityOrdering[subEntityDim][subEntityOrdinal], subEntityNodes);
}

pair<IndexType,IndexType> MeshTopology::leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(unsigned d, unsigned constrainingEntityIndex) {
  unsigned leastActiveCellIndex = (unsigned)-1; // unsigned cast of -1 makes maximal unsigned #
  set<IndexType> constrainedEntities = descendants(d,constrainingEntityIndex);

  IndexType leastActiveCellConstrainedEntityIndex;
  for (set<IndexType>::iterator constrainedEntityIt = constrainedEntities.begin(); constrainedEntityIt != constrainedEntities.end(); constrainedEntityIt++) {
    IndexType constrainedEntityIndex = *constrainedEntityIt;
    if (_sidesForEntities[d].find(constrainingEntityIndex) == _sidesForEntities[d].end()) {
      cout << "ERROR: no sides found containing entityIndex " << constrainingEntityIndex << " of dimension " << d << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: no sides found containing entity");
    }
    set<IndexType> sideEntityIndices = _sidesForEntities[d][constrainedEntityIndex];
    for (set<IndexType>::iterator sideEntityIt = sideEntityIndices.begin(); sideEntityIt != sideEntityIndices.end(); sideEntityIt++) {
      IndexType sideEntityIndex = *sideEntityIt;
      typedef pair<IndexType, unsigned> CellPair;
      pair<CellPair,CellPair> cellsForSide = _cellsForSideEntities[sideEntityIndex];
      IndexType firstCellIndex = cellsForSide.first.first;
      if (_activeCells.find(firstCellIndex) != _activeCells.end()) {
        if (firstCellIndex < leastActiveCellIndex) {
          leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
          leastActiveCellIndex = firstCellIndex;
        }
      }
      IndexType secondCellIndex = cellsForSide.second.first;
      if (_activeCells.find(secondCellIndex) != _activeCells.end()) {
        if (secondCellIndex < leastActiveCellIndex) {
          leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
          leastActiveCellIndex = secondCellIndex;
        }
      }
    }
  }
  if (leastActiveCellIndex == -1) {
    cout << "WARNING: least active cell index not found.\n";
  }
  
  return make_pair(leastActiveCellIndex, leastActiveCellConstrainedEntityIndex);
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

void MeshTopology::printConstraintReport(unsigned d) {
  IndexType entityCount = _entities[d].size();
  cout << "******* MeshTopology, constraints for d = " << d << " *******\n";
  for (IndexType entityIndex=0; entityIndex<entityCount; entityIndex++) {
    pair<IndexType, unsigned> constrainingEntity = getConstrainingEntity(d, entityIndex);
    if ((d != constrainingEntity.second) || (entityIndex != constrainingEntity.first))
      cout << "Entity " << entityIndex << " is constrained by entity " << constrainingEntity.first << " of dimension " << constrainingEntity.second << endl;
    else
      cout << "Entity " << entityIndex << " is unconstrained.\n";
  }
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
  if (d==0) {
    printVertex(entityIndex);
    return;
  }
  vector<unsigned> entityVertices = _canonicalEntityOrdering[d][entityIndex];
  for (vector<unsigned>::iterator vertexIt=entityVertices.begin(); vertexIt !=entityVertices.end(); vertexIt++) {
    printVertex(*vertexIt);
  }
}

void MeshTopology::printAllEntities() {
  for (int d=0; d<_spaceDim; d++) {
    string entityTypeString;
    if (d==0) {
      entityTypeString = "Vertex";
    } else if (d==1) {
      entityTypeString = "Edge";
    } else if (d==2) {
      entityTypeString = "Face";
    } else if (d==3) {
      entityTypeString = "Solid";
    }
    cout << "****************************  ";
    cout << entityTypeString << " entities:";
    cout << "  ****************************\n";
    
    int entityCount = getEntityCount(d);
    for (int entityIndex=0; entityIndex < entityCount; entityIndex++) {
      if (d != 0) cout << entityTypeString << " " << entityIndex << ":" << endl;
      printEntityVertices(d, entityIndex);
    }
  }
  
  cout << "****************************      ";
  cout << "Cells:";
  cout << "      ****************************\n";
  
  int numCells = cellCount();
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    CellPtr cell = getCell(cellIndex);
    cout << "Cell " << cellIndex << ":\n";
    int vertexCount = cell->vertices().size();
    for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++) {
      printVertex(cell->vertices()[vertexOrdinal]);
    }
    for (int d=1; d<_spaceDim; d++) {
      int subcellCount = cell->topology()->getSubcellCount(d);
      for (int subcord=0; subcord<subcellCount; subcord++) {
        ostringstream labelStream;
        if (d==1) {
          labelStream << "Edge";
        } else if (d==2) {
          labelStream << "Face";
        } else if (d==3) {
          labelStream << "Solid";
        }
        labelStream << " " << subcord << " nodes";
        Camellia::print(labelStream.str(), cell->getEntityVertexIndices(d, subcord));
      }
    }
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
  map<unsigned, IndexType> vertexOrdinalToVertexIndex = getVertexIndicesMap(vertices); // key: index in vertices; value: index in _vertices
  map<unsigned, GlobalIndexType> localToGlobalVertexIndex(vertexOrdinalToVertexIndex.begin(),vertexOrdinalToVertexIndex.end());
  
  // get the children, as vectors of vertex indices:
  vector< vector<GlobalIndexType> > childVerticesGlobalType = refPattern->children(localToGlobalVertexIndex);
  vector< vector<IndexType> > childVertices(childVerticesGlobalType.begin(),childVerticesGlobalType.end());
  
  int numChildren = childVertices.size();
  // this is where we assume all the children have same topology as parent:
  vector< CellTopoPtr > childTopos(numChildren,cell->topology());
  
  refineCellEntities(cell, refPattern);
  cell->setRefinementPattern(refPattern);
  
  deactivateCell(cell);
  addChildren(cell, childTopos, childVertices);
  
  determineGeneralizedParentsForRefinement(cell, refPattern);
  
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
      FieldContainer<double> refinedNodes = subcellRefPattern->refinedNodes(); // NOTE: refinedNodes implicitly assumes that all child topos are the same
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
          
          
          // debugging:
//          if ((_cells.size() == 2) && (cell->cellIndex() == 0) && (d==2) && (subcord==2)) {
//            cout << "cellNodes:\n" << cellNodes;
//            cout << "For childOrdinal " << childIndex << " of face 2 on cell 0, details:\n";
//            cout << "nodesOnSubcell:\n" << nodesOnSubcell;
//            cout << "nodesOnRefCell:\n" << nodesOnRefCell;
//            cout << "physicalNodes:\n" << physicalNodes;
//          }
          
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

void MeshTopology::determineGeneralizedParentsForRefinement(CellPtr cell, RefinementPatternPtr refPattern) {
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
      
      // now, establish generalized parent relationships
      vector< IndexType > parentVertexIndices = this->getEntityVertexIndices(d, parentIndex);
      set<IndexType> parentVertexIndexSet(parentVertexIndices.begin(),parentVertexIndices.end());
      vector< pair<RefinementPatternPtr,vector<IndexType> > > childEntities = _childEntities[d][parentIndex];
      for (vector< pair<RefinementPatternPtr,vector<IndexType> > >::iterator refIt = childEntities.begin();
           refIt != childEntities.end(); refIt++) {
        vector<IndexType> childEntityIndices = refIt->second;
        for (int childOrdinal=0; childOrdinal<childEntityIndices.size(); childOrdinal++) {
          IndexType childEntityIndex = childEntityIndices[childOrdinal];
          if (parentIndex == childEntityIndex) { // "null" refinement pattern -- nothing to do here.
            continue;
          }
          setEntityGeneralizedParent(d, childEntityIndex, d, parentIndex); // TODO: change this to consider anisotropic refinements/ recipes...  (need to choose nearest of the possible ancestors, in my view)
          for (int subcdim=0; subcdim<d; subcdim++) {
            int subcCount = this->getSubEntityCount(d, childEntityIndex, subcdim);
            for (int subcord=0; subcord < subcCount; subcord++) {
              IndexType subcellEntityIndex = this->getSubEntityIndex(d, childEntityIndex, subcdim, subcord);
              
              // if this is a vertex that also belongs to the parent, then its parentage will already be handled...
              if ((subcdim==0) && (parentVertexIndexSet.find(subcellEntityIndex) != parentVertexIndexSet.end() )) {
                continue;
              }
              
              // if there was a previous entry, have a look at it...
              if (_generalizedParentEntities[subcdim].find(subcellEntityIndex) != _generalizedParentEntities[subcdim].end()) {
                pair<IndexType, unsigned> previousParent = _generalizedParentEntities[subcdim][subcellEntityIndex];
                if (previousParent.second <= d) { // then the previous parent is a better (nearer) parent
                  continue;
                }
              }
              
              // if we get here, then we're ready to establish the generalized parent relationship
              setEntityGeneralizedParent(subcdim, subcellEntityIndex, d, parentIndex);
            }
          }
        }
      }
    }
  }
}

const set<unsigned> &MeshTopology::getRootCellIndices() {
  return _rootCells;
}

void MeshTopology::setEdgeToCurveMap(const map< pair<IndexType, IndexType>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh) {
  _edgeToCurveMap.clear();
  map< pair<IndexType, IndexType>, ParametricCurvePtr >::const_iterator edgeIt;
  _cellIDsWithCurves.clear();
  
  for (edgeIt = edgeToCurveMap.begin(); edgeIt != edgeToCurveMap.end(); edgeIt++) {
    addEdgeCurve(edgeIt->first, edgeIt->second);
  }
  // mesh transformation function expects global ID type
  set<GlobalIndexType> cellIDsGlobal(_cellIDsWithCurves.begin(),_cellIDsWithCurves.end());
  _transformationFunction = Teuchos::rcp(new MeshTransformationFunction(mesh, cellIDsGlobal));
}

void MeshTopology::setEntityGeneralizedParent(unsigned entityDim, IndexType entityIndex, unsigned parentDim, IndexType parentEntityIndex) {
  _generalizedParentEntities[entityDim][entityIndex] = make_pair(parentEntityIndex,parentDim);
  if (entityDim == 0) { // vertex --> should set parent relationships for any vertices that are equivalent via periodic BCs
    if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end()) {
      for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[entityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[entityIndex].end(); bcIt++) {
        IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, *bcIt)];
        _generalizedParentEntities[entityDim][equivalentNode] = make_pair(parentEntityIndex,parentDim);
      }
    }
  }
}

Teuchos::RCP<MeshTransformationFunction> MeshTopology::transformationFunction() {
  return _transformationFunction;
}
