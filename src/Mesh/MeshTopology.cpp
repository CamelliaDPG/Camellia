//
//  MeshTopology.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#include "CamelliaCellTools.h"
#include "CellTopology.h"
#include "GlobalDofAssignment.h"
#include "MeshTopology.h"
#include "MeshTransformationFunction.h"

#include "Intrepid_CellTools.hpp"

#include <algorithm>

using namespace Intrepid;
using namespace Camellia;
using namespace std;

void MeshTopology::init(unsigned spaceDim)
{
  if (spaceDim >= 2) RefinementPattern::initializeAnisotropicRelationships(); // not sure this is the optimal place for this call

  _spaceDim = spaceDim;
  // for nontrivial mesh topology, we store entities with dimension sideDim down to vertices, so _spaceDim total possibilities
  // for trivial mesh topology (just a node), we allow storage of 0-dimensional (vertex) entity
  int numEntityDimensions = (_spaceDim > 0) ? _spaceDim : 1;
  _entities = vector< vector< vector< unsigned > > >(numEntityDimensions);
  _knownEntities = vector< map< vector<unsigned>, unsigned > >(numEntityDimensions); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< vector< vector<unsigned> > >(numEntityDimensions);
  _activeCellsForEntities = vector< vector< vector< pair<unsigned, unsigned> > > >(numEntityDimensions); // pair entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _sidesForEntities = vector< vector< vector< unsigned > > >(numEntityDimensions);
  _parentEntities = vector< map< unsigned, vector< pair<unsigned, unsigned> > > >(numEntityDimensions); // map to possible parents
  _generalizedParentEntities = vector< map<unsigned, pair<unsigned,unsigned> > >(numEntityDimensions);
  _childEntities = vector< map< unsigned, vector< pair<RefinementPatternPtr, vector<unsigned> > > > >(numEntityDimensions);
  _entityCellTopologyKeys = vector< vector< CellTopologyKey > >(numEntityDimensions);

  _gda = NULL;
}

MeshTopology::MeshTopology(unsigned spaceDim, vector<PeriodicBCPtr> periodicBCs)
{
  init(spaceDim);
  _periodicBCs = periodicBCs;
}

MeshTopology::MeshTopology(MeshGeometryPtr meshGeometry, vector<PeriodicBCPtr> periodicBCs)
{
  unsigned spaceDim = meshGeometry->vertices()[0].size();

  init(spaceDim);
  _periodicBCs = periodicBCs;

  vector< vector<double> > vertices = meshGeometry->vertices();

  vector<int> myVertexIndexForMeshGeometryIndex(vertices.size());
  for (int i=0; i<vertices.size(); i++)
  {
    myVertexIndexForMeshGeometryIndex[i] = getVertexIndexAdding(vertices[i], 1e-14);
  }
  //  _vertices = meshGeometry->vertices();

  //  for (int vertexIndex=0; vertexIndex<_vertices.size(); vertexIndex++) {
  //    _vertexMap[_vertices[vertexIndex]] = vertexIndex;
  //  }

  TEUCHOS_TEST_FOR_EXCEPTION(meshGeometry->cellTopos().size() != meshGeometry->elementVertices().size(), std::invalid_argument,
                             "length of cellTopos != length of elementVertices");

  int numElements = meshGeometry->cellTopos().size();

  GlobalIndexType cellID = 0;
  for (int i=0; i<numElements; i++)
  {
    CellTopoPtr cellTopo = meshGeometry->cellTopos()[i];
    vector< unsigned > cellVerticesInMeshGeometry = meshGeometry->elementVertices()[i];
    vector<unsigned> cellVertices;
    for (int j=0; j<cellVerticesInMeshGeometry.size(); j++)
    {
      cellVertices.push_back(myVertexIndexForMeshGeometryIndex[cellVerticesInMeshGeometry[j]]);
    }
    addCell(cellID, cellTopo, cellVertices);
    cellID++;
  }
}

unsigned MeshTopology::activeCellCount()
{
  return _activeCells.size();
}

const set<unsigned> & MeshTopology::getActiveCellIndices()
{
  return _activeCells;
}

// LLVM memory approximations come from http://info.prelert.com/blog/stl-container-memory-usage
template<typename A, typename B>
long long approximateMapSizeLLVM(map<A,B> &someMap)   // in bytes
{
  // 24 bytes for the map itself; nodes are 32 bytes + sizeof(pair<A,B>) each
  // if A and B are containers, this won't count their contents...

  map<int, int> emptyMap;

  int MAP_OVERHEAD = sizeof(emptyMap);
  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal

  return MAP_OVERHEAD + (MAP_NODE_OVERHEAD + sizeof(pair<A,B>)) * someMap.size();
}

template<typename A>
long long approximateSetSizeLLVM(set<A> &someSet)   // in bytes
{
  // 48 bytes for the set itself; nodes are 32 bytes + sizeof(pair<A,B>) each
  // if A and B are containers, this won't count their contents...

  set<int> emptySet;
  int SET_OVERHEAD = sizeof(emptySet);

  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal

  return SET_OVERHEAD + (MAP_NODE_OVERHEAD + sizeof(A)) * someSet.size();
}

template<typename A>
long long approximateVectorSizeLLVM(vector<A> &someVector)   // in bytes
{
  // 24 bytes for the vector itself; nodes are 32 bytes + sizeof(pair<A,B>) each
  // if A and B are containers, this won't count their contents...
  vector<int> emptyVector;
  int VECTOR_OVERHEAD = sizeof(someVector);

  return VECTOR_OVERHEAD + sizeof(A) * someVector.capacity();
}

map<string, long long> MeshTopology::approximateMemoryCosts()
{
  map<string, long long> variableCost;

  // calibrate by computing some sizes
  map<int, int> emptyMap;
  set<int> emptySet;
  vector<int> emptyVector;

  int MAP_OVERHEAD = sizeof(emptyMap);
  int SET_OVERHEAD = sizeof(emptySet);
  int VECTOR_OVERHEAD = sizeof(emptyVector);

  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal

  variableCost["_spaceDim"] = sizeof(_spaceDim);

  variableCost["_vertexMap"] = approximateMapSizeLLVM(_vertexMap);

  variableCost["_vertices"] = VECTOR_OVERHEAD; // for the outer vector _vertices.
  for (vector< vector<double> >::iterator entryIt = _vertices.begin(); entryIt != _vertices.end(); entryIt++)
  {
    variableCost["_vertices"] += approximateVectorSizeLLVM(*entryIt);
  }
  variableCost["_vertices"] += VECTOR_OVERHEAD * (_vertices.capacity() - _vertices.size());

  variableCost["_periodicBCs"] = approximateVectorSizeLLVM(_periodicBCs);

  variableCost["_periodicBCIndicesMatchingNode"] = MAP_OVERHEAD; // for map _periodicBCIndicesMatchingNode
  for (map<IndexType, set< pair<int, int> > >::iterator entryIt = _periodicBCIndicesMatchingNode.begin(); entryIt != _periodicBCIndicesMatchingNode.end(); entryIt++)
  {
    variableCost["_periodicBCIndicesMatchingNode"] += MAP_NODE_OVERHEAD;
    variableCost["_periodicBCIndicesMatchingNode"] += sizeof(IndexType);
    variableCost["_periodicBCIndicesMatchingNode"] += approximateSetSizeLLVM(entryIt->second);
  }

  variableCost["_equivalentNodeViaPeriodicBC"] = approximateMapSizeLLVM(_equivalentNodeViaPeriodicBC); // for map _equivalentNodeViaPeriodicBC

  variableCost["_entities"] = VECTOR_OVERHEAD; // for outer vector _entities
  for (vector< vector< vector<IndexType> > >::iterator entryIt = _entities.begin(); entryIt != _entities.end(); entryIt++)
  {
    variableCost["_entities"] += VECTOR_OVERHEAD; //
    for (vector< vector<IndexType> >::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_entities"] += approximateVectorSizeLLVM(*entry2It);
    }
    variableCost["_entities"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
  variableCost["_entities"] += VECTOR_OVERHEAD * (_entities.capacity() - _entities.size());

  variableCost["_knownEntities"] = VECTOR_OVERHEAD; // for outer vector _knownEntities
  for (vector< map< vector<IndexType>, IndexType > >::iterator entryIt = _knownEntities.begin(); entryIt != _knownEntities.end(); entryIt++)
  {
    variableCost["_knownEntities"] += MAP_OVERHEAD; // for inner map
    for (map< vector<IndexType>, IndexType >::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      vector<IndexType> entryVector = entry2It->first;
      variableCost["_knownEntities"] += approximateVectorSizeLLVM(entryVector) + sizeof(IndexType);
    }
  }
  variableCost["_knownEntities"] += MAP_OVERHEAD * (_knownEntities.capacity() - _knownEntities.size());

  variableCost["_canonicalEntityOrdering"] = VECTOR_OVERHEAD; // for outer vector _canonicalEntityOrdering
  for (vector< vector< vector<IndexType> > >::iterator entryIt = _canonicalEntityOrdering.begin(); entryIt != _canonicalEntityOrdering.end(); entryIt++)
  {
    variableCost["_canonicalEntityOrdering"] += VECTOR_OVERHEAD;
    for (vector< vector<IndexType> >::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_canonicalEntityOrdering"] += approximateVectorSizeLLVM(*entry2It);
    }
    variableCost["_canonicalEntityOrdering"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
  variableCost["_canonicalEntityOrdering"] += MAP_OVERHEAD * (_canonicalEntityOrdering.capacity() - _canonicalEntityOrdering.size());

  variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD; // for outer vector _activeCellsForEntities
  for (vector< vector< vector< pair<IndexType, unsigned> > > >::iterator entryIt = _activeCellsForEntities.begin(); entryIt != _activeCellsForEntities.end(); entryIt++)
  {
    variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD; // inner vector
    for (vector< vector< pair<IndexType, unsigned> > >::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_activeCellsForEntities"] += approximateVectorSizeLLVM(*entry2It);
    }
    variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
  variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD * (_activeCellsForEntities.capacity() - _activeCellsForEntities.size());

  variableCost["_sidesForEntities"] = VECTOR_OVERHEAD; // _sidesForEntities
  for (vector< vector< vector<IndexType> > >::iterator entryIt = _sidesForEntities.begin(); entryIt != _sidesForEntities.end(); entryIt++)
  {
    variableCost["_sidesForEntities"] += VECTOR_OVERHEAD;
    for (vector< vector<IndexType> >::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_sidesForEntities"] += sizeof(IndexType);
      variableCost["_sidesForEntities"] += approximateVectorSizeLLVM(*entry2It);
    }
    variableCost["_sidesForEntities"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
  variableCost["_sidesForEntities"] += VECTOR_OVERHEAD * (_sidesForEntities.capacity() - _sidesForEntities.size());

  variableCost["_cellsForSideEntities"] = approximateMapSizeLLVM(_cellsForSideEntities);

  variableCost["_boundarySides"] = approximateSetSizeLLVM(_boundarySides);

  variableCost["_parentEntities"] = VECTOR_OVERHEAD; // vector _parentEntities
  for (vector< map< IndexType, vector< pair<IndexType, unsigned> > > >::iterator entryIt = _parentEntities.begin(); entryIt != _parentEntities.end(); entryIt++)
  {
    variableCost["_parentEntities"] += MAP_OVERHEAD; // map
    for (map< IndexType, vector< pair<IndexType, unsigned> > > ::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_parentEntities"] += MAP_NODE_OVERHEAD; // map node
      variableCost["_parentEntities"] += sizeof(IndexType);
      variableCost["_parentEntities"] += approximateVectorSizeLLVM(entry2It->second);
    }
  }
  variableCost["_parentEntities"] += MAP_OVERHEAD * (_parentEntities.capacity() - _parentEntities.size());

  variableCost["_generalizedParentEntities"] = VECTOR_OVERHEAD; // vector _generalizedParentEntities
  for (vector< map< IndexType, pair<IndexType, unsigned> > >::iterator entryIt = _generalizedParentEntities.begin(); entryIt != _generalizedParentEntities.end(); entryIt++)
  {
    variableCost["_generalizedParentEntities"] += approximateMapSizeLLVM(*entryIt);
  }
  variableCost["_generalizedParentEntities"] += MAP_OVERHEAD * (_generalizedParentEntities.capacity() - _generalizedParentEntities.size());

  variableCost["_childEntities"] = VECTOR_OVERHEAD; // vector _childEntities
  for (vector< map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > > >::iterator entryIt = _childEntities.begin(); entryIt != _childEntities.end(); entryIt++)
  {
    variableCost["_childEntities"] += MAP_OVERHEAD; // map
    for (map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > >::iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_childEntities"] += MAP_NODE_OVERHEAD; // map node
      variableCost["_childEntities"] += sizeof(IndexType);

      variableCost["_childEntities"] += VECTOR_OVERHEAD; // vector
      for (vector< pair< RefinementPatternPtr, vector<IndexType> > >::iterator entry3It = entry2It->second.begin(); entry3It != entry2It->second.end(); entry3It++)
      {
        variableCost["_childEntities"] += sizeof(RefinementPatternPtr);
        variableCost["_childEntities"] += approximateVectorSizeLLVM(entry3It->second);
      }
      variableCost["_childEntities"] += sizeof(pair< RefinementPatternPtr, vector<IndexType> >) * (entry2It->second.capacity() - entry2It->second.size());
    }
  }
  variableCost["_childEntities"] += MAP_OVERHEAD * (_childEntities.capacity() - _childEntities.size());

  variableCost["_entityCellTopologyKeys"] = VECTOR_OVERHEAD; // _entityCellTopologyKeys vector
  for (vector< vector< CellTopologyKey > >::iterator entryIt = _entityCellTopologyKeys.begin(); entryIt != _entityCellTopologyKeys.end(); entryIt++)
  {
    variableCost["_entityCellTopologyKeys"] += approximateVectorSizeLLVM(*entryIt);
  }
  variableCost["_entityCellTopologyKeys"] += VECTOR_OVERHEAD * (_entityCellTopologyKeys.capacity() - _entityCellTopologyKeys.size());

  variableCost["_cells"] = approximateMapSizeLLVM(_cells); // _cells map
  for (auto cellEntry = _cells.begin(); cellEntry != _cells.end(); cellEntry++)
  {
    variableCost["_cells"] += cellEntry->second->approximateMemoryFootprint();
  }

  variableCost["_activeCells"] = approximateSetSizeLLVM(_activeCells);
  variableCost["_rootCells"] = approximateSetSizeLLVM(_rootCells);

  variableCost["_cellIDsWithCurves"] = approximateSetSizeLLVM(_cellIDsWithCurves);

  variableCost["_edgeToCurveMap"] = approximateMapSizeLLVM(_edgeToCurveMap);

  variableCost["_knownTopologies"] = approximateMapSizeLLVM( _knownTopologies );

  return variableCost;
}

long long MeshTopology::approximateMemoryFootprint()
{
  long long memSize = 0; // in bytes

  map<string, long long> variableCost = approximateMemoryCosts();
  for (map<string, long long>::iterator entryIt = variableCost.begin(); entryIt != variableCost.end(); entryIt++)
  {
    //    cout << entryIt->first << ": " << entryIt->second << endl;
    memSize += entryIt->second;
  }
  return memSize;
}

CellPtr MeshTopology::addCell(IndexType cellIndex, CellTopoPtr cellTopo, const FieldContainer<double> &cellVertices)
{
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo->getDimension() != _spaceDim, std::invalid_argument, "cellTopo dimension must match mesh topology dimension");
  TEUCHOS_TEST_FOR_EXCEPTION(cellVertices.dimension(0) != cellTopo->getVertexCount(), std::invalid_argument, "cellVertices must have shape (P,D)");
  TEUCHOS_TEST_FOR_EXCEPTION(cellVertices.dimension(1) != cellTopo->getDimension(), std::invalid_argument, "cellVertices must have shape (P,D)");

  int vertexCount = cellVertices.dimension(0);
  vector< vector<double> > cellVertexVector(vertexCount,vector<double>(_spaceDim));
  for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
  {
    for (int d=0; d<_spaceDim; d++)
    {
      cellVertexVector[vertexOrdinal][d] = cellVertices(vertexOrdinal,d);
    }
  }
  return addCell(cellIndex, cellTopo, cellVertexVector);
}

CellPtr MeshTopology::addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<vector<double> > &cellVertices)
{
  if (cellTopo->getNodeCount() != cellVertices.size())
  {
    cout << "ERROR: cellTopo->getNodeCount() != cellVertices.size().\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"cellTopo->getNodeCount() != cellVertices.size()");
  }

  vector<unsigned> vertexIndices = getVertexIndices(cellVertices);
  addCell(cellIndex, cellTopo, vertexIndices);
  return _cells[cellIndex];
}

CellPtr MeshTopology::addCell(IndexType cellIndex, CellTopoPtrLegacy shardsTopo, const vector<vector<double> > &cellVertices)
{
  CellTopoPtr cellTopo = CellTopology::cellTopology(*shardsTopo);
  return addCell(cellIndex, cellTopo, cellVertices);
}

unsigned MeshTopology::addCell(IndexType cellIndex, CellTopoPtrLegacy shardsTopo, const vector<unsigned> &cellVertices, unsigned parentCellIndex)
{
  CellTopoPtr cellTopo = CellTopology::cellTopology(*shardsTopo);
  return addCell(cellIndex, cellTopo, cellVertices, parentCellIndex);
}

unsigned MeshTopology::addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<unsigned> &cellVertices, unsigned parentCellIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_cells.find(cellIndex) != _cells.end(), std::invalid_argument, "addCell: cell with specified cellIndex already exists!");
  
  vector< vector< unsigned > > cellEntityPermutations;
  
  vector< vector<unsigned> > cellEntityIndices(_spaceDim); // subcdim, subcord
  for (int d=0; d<_spaceDim; d++)   // start with vertices, and go up to sides
  {
    int entityCount = cellTopo->getSubcellCount(d);
    if (d > 0) cellEntityPermutations.push_back(vector<unsigned>(entityCount));
    else cellEntityPermutations.push_back(vector<unsigned>(0)); // empty vector for d=0 -- we don't track permutations here...
    cellEntityIndices[d] = vector<unsigned>(entityCount);
    for (int j=0; j<entityCount; j++)
    {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      unsigned entityIndex, entityPermutation;
      vector< unsigned > nodes;
      if (d != 0)
      {
        int entityNodeCount = cellTopo->getNodeCount(d, j);
        for (int node=0; node<entityNodeCount; node++)
        {
          unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
          nodes.push_back(cellVertices[nodeIndexInCell]);
        }
      }
      else
      {
        nodes.push_back(cellVertices[j]);
      }

      entityIndex = addEntity(cellTopo->getSubcell(d, j), nodes, entityPermutation);
      cellEntityIndices[d][j] = entityIndex;

      // if d==0, then we don't need permutation info
      if (d != 0) cellEntityPermutations[d][j] = entityPermutation;
      if (_activeCellsForEntities[d].size() <= entityIndex)   // expand container
      {
        _activeCellsForEntities[d].resize(entityIndex + 1, vector< pair<IndexType, unsigned> >());
      }
      _activeCellsForEntities[d][entityIndex].push_back(make_pair(cellIndex,j));

      // now that we've added, sort:
      std::sort(_activeCellsForEntities[d][entityIndex].begin(), _activeCellsForEntities[d][entityIndex].end());

      if (d == 0)   // vertex --> should set parent relationships for any vertices that are equivalent via periodic BCs
      {
        if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end())
        {
          for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[entityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[entityIndex].end(); bcIt++)
          {
            IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, *bcIt)];
            if (_activeCellsForEntities[d].size() <= equivalentNode)   // expand container
            {
              _activeCellsForEntities[d].resize(equivalentNode + 1, vector< pair<IndexType, unsigned> >());
            }
            _activeCellsForEntities[d][equivalentNode].push_back(make_pair(cellIndex, j));
            // now that we've added, sort:
            std::sort(_activeCellsForEntities[d][equivalentNode].begin(), _activeCellsForEntities[d][equivalentNode].end());
          }
        }
      }
    }
  }
  CellPtr cell = Teuchos::rcp( new Cell(cellTopo, cellVertices, cellEntityPermutations, cellIndex, this) );
  _cells[cellIndex] = cell;
  _activeCells.insert(cellIndex);
  _rootCells.insert(cellIndex); // will remove if a parent relationship is established
  if (parentCellIndex != -1)
  {
    cell->setParent(getCell(parentCellIndex));
  }

  // set neighbors:
  unsigned sideDim = _spaceDim - 1;
  unsigned sideCount = cellTopo->getSideCount();
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
    addCellForSide(cellIndex,sideOrdinal,sideEntityIndex);
  }
  bool allowSameCellIndices = (_periodicBCs.size() > 0); // for periodic BCs, we allow a cell to be its own neighbor
  
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
    unsigned cellCountForSide = getCellCountForSide(sideEntityIndex);
    if (cellCountForSide == 2)   // compatible neighbors
    {
      pair<unsigned,unsigned> firstNeighbor  = getFirstCellForSide(sideEntityIndex);
      pair<unsigned,unsigned> secondNeighbor = getSecondCellForSide(sideEntityIndex);
      CellPtr firstCell = _cells[firstNeighbor.first];
      CellPtr secondCell = _cells[secondNeighbor.first];
      firstCell->setNeighbor(firstNeighbor.second, secondNeighbor.first, secondNeighbor.second, allowSameCellIndices);
      secondCell->setNeighbor(secondNeighbor.second, firstNeighbor.first, firstNeighbor.second, allowSameCellIndices);
      if (_boundarySides.find(sideEntityIndex) != _boundarySides.end())
      {
        if (_childEntities[sideDim].find(sideEntityIndex) != _childEntities[sideDim].end())
        {
          cout << "Unhandled case: boundary side acquired neighbor after being refined.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled case: boundary side acquired neighbor after being refined");
        }
        _boundarySides.erase(sideEntityIndex);
      }
      // if the pre-existing neighbor is refined, set its descendants to have the appropriate neighbor.
      MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
      if (firstCell->isParent(thisPtr))
      {
        vector< pair< GlobalIndexType, unsigned> > firstCellDescendants = firstCell->getDescendantsForSide(firstNeighbor.second, thisPtr);
        for (vector< pair< GlobalIndexType, unsigned> >::iterator descIt = firstCellDescendants.begin(); descIt != firstCellDescendants.end(); descIt++)
        {
          unsigned childCellIndex = descIt->first;
          unsigned childSideIndex = descIt->second;
          getCell(childCellIndex)->setNeighbor(childSideIndex, secondNeighbor.first, secondNeighbor.second);
        }
      }
      if (secondCell->isParent(thisPtr))   // I don't think we should ever get here
      {
        vector< pair< GlobalIndexType, unsigned> > secondCellDescendants = secondCell->getDescendantsForSide(secondNeighbor.first, thisPtr);
        for (vector< pair< GlobalIndexType, unsigned> >::iterator descIt = secondCellDescendants.begin(); descIt != secondCellDescendants.end(); descIt++)
        {
          GlobalIndexType childCellIndex = descIt->first;
          unsigned childSideOrdinal = descIt->second;
          getCell(childCellIndex)->setNeighbor(childSideOrdinal, firstNeighbor.first, firstNeighbor.second);
        }
      }
    }
    else if (cellCountForSide == 1)     // just this side
    {
      if (parentCellIndex == -1)   // for now anyway, we are on the boundary...
      {
        _boundarySides.insert(sideEntityIndex);
      }
      else
      {
        vector< pair<unsigned, unsigned> > sideAncestry = getConstrainingSideAncestry(sideEntityIndex);
        // the last entry, if any, should refer to an active cell's side...
        if (sideAncestry.size() > 0)
        {
          unsigned sideAncestorIndex = sideAncestry[sideAncestry.size()-1].first;
          vector< pair<unsigned, unsigned> > activeCellEntries = _activeCellsForEntities[sideDim][sideAncestorIndex];
          if (activeCellEntries.size() != 1)
          {
            cout << "Internal error: activeCellEntries does not have the expected size.\n";
            cout << "sideEntityIndex: " << sideEntityIndex << endl;
            cout << "sideAncestorIndex: " << sideAncestorIndex << endl;

            printEntityVertices(sideDim, sideEntityIndex);
            printEntityVertices(sideDim, sideAncestorIndex);

            TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Internal error: activeCellEntries does not have the expected size.\n");
          }
          pair<unsigned,unsigned> activeCellEntry = activeCellEntries[0];
          unsigned neighborCellIndex = activeCellEntry.first;
          unsigned sideIndexInNeighbor = activeCellEntry.second;
          cell->setNeighbor(sideOrdinal, neighborCellIndex, sideIndexInNeighbor);
        }
      }
    }

    for (int d=0; d<sideDim; d++)
    {
      set<unsigned> sideSubcellIndices = getEntitiesForSide(sideEntityIndex, d);
      for (set<unsigned>::iterator subcellIt = sideSubcellIndices.begin(); subcellIt != sideSubcellIndices.end(); subcellIt++)
      {
        unsigned subcellEntityIndex = *subcellIt;
        addSideForEntity(d, subcellEntityIndex, sideEntityIndex);
        if (d==0)
        {
          if (_periodicBCIndicesMatchingNode.find(subcellEntityIndex) != _periodicBCIndicesMatchingNode.end())
          {
            for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[subcellEntityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[subcellEntityIndex].end(); bcIt++)
            {
              IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(subcellEntityIndex, *bcIt)];

              addSideForEntity(d, equivalentNode, sideEntityIndex);
            }
          }
        }
      }
    }
    // for convenience, include the side itself in the _sidesForEntities lookup:
    addSideForEntity(sideDim, sideEntityIndex, sideEntityIndex);
  }

  return cellIndex;
}

void MeshTopology::addCellForSide(unsigned int cellIndex, unsigned int sideOrdinal, unsigned int sideEntityIndex)
{
  if (_cellsForSideEntities.find(sideEntityIndex) == _cellsForSideEntities.end())
  {
    pair< unsigned, unsigned > cell1 = make_pair(cellIndex, sideOrdinal);
    pair< unsigned, unsigned > cell2 = {-1,-1};
    
    // check for equivalent side that matches periodic BCs
    
    
    _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
  }
  else
  {
    pair< unsigned, unsigned > cell1 = _cellsForSideEntities[sideEntityIndex].first;
    pair< unsigned, unsigned > cell2 = _cellsForSideEntities[sideEntityIndex].second;

    CellPtr cellToAdd = getCell(cellIndex);
    unsigned parentCellIndex;
    if ( cellToAdd->getParent().get() == NULL)
    {
      parentCellIndex = -1;
    }
    else
    {
      parentCellIndex = cellToAdd->getParent()->cellIndex();
    }
    if (parentCellIndex == cell1.first)
    {
      // then replace cell1's entry with the new one
      cell1.first = cellIndex;
      cell1.second = sideOrdinal;
    }
    else if ((cell2.first == -1) || (parentCellIndex == cell2.first))
    {
      cell2.first = cellIndex;
      cell2.second = sideOrdinal;
    }
    else
    {
      cout << "Internal error: attempt to add 3rd cell for side with entity index " << sideEntityIndex << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: attempt to add 3rd cell for side");
    }
    _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
  }
}

void MeshTopology::addEdgeCurve(pair<unsigned,unsigned> edge, ParametricCurvePtr curve)
{
  // note: does NOT update the MeshTransformationFunction.  That's caller's responsibility,
  // because we don't know whether there are more curves coming for the affected elements.

  unsigned edgeDim = 1;
  vector<unsigned> edgeNodes;
  edgeNodes.push_back(edge.first);
  edgeNodes.push_back(edge.second);

  std::sort(edgeNodes.begin(), edgeNodes.end());

  if (_knownEntities[edgeDim].find(edgeNodes) == _knownEntities[edgeDim].end() )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "edge not found.");
  }
  unsigned edgeIndex = _knownEntities[edgeDim][edgeNodes];
  if (getChildEntities(edgeDim, edgeIndex).size() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "setting curves along broken edges not supported.  Should set for each piece separately.");
  }

  // check that the curve agrees with the vertices in the mesh:
  vector<double> v0 = getVertex(edge.first);
  vector<double> v1 = getVertex(edge.second);

  int spaceDim = 2; // v0.size();
  FieldContainer<double> curve0(spaceDim);
  FieldContainer<double> curve1(spaceDim);
  curve->value(0, curve0(0), curve0(1));
  curve->value(1, curve1(0), curve1(1));
  double maxDiff = 0;
  double tol = 1e-14;
  for (int d=0; d<spaceDim; d++)
  {
    maxDiff = std::max(maxDiff, abs(curve0(d)-v0[d]));
    maxDiff = std::max(maxDiff, abs(curve1(d)-v1[d]));
  }
  if (maxDiff > tol)
  {
    cout << "Error: curve's endpoints do not match edge vertices (maxDiff in coordinates " << maxDiff << ")" << endl;
    cout << "curve0:\n" << curve0;
    cout << "v0: (" << v0[0] << ", " << v0[1] << ")\n";
    cout << "curve1:\n" << curve1;
    cout << "v1: (" << v1[0] << ", " << v1[1] << ")\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Curve does not match vertices");
  }

  _edgeToCurveMap[edge] = curve;
  pair<IndexType,IndexType> reverseEdge = {edge.second,edge.first};
  _edgeToCurveMap[reverseEdge] = ParametricCurve::reverse(curve);

  vector< pair<IndexType, unsigned> > cellsForEdge = _activeCellsForEntities[edgeDim][edgeIndex];
  //  (cellIndex, entityOrdinalInCell)
  for (auto cellForEdge : cellsForEdge)
  {
    IndexType cellIndex = cellForEdge.first;
    _cellIDsWithCurves.insert(cellIndex);
    
    if (this->getDimension() == 3)
    {
      pair<unsigned,unsigned> otherEdge;
      // then we must be doing space-time, and we should check that the corresponding edge on the
      // other side gets the same curve
      CellPtr cell = getCell(cellIndex);
      unsigned spaceTimeEdgeOrdinal = cell->findSubcellOrdinal(edgeDim, edgeIndex);
      
      vector<IndexType> cellEdgeVertexNodes = cell->getEntityVertexIndices(edgeDim, spaceTimeEdgeOrdinal);
      bool swapped; // in cell relative to the edge we got called with
      if ((cellEdgeVertexNodes[0] == edge.first) && (cellEdgeVertexNodes[1] == edge.second))
      {
        swapped = false;
      }
      else if ((cellEdgeVertexNodes[1] == edge.first) && (cellEdgeVertexNodes[0] == edge.second))
      {
        swapped = true;
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "internal error: cellEdgeVertexNodes do not match edge");
      }
      
      CellTopoPtr spaceTopo = cell->topology()->getTensorialComponent();

      int spaceDim = this->getDimension() - 1;
      unsigned vertexOrdinal0 = cell->topology()->getNodeMap(edgeDim, spaceTimeEdgeOrdinal, 0);
      unsigned vertexOrdinal1 = cell->topology()->getNodeMap(edgeDim, spaceTimeEdgeOrdinal, 1);

      bool atTimeZero = (vertexOrdinal0 < spaceTopo->getNodeCount()); // a bit hackish: uses knowledge of how the vertices are numbered in CellTopology
      
      TEUCHOS_TEST_FOR_EXCEPTION(atTimeZero && (vertexOrdinal1 >= spaceTopo->getNodeCount()), std::invalid_argument, "Looks like a curvilinear edge goes from one temporal side to a different one.  This is not allowed!");
      
      TEUCHOS_TEST_FOR_EXCEPTION(!atTimeZero && (vertexOrdinal1 < spaceTopo->getNodeCount()), std::invalid_argument, "Looks like a curvilinear edge goes from one temporal side to a different one.  This is not allowed!");
      
      unsigned timeSide0 = cell->topology()->getTemporalSideOrdinal(0);
      unsigned timeSide1 = cell->topology()->getTemporalSideOrdinal(1);
      
      int vertexDim = 0;
      
      unsigned otherVertexOrdinal0InSpaceTimeTopology, otherVertexOrdinal1InSpaceTimeTopology;
      if (atTimeZero)
      {
        unsigned vertexOrdinal0InTimeSide = CamelliaCellTools::subcellReverseOrdinalMap(cell->topology(), spaceDim, timeSide0, vertexDim, vertexOrdinal0);
        unsigned vertexOrdinal1InTimeSide = CamelliaCellTools::subcellReverseOrdinalMap(cell->topology(), spaceDim, timeSide0, vertexDim, vertexOrdinal1);
        otherVertexOrdinal0InSpaceTimeTopology = CamelliaCellTools::subcellOrdinalMap(cell->topology(), spaceDim, timeSide1, vertexDim, vertexOrdinal0InTimeSide);
        otherVertexOrdinal1InSpaceTimeTopology = CamelliaCellTools::subcellOrdinalMap(cell->topology(), spaceDim, timeSide1, vertexDim, vertexOrdinal1InTimeSide);
      }
      else
      {
        unsigned vertexOrdinal0InTimeSide = CamelliaCellTools::subcellReverseOrdinalMap(cell->topology(), spaceDim, timeSide1, vertexDim, vertexOrdinal0);
        unsigned vertexOrdinal1InTimeSide = CamelliaCellTools::subcellReverseOrdinalMap(cell->topology(), spaceDim, timeSide1, vertexDim, vertexOrdinal1);
        otherVertexOrdinal0InSpaceTimeTopology = CamelliaCellTools::subcellOrdinalMap(cell->topology(), spaceDim, timeSide0, vertexDim, vertexOrdinal0InTimeSide);
        otherVertexOrdinal1InSpaceTimeTopology = CamelliaCellTools::subcellOrdinalMap(cell->topology(), spaceDim, timeSide0, vertexDim, vertexOrdinal1InTimeSide);
      }
      IndexType otherVertex0EntityIndex = cell->entityIndex(vertexDim, otherVertexOrdinal0InSpaceTimeTopology);
      IndexType otherVertex1EntityIndex = cell->entityIndex(vertexDim, otherVertexOrdinal1InSpaceTimeTopology);
      otherEdge = {otherVertex0EntityIndex,otherVertex1EntityIndex};
      if (swapped)
      {
        otherEdge = {otherEdge.second,otherEdge.first};
      }
      if (_edgeToCurveMap.find(otherEdge) == _edgeToCurveMap.end())
      {
        addEdgeCurve(otherEdge, curve);
      }
    }
  }
}

unsigned MeshTopology::addEntity(CellTopoPtr entityTopo, const vector<unsigned> &entityVertices, unsigned &entityPermutation)
{
  set< unsigned > nodeSet;
  nodeSet.insert(entityVertices.begin(),entityVertices.end());

  if (nodeSet.size() != entityVertices.size())
  {
    for (IndexType vertexIndex : entityVertices)
      printVertex(vertexIndex);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Entities may not have repeated vertices");
  }
  unsigned d  = entityTopo->getDimension();
  unsigned entityIndex = getEntityIndex(d, nodeSet);

  vector<unsigned> sortedVertices(nodeSet.begin(),nodeSet.end());

  if ( entityIndex == -1 )
  {
    // new entity
    entityIndex = _entities[d].size();
    _entities[d].push_back(sortedVertices);
    _knownEntities[d].insert(make_pair(sortedVertices, entityIndex));
    if (d != 0) _canonicalEntityOrdering[d].push_back(entityVertices);
    entityPermutation = 0;
    if (_knownTopologies.find(entityTopo->getKey()) == _knownTopologies.end())
    {
      _knownTopologies[entityTopo->getKey()] = entityTopo;
    }
    _entityCellTopologyKeys[d].push_back(entityTopo->getKey());
  }
  else
  {
    // existing entity
    // maintain order but relabel nodes according to periodic BCs:
    vector<IndexType> canonicalVerticesNewOrdering = getCanonicalEntityNodesViaPeriodicBCs(d, entityVertices);
    //
    //    Camellia::print("canonicalEntityOrdering",_canonicalEntityOrdering[d][entityIndex]);
    if (d==0) entityPermutation = 0;
    else entityPermutation = CamelliaCellTools::permutationMatchingOrder(entityTopo, _canonicalEntityOrdering[d][entityIndex], canonicalVerticesNewOrdering);
  }
  return entityIndex;
}

void MeshTopology::addChildren(IndexType firstChildIndex, CellPtr parentCell, const vector< CellTopoPtr > &childTopos, const vector< vector<unsigned> > &childVertices)
{
  int numChildren = childTopos.size();
  TEUCHOS_TEST_FOR_EXCEPTION(numChildren != childVertices.size(), std::invalid_argument, "childTopos and childVertices must be the same size");
  vector< CellPtr > children;
  IndexType cellIndex = firstChildIndex; // children get continguous cell indices
  for (int childIndex=0; childIndex<numChildren; childIndex++)
  {
    addCell(cellIndex, childTopos[childIndex], childVertices[childIndex],parentCell->cellIndex());
    children.push_back(_cells[cellIndex]);
    _rootCells.erase(cellIndex);
    cellIndex++;
  }
  parentCell->setChildren(children);
  
  // if any entity sets contain parent cell, add child cells, too
  for (auto entry : _entitySets)
  {
    EntitySetPtr entitySet = entry.second;
    if (entitySet->containsEntity(this->getDimension(), parentCell->cellIndex()))
    {
      for (CellPtr child : children)
      {
        entitySet->addEntity(this->getDimension(), child->cellIndex());
      }
    }
  }
}

void MeshTopology::addSideForEntity(unsigned int entityDim, IndexType entityIndex, IndexType sideEntityIndex)
{
  if (_sidesForEntities[entityDim].size() <= entityIndex)
  {
    _sidesForEntities[entityDim].resize(entityIndex + 1);
  }

  std::vector<IndexType>::iterator searchResult = std::find(_sidesForEntities[entityDim][entityIndex].begin(), _sidesForEntities[entityDim][entityIndex].end(), sideEntityIndex);
  if (searchResult == _sidesForEntities[entityDim][entityIndex].end())
  {
    _sidesForEntities[entityDim][entityIndex].push_back(sideEntityIndex);
  }
}

void MeshTopology::addVertex(const vector<double> &vertex)
{
  double tol = 1e-15;
  getVertexIndexAdding(vertex, tol);
}

void MeshTopology::applyTag(std::string tagName, int tagID, EntitySetPtr entitySet)
{
  _tagSetsInteger[tagName].push_back({entitySet->getHandle(), tagID});
}

vector<IndexType> MeshTopology::getCanonicalEntityNodesViaPeriodicBCs(unsigned d, const vector<IndexType> &myEntityNodes)
{
  vector<IndexType> sortedNodes(myEntityNodes.begin(),myEntityNodes.end());
  std::sort(sortedNodes.begin(), sortedNodes.end());
  
  if (_knownEntities[d].find(sortedNodes) != _knownEntities[d].end())
  {
    return myEntityNodes;
  }
  else
  {
    if (d==0)
    {
      IndexType vertexIndex = myEntityNodes[0];
      if (_canonicalVertexPeriodic.find(vertexIndex) != _canonicalVertexPeriodic.end())
      {
        return {_canonicalVertexPeriodic[vertexIndex]};
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTopology error: vertex not found.");
      }
    }
    
    // compute the intersection of the periodic BCs that match each node in nodeSet
    set< pair<int, int> > matchingPeriodicBCsIntersection;
    bool firstNode = true;
    for (vector<IndexType>::const_iterator nodeIt=myEntityNodes.begin(); nodeIt!=myEntityNodes.end(); nodeIt++)
    {
      if (_periodicBCIndicesMatchingNode.find(*nodeIt) == _periodicBCIndicesMatchingNode.end())
      {
        matchingPeriodicBCsIntersection.clear();
        break;
      }
      if (firstNode)
      {
        matchingPeriodicBCsIntersection = _periodicBCIndicesMatchingNode[*nodeIt];
        firstNode = false;
      }
      else
      {
        set< pair<int, int> > newSet;
        set< pair<int, int> > matchesForThisNode = _periodicBCIndicesMatchingNode[*nodeIt];
        for (set< pair<int, int> >::iterator prevMatchIt=matchingPeriodicBCsIntersection.begin();
             prevMatchIt != matchingPeriodicBCsIntersection.end(); prevMatchIt++)
        {
          if (matchesForThisNode.find(*prevMatchIt) != matchesForThisNode.end())
          {
            newSet.insert(*prevMatchIt);
          }
        }
        matchingPeriodicBCsIntersection = newSet;
      }
    }
    // for each periodic BC that remains, convert the nodeSet using that periodic BC
    for (set< pair<int, int> >::iterator bcIt=matchingPeriodicBCsIntersection.begin();
         bcIt != matchingPeriodicBCsIntersection.end(); bcIt++)
    {
      pair<int,int> matchingBC = *bcIt;
      vector<IndexType> equivalentNodeVector;
      for (vector<IndexType>::const_iterator nodeIt=myEntityNodes.begin(); nodeIt!=myEntityNodes.end(); nodeIt++)
      {
        equivalentNodeVector.push_back(_equivalentNodeViaPeriodicBC[make_pair(*nodeIt, matchingBC)]);
      }

      vector<IndexType> sortedEquivalentNodeVector = equivalentNodeVector;
      std::sort(sortedEquivalentNodeVector.begin(), sortedEquivalentNodeVector.end());

      if (_knownEntities[d].find(sortedEquivalentNodeVector) != _knownEntities[d].end())
      {
        return equivalentNodeVector;
      }
    }
  }
  return vector<IndexType>(); // empty result meant to indicate not found...
}

bool MeshTopology::cellHasCurvedEdges(unsigned cellIndex)
{
  CellPtr cell = getCell(cellIndex);
  unsigned edgeCount = cell->topology()->getEdgeCount();
  unsigned edgeDim = 1;
  for (int edgeOrdinal=0; edgeOrdinal<edgeCount; edgeOrdinal++)
  {
    unsigned edgeIndex = cell->entityIndex(edgeDim, edgeOrdinal);
    unsigned v0 = _canonicalEntityOrdering[edgeDim][edgeIndex][0];
    unsigned v1 = _canonicalEntityOrdering[edgeDim][edgeIndex][1];
    pair<unsigned, unsigned> edge = make_pair(v0, v1);
    pair<unsigned, unsigned> edgeReversed = make_pair(v1, v0);
    if (_edgeToCurveMap.find(edge) != _edgeToCurveMap.end())
    {
      return true;
    }
    if (_edgeToCurveMap.find(edgeReversed) != _edgeToCurveMap.end())
    {
      return true;
    }
  }
  return false;
}

bool MeshTopology::cellContainsPoint(GlobalIndexType cellID, const vector<double> &point, int cubatureDegree)
{
  // note that this design, with a single point being passed in, will be quite inefficient
  // if there are many points.  TODO: revise to allow multiple points (returning vector<bool>, maybe)
  int numCells = 1, numPoints = 1;
  FieldContainer<double> physicalPoints(numCells,numPoints,_spaceDim);
  for (int d=0; d<_spaceDim; d++)
  {
    physicalPoints(0,0,d) = point[d];
  }
  //  cout << "cell " << elem->cellID() << ": (" << x << "," << y << ") --> ";
  FieldContainer<double> refPoints(numCells,numPoints,_spaceDim);
  MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  CamelliaCellTools::mapToReferenceFrame(refPoints, physicalPoints, thisPtr, cellID, cubatureDegree);

  CellTopoPtr cellTopo = getCell(cellID)->topology();

  int result = CamelliaCellTools::checkPointInclusion(&refPoints[0], _spaceDim, cellTopo);
  return result == 1;
}

IndexType MeshTopology::cellCount()
{
  return _cells.size();
}

vector<IndexType> MeshTopology::cellIDsForPoints(const FieldContainer<double> &physicalPoints)
{
  // returns a vector of an active element per point, or null if there is no element including that point
  vector<GlobalIndexType> cellIDs;
  //  cout << "entered elementsForPoints: \n" << physicalPoints;
  int numPoints = physicalPoints.dimension(0);

  int spaceDim = this->getDimension();

  set<GlobalIndexType> rootCellIndices = this->getRootCellIndices();

  // NOTE: the above does depend on the domain of the mesh remaining fixed after refinements begin.

  for (int pointIndex=0; pointIndex<numPoints; pointIndex++)
  {
    vector<double> point;

    for (int d=0; d<spaceDim; d++)
    {
      point.push_back(physicalPoints(pointIndex,d));
    }

    // find the element from the original mesh that contains this point
    CellPtr cell;
    for (set<GlobalIndexType>::iterator cellIt = rootCellIndices.begin(); cellIt != rootCellIndices.end(); cellIt++)
    {
      GlobalIndexType cellID = *cellIt;
      int cubatureDegreeForCell = 1;
      if (_gda != NULL)
      {
        cubatureDegreeForCell = _gda->getCubatureDegree(cellID);
      }
      if (cellContainsPoint(cellID,point,cubatureDegreeForCell))
      {
        cell = getCell(cellID);
        break;
      }
    }
    if (cell.get() != NULL)
    {
      MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
      while ( cell->isParent(thisPtr) )
      {
        int numChildren = cell->numChildren();
        bool foundMatchingChild = false;
        for (int childOrdinal = 0; childOrdinal < numChildren; childOrdinal++)
        {
          CellPtr child = cell->children()[childOrdinal];
          int cubatureDegreeForCell = 1;
          if (_gda != NULL)
          {
            cubatureDegreeForCell = _gda->getCubatureDegree(child->cellIndex());
          }
          if ( cellContainsPoint(child->cellIndex(),point,cubatureDegreeForCell) )
          {
            cell = child;
            foundMatchingChild = true;
            break;
          }
        }
        if (!foundMatchingChild)
        {
          cout << "parent matches, but none of its children do... will return nearest cell centroid\n";
          int numVertices = cell->vertices().size();
          FieldContainer<double> vertices(numVertices,spaceDim);
          vector<unsigned> vertexIndices = cell->vertices();

          //vertices.resize(numVertices,dimension);
          for (unsigned vertexOrdinal = 0; vertexOrdinal < numVertices; vertexOrdinal++)
          {
            for (int d=0; d<spaceDim; d++)
            {
              vertices(vertexOrdinal,d) = getVertex(vertexIndices[vertexOrdinal])[d];
            }
          }

          cout << "parent vertices:\n" << vertices;
          double minDistance = numeric_limits<double>::max();
          int childSelected = -1;
          for (int childIndex = 0; childIndex < numChildren; childIndex++)
          {
            CellPtr child = cell->children()[childIndex];
            int numVertices = child->vertices().size();
            FieldContainer<double> vertices(numVertices,spaceDim);
            vector<unsigned> vertexIndices = child->vertices();

            //vertices.resize(numVertices,dimension);
            for (unsigned vertexOrdinal = 0; vertexOrdinal < numVertices; vertexOrdinal++)
            {
              for (int d=0; d<spaceDim; d++)
              {
                vertices(vertexOrdinal,d) = getVertex(vertexIndices[vertexOrdinal])[d];
              }
            }
            cout << "child " << childIndex << ", vertices:\n" << vertices;
            vector<double> cellCentroid = getCellCentroid(child->cellIndex());
            double squaredDistance = 0;
            for (int d=0; d<spaceDim; d++)
            {
              squaredDistance += (cellCentroid[d] - physicalPoints(pointIndex,d)) * (cellCentroid[d] - physicalPoints(pointIndex,d));
            }

            double distance = sqrt(squaredDistance);
            if (distance < minDistance)
            {
              minDistance = distance;
              childSelected = childIndex;
            }
          }
          cell = cell->children()[childSelected];
        }
      }
    }
    GlobalIndexType cellID = -1;
    if (cell.get() != NULL)
    {
      cellID = cell->cellIndex();
    }
    cellIDs.push_back(cellID);
  }
  return cellIDs;
}

EntitySetPtr MeshTopology::createEntitySet()
{
  // at some point, we might want to use MOAB for entity sets, etc., but for now, we just use an
  // EntityHandle equal to the ordinal of the entity set: start at 0 and increment as new ones are created...
  EntityHandle handle = _entitySets.size();
  
  EntitySetPtr entitySet = Teuchos::rcp( new EntitySet(handle) );
  _entitySets[handle] = entitySet;
  
  return entitySet;
}

CellPtr MeshTopology::findCellWithVertices(const vector< vector<double> > &cellVertices)
{
  CellPtr cell;
  vector<IndexType> vertexIndices;
  bool firstVertex = true;
  unsigned vertexDim = 0;
  set<IndexType> matchingCells;
  for (vector< vector<double> >::const_iterator vertexIt = cellVertices.begin(); vertexIt != cellVertices.end(); vertexIt++)
  {
    vector<double> vertex = *vertexIt;
    IndexType vertexIndex;
    if (! getVertexIndex(vertex, vertexIndex) )
    {
      cout << "vertex not found. returning NULL.\n";
      return cell;
    }
    // otherwise, vertexIndex has been populated
    vertexIndices.push_back(vertexIndex);

    set< pair<IndexType, unsigned> > matchingCellPairs = getCellsContainingEntity(vertexDim, vertexIndex);
    set<IndexType> matchingCellsIntersection;
    for (set< pair<IndexType, unsigned> >::iterator cellPairIt = matchingCellPairs.begin(); cellPairIt != matchingCellPairs.end(); cellPairIt++)
    {
      IndexType cellID = cellPairIt->first;
      if (firstVertex)
      {
        matchingCellsIntersection.insert(cellID);
      }
      else
      {
        if (matchingCells.find(cellID) != matchingCells.end())
        {
          matchingCellsIntersection.insert(cellID);
        }
      }
    }
    matchingCells = matchingCellsIntersection;
    firstVertex = false;
  }
  if (matchingCells.size() == 0)
  {
    return cell; // null
  }
  if (matchingCells.size() > 1)
  {
    cout << "WARNING: multiple matching cells found.  Returning first one that matches.\n";
  }
  cell = getCell(*matchingCells.begin());
  return cell;
}

set< pair<IndexType, unsigned> > MeshTopology::getActiveBoundaryCells()   // (cellIndex, sideOrdinal)
{
  set< pair<IndexType, unsigned> > boundaryCells;
  for (set<IndexType>::iterator boundarySideIt = _boundarySides.begin(); boundarySideIt != _boundarySides.end(); boundarySideIt++)
  {
    IndexType sideEntityIndex = *boundarySideIt;
    int cellCount = getCellCountForSide(sideEntityIndex);
    if (cellCount == 1)
    {
      pair<IndexType, unsigned> cellInfo = _cellsForSideEntities[sideEntityIndex].first;
      if (cellInfo.first == -1)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid cellIndex for side boundary.");
      }
      if (_activeCells.find(cellInfo.first) != _activeCells.end())
      {
        boundaryCells.insert(cellInfo);
        // DEBUGGING:
        //        if (getCell(cellInfo.first)->isParent()) {
        //          cout << "ERROR: cell is parent, but is stored as an active cell in the mesh...\n";
        //          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cell is parent, but is stored as an active cell in the mesh...");
        //        }
      }
    }
    else if (cellCount > 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "boundary side has more than 1 cell!");
    } // cellCount = 0 just means that the side has been refined; that's acceptable
  }
  return boundaryCells;
}

vector<double> MeshTopology::getCellCentroid(IndexType cellIndex)
{
  // average of the cell vertices
  vector<double> centroid(_spaceDim);
  CellPtr cell = getCell(cellIndex);
  unsigned vertexCount = cell->vertices().size();
  for (unsigned vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
  {
    unsigned vertexIndex = cell->vertices()[vertexOrdinal];
    for (unsigned d=0; d<_spaceDim; d++)
    {
      centroid[d] += _vertices[vertexIndex][d];
    }
  }
  for (unsigned d=0; d<_spaceDim; d++)
  {
    centroid[d] /= vertexCount;
  }
  return centroid;
}

unsigned MeshTopology::getCellCountForSide(IndexType sideEntityIndex)
{
  if (_cellsForSideEntities.find(sideEntityIndex) == _cellsForSideEntities.end())
  {
    return 0;
  }
  else
  {
    pair<IndexType, unsigned> cell1 = _cellsForSideEntities[sideEntityIndex].first;
    pair<IndexType, unsigned> cell2 = _cellsForSideEntities[sideEntityIndex].second;
    if (cell2.first == -1)
    {
      return 1;
    }
    else
    {
      return 2;
    }
  }
}

vector<EntitySetPtr> MeshTopology::getEntitySetsForTagID(string tagName, int tagID)
{
  if (_tagSetsInteger.find(tagName) == _tagSetsInteger.end()) return vector<EntitySetPtr>();
  
  vector<EntitySetPtr> entitySets;
  vector<pair<EntityHandle,int>> tagEntries = _tagSetsInteger[tagName];
  for (pair<EntityHandle,int> tagEntry : tagEntries)
  {
    if (tagEntry.second == tagID)
    {
      entitySets.push_back(getEntitySet(tagEntry.first));
    }
  }
  
  return entitySets;
}

EntitySetPtr MeshTopology::getEntitySet(EntityHandle entitySetHandle) const
{
  auto entry = _entitySets.find(entitySetHandle);
  if (entry == _entitySets.end()) return Teuchos::null;
  return entry->second;
}

EntitySetPtr MeshTopology::getEntitySetInitialTime() const
{
  if (_initialTimeEntityHandle == -1) return Teuchos::null;
  return getEntitySet(_initialTimeEntityHandle);
}

pair<IndexType, unsigned> MeshTopology::getFirstCellForSide(IndexType sideEntityIndex)
{
  if (_cellsForSideEntities.find(sideEntityIndex) == _cellsForSideEntities.end()) return {-1,-1};
  return _cellsForSideEntities[sideEntityIndex].first;
}

pair<IndexType, unsigned> MeshTopology::getSecondCellForSide(IndexType sideEntityIndex)
{
  if (_cellsForSideEntities.find(sideEntityIndex) == _cellsForSideEntities.end()) return {-1,-1};
  return _cellsForSideEntities[sideEntityIndex].second;
}

void MeshTopology::deactivateCell(CellPtr cell)
{
  //  cout << "deactivating cell " << cell->cellIndex() << endl;
  CellTopoPtr cellTopo = cell->topology();
  for (int d=0; d<_spaceDim; d++)   // start with vertices, and go up to sides
  {
    int entityCount = cellTopo->getSubcellCount(d);
    for (int j=0; j<entityCount; j++)
    {
      // for now, we treat vertices just like all the others--could save a bit of memory, etc. by not storing in _knownEntities[0], etc.
      int entityNodeCount = cellTopo->getNodeCount(d, j);
      set< unsigned > nodeSet;
      if (d != 0)
      {
        for (int node=0; node<entityNodeCount; node++)
        {
          unsigned nodeIndexInCell = cellTopo->getNodeMap(d, j, node);
          nodeSet.insert(cell->vertices()[nodeIndexInCell]);
        }
      }
      else
      {
        nodeSet.insert(cell->vertices()[j]);
      }

      unsigned entityIndex = getEntityIndex(d, nodeSet);
      if (entityIndex == -1)
      {
        // entity not found: an error
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cell entity not found!");
      }

      // delete from the _activeCellsForEntities store
      if (_activeCellsForEntities[d].size() <= entityIndex)
      {
        cout << "WARNING: Entity index is out of bounds for _activeCellsForEntities[" << d << "][" << entityIndex << "]\n";
      }
      else
      {
        vector<unsigned> indicesToDelete;
        int i = 0;
        for (vector< pair<IndexType, unsigned> >::iterator entryIt = _activeCellsForEntities[d][entityIndex].begin();
             entryIt != _activeCellsForEntities[d][entityIndex].end(); entryIt++, i++)
        {
          if ((entryIt->first == cell->cellIndex()) && (entryIt->second  == j))
          {
            indicesToDelete.push_back(i);
          }
        }
        // delete in reverse order
        for (int j=indicesToDelete.size()-1; j >= 0; j--)
        {
          int i = indicesToDelete[j];
          _activeCellsForEntities[d][entityIndex].erase(_activeCellsForEntities[d][entityIndex].begin() + i);
        }

        unsigned eraseCount = indicesToDelete.size();
        if (eraseCount==0)
        {
          cout << "WARNING: attempt was made to deactivate a non-active subcell topology...\n";
        }
        else
        {
          //        cout << "Erased _activeCellsForEntities[" << d << "][" << entityIndex << "] entry for (";
          //        cout << cell->cellIndex() << "," << j << ").  Remaining entries: ";
          //        set< pair<unsigned,unsigned> > remainingEntries = _activeCellsForEntities[d][entityIndex];
          //        for (set< pair<unsigned,unsigned> >::iterator entryIt = remainingEntries.begin(); entryIt != remainingEntries.end(); entryIt++) {
          //          cout << "(" << entryIt->first << "," << entryIt->second << ") ";
          //        }
          //        cout << endl;
        }
      }
      if (d == 0)   // vertex --> should delete entries for any that are equivalent via periodic BCs
      {
        if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end())
        {
          for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[entityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[entityIndex].end(); bcIt++)
          {
            IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, *bcIt)];
            if (_activeCellsForEntities[d].size() <= equivalentNode)
            {
              cout << "WARNING: Entity index is out of bounds for _activeCellsForEntities[" << d << "][" << equivalentNode << "]\n";
            }
            else
            {

              vector<unsigned> indicesToDelete;
              int i = 0;
              for (vector< pair<IndexType, unsigned> >::iterator entryIt = _activeCellsForEntities[d][equivalentNode].begin();
                   entryIt != _activeCellsForEntities[d][equivalentNode].end(); entryIt++, i++)
              {
                if ((entryIt->first == cell->cellIndex()) && (entryIt->second  == j))
                {
                  indicesToDelete.push_back(i);
                }
              }
              // delete in reverse order
              for (int j=indicesToDelete.size()-1; j > 0; j--)
              {
                int i = indicesToDelete[j];
                _activeCellsForEntities[d][equivalentNode].erase(_activeCellsForEntities[d][equivalentNode].begin() + i);
              }

              unsigned eraseCount = indicesToDelete.size();

              if (eraseCount==0)
              {
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

MeshTopologyPtr MeshTopology::deepCopy()
{
  MeshTopologyPtr meshTopoCopy = Teuchos::rcp( new MeshTopology(*this) );
  meshTopoCopy->deepCopyCells();
  return meshTopoCopy;
}

void MeshTopology::deepCopyCells()
{
  map<IndexType, CellPtr> oldCells = _cells;
  
  Teuchos::RCP<MeshTopology> thisPtr = Teuchos::rcp(this,false);

  // first pass: construct cells
  for (auto oldCellEntry : oldCells)
  {
    CellPtr oldCell = oldCellEntry.second;
    IndexType oldCellIndex = oldCellEntry.first;
    _cells[oldCellIndex] = Teuchos::rcp( new Cell(oldCell->topology(), oldCell->vertices(), oldCell->subcellPermutations(), oldCell->cellIndex(), this) );
    for (int sideOrdinal=0; sideOrdinal<oldCell->getSideCount(); sideOrdinal++)
    {
      pair<GlobalIndexType, unsigned> neighborInfo = oldCell->getNeighborInfo(sideOrdinal, thisPtr);
      _cells[oldCellIndex]->setNeighbor(sideOrdinal, neighborInfo.first, neighborInfo.second);
    }
  }

  // second pass: establish parent-child relationships
  for (auto oldCellEntry : oldCells)
  {
    IndexType oldCellIndex = oldCellEntry.first;
    CellPtr oldCell = oldCellEntry.second;

    CellPtr oldParent = oldCell->getParent();
    if (oldParent != Teuchos::null)
    {
      CellPtr newParent = _cells[oldParent->cellIndex()];
      newParent->setRefinementPattern(oldParent->refinementPattern());
      _cells[oldCellIndex]->setParent(newParent);
    }
    vector<CellPtr> children;
    for (int childOrdinal=0; childOrdinal<oldCell->children().size(); childOrdinal++)
    {
      CellPtr newChild = _cells[oldCell->children()[childOrdinal]->cellIndex()];
      children.push_back(newChild);
    }
    _cells[oldCellIndex]->setChildren(children);
  }
}

set<unsigned> MeshTopology::descendants(unsigned d, unsigned entityIndex)
{
  set<unsigned> allDescendants;

  allDescendants.insert(entityIndex);
  if (_childEntities[d].find(entityIndex) != _childEntities[d].end())
  {
    set<unsigned> unfollowedDescendants;
    for (unsigned i=0; i<_childEntities[d][entityIndex].size(); i++)
    {
      vector<unsigned> immediateChildren = _childEntities[d][entityIndex][i].second;
      unfollowedDescendants.insert(immediateChildren.begin(), immediateChildren.end());
    }
    for (set<unsigned>::iterator descIt=unfollowedDescendants.begin(); descIt!=unfollowedDescendants.end(); descIt++)
    {
      set<unsigned> myDescendants = descendants(d,*descIt);
      allDescendants.insert(myDescendants.begin(),myDescendants.end());
    }
  }

  return allDescendants;
}

bool MeshTopology::entityHasChildren(unsigned int d, IndexType entityIndex)
{
  if (_childEntities[d].find(entityIndex) == _childEntities[d].end()) return false;
  return _childEntities[d][entityIndex].size() > 0;
}

bool MeshTopology::entityHasParent(unsigned d, unsigned entityIndex)
{
  if (_parentEntities[d].find(entityIndex) == _parentEntities[d].end()) return false;
  return _parentEntities[d][entityIndex].size() > 0;
}

bool MeshTopology::entityHasGeneralizedParent(unsigned d, IndexType entityIndex)
{
  return _generalizedParentEntities[d].find(entityIndex) != _generalizedParentEntities[d].end();
}

bool MeshTopology::entityIsAncestor(unsigned d, unsigned ancestor, unsigned descendent)
{
  if (ancestor == descendent) return true;
  map< unsigned, vector< pair<unsigned, unsigned> > >::iterator parentIt = _parentEntities[d].find(descendent);
  while (parentIt != _parentEntities[d].end())
  {
    vector< pair<unsigned, unsigned> > parents = parentIt->second;
    unsigned parentEntityIndex = -1;
    for (vector< pair<unsigned, unsigned> >::iterator entryIt = parents.begin(); entryIt != parents.end(); entryIt++)
    {
      parentEntityIndex = entryIt->first;
      if (parentEntityIndex==ancestor)
      {
        return true;
      }
    }
    parentIt = _parentEntities[d].find(parentEntityIndex);
  }
  return false;
}

bool MeshTopology::entityIsGeneralizedAncestor(unsigned ancestorDimension, IndexType ancestor,
    unsigned descendentDimension, IndexType descendent)
{
  // note that this method does not treat the possibility of multiple parents, which can happen in
  // the context of anisotropic refinements.
  if (ancestorDimension == descendentDimension)
  {
    return entityIsAncestor(ancestorDimension, ancestor, descendent);
  }
  if (ancestorDimension < descendentDimension) return false;

  while (_generalizedParentEntities[descendentDimension].find(descendent) != _generalizedParentEntities[descendentDimension].end())
  {
    pair<IndexType, unsigned> generalizedParent = _generalizedParentEntities[descendentDimension][descendent];
    descendentDimension = generalizedParent.second;
    descendent = generalizedParent.first;
    if (descendent == ancestor) return true;
  }
  return false;
}

unsigned MeshTopology::getActiveCellCount(unsigned int d, unsigned int entityIndex)
{
  if (_activeCellsForEntities[d].size() <= entityIndex)
  {
    return 0;
  }
  else
  {
    return _activeCellsForEntities[d][entityIndex].size();
  }
}

vector< pair<unsigned,unsigned> > MeshTopology::getActiveCellIndices(unsigned d, unsigned entityIndex)
{
  return _activeCellsForEntities[d][entityIndex];
}

CellPtr MeshTopology::getCell(unsigned cellIndex)
{
  if (cellIndex > _cells.size())
  {
    cout << "MeshTopology::getCell: cellIndex " << cellIndex << " out of bounds (0, " << _cells.size() - 1 << ").\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex out of bounds.\n");
  }
  return _cells[cellIndex];
}

vector<IndexType> MeshTopology::getCellsForSide(IndexType sideEntityIndex)
{
  vector<IndexType> cells;
  IndexType cellIndex = this->getFirstCellForSide(sideEntityIndex).first;
  if (cellIndex != -1) cells.push_back(cellIndex);
  cellIndex = this->getSecondCellForSide(sideEntityIndex).first;
  if (cellIndex != -1) cells.push_back(cellIndex);
  return cells;
}

unsigned MeshTopology::getEntityCount(unsigned int d)
{
  if (d==0) return _vertices.size();
  return _entities[d].size();
}

pair<IndexType, unsigned> MeshTopology::getEntityGeneralizedParent(unsigned int d, IndexType entityIndex)
{
  if ((d < _generalizedParentEntities.size()) && (_generalizedParentEntities[d].find(entityIndex) != _generalizedParentEntities[d].end()))
    return _generalizedParentEntities[d][entityIndex];
  else
  {
    // entity may be a cell, in which case parent is also a cell (if there is a parent)
    if (d == _spaceDim)
    {
      if (entityIndex >= _cells.size())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "entityIndex is out of bounds");
      }
      CellPtr cell = _cells[entityIndex];
      if (cell->getParent() != Teuchos::null)
      {
        return make_pair(cell->getParent()->cellIndex(), _spaceDim);
      }
    }
    else
    {
      // generalized parent may be a cell
      set< pair<IndexType, unsigned> > cellsForEntity = getCellsContainingEntity(d, entityIndex);
      if (cellsForEntity.size() > 0)
      {
        IndexType cellIndex = cellsForEntity.begin()->first;
        if (_cells[cellIndex]->getParent() != Teuchos::null)
        {
          return make_pair(_cells[cellIndex]->getParent()->cellIndex(), _spaceDim);
        }
      }
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Entity generalized parent not found...");
  return make_pair(-1,-1);
}

unsigned MeshTopology::getEntityIndex(unsigned d, const set<unsigned> &nodeSet)
{
  if (d==0)
  {
    if (nodeSet.size()==1)
    {
      if (_periodicBCs.size() == 0)
      {
        return *nodeSet.begin();
      }
      else
      {
        // NEW 2-11-16: for periodic BCs, return a "canonical" vertex here
        //              Notion is that the result of getEntityIndex is used by GDAMinimumRule, etc.; we need to know
        //              which cells contain this particular vertex.  This is analogous to what we do below with edges, etc.;
        //              the only distinction is that there *are* two vertices stored, so that the physical location of the
        //              cell can still be meaningfully determined.
        
        vector<IndexType> nodeVector(nodeSet.begin(),nodeSet.end());
        vector<IndexType> equivalentNodeVector = getCanonicalEntityNodesViaPeriodicBCs(d, nodeVector);
        return equivalentNodeVector[0];
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "node set for vertex should not have more than one entry!");
    }
  }
  vector<unsigned> sortedNodes(nodeSet.begin(),nodeSet.end());
  if (_knownEntities[d].find(sortedNodes) != _knownEntities[d].end())
  {
    return _knownEntities[d][sortedNodes];
  }
  else if (_periodicBCs.size() > 0)
  {
    // look for alternative, equivalent nodeSets, arrived at via periodic BCs
    vector<IndexType> nodeVector(nodeSet.begin(),nodeSet.end());
    vector<IndexType> equivalentNodeVector = getCanonicalEntityNodesViaPeriodicBCs(d, nodeVector);

    if (equivalentNodeVector.size() > 0)
    {
      vector<IndexType> sortedEquivalentNodeVector = equivalentNodeVector;
      std::sort(sortedEquivalentNodeVector.begin(), sortedEquivalentNodeVector.end());

//      set<IndexType> equivalentNodeSet(equivalentNodeVector.begin(),equivalentNodeVector.end());
      if (_knownEntities[d].find(sortedEquivalentNodeVector) != _knownEntities[d].end())
      {
        return _knownEntities[d][sortedEquivalentNodeVector];
      }
    }
  }
  return -1;
}

unsigned MeshTopology::getEntityParent(unsigned d, unsigned entityIndex, unsigned parentOrdinal)
{
  TEUCHOS_TEST_FOR_EXCEPTION(! entityHasParent(d, entityIndex), std::invalid_argument, "entity does not have parent");
  return _parentEntities[d][entityIndex][parentOrdinal].first;
}

CellTopoPtr MeshTopology::getEntityTopology(unsigned d, IndexType entityIndex)
{
  CellTopoPtr entityTopo;
  if (d < _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(entityIndex >= _entityCellTopologyKeys[d].size(), std::invalid_argument, "entityIndex is out of bounds");
    return _knownTopologies[_entityCellTopologyKeys[d][entityIndex]];
  }
  else
  {
    return getCell(entityIndex)->topology();
  }
}

vector<unsigned> MeshTopology::getEntityVertexIndices(unsigned d, unsigned entityIndex)
{
  if (d==0)
  {
    return vector<IndexType>(1,entityIndex);
  }
  if (d==_spaceDim)
  {
    return getCell(entityIndex)->vertices();
  }
  if (d > _canonicalEntityOrdering.size())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "d out of bounds");
  }
  if (_canonicalEntityOrdering[d].size() <= entityIndex)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "entityIndex out of bounds");
  }
  return _canonicalEntityOrdering[d][entityIndex];
}

set<unsigned> MeshTopology::getEntitiesForSide(unsigned sideEntityIndex, unsigned d)
{
  unsigned sideDim = _spaceDim - 1;
  unsigned subEntityCount = getSubEntityCount(sideDim, sideEntityIndex, d);
  set<unsigned> subEntities;
  for (int subEntityOrdinal=0; subEntityOrdinal<subEntityCount; subEntityOrdinal++)
  {
    subEntities.insert(getSubEntityIndex(sideDim, sideEntityIndex, d, subEntityOrdinal));
  }
  return subEntities;
}

unsigned MeshTopology::getFaceEdgeIndex(unsigned int faceIndex, unsigned int edgeOrdinalInFace)
{
  return getSubEntityIndex(2, faceIndex, 1, edgeOrdinalInFace);
}

MeshTopologyPtr MeshTopology::getRootMeshTopology()
{
  // TODO: add support for curvilinear elements, periodic BCs to this method.
  set<IndexType> vertexIndicesSet;

  vector< vector<IndexType> > allCellsVertexIndices;
  vector< CellTopoPtr > cellTopos;

  for (set<IndexType>::iterator rootCellIt = _rootCells.begin(); rootCellIt != _rootCells.end(); rootCellIt++)
  {
    IndexType rootCellIndex = *rootCellIt;
    CellPtr cell = getCell(rootCellIndex);

    cellTopos.push_back(cell->topology());

    vector<unsigned> cellVertexIndices = cell->vertices();
    allCellsVertexIndices.push_back(cellVertexIndices);

    vertexIndicesSet.insert(cellVertexIndices.begin(),cellVertexIndices.end());
  }

  // we require that the vertices be contiguously numbered (we want to enforce that root MeshTopology shares vertex numbers with this).

  vector< vector<double> > vertices( vertexIndicesSet.size() );
  for (set<IndexType>::iterator vertexIndexIt = vertexIndicesSet.begin(); vertexIndexIt != vertexIndicesSet.end(); vertexIndexIt++)
  {
    IndexType vertexIndex = *vertexIndexIt;
    vertices[vertexIndex] = getVertex(vertexIndex);
  }

  MeshGeometryPtr meshGeometry = Teuchos::rcp(new MeshGeometry(vertices, allCellsVertexIndices, cellTopos) );

  MeshTopologyPtr rootTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );
  return rootTopology;
}

unsigned MeshTopology::getDimension()
{
  return _spaceDim;
}

unsigned MeshTopology::getSubEntityCount(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim)
{
  if (d==0)
  {
    if (subEntityDim==0)
    {
      return 1; // the vertex is its own sub-entity then
    }
    else
    {
      return 0;
    }
  }
  CellTopoPtr entityTopo = getEntityTopology(d, entityIndex);
  return entityTopo->getSubcellCount(subEntityDim);
}

unsigned MeshTopology::getSubEntityIndex(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim, unsigned int subEntityOrdinal)
{
  if (d==0)
  {
    if ((subEntityDim==0) && (subEntityOrdinal==0))
    {
      return entityIndex; // the vertex is its own sub-entity then
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-entity not found for vertex");
    }
  }
  else if (d==_spaceDim)
  {
    // entity is a cell
    return getCell(entityIndex)->entityIndex(subEntityDim, subEntityOrdinal);
  }

  CellTopoPtr entityTopo = getEntityTopology(d, entityIndex);
  set<unsigned> subEntityNodes;
  unsigned subEntityNodeCount = (subEntityDim > 0) ? entityTopo->getNodeCount(subEntityDim, subEntityOrdinal) : 1; // vertices are by definition just one node
  vector<unsigned> entityNodes = getEntityVertexIndices(d, entityIndex);

  for (unsigned nodeOrdinal=0; nodeOrdinal<subEntityNodeCount; nodeOrdinal++)
  {
    unsigned nodeOrdinalInEntity = entityTopo->getNodeMap(subEntityDim, subEntityOrdinal, nodeOrdinal);
    unsigned nodeIndexInMesh = entityNodes[nodeOrdinalInEntity];
    if (subEntityDim == 0)
    {
      return nodeIndexInMesh;
    }
    subEntityNodes.insert(nodeIndexInMesh);
  }
  unsigned subEntityIndex = getEntityIndex(subEntityDim, subEntityNodes);
  if (subEntityIndex == -1)
  {
    cout << "sub-entity not found with vertices:\n";
    printVertices(subEntityNodes);
    cout << "entity vertices:\n";
    set<unsigned> entityNodeSet(entityNodes.begin(),entityNodes.end());
    printVertices(entityNodeSet);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-entity not found");
  }
  return subEntityIndex;
}

const vector<double>& MeshTopology::getVertex(unsigned vertexIndex)
{
  return _vertices[vertexIndex];
}

bool MeshTopology::getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol)
{
  if (_vertexMap.find(vertex) != _vertexMap.end())
  {
    vertexIndex = _vertexMap[vertex];
    return true;
  }
  
  // if we don't have an exact match, we look for one that meets the tolerance.
  // (this is inefficient, and perhaps should be revisited.)
  
  vector<double> vertexForLowerBound;
  for (int d=0; d<_spaceDim; d++)
  {
    vertexForLowerBound.push_back(vertex[d]-tol);
  }

  map< vector<double>, unsigned >::iterator lowerBoundIt = _vertexMap.lower_bound(vertexForLowerBound);
  long bestMatchIndex = -1;
  double bestMatchDistance = tol;
  double xDist = 0; // xDist because vector<double> sorts according to the first entry: so we'll end up looking at
  // all the vertices that are near (x,...) in x...
  
  while ((lowerBoundIt != _vertexMap.end()) && (xDist < tol))
  {
    double dist = 0;
    for (int d=0; d<_spaceDim; d++)
    {
      double ddist = (lowerBoundIt->first[d] - vertex[d]);
      dist += ddist * ddist;
    }
    dist = sqrt( dist );
    if (dist < bestMatchDistance)
    {
      bestMatchDistance = dist;
      bestMatchIndex = lowerBoundIt->second;
    }
    xDist = abs(lowerBoundIt->first[0] - vertex[0]);
    lowerBoundIt++;
  }
  if (bestMatchIndex == -1)
  {
    return false;
  }
  else
  {
    vertexIndex = bestMatchIndex;
    return true;
  }
}

// Here, we assume that the initial coordinates provided are exactly equal (no round-off error) to the ones sought
vector<IndexType> MeshTopology::getVertexIndicesMatching(const vector<double> &vertexInitialCoordinates, double tol)
{
  int numCoords = vertexInitialCoordinates.size();
  vector<double> vertexForLowerBound;
  for (int d=0; d<numCoords; d++)
  {
    vertexForLowerBound.push_back(vertexInitialCoordinates[d]-tol);
  }
  
  double xDist = 0; // xDist because vector<double> sorts according to the first entry: so we'll end up looking at
  // all the vertices that are near (x,...) in x...
  
  map< vector<double>, unsigned >::iterator lowerBoundIt = _vertexMap.lower_bound(vertexInitialCoordinates);
  vector<IndexType> matches;
  while (lowerBoundIt != _vertexMap.end() && (xDist < tol))
  {
    double dist = 0; // distance in the first numCoords coordinates
    for (int d=0; d<numCoords; d++)
    {
      double ddist = (lowerBoundIt->first[d] - vertexInitialCoordinates[d]);
      dist += ddist * ddist;
    }
    dist = sqrt( dist );
    
    if (dist < tol) // counts as a match
    {
      IndexType matchIndex = lowerBoundIt->second;
      matches.push_back(matchIndex);
    }
    
    xDist = abs(lowerBoundIt->first[0] - vertexInitialCoordinates[0]);
    lowerBoundIt++;
  }
  return matches;
}

unsigned MeshTopology::getVertexIndexAdding(const vector<double> &vertex, double tol)
{
  unsigned vertexIndex;
  if (getVertexIndex(vertex, vertexIndex, tol))
  {
    return vertexIndex;
  }
  // if we get here, then we should add
  vertexIndex = _vertices.size();
  _vertices.push_back(vertex);

  if (_vertexMap.find(vertex) != _vertexMap.end() )
  {
    cout << "Mesh error: attempting to add existing vertex.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Mesh error: attempting to add existing vertex");
  }
  _vertexMap[vertex] = vertexIndex;
  
  // update the various entity containers
  int vertexDim = 0;
  vector<IndexType> nodeVector(1,vertexIndex);
  _entities[vertexDim].push_back(nodeVector);
  vector<IndexType> entityVertices;
  entityVertices.push_back(vertexIndex);
  //_canonicalEntityOrdering[vertexDim][vertexIndex] = entityVertices;
  CellTopoPtr nodeTopo = CellTopology::point();
  if (_knownTopologies.find(nodeTopo->getKey()) == _knownTopologies.end())
  {
    _knownTopologies[nodeTopo->getKey()] = nodeTopo;
  }
  _entityCellTopologyKeys[vertexDim].push_back(nodeTopo->getKey());
  
  // new 2-11-16: when using periodic BCs, only add vertex to _knownEntities if it is the original matching point
  bool matchFound = false;
  for (int i=0; i<_periodicBCs.size(); i++)
  {
    vector<int> matchingSides = _periodicBCs[i]->getMatchingSides(vertex);
    for (int j=0; j<matchingSides.size(); j++)
    {
      int matchingSide = matchingSides[j];
      pair<int,int> matchingBC{i, matchingSide};
      pair<int,int> matchingBCForEquivalentVertex = {i, 1 - matchingBC.second}; // the matching side 0 or 1, depending on whether it's "to" or "from"
      vector<double> matchingPoint = _periodicBCs[i]->getMatchingPoint(vertex, matchingSide);
      unsigned equivalentVertexIndex;
      if ( getVertexIndex(matchingPoint, equivalentVertexIndex, tol) )
      {
        if (_canonicalVertexPeriodic.find(equivalentVertexIndex) == _canonicalVertexPeriodic.end())
        {
          _canonicalVertexPeriodic[vertexIndex] = equivalentVertexIndex;
        }
        else
        {
          _canonicalVertexPeriodic[vertexIndex] = _canonicalVertexPeriodic[equivalentVertexIndex];
        }
        // we do still need to keep track of _equivalentNodeViaPeriodicBC, _periodicBCIndicesMatchingNode,
        // since this is how we can decide that two sides are the same...
        _equivalentNodeViaPeriodicBC[make_pair(vertexIndex, matchingBC)] = equivalentVertexIndex;
        _equivalentNodeViaPeriodicBC[make_pair(equivalentVertexIndex, matchingBCForEquivalentVertex)] = vertexIndex;
        _periodicBCIndicesMatchingNode[vertexIndex].insert(matchingBC);
        _periodicBCIndicesMatchingNode[equivalentVertexIndex].insert(matchingBCForEquivalentVertex);
        matchFound = true;
      }
    }
  }
  if (!matchFound)
  {
    _knownEntities[vertexDim][nodeVector] = vertexIndex;
  }

  return vertexIndex;
}

// key: index in vertices; value: index in _vertices
vector<unsigned> MeshTopology::getVertexIndices(const FieldContainer<double> &vertices)
{
  double tol = 1e-14; // tolerance for vertex equality

  int numVertices = vertices.dimension(0);
  vector<unsigned> localToGlobalVertexIndex(numVertices);
  for (int i=0; i<numVertices; i++)
  {
    vector<double> vertex;
    for (int d=0; d<_spaceDim; d++)
    {
      vertex.push_back(vertices(i,d));
    }
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertex,tol);
  }
  return localToGlobalVertexIndex;
}

// key: index in vertices; value: index in _vertices
map<unsigned, IndexType> MeshTopology::getVertexIndicesMap(const FieldContainer<double> &vertices)
{
  map<unsigned, IndexType> vertexMap;
  vector<IndexType> vertexVector = getVertexIndices(vertices);
  unsigned numVertices = vertexVector.size();
  for (unsigned i=0; i<numVertices; i++)
  {
    vertexMap[i] = vertexVector[i];
  }
  return vertexMap;
}

vector<IndexType> MeshTopology::getVertexIndices(const vector< vector<double> > &vertices)
{
  double tol = 1e-14; // tolerance for vertex equality

  int numVertices = vertices.size();
  vector<IndexType> localToGlobalVertexIndex(numVertices);
  for (int i=0; i<numVertices; i++)
  {
    localToGlobalVertexIndex[i] = getVertexIndexAdding(vertices[i],tol);
  }
  return localToGlobalVertexIndex;
}

vector<IndexType> MeshTopology::getChildEntities(unsigned int d, IndexType entityIndex)
{
  vector<IndexType> childIndices;
  if (d==0) return childIndices;
  if (d==_spaceDim)
  {
    MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
    return getCell(entityIndex)->getChildIndices(thisPtr);
  }
  if (_childEntities[d].find(entityIndex) == _childEntities[d].end()) return childIndices;
  vector< pair< RefinementPatternPtr, vector<unsigned> > > childEntries = _childEntities[d][entityIndex];
  for (vector< pair< RefinementPatternPtr, vector<unsigned> > >::iterator entryIt = childEntries.begin();
       entryIt != childEntries.end(); entryIt++)
  {
    childIndices.insert(childIndices.end(),entryIt->second.begin(),entryIt->second.end());
  }
  return childIndices;
}

set<unsigned> MeshTopology::getChildEntitiesSet(unsigned int d, unsigned int entityIndex)
{
  set<unsigned> childIndices;
  if (d==0) return childIndices;
  if (_childEntities[d].find(entityIndex) == _childEntities[d].end()) return childIndices;
  vector< pair< RefinementPatternPtr, vector<unsigned> > > childEntries = _childEntities[d][entityIndex];
  for (vector< pair< RefinementPatternPtr, vector<unsigned> > >::iterator entryIt = childEntries.begin();
       entryIt != childEntries.end(); entryIt++)
  {
    childIndices.insert(entryIt->second.begin(),entryIt->second.end());
  }
  return childIndices;
}

pair<IndexType, unsigned> MeshTopology::getConstrainingEntity(unsigned d, IndexType entityIndex)
{
  unsigned sideDim = _spaceDim - 1;

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
        sidesForEntity = _sidesForEntities[generalizedAncestorDim][generalizedAncestorEntityIndex];
      }
      for (vector<IndexType>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++)
      {
        IndexType sideEntityIndex = *sideEntityIt;
        if (getActiveCellCount(sideDim, sideEntityIndex) > 0)
        {
          constrainingEntity.second = generalizedAncestorDim;
          constrainingEntity.first = possibleConstrainingEntityIndex;
          break;
        }
      }
    }
    while (entityHasParent(generalizedAncestorDim, generalizedAncestorEntityIndex))   // parent of like dimension
    {
      generalizedAncestorEntityIndex = getEntityParent(generalizedAncestorDim, generalizedAncestorEntityIndex);
    }
    if (_generalizedParentEntities[generalizedAncestorDim].find(generalizedAncestorEntityIndex)
        != _generalizedParentEntities[generalizedAncestorDim].end())
    {
      pair< IndexType, unsigned > generalizedParent = _generalizedParentEntities[generalizedAncestorDim][generalizedAncestorEntityIndex];
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

unsigned MeshTopology::getConstrainingEntityIndexOfLikeDimension(unsigned int d, unsigned int entityIndex)
{
  unsigned constrainingEntityIndex = entityIndex;

  if (d==0)   // one vertex can't constrain another...
  {
    return entityIndex;
  }

  vector<unsigned> sidesForEntity;
  unsigned sideDim = _spaceDim - 1;
  if (d==sideDim)
  {
    sidesForEntity.push_back(entityIndex);
  }
  else
  {
    sidesForEntity = _sidesForEntities[d][entityIndex];
  }
  for (vector<unsigned>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++)
  {
    unsigned sideEntityIndex = *sideEntityIt;
    vector< pair<unsigned,unsigned> > sideAncestry = getConstrainingSideAncestry(sideEntityIndex);
    unsigned constrainingEntityIndexForSide = entityIndex;
    if (sideAncestry.size() > 0)
    {
      // need to find the subcellEntity for the constraining side that overlaps with the one on our present side
      for (vector< pair<unsigned,unsigned> >::iterator entryIt=sideAncestry.begin(); entryIt != sideAncestry.end(); entryIt++)
      {
        // need to map constrained entity index from the current side to its parent in sideAncestry
        unsigned parentSideEntityIndex = entryIt->first;
        if (_parentEntities[d].find(constrainingEntityIndexForSide) == _parentEntities[d].end())
        {
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
vector< pair<unsigned,unsigned> > MeshTopology::getConstrainingSideAncestry(unsigned int sideEntityIndex)
{
  // three possibilities: 1) compatible side, 2) side is parent, 3) side is child
  // 1) and 2) mean unconstrained.  3) means constrained (by parent)
  unsigned sideDim = _spaceDim - 1;
  vector< pair<unsigned, unsigned> > ancestry;
  if (_boundarySides.find(sideEntityIndex) != _boundarySides.end())
  {
    return ancestry; // sides on boundary are unconstrained...
  }

  vector< pair<unsigned,unsigned> > sideCellEntries = _activeCellsForEntities[sideDim][sideEntityIndex];
  int activeCellCountForSide = sideCellEntries.size();
  if (activeCellCountForSide == 2)
  {
    // compatible side
    return ancestry; // will be empty
  }
  else if ((activeCellCountForSide == 0) || (activeCellCountForSide == 1))
  {
    // then we're either parent or child of an active side
    // if we are a child, then we should find and return an ancestral path that ends in an active side
    map< unsigned, vector< pair<unsigned, unsigned> > >::iterator parentIt = _parentEntities[sideDim].find(sideEntityIndex);
    while (parentIt != _parentEntities[sideDim].end())
    {
      vector< pair<unsigned, unsigned> > parents = parentIt->second;
      unsigned parentEntityIndex, refinementIndex;
      for (vector< pair<unsigned, unsigned> >::iterator entryIt = parents.begin(); entryIt != parents.end(); entryIt++)
      {
        parentEntityIndex = entryIt->first;
        refinementIndex = entryIt->second;
        if (getActiveCellCount(sideDim, parentEntityIndex) > 0)
        {
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
  }
  else
  {
    cout << "MeshTopology internal error: # active cells for side is not 0, 1, or 2\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTopology internal error: # active cells for side is not 0, 1, or 2\n");
    return ancestry; // unreachable, but most compilers don't seem to know that.
  }
}

RefinementBranch MeshTopology::getSideConstraintRefinementBranch(IndexType sideEntityIndex)
{
  // Returns a RefinementBranch that goes from the constraining side to the side indicated.
  vector< pair<IndexType,unsigned> > constrainingSideAncestry = getConstrainingSideAncestry(sideEntityIndex);
  pair< RefinementPattern*, unsigned > branchEntry;
  unsigned sideDim = _spaceDim - 1;
  IndexType previousChild = sideEntityIndex;
  RefinementBranch refBranch;
  for (vector< pair<IndexType,unsigned> >::iterator ancestorIt = constrainingSideAncestry.begin();
       ancestorIt != constrainingSideAncestry.end(); ancestorIt++)
  {
    IndexType ancestorSideEntityIndex = ancestorIt->first;
    unsigned refinementIndex = ancestorIt->second;
    pair<RefinementPatternPtr, vector<IndexType> > children = _childEntities[sideDim][ancestorSideEntityIndex][refinementIndex];
    branchEntry.first = children.first.get();
    for (int i=0; i<children.second.size(); i++)
    {
      if (children.second[i]==previousChild)
      {
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
    unsigned parentSideEntityIndex)
{
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

  if (entitiesForParentSide.find(entityIndex) != entitiesForParentSide.end())
  {
    return entityIndex;
  }
  vector< pair<unsigned, unsigned> > entityParents = _parentEntities[d][entityIndex];
  //  cout << "parent entities of entity " << entityIndex << ": ";
  for (vector< pair<unsigned, unsigned> >::iterator parentIt = entityParents.begin(); parentIt != entityParents.end(); parentIt++)
  {
    unsigned parentEntityIndex = parentIt->first;
    //    cout << parentEntityIndex << " ";
    if (entitiesForParentSide.find(parentEntityIndex) != entitiesForParentSide.end())
    {
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

unsigned MeshTopology::getEntityParentCount(unsigned d, IndexType entityIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION(d >= _parentEntities.size(), std::invalid_argument, "dimension is out of bounds");
  TEUCHOS_TEST_FOR_EXCEPTION(_parentEntities[d].find(entityIndex) == _parentEntities[d].end(), std::invalid_argument, "entityIndex not found in _parentEntities[d]");
  return _parentEntities[d][entityIndex].size();
}

// ! pairs are (cellIndex, sideOrdinal) where the sideOrdinal is a side that contains the entity
set< pair<IndexType, unsigned> > MeshTopology::getCellsContainingEntity(unsigned d, unsigned entityIndex)   // not *all* cells, but within any refinement branch, the most refined cell that contains the entity will be present in this set.  The unsigned value is the ordinal of a *side* in the cell containing this entity.  There may be multiple sides in a cell that contain the entity; this method will return just one entry per cell.
{
  if (d==getDimension())
  {
    // entityIndex is a cell; the side then is contained within the cell; we'll flag this fact by setting the side ordinal to -1.
    return {{entityIndex,-1}};
  }
  vector<IndexType> sidesForEntity = _sidesForEntities[d][entityIndex];
  typedef pair<IndexType,unsigned> CellPair;
  set< CellPair > cells;
  set< IndexType > cellIndices;  // container to keep track of which cells we've already counted -- we only return one (cell, side) pair per cell that contains the entity...
  for (vector<IndexType>::iterator sideEntityIt = sidesForEntity.begin(); sideEntityIt != sidesForEntity.end(); sideEntityIt++)
  {
    IndexType sideEntityIndex = *sideEntityIt;
    int numCellsForSide = getCellCountForSide(sideEntityIndex);
    if (numCellsForSide == 2)
    {
      CellPair cell1 = getFirstCellForSide(sideEntityIndex);
      if (cellIndices.find(cell1.first) == cellIndices.end())
      {
        cells.insert(cell1);
        cellIndices.insert(cell1.first);
      }
      CellPair cell2 = getSecondCellForSide(sideEntityIndex);
      if (cellIndices.find(cell2.first) == cellIndices.end())
      {
        cells.insert(cell2);
        cellIndices.insert(cell2.first);
      }
    }
    else if (numCellsForSide == 1)
    {
      CellPair cell1 = getFirstCellForSide(sideEntityIndex);
      if (cellIndices.find(cell1.first) == cellIndices.end())
      {
        cells.insert(cell1);
        cellIndices.insert(cell1.first);
      }
    }
    else
    {
      cout << "Unexpected cell count for side.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unexpected cell count for side.");
    }
  }
  return cells;
}

bool MeshTopology::isBoundarySide(IndexType sideEntityIndex)
{
  return _boundarySides.find(sideEntityIndex) != _boundarySides.end();
}

bool MeshTopology::isValidCellIndex(IndexType cellIndex)
{
  return cellIndex < _cells.size();
}

pair<IndexType,IndexType> MeshTopology::owningCellIndexForConstrainingEntity(unsigned d, IndexType constrainingEntityIndex)
{
  // sorta like the old leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity, but now prefers larger cells
  // -- the first level of the entity refinement hierarchy that has an active cell containing an entity in that level is the one from
  // which we choose the owning cell (and we do take the least such cellIndex)
  unsigned leastActiveCellIndex = (unsigned)-1; // unsigned cast of -1 makes maximal unsigned #
  set<IndexType> constrainedEntities;
  constrainedEntities.insert(constrainingEntityIndex);

  IndexType leastActiveCellConstrainedEntityIndex;
  while (true)
  {
    set<IndexType> nextTierConstrainedEntities;

    for (set<IndexType>::iterator constrainedEntityIt = constrainedEntities.begin(); constrainedEntityIt != constrainedEntities.end(); constrainedEntityIt++)
    {
      IndexType constrainedEntityIndex = *constrainedEntityIt;

      // get this entity's immediate children, in case we don't find an active cell on this tier
      if (_childEntities[d].find(constrainedEntityIndex) != _childEntities[d].end())
      {
        for (unsigned i=0; i<_childEntities[d][constrainedEntityIndex].size(); i++)
        {
          vector<unsigned> immediateChildren = _childEntities[d][constrainedEntityIndex][i].second;
          nextTierConstrainedEntities.insert(immediateChildren.begin(), immediateChildren.end());
        }
      }

      if (_sidesForEntities[d].size() <= constrainingEntityIndex)
      {
        cout << "ERROR: entityIndex " << constrainingEntityIndex << " of dimension " << d << " is beyond bounds of _sidesForEntities" << endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: constrainingEntityIndex is out of bounds of _sidesForEntities");
      }
      vector<IndexType> sideEntityIndices = _sidesForEntities[d][constrainedEntityIndex];
      for (vector<IndexType>::iterator sideEntityIt = sideEntityIndices.begin(); sideEntityIt != sideEntityIndices.end(); sideEntityIt++)
      {
        IndexType sideEntityIndex = *sideEntityIt;
        typedef pair<IndexType, unsigned> CellPair;
        pair<CellPair,CellPair> cellsForSide = _cellsForSideEntities[sideEntityIndex];
        IndexType firstCellIndex = cellsForSide.first.first;
        if (_activeCells.find(firstCellIndex) != _activeCells.end())
        {
          if (firstCellIndex < leastActiveCellIndex)
          {
            leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
            leastActiveCellIndex = firstCellIndex;
          }
        }
        IndexType secondCellIndex = cellsForSide.second.first;
        if (_activeCells.find(secondCellIndex) != _activeCells.end())
        {
          if (secondCellIndex < leastActiveCellIndex)
          {
            leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
            leastActiveCellIndex = secondCellIndex;
          }
        }
      }
    }
    if (leastActiveCellIndex == -1)
    {
      // try the next refinement level down
      if (nextTierConstrainedEntities.size() == 0)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No active cell found containing entity constrained by constraining entity");
      }
      constrainedEntities = nextTierConstrainedEntities;
    }
    else
    {
      return make_pair(leastActiveCellIndex, leastActiveCellConstrainedEntityIndex);
    }
  }

  return make_pair(leastActiveCellIndex, leastActiveCellConstrainedEntityIndex);
}

vector< IndexType > MeshTopology::getSidesContainingEntity(unsigned d, IndexType entityIndex)
{
  unsigned sideDim = getDimension() - 1;
  if (d == sideDim) return {entityIndex};
  
  if (_sidesForEntities[d].size() <= entityIndex)
  {
    return {};
  }
  return _sidesForEntities[d][entityIndex];
}

unsigned MeshTopology::getSubEntityPermutation(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal)
{
  if (subEntityDim==0) return 0;

  if (subEntityDim >= d)
  {
    cout << "subEntityDim cannot be greater than d!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subEntityDim cannot be greater than d!");
  }

  vector<unsigned> entityNodes = getEntityVertexIndices(d,entityIndex);
  CellTopoPtr topo = getEntityTopology(d, entityIndex);
  vector<unsigned> subEntityNodes;
  int subEntityNodeCount = topo->getNodeCount(subEntityDim, subEntityOrdinal);
  for (int seNodeOrdinal = 0; seNodeOrdinal<subEntityNodeCount; seNodeOrdinal++)
  {
    unsigned entityNodeOrdinal = topo->getNodeMap(subEntityDim, subEntityOrdinal, seNodeOrdinal);
    subEntityNodes.push_back(entityNodes[entityNodeOrdinal]);
  }
  subEntityNodes = getCanonicalEntityNodesViaPeriodicBCs(subEntityDim, subEntityNodes);
  unsigned subEntityIndex = getSubEntityIndex(d, entityIndex, subEntityDim, subEntityOrdinal);
  CellTopoPtr subEntityTopo = getEntityTopology(subEntityDim, subEntityIndex);
  return CamelliaCellTools::permutationMatchingOrder(subEntityTopo, _canonicalEntityOrdering[subEntityDim][subEntityOrdinal], subEntityNodes);
}

//pair<IndexType,IndexType> MeshTopology::leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(unsigned d, unsigned constrainingEntityIndex) {
//  unsigned leastActiveCellIndex = (unsigned)-1; // unsigned cast of -1 makes maximal unsigned #
//  set<IndexType> constrainedEntities = descendants(d,constrainingEntityIndex);
//
//  IndexType leastActiveCellConstrainedEntityIndex;
//  for (set<IndexType>::iterator constrainedEntityIt = constrainedEntities.begin(); constrainedEntityIt != constrainedEntities.end(); constrainedEntityIt++) {
//    IndexType constrainedEntityIndex = *constrainedEntityIt;
//    if (_sidesForEntities[d].find(constrainingEntityIndex) == _sidesForEntities[d].end()) {
//      cout << "ERROR: no sides found containing entityIndex " << constrainingEntityIndex << " of dimension " << d << endl;
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: no sides found containing entity");
//    }
//    set<IndexType> sideEntityIndices = _sidesForEntities[d][constrainedEntityIndex];
//    for (set<IndexType>::iterator sideEntityIt = sideEntityIndices.begin(); sideEntityIt != sideEntityIndices.end(); sideEntityIt++) {
//      IndexType sideEntityIndex = *sideEntityIt;
//      typedef pair<IndexType, unsigned> CellPair;
//      pair<CellPair,CellPair> cellsForSide = _cellsForSideEntities[sideEntityIndex];
//      IndexType firstCellIndex = cellsForSide.first.first;
//      if (_activeCells.find(firstCellIndex) != _activeCells.end()) {
//        if (firstCellIndex < leastActiveCellIndex) {
//          leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
//          leastActiveCellIndex = firstCellIndex;
//        }
//      }
//      IndexType secondCellIndex = cellsForSide.second.first;
//      if (_activeCells.find(secondCellIndex) != _activeCells.end()) {
//        if (secondCellIndex < leastActiveCellIndex) {
//          leastActiveCellConstrainedEntityIndex = constrainedEntityIndex;
//          leastActiveCellIndex = secondCellIndex;
//        }
//      }
//    }
//  }
//  if (leastActiveCellIndex == -1) {
//    cout << "WARNING: least active cell index not found.\n";
//  }
//
//  return make_pair(leastActiveCellIndex, leastActiveCellConstrainedEntityIndex);
//}

IndexType MeshTopology::maxConstraint(unsigned d, IndexType entityIndex1, IndexType entityIndex2)
{
  // if one of the entities is the ancestor of the other, returns that one.  Otherwise returns (unsigned) -1.

  if (entityIndex1==entityIndex2) return entityIndex1;

  // a good guess is that the entity with lower index is the ancestor
  unsigned smallerEntityIndex = std::min(entityIndex1, entityIndex2);
  unsigned largerEntityIndex = std::max(entityIndex1, entityIndex2);
  if (entityIsAncestor(d,smallerEntityIndex,largerEntityIndex))
  {
    return smallerEntityIndex;
  }
  else if (entityIsAncestor(d,largerEntityIndex,smallerEntityIndex))
  {
    return largerEntityIndex;
  }
  return -1;
}

vector< ParametricCurvePtr > MeshTopology::parametricEdgesForCell(unsigned cellIndex, bool neglectCurves)
{
  vector< ParametricCurvePtr > edges;
  CellPtr cell = getCell(cellIndex);
  
  vector<unsigned> vertices;
  if (cell->topology()->getTensorialDegree() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim != 2, std::invalid_argument, "Only 2D supported right now.");
    vertices = cell->vertices();
  }
  else
  {
    // for space-time, we assume that:
    // (a) only the pure-spatial edges (i.e. those that have no temporal extension) are curved
    // (b) the vertices and parametric curves at both time nodes are identical (so that the curves are independent of time)
    // At some point, it would be desirable to revisit these assumptions.  Having moving meshes, including mesh movement
    // that follows a curved path, would be pretty neat.
    // we take the first temporal side:
    unsigned temporalSideOrdinal = cell->topology()->getTemporalSideOrdinal(0);
    int sideDim = _spaceDim - 1;
    vertices = cell->getEntityVertexIndices(sideDim, temporalSideOrdinal);
  }
  
  int numNodes = vertices.size();
  
  for (int nodeIndex=0; nodeIndex<numNodes; nodeIndex++)
  {
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

    if (neglectCurves)
    {
      edgeFxn = straightEdgeFxn;
    }
    if ( _edgeToCurveMap.find(edge) != _edgeToCurveMap.end() )
    {
      edgeFxn = _edgeToCurveMap[edge];
    }
    else if ( _edgeToCurveMap.find(reverse_edge) != _edgeToCurveMap.end() )
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: reverse_edge found, but edge not found in edgeToCurveMap.");
    }
    else
    {
      edgeFxn = straightEdgeFxn;
    }
    edges.push_back(edgeFxn);
  }
  return edges;
}

void MeshTopology::printApproximateMemoryReport()
{
  cout << "**** MeshTopology Memory Report ****\n";
  cout << "Memory sizes are in bytes.\n";

  long long memSize = 0;

  map<string, long long> variableCost = approximateMemoryCosts();

  map<long long, vector<string> > variableOrderedByCost;
  for (map<string, long long>::iterator entryIt = variableCost.begin(); entryIt != variableCost.end(); entryIt++)
  {
    variableOrderedByCost[entryIt->second].push_back(entryIt->first);
  }

  for (map<long long, vector<string> >::iterator entryIt = variableOrderedByCost.begin(); entryIt != variableOrderedByCost.end(); entryIt++)
  {
    for (int i=0; i< entryIt->second.size(); i++)
    {
      cout << setw(30) << (entryIt->second)[i] << setw(30) << entryIt->first << endl;
      memSize += entryIt->first;
    }
  }
  cout << "Total: " << memSize << " bytes.\n";
}

void MeshTopology::printConstraintReport(unsigned d)
{
  if (_entities.size() <= d)
  {
    cout << "No entities of dimension " << d << " in MeshTopology.\n";
    return;
  }
  IndexType entityCount = _entities[d].size();
  cout << "******* MeshTopology, constraints for d = " << d << " *******\n";
  for (IndexType entityIndex=0; entityIndex<entityCount; entityIndex++)
  {
    pair<IndexType, unsigned> constrainingEntity = getConstrainingEntity(d, entityIndex);
    if ((d != constrainingEntity.second) || (entityIndex != constrainingEntity.first))
      cout << "Entity " << entityIndex << " is constrained by entity " << constrainingEntity.first << " of dimension " << constrainingEntity.second << endl;
    else
      cout << "Entity " << entityIndex << " is unconstrained.\n";
  }
}

void MeshTopology::printVertex(unsigned int vertexIndex)
{
  cout << "vertex " << vertexIndex << ": (";
  for (unsigned d=0; d<_spaceDim; d++)
  {
    cout << _vertices[vertexIndex][d];
    if (d != _spaceDim-1) cout << ",";
  }
  cout << ")\n";
}

void MeshTopology::printVertices(set<unsigned int> vertexIndices)
{
  for (set<unsigned>::iterator indexIt=vertexIndices.begin(); indexIt!=vertexIndices.end(); indexIt++)
  {
    unsigned vertexIndex = *indexIt;
    printVertex(vertexIndex);
  }
}

void MeshTopology::printEntityVertices(unsigned int d, unsigned int entityIndex)
{
  if (d==0)
  {
    printVertex(entityIndex);
    return;
  }
  vector<unsigned> entityVertices = _canonicalEntityOrdering[d][entityIndex];
  for (vector<unsigned>::iterator vertexIt=entityVertices.begin(); vertexIt !=entityVertices.end(); vertexIt++)
  {
    printVertex(*vertexIt);
  }
}

void MeshTopology::printAllEntities()
{
  for (int d=0; d<_spaceDim; d++)
  {
    string entityTypeString;
    if (d==0)
    {
      entityTypeString = "Vertex";
    }
    else if (d==1)
    {
      entityTypeString = "Edge";
    }
    else if (d==2)
    {
      entityTypeString = "Face";
    }
    else if (d==3)
    {
      entityTypeString = "Solid";
    }
    cout << "****************************  ";
    cout << entityTypeString << " entities:";
    cout << "  ****************************\n";

    int entityCount = getEntityCount(d);
    for (int entityIndex=0; entityIndex < entityCount; entityIndex++)
    {
      if (d != 0) cout << entityTypeString << " " << entityIndex << ":" << endl;
      printEntityVertices(d, entityIndex);
    }
  }

  cout << "****************************      ";
  cout << "Cells:";
  cout << "      ****************************\n";

  int numCells = _cells.size();
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    CellPtr cell = getCell(cellIndex);
    cout << "Cell " << cellIndex << ":\n";
    int vertexCount = cell->vertices().size();
    for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
    {
      printVertex(cell->vertices()[vertexOrdinal]);
    }
    for (int d=1; d<_spaceDim; d++)
    {
      int subcellCount = cell->topology()->getSubcellCount(d);
      for (int subcord=0; subcord<subcellCount; subcord++)
      {
        ostringstream labelStream;
        if (d==1)
        {
          labelStream << "Edge";
        }
        else if (d==2)
        {
          labelStream << "Face";
        }
        else if (d==3)
        {
          labelStream << "Solid";
        }
        labelStream << " " << subcord << " nodes";
        Camellia::print(labelStream.str(), cell->getEntityVertexIndices(d, subcord));
      }
    }
  }
}

FieldContainer<double> MeshTopology::physicalCellNodesForCell(unsigned int cellIndex, bool includeCellDimension)
{
  CellPtr cell = getCell(cellIndex);
  unsigned vertexCount = cell->vertices().size();
  FieldContainer<double> nodes(vertexCount, _spaceDim);
  for (unsigned vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
  {
    unsigned vertexIndex = cell->vertices()[vertexOrdinal];
    for (unsigned d=0; d<_spaceDim; d++)
    {
      nodes(vertexOrdinal,d) = _vertices[vertexIndex][d];
    }
  }
  if (includeCellDimension)
  {
    nodes.resize(1,nodes.dimension(0),nodes.dimension(1));
  }
  return nodes;
}

void MeshTopology::refineCell(IndexType cellIndex, RefinementPatternPtr refPattern, IndexType firstChildCellIndex)
{
  // TODO: worry about the case (currently unsupported in RefinementPattern) of children that do not share topology with the parent.  E.g. quad broken into triangles.  (3D has better examples.)

//  { // DEBUGGING
//    if (cellIndex == 39)
//    {
//      cout << "refining cell " << cellIndex << endl;
//    }
//  }
  
  CellPtr cell = _cells[cellIndex];
  FieldContainer<double> cellNodes(cell->vertices().size(), _spaceDim);

  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(0); vertexIndex++)
  {
    for (int d=0; d<_spaceDim; d++)
    {
      cellNodes(vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }

  FieldContainer<double> vertices = refPattern->verticesForRefinement(cellNodes);
  if (_transformationFunction.get())
  {
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
  addChildren(firstChildCellIndex, cell, childTopos, childVertices);

  determineGeneralizedParentsForRefinement(cell, refPattern);

  if (_edgeToCurveMap.size() > 0)
  {
    vector< vector< pair< unsigned, unsigned> > > childrenForSides = refPattern->childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
    // handle any broken curved edges
    //    set<int> childrenWithCurvedEdges;
    int edgeCount = cell->topology()->getEdgeCount();
    int edgeDim = 1;
    for (int edgeOrdinal=0; edgeOrdinal < edgeCount; edgeOrdinal++)
    {
      IndexType edgeEntityIndex = cell->entityIndex(edgeDim, edgeOrdinal);
      if (!entityHasChildren(edgeDim, edgeEntityIndex)) continue; // unbroken edge: no treatment necessary
      
      vector<IndexType> childEntities = getChildEntities(edgeDim, edgeEntityIndex);
      int edgeChildCount = childEntities.size();
      TEUCHOS_TEST_FOR_EXCEPTION(edgeChildCount != 2, std::invalid_argument, "unexpected number of edge children");
      
      vector<IndexType> parentEdgeVertexIndices = getEntityVertexIndices(edgeDim, edgeEntityIndex);
      int v0 = parentEdgeVertexIndices[0];
      int v1 = parentEdgeVertexIndices[1];
      pair<int,int> edge = make_pair(v0, v1);
      if (_edgeToCurveMap.find(edge) != _edgeToCurveMap.end())
      {
        // then define the new curves
        for (int i=0; i<edgeChildCount; i++)
        {
          IndexType childEdgeEntityIndex = childEntities[i];
          vector<IndexType> childEdgeVertexIndices = getEntityVertexIndices(edgeDim, childEdgeEntityIndex);
          double child_t0, child_t1;
          if (childEdgeVertexIndices[0] == parentEdgeVertexIndices[0])
          {
            child_t0 = 0.0;
            child_t1 = 1.0 / edgeChildCount;
          }
          else if (childEdgeVertexIndices[0] == parentEdgeVertexIndices[1])
          {
            child_t0 = 1.0;
            child_t1 = 1.0 / edgeChildCount;
          }
          else if (childEdgeVertexIndices[1] == parentEdgeVertexIndices[0])
          {
            child_t0 = 1.0 / edgeChildCount;
            child_t1 = 0.0;
          }
          else if (childEdgeVertexIndices[1] == parentEdgeVertexIndices[1])
          {
            child_t0 = 1.0 / edgeChildCount;
            child_t1 = 1.0;
          }
          else
          {
            printAllEntities();
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child edge not in expected relationship to parent");
          }
          
          ParametricCurvePtr parentCurve = _edgeToCurveMap[edge];
          ParametricCurvePtr childCurve = ParametricCurve::subCurve(parentCurve, child_t0, child_t1);
          
          pair<unsigned, unsigned> childEdge = {childEdgeVertexIndices[0],childEdgeVertexIndices[1]};
          addEdgeCurve(childEdge, childCurve);
        }
      }
    }
    //    if (_transformationFunction.get()) {
    //      _transformationFunction->updateCells(childrenWithCurvedEdges);
    //    }
  }
}

void MeshTopology::refineCellEntities(CellPtr cell, RefinementPatternPtr refPattern)
{
  // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities

  FieldContainer<double> cellNodes(1,cell->vertices().size(), _spaceDim);

  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(1); vertexIndex++)
  {
    for (int d=0; d<_spaceDim; d++)
    {
      cellNodes(0,vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }

  vector< RefinementPatternRecipe > relatedRecipes = refPattern->relatedRecipes();
  if (relatedRecipes.size()==0)
  {
    RefinementPatternRecipe recipe;
    vector<unsigned> initialCell;
    recipe.push_back(make_pair(refPattern.get(),vector<unsigned>()));
    relatedRecipes.push_back(recipe);
  }

  // TODO generalize the below code to apply recipes instead of just the refPattern...

  CellTopoPtr cellTopo = cell->topology();
  for (unsigned d=1; d<_spaceDim; d++)
  {
    unsigned subcellCount = cellTopo->getSubcellCount(d);
    for (unsigned subcord = 0; subcord < subcellCount; subcord++)
    {
      RefinementPatternPtr subcellRefPattern = refPattern->patternForSubcell(d, subcord);
      FieldContainer<double> refinedNodes = subcellRefPattern->refinedNodes(); // NOTE: refinedNodes implicitly assumes that all child topos are the same
      unsigned childCount = refinedNodes.dimension(0);
      if (childCount==1) continue; // we already have the appropriate entities and parent relationships defined...

      //      cout << "Refined nodes:\n" << refinedNodes;

      unsigned parentIndex = cell->entityIndex(d, subcord);
      // determine matching EntitySets--we add to these on refinement
      vector<EntitySetPtr> parentEntitySets;
      for (auto entry : _entitySets)
      {
        if (entry.second->containsEntity(d, parentIndex)) parentEntitySets.push_back(entry.second);
      }
      
      // if we ever allow multiple parentage, then we'll need to record things differently in both _childEntities and _parentEntities
      // (and the if statement just below will need to change in a corresponding way, indexed by the particular refPattern in question maybe
      if (_childEntities[d].find(parentIndex) == _childEntities[d].end())
      {
        vector<unsigned> childEntityIndices(childCount);
        for (unsigned childIndex=0; childIndex<childCount; childIndex++)
        {
          unsigned nodeCount = refinedNodes.dimension(1);
          FieldContainer<double> nodesOnSubcell(nodeCount,d);
          for (int nodeIndex=0; nodeIndex<nodeCount; nodeIndex++)
          {
            for (int dimIndex=0; dimIndex<d; dimIndex++)
            {
              nodesOnSubcell(nodeIndex,dimIndex) = refinedNodes(childIndex,nodeIndex,dimIndex);
            }
          }
          //          cout << "nodesOnSubcell:\n" << nodesOnSubcell;
          FieldContainer<double> nodesOnRefCell(nodeCount,_spaceDim);
          CamelliaCellTools::mapToReferenceSubcell(nodesOnRefCell, nodesOnSubcell, d, subcord, cellTopo);
          //          cout << "nodesOnRefCell:\n" << nodesOnRefCell;
          FieldContainer<double> physicalNodes(1,nodeCount,_spaceDim);
          // map to physical space:
          CamelliaCellTools::mapToPhysicalFrame(physicalNodes, nodesOnRefCell, cellNodes, cellTopo);
          //          cout << "physicalNodes:\n" << physicalNodes;


          // debugging:
          //          if ((_cells.size() == 2) && (cell->cellIndex() == 0) && (d==2) && (subcord==2)) {
          //            cout << "cellNodes:\n" << cellNodes;
          //            cout << "For childOrdinal " << childIndex << " of face 2 on cell 0, details:\n";
          //            cout << "nodesOnSubcell:\n" << nodesOnSubcell;
          //            cout << "nodesOnRefCell:\n" << nodesOnRefCell;
          //            cout << "physicalNodes:\n" << physicalNodes;
          //          }

          if (_transformationFunction.get())
          {
            physicalNodes.resize(nodeCount,_spaceDim);
            bool changedVertices = _transformationFunction->mapRefCellPointsUsingExactGeometry(physicalNodes, nodesOnRefCell, cell->cellIndex());
            //            cout << "physicalNodes after transformation:\n" << physicalNodes;
          }
          //          cout << "cellNodes:\n" << cellNodes;

          // add vertices as necessary and get their indices
          physicalNodes.resize(nodeCount,_spaceDim);
          vector<unsigned> childEntityVertices = getVertexIndices(physicalNodes); // key: index in physicalNodes; value: index in _vertices

//          cout << "nodesOnRefCell:\n" << nodesOnRefCell;
//          cout << "physicalNodes:\n" << physicalNodes;
          
          unsigned entityPermutation;
          CellTopoPtr childTopo = cellTopo->getSubcell(d, subcord);
          unsigned childEntityIndex = addEntity(childTopo, childEntityVertices, entityPermutation);
          //          cout << "for d=" << d << ", entity index " << childEntityIndex << " is child of " << parentIndex << endl;
          if (childEntityIndex != parentIndex) // anisotropic and null refinements can leave the entity unrefined
          {
            _parentEntities[d][childEntityIndex] = vector< pair<unsigned,unsigned> >(1, make_pair(parentIndex,0)); // TODO: this is where we want to fill in a proper list of possible parents once we work through recipes
          }
          childEntityIndices[childIndex] = childEntityIndex;
          vector< pair<unsigned, unsigned> > parentActiveCells = _activeCellsForEntities[d][parentIndex];
          // TODO: ?? do something with parentActiveCells?  Seems like we just trailed off here...
        }
        _childEntities[d][parentIndex] = vector< pair<RefinementPatternPtr,vector<unsigned> > >(1, make_pair(subcellRefPattern, childEntityIndices) ); // TODO: this also needs to change when we work through recipes.  Note that the correct parent will vary here...  i.e. in the anisotropic case, the child we're ultimately interested in will have an anisotropic parent, and *its* parent would be the bigger guy referred to here.

        // add the child entities to the parent's entity sets
        for (EntitySetPtr entitySet : parentEntitySets)
          for (IndexType childEntityIndex : childEntityIndices)
            entitySet->addEntity(d, childEntityIndex);
        
        if (d==_spaceDim-1)   // side
        {
          if (_boundarySides.find(parentIndex) != _boundarySides.end())   // parent is a boundary side, so children are, too
          {
            _boundarySides.insert(childEntityIndices.begin(),childEntityIndices.end());
          }
        }
      }
    }
  }
}

void MeshTopology::determineGeneralizedParentsForRefinement(CellPtr cell, RefinementPatternPtr refPattern)
{
  FieldContainer<double> cellNodes(1,cell->vertices().size(), _spaceDim);

  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(1); vertexIndex++)
  {
    for (int d=0; d<_spaceDim; d++)
    {
      cellNodes(0,vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }

  vector< RefinementPatternRecipe > relatedRecipes = refPattern->relatedRecipes();
  if (relatedRecipes.size()==0)
  {
    RefinementPatternRecipe recipe;
    vector<unsigned> initialCell;
    recipe.push_back(make_pair(refPattern.get(),vector<unsigned>()));
    relatedRecipes.push_back(recipe);
  }

  // TODO generalize the below code to apply recipes instead of just the refPattern...

  CellTopoPtr cellTopo = cell->topology();
  for (unsigned d=1; d<_spaceDim; d++)
  {
    unsigned subcellCount = cellTopo->getSubcellCount(d);
    for (unsigned subcord = 0; subcord < subcellCount; subcord++)
    {
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
           refIt != childEntities.end(); refIt++)
      {
        vector<IndexType> childEntityIndices = refIt->second;
        for (int childOrdinal=0; childOrdinal<childEntityIndices.size(); childOrdinal++)
        {
          IndexType childEntityIndex = childEntityIndices[childOrdinal];
          if (parentIndex == childEntityIndex)   // "null" refinement pattern -- nothing to do here.
          {
            continue;
          }
          setEntityGeneralizedParent(d, childEntityIndex, d, parentIndex); // TODO: change this to consider anisotropic refinements/ recipes...  (need to choose nearest of the possible ancestors, in my view)
          for (int subcdim=0; subcdim<d; subcdim++)
          {
            int subcCount = this->getSubEntityCount(d, childEntityIndex, subcdim);
            for (int subcord=0; subcord < subcCount; subcord++)
            {
              IndexType subcellEntityIndex = this->getSubEntityIndex(d, childEntityIndex, subcdim, subcord);

              // if this is a vertex that also belongs to the parent, then its parentage will already be handled...
              if ((subcdim==0) && (parentVertexIndexSet.find(subcellEntityIndex) != parentVertexIndexSet.end() ))
              {
                continue;
              }

              // if there was a previous entry, have a look at it...
              if (_generalizedParentEntities[subcdim].find(subcellEntityIndex) != _generalizedParentEntities[subcdim].end())
              {
                pair<IndexType, unsigned> previousParent = _generalizedParentEntities[subcdim][subcellEntityIndex];
                if (previousParent.second <= d)   // then the previous parent is a better (nearer) parent
                {
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

const set<unsigned> &MeshTopology::getRootCellIndices()
{
  return _rootCells;
}

void MeshTopology::setEdgeToCurveMap(const map< pair<IndexType, IndexType>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh)
{
  _edgeToCurveMap.clear();
  map< pair<IndexType, IndexType>, ParametricCurvePtr >::const_iterator edgeIt;
  _cellIDsWithCurves.clear();

  for (edgeIt = edgeToCurveMap.begin(); edgeIt != edgeToCurveMap.end(); edgeIt++)
  {
    addEdgeCurve(edgeIt->first, edgeIt->second);
  }
  // mesh transformation function expects global ID type
  set<GlobalIndexType> cellIDsGlobal(_cellIDsWithCurves.begin(),_cellIDsWithCurves.end());
  if (cellIDsGlobal.size() > 0)
    _transformationFunction = Teuchos::rcp(new MeshTransformationFunction(mesh, cellIDsGlobal));
  else
    _transformationFunction = Teuchos::null;
}

void MeshTopology::setGlobalDofAssignment(GlobalDofAssignment* gda)   // for cubature degree lookups
{
  _gda = gda;
}

void MeshTopology::setEntityGeneralizedParent(unsigned entityDim, IndexType entityIndex, unsigned parentDim, IndexType parentEntityIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION((entityDim==parentDim) && (parentEntityIndex==entityIndex), std::invalid_argument, "entity cannot be its own parent!");
  _generalizedParentEntities[entityDim][entityIndex] = make_pair(parentEntityIndex,parentDim);
  if (entityDim == 0)   // vertex --> should set parent relationships for any vertices that are equivalent via periodic BCs
  {
    if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end())
    {
      for (set< pair<int, int> >::iterator bcIt = _periodicBCIndicesMatchingNode[entityIndex].begin(); bcIt != _periodicBCIndicesMatchingNode[entityIndex].end(); bcIt++)
      {
        IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, *bcIt)];
        _generalizedParentEntities[entityDim][equivalentNode] = make_pair(parentEntityIndex,parentDim);
      }
    }
  }
}

void MeshTopology::setEntitySetInitialTime(EntitySetPtr entitySet)
{
  _initialTimeEntityHandle = entitySet->getHandle();
}

Teuchos::RCP<MeshTransformationFunction> MeshTopology::transformationFunction()
{
  return _transformationFunction;
}

void MeshTopology::verticesForCell(FieldContainer<double>& vertices, GlobalIndexType cellID)
{
  CellPtr cell = getCell(cellID);
  vector<IndexType> vertexIndices = cell->vertices();
  int numVertices = vertexIndices.size();
  int spaceDim = getDimension();

  //vertices.resize(numVertices,dimension);
  for (unsigned vertexOrdinal = 0; vertexOrdinal < numVertices; vertexOrdinal++)
  {
    for (int d=0; d<spaceDim; d++)
    {
      vertices(vertexOrdinal,d) = getVertex(vertexIndices[vertexOrdinal])[d];
    }
  }
}

MeshTopologyViewPtr MeshTopology::getView(const set<IndexType> &activeCells)
{
  MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  return Teuchos::rcp( new MeshTopologyView(thisPtr, activeCells) );
}