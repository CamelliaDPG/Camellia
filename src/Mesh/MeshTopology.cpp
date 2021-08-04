//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  MeshTopology.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//

#include "CamelliaCellTools.h"
#include "CellDataMigration.h"
#include "CamelliaMemoryUtility.h"
#include "CellTopology.h"
#include "GlobalDofAssignment.h"
#include "MeshTopology.h"
#include "MeshTransformationFunction.h"

#include <algorithm>

using namespace Camellia;
using namespace std;

void MeshTopology::init(unsigned spaceDim)
{
  if (spaceDim >= 2) RefinementPattern::initializeAnisotropicRelationships(); // not sure this is the optimal place for this call

  _spaceDim = spaceDim;
  // for nontrivial mesh topology, we store entities with dimension sideDim down to vertices, so _spaceDim total possibilities
  // for trivial mesh topology (just a node), we allow storage of 0-dimensional (vertex) entity
  int numEntityDimensions = (_spaceDim > 0) ? _spaceDim : 1;
  _entities = vector< vector< vector< IndexType > > >(numEntityDimensions);
  _knownEntities = vector< map< vector<IndexType>, IndexType > >(numEntityDimensions); // map keys are sets of vertices, values are entity indices in _entities[d]
  _canonicalEntityOrdering = vector< vector< vector<IndexType> > >(numEntityDimensions);
  _activeCellsForEntities = vector< vector< vector< pair<IndexType, unsigned> > > >(numEntityDimensions); // pair entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  _sidesForEntities = vector< vector< vector< IndexType > > >(numEntityDimensions);
  _parentEntities = vector< map< IndexType, vector< pair<IndexType, unsigned> > > >(numEntityDimensions); // map to possible parents
  _generalizedParentEntities = vector< map<IndexType, pair<IndexType,unsigned> > >(numEntityDimensions);
  _childEntities = vector< map< IndexType, vector< pair<RefinementPatternPtr, vector<IndexType> > > > >(numEntityDimensions);
  _entityCellTopologyKeys = vector< map<CellTopologyKey, RangeList<IndexType> > >(numEntityDimensions);
  _nextCellIndex = 0;
  _activeCellCount = 0;
  _cellCount = 0;
  
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

  GlobalIndexType cellID = _nextCellIndex;
  for (int i=0; i<numElements; i++)
  {
    CellTopoPtr cellTopo = meshGeometry->cellTopos()[i];
    vector<IndexType> cellVerticesInMeshGeometry = meshGeometry->elementVertices()[i];
    vector<IndexType> cellVertices;
    for (int j=0; j<cellVerticesInMeshGeometry.size(); j++)
    {
      cellVertices.push_back(myVertexIndexForMeshGeometryIndex[cellVerticesInMeshGeometry[j]]);
    }
    addCell(cellID, cellTopo, cellVertices);
    cellID++;
  }
}

MeshTopology::MeshTopology(Epetra_CommPtr Comm, const MeshGeometryInfo &meshGeometryInfo)
{
  // initialize data structures with spaceDim
  // TODO: consider storing spaceDim in meshGeometryInfo
  int mySpaceDim = 0;
  if (meshGeometryInfo.rootCellIDs.size() != 0)
  {
    CellTopologyKey sampleCellTopoKey = meshGeometryInfo.rootCellTopos[0];
    mySpaceDim = CellTopology::cellTopology(sampleCellTopoKey)->getDimension();
  }
  int globalSpaceDim;
  Comm->MaxAll(&mySpaceDim, &globalSpaceDim, 1);
  init(globalSpaceDim);
  
  _Comm = Comm;
  _activeCellCount = meshGeometryInfo.globalActiveCellCount;
  _nextCellIndex = meshGeometryInfo.globalCellCount;
  int rootCellOrdinal = 0;
  for (IndexType rootCellID : meshGeometryInfo.rootCellIDs)
  {
    const vector<vector<double>>* vertices = &meshGeometryInfo.rootVertices[rootCellOrdinal];
    CellTopoPtr cellTopo = CellTopology::cellTopology(meshGeometryInfo.rootCellTopos[rootCellOrdinal]);
    this->addCell(rootCellID, cellTopo, *vertices);
    rootCellOrdinal++;
  }
  int numLevels = meshGeometryInfo.refinementLevels.size();
  for (int level=0; level<numLevels; level++)
  {
    for (auto entry : meshGeometryInfo.refinementLevels[level])
    {
      RefinementPatternKey refPatternKey = entry.first;
      RefinementPatternPtr refPattern = RefinementPattern::refinementPattern(refPatternKey);
      for (pair<GlobalIndexType,GlobalIndexType> parentAndFirstChildID : entry.second)
      {
        GlobalIndexType parentCellID = parentAndFirstChildID.first;
        GlobalIndexType firstChildCellID = parentAndFirstChildID.second;
        refineCell(parentCellID, refPattern, firstChildCellID);
      }
    }
  }
  _ownedCellIndices.insert(meshGeometryInfo.myCellIDs.begin(), meshGeometryInfo.myCellIDs.end());
}

IndexType MeshTopology::activeCellCount() const
{
  return _activeCellCount;
}

const set<IndexType> & MeshTopology::getLocallyKnownActiveCellIndices() const
{
  return _activeCells;
}

const set<IndexType> & MeshTopology::getMyActiveCellIndices() const
{
  return _ownedCellIndices;
}

vector<IndexType> MeshTopology::getActiveCellIndicesGlobal() const
{
  if (Comm() == Teuchos::null)
  {
    // then the MeshTopology should be *replicated*, and _activeCells will do the trick
    vector<IndexType> activeCellsVector(_activeCells.begin(), _activeCells.end());
    return activeCellsVector;
  }
  
  const set<GlobalIndexType>* myCellIDs = &_ownedCellIndices;
  int myCellCount = myCellIDs->size();
  int priorCellCount = 0;
  Comm()->ScanSum(&myCellCount, &priorCellCount, 1);
  priorCellCount -= myCellCount;
  int globalCellCount = 0;
  Comm()->SumAll(&myCellCount, &globalCellCount, 1);
  vector<int> allCellIDsInt(globalCellCount);
  auto myCellIDPtr = myCellIDs->begin();
  for (int i=priorCellCount; i<priorCellCount+myCellCount; i++)
  {
    allCellIDsInt[i] = *myCellIDPtr;
    myCellIDPtr++;
  }
  vector<int> gatheredCellIDs(globalCellCount);
  Comm()->SumAll(&allCellIDsInt[0], &gatheredCellIDs[0], globalCellCount);
  vector<IndexType> allCellIDs(gatheredCellIDs.begin(),gatheredCellIDs.end());
  return allCellIDs;
}

map<string, long long> MeshTopology::approximateMemoryCosts() const
{
  map<string, long long> variableCost;

  // calibrate by computing some sizes
  map<int, int> emptyMap;
//  set<int> emptySet;
  vector<int> emptyVector;

  int MAP_OVERHEAD = sizeof(emptyMap);
//  int SET_OVERHEAD = sizeof(emptySet);
  int VECTOR_OVERHEAD = sizeof(emptyVector);

  int MAP_NODE_OVERHEAD = 32; // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal

  variableCost["_spaceDim"] = sizeof(_spaceDim);

  variableCost["_vertexMap"] = approximateMapSizeLLVM(_vertexMap);

  variableCost["_vertices"] = VECTOR_OVERHEAD; // for the outer vector _vertices.
  for (vector< vector<double> >::const_iterator entryIt = _vertices.begin(); entryIt != _vertices.end(); entryIt++)
  {
    variableCost["_vertices"] += approximateVectorSizeLLVM(*entryIt);
  }
//  variableCost["_vertices"] += VECTOR_OVERHEAD * (_vertices.capacity() - _vertices.size());

  variableCost["_periodicBCs"] = approximateVectorSizeLLVM(_periodicBCs);

  variableCost["_periodicBCIndicesMatchingNode"] = MAP_OVERHEAD; // for map _periodicBCIndicesMatchingNode
  for (map<IndexType, set< pair<int, int> > >::const_iterator entryIt = _periodicBCIndicesMatchingNode.begin(); entryIt != _periodicBCIndicesMatchingNode.end(); entryIt++)
  {
    variableCost["_periodicBCIndicesMatchingNode"] += MAP_NODE_OVERHEAD;
    variableCost["_periodicBCIndicesMatchingNode"] += sizeof(IndexType);
    variableCost["_periodicBCIndicesMatchingNode"] += approximateSetSizeLLVM(entryIt->second);
  }

  variableCost["_equivalentNodeViaPeriodicBC"] = approximateMapSizeLLVM(_equivalentNodeViaPeriodicBC); // for map _equivalentNodeViaPeriodicBC

  variableCost["_entities"] = VECTOR_OVERHEAD; // for outer vector _entities
  for (vector< vector< vector<IndexType> > >::const_iterator entryIt = _entities.begin(); entryIt != _entities.end(); entryIt++)
  {
    variableCost["_entities"] += VECTOR_OVERHEAD; //
    for (vector< vector<IndexType> >::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_entities"] += approximateVectorSizeLLVM(*entry2It);
    }
//    variableCost["_entities"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
//  variableCost["_entities"] += VECTOR_OVERHEAD * (_entities.capacity() - _entities.size());

  variableCost["_knownEntities"] = VECTOR_OVERHEAD; // for outer vector _knownEntities
  for (vector< map< vector<IndexType>, IndexType > >::const_iterator entryIt = _knownEntities.begin(); entryIt != _knownEntities.end(); entryIt++)
  {
    variableCost["_knownEntities"] += MAP_OVERHEAD; // for inner map
    for (map< vector<IndexType>, IndexType >::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      vector<IndexType> entryVector = entry2It->first;
      variableCost["_knownEntities"] += approximateVectorSizeLLVM(entryVector) + sizeof(IndexType);
    }
  }
//  variableCost["_knownEntities"] += MAP_OVERHEAD * (_knownEntities.capacity() - _knownEntities.size());

  variableCost["_canonicalEntityOrdering"] = VECTOR_OVERHEAD; // for outer vector _canonicalEntityOrdering
  for (vector< vector< vector<IndexType> > >::const_iterator entryIt = _canonicalEntityOrdering.begin(); entryIt != _canonicalEntityOrdering.end(); entryIt++)
  {
    variableCost["_canonicalEntityOrdering"] += VECTOR_OVERHEAD;
    for (vector< vector<IndexType> >::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_canonicalEntityOrdering"] += approximateVectorSizeLLVM(*entry2It);
    }
//    variableCost["_canonicalEntityOrdering"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
//  variableCost["_canonicalEntityOrdering"] += MAP_OVERHEAD * (_canonicalEntityOrdering.capacity() - _canonicalEntityOrdering.size());

  variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD; // for outer vector _activeCellsForEntities
  for (vector< vector< vector< pair<IndexType, unsigned> > > >::const_iterator entryIt = _activeCellsForEntities.begin(); entryIt != _activeCellsForEntities.end(); entryIt++)
  {
    variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD; // inner vector
    for (vector< vector< pair<IndexType, unsigned> > >::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_activeCellsForEntities"] += approximateVectorSizeLLVM(*entry2It);
    }
//    variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
//  variableCost["_activeCellsForEntities"] += VECTOR_OVERHEAD * (_activeCellsForEntities.capacity() - _activeCellsForEntities.size());

  variableCost["_sidesForEntities"] = VECTOR_OVERHEAD; // _sidesForEntities
  for (vector< vector< vector<IndexType> > >::const_iterator entryIt = _sidesForEntities.begin(); entryIt != _sidesForEntities.end(); entryIt++)
  {
    variableCost["_sidesForEntities"] += VECTOR_OVERHEAD;
    for (vector< vector<IndexType> >::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_sidesForEntities"] += sizeof(IndexType);
      variableCost["_sidesForEntities"] += approximateVectorSizeLLVM(*entry2It);
    }
//    variableCost["_sidesForEntities"] += VECTOR_OVERHEAD * (entryIt->capacity() - entryIt->size());
  }
//  variableCost["_sidesForEntities"] += VECTOR_OVERHEAD * (_sidesForEntities.capacity() - _sidesForEntities.size());

  variableCost["_cellsForSideEntities"] = approximateVectorSizeLLVM(_cellsForSideEntities);

  variableCost["_boundarySides"] = approximateSetSizeLLVM(_boundarySides);

  variableCost["_parentEntities"] = VECTOR_OVERHEAD; // vector _parentEntities
  for (vector< map< IndexType, vector< pair<IndexType, unsigned> > > >::const_iterator entryIt = _parentEntities.begin(); entryIt != _parentEntities.end(); entryIt++)
  {
    variableCost["_parentEntities"] += MAP_OVERHEAD; // map
    for (map< IndexType, vector< pair<IndexType, unsigned> > > ::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_parentEntities"] += MAP_NODE_OVERHEAD; // map node
      variableCost["_parentEntities"] += sizeof(IndexType);
      variableCost["_parentEntities"] += approximateVectorSizeLLVM(entry2It->second);
    }
  }
//  variableCost["_parentEntities"] += MAP_OVERHEAD * (_parentEntities.capacity() - _parentEntities.size());

  variableCost["_generalizedParentEntities"] = VECTOR_OVERHEAD; // vector _generalizedParentEntities
  for (vector< map< IndexType, pair<IndexType, unsigned> > >::const_iterator entryIt = _generalizedParentEntities.begin(); entryIt != _generalizedParentEntities.end(); entryIt++)
  {
    variableCost["_generalizedParentEntities"] += approximateMapSizeLLVM(*entryIt);
  }
//  variableCost["_generalizedParentEntities"] += MAP_OVERHEAD * (_generalizedParentEntities.capacity() - _generalizedParentEntities.size());

  variableCost["_childEntities"] = VECTOR_OVERHEAD; // vector _childEntities
  for (vector< map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > > >::const_iterator entryIt = _childEntities.begin(); entryIt != _childEntities.end(); entryIt++)
  {
    variableCost["_childEntities"] += MAP_OVERHEAD; // map
    for (map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > >::const_iterator entry2It = entryIt->begin(); entry2It != entryIt->end(); entry2It++)
    {
      variableCost["_childEntities"] += MAP_NODE_OVERHEAD; // map node
      variableCost["_childEntities"] += sizeof(IndexType);

      variableCost["_childEntities"] += VECTOR_OVERHEAD; // vector
      for (vector< pair< RefinementPatternPtr, vector<IndexType> > >::const_iterator entry3It = entry2It->second.begin(); entry3It != entry2It->second.end(); entry3It++)
      {
        variableCost["_childEntities"] += sizeof(RefinementPatternPtr);
        variableCost["_childEntities"] += approximateVectorSizeLLVM(entry3It->second);
      }
//      variableCost["_childEntities"] += sizeof(pair< RefinementPatternPtr, vector<IndexType> >) * (entry2It->second.capacity() - entry2It->second.size());
    }
  }
//  variableCost["_childEntities"] += MAP_OVERHEAD * (_childEntities.capacity() - _childEntities.size());

  variableCost["_entityCellTopologyKeys"] = VECTOR_OVERHEAD; // _entityCellTopologyKeys vector
  for (vector< map< CellTopologyKey, RangeList<IndexType> > >::const_iterator entryIt = _entityCellTopologyKeys.begin(); entryIt != _entityCellTopologyKeys.end(); entryIt++)
  {
    variableCost["_entityCellTopologyKeys"] += MAP_OVERHEAD;
    for (auto mapEntry : *entryIt)
    {
      variableCost["_entityCellTopologyKeys"] += MAP_NODE_OVERHEAD;
      variableCost["_entityCellTopologyKeys"] += sizeof(CellTopologyKey);
      // RangeList is two vectors of length RangeList.length(), plus an int _size value
      variableCost["_entityCellTopologyKeys"] += sizeof(int);
      variableCost["_entityCellTopologyKeys"] += VECTOR_OVERHEAD * 2;
      variableCost["_entityCellTopologyKeys"] += sizeof(IndexType) * 2 * mapEntry.second.length();
    }
  }
//  variableCost["_entityCellTopologyKeys"] += VECTOR_OVERHEAD * (_entityCellTopologyKeys.capacity() - _entityCellTopologyKeys.size());

  variableCost["_cells"] = approximateMapSizeLLVM(_cells); // _cells map
  for (auto cellEntry = _cells.begin(); cellEntry != _cells.end(); cellEntry++)
  {
    variableCost["_cells"] += cellEntry->second->approximateMemoryFootprint();
  }

  variableCost["_activeCells"] = approximateSetSizeLLVM(_activeCells);
  variableCost["_rootCells"] = approximateSetSizeLLVM(_rootCells);

  variableCost["_cellIDsWithCurves"] = approximateSetSizeLLVM(_cellIDsWithCurves);

  variableCost["_edgeToCurveMap"] = approximateMapSizeLLVM(_edgeToCurveMap);

  return variableCost;
}

long long MeshTopology::approximateMemoryFootprint() const
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

CellPtr MeshTopology::addCell(CellTopoPtr cellTopo, const vector< vector<double> > &cellVertices)
{
  return addCell(_nextCellIndex, cellTopo, cellVertices);
}

CellPtr MeshTopology::addCell(CellTopoPtr cellTopo, const Camellia::FieldContainer<double> &cellVertices)
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
  return addCell(cellTopo, cellVertexVector);
}

CellPtr MeshTopology::addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<vector<double> > &cellVertices)
{
  if (cellTopo->getNodeCount() != cellVertices.size())
  {
    cout << "ERROR: cellTopo->getNodeCount() != cellVertices.size().\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"cellTopo->getNodeCount() != cellVertices.size()");
  }

  vector<IndexType> vertexIndices = getVertexIndices(cellVertices);
  addCell(cellIndex, cellTopo, vertexIndices);
  return _cells[cellIndex];
}

IndexType MeshTopology::addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_cells.find(cellIndex) != _cells.end(), std::invalid_argument, "addCell: cell with specified cellIndex already exists!");
 
  if (cellIndex < _nextCellIndex)
  {
    // then we take it that we're being told about a previously existing cell
    // We may have pruned away either this cell or its ancestor.
    // Upshot: we don't need to increase _nextCellIndex...
  }
  else if (cellIndex == _nextCellIndex)
  {
    _nextCellIndex++;
    _activeCellCount++;
    _cellCount++;
  }
  else
  {
    // should not get here
    cout << "Error: adding cell " << cellIndex << ", which is greater than _nextCellIndex.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex must be <= _nextCellIndex.");
  }
  
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
      IndexType entityIndex;
      unsigned entityPermutation;
      vector< IndexType > nodes;
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
        _activeCellsForEntities[d].resize(entityIndex + 100, vector< pair<IndexType, unsigned> >());
      }
      _activeCellsForEntities[d][entityIndex].push_back(make_pair(cellIndex,j));

      // now that we've added, sort:
      std::sort(_activeCellsForEntities[d][entityIndex].begin(), _activeCellsForEntities[d][entityIndex].end());

      if (d == 0)   // vertex --> should set parent relationships for any vertices that are equivalent via periodic BCs
      {
//        {
//          std::cout << "added {" << cellIndex << "," << j << "} as active cell entry for vertex " << entityIndex << std::endl;
//        }
        if (_periodicBCIndicesMatchingNode.find(entityIndex) != _periodicBCIndicesMatchingNode.end())
        {
          for (const auto &bcMatch : _periodicBCIndicesMatchingNode[entityIndex])
          {
            IndexType equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(entityIndex, bcMatch)];
            if (_activeCellsForEntities[d].size() <= equivalentNode)   // expand container
            {
              _activeCellsForEntities[d].resize(equivalentNode + 100, vector< pair<IndexType, unsigned> >());
            }
            auto & activeCellsForCanonicalVertex = _activeCellsForEntities[d][equivalentNode];
            activeCellsForCanonicalVertex.push_back(make_pair(cellIndex, j));
            // now that we've added, sort:
            std::sort(activeCellsForCanonicalVertex.begin(), activeCellsForCanonicalVertex.end());
          }
        }
      }
    }
  }
  CellPtr cell = Teuchos::rcp( new Cell(cellTopo, cellVertices, cellEntityPermutations, cellIndex, this) );
  _cells[cellIndex] = cell;
  _validCells.insert(cellIndex);
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
      pair<IndexType,unsigned> firstNeighbor  = getFirstCellForSide(sideEntityIndex);
      pair<IndexType,unsigned> secondNeighbor = getSecondCellForSide(sideEntityIndex);
      CellPtr firstCell = _cells[firstNeighbor.first];
      CellPtr secondCell = _cells[secondNeighbor.first];
      firstCell->setNeighbor(firstNeighbor.second, secondNeighbor.first, secondNeighbor.second, allowSameCellIndices);
      secondCell->setNeighbor(secondNeighbor.second, firstNeighbor.first, firstNeighbor.second, allowSameCellIndices);
      // TODO: Consider--might want to get rid of _boundarySides container altogether.  With distributed MeshTopology, it
      // also contains the sides of ghost cells that do not actually lie on the boundary; they're just on the boundary of
      // the region we know about...  The right way to know if a side is on the boundary is if it belongs to a cell we own
      // *AND* that cell has no neighbor on that side.
      if (_boundarySides.find(sideEntityIndex) != _boundarySides.end())
      {
        if (_childEntities[sideDim].find(sideEntityIndex) != _childEntities[sideDim].end())
        {
          // this can happen in context of migrated geometry
          // then we also should erase all descendants of sideEntityIndex from _boundarySides
          set<IndexType> childSideEntityIndices = descendants(sideDim, sideEntityIndex);
          for (IndexType childSideEntityIndex : childSideEntityIndices)
          {
            _boundarySides.erase(childSideEntityIndex);
          }
        }
        _boundarySides.erase(sideEntityIndex);
      }
      // if the pre-existing neighbor is refined, set its descendants to have the appropriate neighbor.
      MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
      if (firstCell->isParent(thisPtr))
      {
        const bool leafNodesOnly = false; // we want to set the neighbor relationship for all levels of refinement hierarchy
        vector< pair< GlobalIndexType, unsigned> > firstCellDescendants = firstCell->getDescendantsForSide(firstNeighbor.second, thisPtr, leafNodesOnly);
        for (pair< GlobalIndexType, unsigned> descendantEntry : firstCellDescendants)
        {
          GlobalIndexType childCellIndex = descendantEntry.first;
          unsigned childSideOrdinal = descendantEntry.second;
          // if the child is not locally known, presumably we don't need the neighbor relation set
          if (isValidCellIndex(childCellIndex))
          {
            getCell(childCellIndex)->setNeighbor(childSideOrdinal, secondNeighbor.first, secondNeighbor.second);
          }
        }
      }
      if (secondCell->isParent(thisPtr))
      {
        const bool leafNodesOnly = false; // we want to set the neighbor relationship for all levels of refinement hierarchy
        vector< pair< GlobalIndexType, unsigned> > secondCellDescendants = secondCell->getDescendantsForSide(secondNeighbor.second, thisPtr, leafNodesOnly);
        for (pair< GlobalIndexType, unsigned> descendantEntry : secondCellDescendants)
        {
          GlobalIndexType childCellIndex = descendantEntry.first;
          unsigned childSideOrdinal = descendantEntry.second;
          // if the child is not locally known, presumably we don't need the neighbor relation set
          if (isValidCellIndex(childCellIndex))
          {
            getCell(childCellIndex)->setNeighbor(childSideOrdinal, firstNeighbor.first, firstNeighbor.second);
          }
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
        // TODO: 3-9-16.  This is now the only use of getConstrainingSideAncestry(), outside of tests.  I think probably we should rewrite the below to eliminate it altogether.
        // (Can we just use getConstrainingEntityOfLikeDimension()?)
        vector< pair<IndexType, unsigned> > sideAncestry = getConstrainingSideAncestry(sideEntityIndex);
        // the last entry, if any, should refer to an active cell's side...
        if (sideAncestry.size() > 0)
        {
          unsigned sideAncestorIndex = sideAncestry[sideAncestry.size()-1].first;
          vector< pair<IndexType, unsigned> > activeCellEntries = _activeCellsForEntities[sideDim][sideAncestorIndex];
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
      set<IndexType> sideSubcellIndices = getEntitiesForSide(sideEntityIndex, d);
      for (IndexType subcellEntityIndex : sideSubcellIndices)
      {
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

void MeshTopology::addCellForSide(IndexType cellIndex, unsigned int sideOrdinal, IndexType sideEntityIndex)
{
//  {
//    // DEBUGGING
//    if ((Comm() != Teuchos::null) && (Comm()->MyPID() == 1) && (_spaceDim == 1))
//    {
//      if ((sideEntityIndex == 0) && (_vertices[sideEntityIndex][0] == 0.0))
//      {
//        cout << "addCellForSide(" << cellIndex << "," << sideOrdinal << "," << sideEntityIndex << ")\n";
//      }
//    }
//  }
  TEUCHOS_TEST_FOR_EXCEPT(sideEntityIndex == MESHTOPOLOGY_INVALID_ENTITY_INDEX);
  
  if (_cellsForSideEntities.size() <= sideEntityIndex)
  {
    _cellsForSideEntities.resize(sideEntityIndex + 100, {{-1,-1},{-1,-1}});
    pair< unsigned, unsigned > cell1 = make_pair(cellIndex, sideOrdinal);
    pair< unsigned, unsigned > cell2 = {-1,-1};
    
    // check for equivalent side that matches periodic BCs
    
    _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
  }
  else
  {
    auto cell1 = _cellsForSideEntities[sideEntityIndex].first;
    auto cell2 = _cellsForSideEntities[sideEntityIndex].second;

    CellPtr cellToAdd = getCell(cellIndex);
    IndexType parentCellIndex;
    if ( cellToAdd->getParent().get() == NULL)
    {
      parentCellIndex = MESHTOPOLOGY_INVALID_CELL_INDEX;
    }
    else
    {
      parentCellIndex = cellToAdd->getParent()->cellIndex();
    }
    if ((cell1.first == MESHTOPOLOGY_INVALID_CELL_INDEX) || (parentCellIndex == cell1.first))
    {
      // then replace cell1's entry with the new one
      cell1.first = cellIndex;
      cell1.second = sideOrdinal;
    }
    else if ((cell2.first == MESHTOPOLOGY_INVALID_CELL_INDEX) || (parentCellIndex == cell2.first))
    {
      cell2.first = cellIndex;
      cell2.second = sideOrdinal;
    }
    else
    {
      cout << "Internal error: attempt to add 3rd cell (" << cellIndex << ") for side with entity index " << sideEntityIndex;
      cout << ", which already has cells " << cell1.first << " and " << cell2.first << endl;
      printAllEntities();
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: attempt to add 3rd cell for side");
    }
    _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
  }
}

void MeshTopology::addEdgeCurve(pair<IndexType,IndexType> edge, ParametricCurvePtr curve)
{
  // note: does NOT update the MeshTransformationFunction.  That's caller's responsibility,
  // because we don't know whether there are more curves coming for the affected elements.

  unsigned edgeDim = 1;
  vector<IndexType> edgeNodes;
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
  Camellia::FieldContainer<double> curve0(spaceDim);
  Camellia::FieldContainer<double> curve1(spaceDim);
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

IndexType MeshTopology::addEntity(CellTopoPtr entityTopo, const vector<IndexType> &entityVertices, unsigned &entityPermutation)
{
  set< IndexType > nodeSet;
  nodeSet.insert(entityVertices.begin(),entityVertices.end());

  if (nodeSet.size() != entityVertices.size())
  {
    for (IndexType vertexIndex : entityVertices)
      printVertex(vertexIndex);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Entities may not have repeated vertices");
  }
  unsigned d  = entityTopo->getDimension();
  IndexType entityIndex = getEntityIndex(d, nodeSet);

  vector<IndexType> sortedVertices(nodeSet.begin(),nodeSet.end());

  if ( entityIndex == MESHTOPOLOGY_INVALID_ENTITY_INDEX )
  {
    // new entity
    entityIndex = _entities[d].size();
    _entities[d].push_back(sortedVertices);
    _knownEntities[d].insert(make_pair(sortedVertices, entityIndex));
    if (d != 0) _canonicalEntityOrdering[d].push_back(entityVertices);
    entityPermutation = 0;
    _entityCellTopologyKeys[d][entityTopo->getKey()].insert(entityIndex);
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

void MeshTopology::addChildren(IndexType firstChildIndex, CellPtr parentCell, const vector< CellTopoPtr > &childTopos, const vector< vector<IndexType> > &childVertices)
{
  int numChildren = childTopos.size();
  TEUCHOS_TEST_FOR_EXCEPTION(numChildren != childVertices.size(), std::invalid_argument, "childTopos and childVertices must be the same size");
  vector<GlobalIndexType> childIndices;
  IndexType childCellIndex = firstChildIndex; // children get continguous cell indices
  for (int childOrdinal=0; childOrdinal<numChildren; childOrdinal++)
  {
    // add if we don't already know about this child (we might already know it and have pruned its siblings away...)
    if (!isValidCellIndex(childCellIndex))
    {
      addCell(childCellIndex, childTopos[childOrdinal], childVertices[childOrdinal],parentCell->cellIndex());
      _rootCells.erase(childCellIndex);
    }
    childIndices.push_back(childCellIndex);
    childCellIndex++;
  }
  parentCell->setChildren(childIndices);
  
  // if any entity sets contain parent cell, add child cells, too
  for (auto entry : _entitySets)
  {
    EntitySetPtr entitySet = entry.second;
    if (entitySet->containsEntity(this->getDimension(), parentCell->cellIndex()))
    {
      for (IndexType childCellIndex : childIndices)
      {
        entitySet->addEntity(this->getDimension(), childCellIndex);
      }
    }
  }
}

CellPtr MeshTopology::addMigratedCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<vector<double>> &cellVertices)
{
  TEUCHOS_TEST_FOR_EXCEPTION(cellIndex >= _nextCellIndex, std::invalid_argument, "migrated cellIndex must be less than _nextCellIndex");
  return this->addCell(cellIndex, cellTopo, cellVertices);
}

void MeshTopology::addSideForEntity(unsigned int entityDim, IndexType entityIndex, IndexType sideEntityIndex)
{
  if (_sidesForEntities[entityDim].size() <= entityIndex)
  {
    _sidesForEntities[entityDim].resize(entityIndex + 100);
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

const MeshTopology* MeshTopology::baseMeshTopology() const
{
  return this;
}

vector<IndexType> MeshTopology::getCanonicalEntityNodesViaPeriodicBCs(unsigned d, const vector<IndexType> &myEntityNodes) const
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
      auto foundVertexIt = _canonicalVertexPeriodic.find(vertexIndex);
      if (foundVertexIt != _canonicalVertexPeriodic.end())
      {
        const auto & canonicalVertexIndex = foundVertexIt->second;
        return {canonicalVertexIndex};
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTopology error: vertex not found.");
      }
    }
    
    // compute the intersection of the periodic BCs that match each node in nodeSet
    set< pair<int, int> > matchingPeriodicBCsIntersection;
    bool firstNode = true;
    for (IndexType node : myEntityNodes)
    {
      auto foundEntry = _periodicBCIndicesMatchingNode.find(node);
      if (foundEntry == _periodicBCIndicesMatchingNode.end())
      {
        matchingPeriodicBCsIntersection.clear();
        break;
      }
      if (firstNode)
      {
        matchingPeriodicBCsIntersection = foundEntry->second;
        firstNode = false;
      }
      else
      {
        set< pair<int, int> > newSet;
        set< pair<int, int> > matchesForThisNode = foundEntry->second;
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
    for (pair<int,int> matchingBC : matchingPeriodicBCsIntersection)
    {
      vector<IndexType> equivalentNodeVector;
      for (IndexType node : myEntityNodes)
      {
        auto equivalentNodeEntry = _equivalentNodeViaPeriodicBC.find(make_pair(node, matchingBC));
        equivalentNodeVector.push_back(equivalentNodeEntry->second);
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

bool MeshTopology::cellHasCurvedEdges(IndexType cellIndex) const
{
  CellPtr cell = getCell(cellIndex);
  if (cell->topology()->getDimension() <= 1) return false; // 0D, 1D cells can't have curved edges
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

bool MeshTopology::cellContainsPoint(GlobalIndexType cellID, const vector<double> &point, int cubatureDegree) const
{
  // note that this design, with a single point being passed in, will be quite inefficient
  // if there are many points.  TODO: revise to allow multiple points (returning vector<bool>, maybe)
  int numCells = 1, numPoints = 1;
  Camellia::FieldContainer<double> physicalPoints(numCells,numPoints,_spaceDim);
  for (int d=0; d<_spaceDim; d++)
  {
    physicalPoints(0,0,d) = point[d];
  }
  //  cout << "cell " << elem->cellID() << ": (" << x << "," << y << ") --> ";
  Camellia::FieldContainer<double> refPoints(numCells,numPoints,_spaceDim);
  ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  CamelliaCellTools::mapToReferenceFrame(refPoints, physicalPoints, thisPtr, cellID, cubatureDegree);

  CellTopoPtr cellTopo = getCell(cellID)->topology();

  int result = CamelliaCellTools::checkPointInclusion(&refPoints[0], _spaceDim, cellTopo);
  return result == 1;
}

IndexType MeshTopology::cellCount() const
{
  return _cellCount;
}

vector<IndexType> MeshTopology::cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const
{
  // returns a vector of an active element per point, or -1 if there is no locally known element including that point
  vector<GlobalIndexType> cellIDs;
  //  cout << "entered elementsForPoints: \n" << physicalPoints;
  int numPoints = physicalPoints.dimension(0);

  int spaceDim = this->getDimension();

  set<GlobalIndexType> rootCellIndices = this->getRootCellIndicesLocal();

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
      ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
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
          Camellia::FieldContainer<double> vertices(numVertices,spaceDim);
          vector<IndexType> vertexIndices = cell->vertices();

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
            Camellia::FieldContainer<double> vertices(numVertices,spaceDim);
            vector<IndexType> vertexIndices = child->vertices();

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

// ! If the MeshTopology is distributed, returns the Comm object used.  Otherwise, returns Teuchos::null, which is meant to indicate that the MeshTopology is replicated on every MPI rank on which it is used.
Epetra_CommPtr MeshTopology::Comm() const
{
  return _Comm;
}

CellPtr MeshTopology::findCellWithVertices(const vector< vector<double> > &cellVertices) const
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
//      cout << "vertex not found. returning NULL.\n";
      return cell;
    }
    // otherwise, vertexIndex has been populated
    vertexIndices.push_back(vertexIndex);

    set< pair<IndexType, unsigned> > matchingLeafCellPairs = getCellsContainingEntity(vertexDim, vertexIndex);
    
    // getCellsContainingEntity() only returns the most refined cells that contain that entity; let's add any ancestors of those cells that contain the vertex
    set<IndexType> matchingCellsForVertex;
    for (const auto & leafEntry : matchingLeafCellPairs)
    {
      const auto & cellID = leafEntry.first;
      
      CellPtr ancestralCell = getCell(cellID);
      while (ancestralCell != Teuchos::null)
      {
        // when periodic BCs are in play, getCellsContainingEntity() returns matches on the other side of the BC from the specified coordinate
        // we don't want findCellWithVertices() to include such matches.
        // similarly, once we hit an ancestor that doesn't include the vertex, no further ancestors will include the vertex, so we can break out.
        bool foundVertexInCell = false;
      
        const vector<IndexType> & cellVertices = ancestralCell->vertices();
        for (auto &cellVertexIndex : cellVertices)
        {
          if (vertexIndex == cellVertexIndex)
          {
            foundVertexInCell = true;
            break;
          }
        }
        
        if (foundVertexInCell)
        {
          matchingCellsForVertex.insert(ancestralCell->cellIndex());
          ancestralCell = ancestralCell->getParent();
        }
        else
        {
          break; // if the leaf cell's vertex was not found in ancestralCell, no ancestors of ancestralCell will contain that vertex, either
        }
      }
    }
    
    set<IndexType> matchingCellsIntersection;
    for (const auto &vertexMatchingCellID : matchingCellsForVertex)
    {
      if (firstVertex)
      {
        matchingCellsIntersection.insert(vertexMatchingCellID);
      }
      else
      {
        if (matchingCells.find(vertexMatchingCellID) != matchingCells.end())
        {
          matchingCellsIntersection.insert(vertexMatchingCellID);
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

set< pair<IndexType, unsigned> > MeshTopology::getActiveBoundaryCells() const  // (cellIndex, sideOrdinal)
{
  set< pair<IndexType, unsigned> > boundaryCells;
  for (IndexType sideEntityIndex : _boundarySides)
  {
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

vector<double> MeshTopology::getCellCentroid(IndexType cellIndex) const
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

unsigned MeshTopology::getCellCountForSide(IndexType sideEntityIndex) const
{
  if (sideEntityIndex >= _cellsForSideEntities.size())
  {
    return 0;
  }
  else
  {
    pair<IndexType, unsigned> cell1 = _cellsForSideEntities[sideEntityIndex].first;
    pair<IndexType, unsigned> cell2 = _cellsForSideEntities[sideEntityIndex].second;
    if ((cell1.first == -1) && (cell2.first == -1))
    {
      return 0;
    }
    else if (cell2.first == -1)
    {
      return 1;
    }
    else
    {
      return 2;
    }
  }
}

vector<EntityHandle> MeshTopology::getEntityHandlesForCell(IndexType cellIndex) const
{
  ConstMeshTopologyViewPtr thisPtr = Teuchos::rcp(this,false);
  vector<EntityHandle> handles;
  for (auto entry : _entitySets)
  {
    EntityHandle handle = entry.first;
    if (entry.second->cellIDsThatMatch(thisPtr, {cellIndex}).size() > 0)
    {
      handles.push_back(handle);
    }
  }
  return handles;
}

vector<EntitySetPtr> MeshTopology::getEntitySetsForTagID(string tagName, int tagID) const
{
  auto foundTagSetsEntry = _tagSetsInteger.find(tagName);
  if (foundTagSetsEntry == _tagSetsInteger.end()) return vector<EntitySetPtr>();
  
  vector<EntitySetPtr> entitySets;
  for (pair<EntityHandle,int> tagEntry : foundTagSetsEntry->second)
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
  if (_initialTimeEntityHandle == INVALID_ENTITY_HANDLE) return Teuchos::null;
  return getEntitySet(_initialTimeEntityHandle);
}

pair<IndexType, unsigned> MeshTopology::getFirstCellForSide(IndexType sideEntityIndex) const
{
  if (sideEntityIndex >= _cellsForSideEntities.size()) return {-1,-1};
  return _cellsForSideEntities[sideEntityIndex].first;
}

pair<IndexType, unsigned> MeshTopology::getSecondCellForSide(IndexType sideEntityIndex) const
{
  if (sideEntityIndex >= _cellsForSideEntities.size()) return {-1,-1};
  return _cellsForSideEntities[sideEntityIndex].second;
}

const map<EntityHandle, EntitySetPtr>& MeshTopology::getEntitySets() const
{
  return _entitySets;
}

const map<string, vector<pair<EntityHandle, int>>>& MeshTopology::getTagSetsInteger() const
{
  // tags with integer value, applied to EntitySets.
  return _tagSetsInteger;
}

vector<IndexType> MeshTopology::getBoundarySidesThatMatch(SpatialFilterPtr spatialFilter) const
{
  int sideDim = getDimension() - 1;
  vector<IndexType> matchingSides;
  for (IndexType sideEntityIndex : _boundarySides)
  {
    const vector<IndexType>* nodesForSide = &_entities[sideDim][sideEntityIndex];
    bool allMatch = true;
    for (IndexType vertexIndex : *nodesForSide)
    {
      if (! spatialFilter->matchesPoint(_vertices[vertexIndex]) )
      {
        allMatch = false;
        break;
      }
    }
    if (allMatch) matchingSides.push_back(sideEntityIndex);
  }
  return matchingSides;
}

const std::vector<PeriodicBCPtr> & MeshTopology::getPeriodicBCs() const
{
  return _periodicBCs;
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
      set< IndexType > nodeSet;
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

      IndexType entityIndex = getEntityIndex(d, nodeSet);
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
          cout << "WARNING: attempt was made to deactivate a non-active subcell topology...";
          cout << " deactivating cell " << cell->cellIndex() << endl;
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
    }
  }
  _activeCells.erase(cell->cellIndex());
}

MeshTopologyPtr MeshTopology::deepCopy() const
{
  MeshTopologyPtr meshTopoCopy = Teuchos::rcp( new MeshTopology(*this) );
  meshTopoCopy->deepCopyCells();
  map<EntityHandle, EntitySetPtr> newEntitySets;
  for (auto entry : _entitySets)
  {
    EntitySetPtr entitySetCopy = Teuchos::rcp( new EntitySet(*entry.second) );
    newEntitySets[entry.first] = entitySetCopy;
  }
  meshTopoCopy->_entitySets = newEntitySets;
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
    vector<GlobalIndexType> childIndices;
    for (int childOrdinal=0; childOrdinal<oldCell->children().size(); childOrdinal++)
    {
      GlobalIndexType newChildIndex = oldCell->children()[childOrdinal]->cellIndex();
      childIndices.push_back(newChildIndex);
    }
    _cells[oldCellIndex]->setChildren(childIndices);
  }
}

set<IndexType> MeshTopology::descendants(unsigned d, IndexType entityIndex) const
{
  set<IndexType> allDescendants;

  allDescendants.insert(entityIndex);
  auto foundChildEntitiesEntry = _childEntities[d].find(entityIndex);
  if (foundChildEntitiesEntry != _childEntities[d].end())
  {
    set<IndexType> unfollowedDescendants;
    for (unsigned i=0; i<foundChildEntitiesEntry->second.size(); i++)
    {
      vector<IndexType> immediateChildren = foundChildEntitiesEntry->second[i].second;
      unfollowedDescendants.insert(immediateChildren.begin(), immediateChildren.end());
    }
    for (set<IndexType>::iterator descIt=unfollowedDescendants.begin(); descIt!=unfollowedDescendants.end(); descIt++)
    {
      set<IndexType> myDescendants = descendants(d,*descIt);
      allDescendants.insert(myDescendants.begin(),myDescendants.end());
    }
  }

  return allDescendants;
}

bool MeshTopology::entityHasChildren(unsigned int d, IndexType entityIndex) const
{
  if (d == _spaceDim)
  {
    // interpret entityIndex as a Cell
    ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
    return getCell(entityIndex)->isParent(thisPtr);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(d >= _childEntities.size(), std::invalid_argument, "d is out of bounds");
  auto foundChildEntitiesEntry = _childEntities[d].find(entityIndex);
  if (foundChildEntitiesEntry == _childEntities[d].end()) return false;
  return foundChildEntitiesEntry->second.size() > 0;
}

bool MeshTopology::entityHasParent(unsigned d, IndexType entityIndex) const
{
  auto foundParentEntitiesEntry = _parentEntities[d].find(entityIndex);
  if (foundParentEntitiesEntry == _parentEntities[d].end()) return false;
  return foundParentEntitiesEntry->second.size() > 0;
}

bool MeshTopology::entityHasGeneralizedParent(unsigned d, IndexType entityIndex) const
{
  return _generalizedParentEntities[d].find(entityIndex) != _generalizedParentEntities[d].end();
}

bool MeshTopology::entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent) const
{
  if (ancestor == descendent) return true;
  auto parentIt = _parentEntities[d].find(descendent);
  while (parentIt != _parentEntities[d].end())
  {
    vector< pair<IndexType, unsigned> > parents = parentIt->second;
    IndexType parentEntityIndex = MESHTOPOLOGY_INVALID_ENTITY_INDEX;
    for (vector< pair<IndexType, unsigned> >::iterator entryIt = parents.begin(); entryIt != parents.end(); entryIt++)
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
    unsigned descendentDimension, IndexType descendent) const
{
  // note that this method does not treat the possibility of multiple parents, which can happen in
  // the context of anisotropic refinements.
  if (ancestorDimension == descendentDimension)
  {
    return entityIsAncestor(ancestorDimension, ancestor, descendent);
  }
  if (ancestorDimension < descendentDimension) return false;

  auto foundEntry = _generalizedParentEntities[descendentDimension].find(descendent);
  while (foundEntry != _generalizedParentEntities[descendentDimension].end())
  {
    pair<IndexType, unsigned> generalizedParent = foundEntry->second;
    descendentDimension = generalizedParent.second;
    descendent = generalizedParent.first;
    if (descendent == ancestor) return true;
    foundEntry = _generalizedParentEntities[descendentDimension].find(descendent);
  }
  return false;
}

IndexType MeshTopology::getActiveCellCount(unsigned int d, IndexType entityIndex) const
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

vector< pair<IndexType,unsigned> > MeshTopology::getActiveCellIndices(unsigned d, IndexType entityIndex) const
{
  return _activeCellsForEntities[d][entityIndex];
}

CellPtr MeshTopology::getCell(IndexType cellIndex) const
{
  auto entry = _cells.find(cellIndex);
  if (entry == _cells.end())
  {
    cout << "MeshTopology::getCell: cellIndex " << cellIndex << " is invalid.\n";
    vector<GlobalIndexType> validIndices;
    for (auto cellEntry : _cells)
    {
      validIndices.push_back(cellEntry.first);
    }
    print("valid cells", validIndices);
    print("owned cells", _ownedCellIndices);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex is invalid.\n");
  }
  return entry->second;
}

vector<IndexType> MeshTopology::getCellsForSide(IndexType sideEntityIndex) const
{
  vector<IndexType> cells;
  IndexType cellIndex = this->getFirstCellForSide(sideEntityIndex).first;
  if (cellIndex != -1) cells.push_back(cellIndex);
  cellIndex = this->getSecondCellForSide(sideEntityIndex).first;
  if (cellIndex != -1) cells.push_back(cellIndex);
  return cells;
}

IndexType MeshTopology::getEntityCount(unsigned int d) const
{
  if (d==0) return _vertices.size();
  return _entities[d].size();
}

pair<IndexType, unsigned> MeshTopology::getEntityGeneralizedParent(unsigned int d, IndexType entityIndex) const
{
  if ((d < _generalizedParentEntities.size()) && (_generalizedParentEntities[d].find(entityIndex) != _generalizedParentEntities[d].end()))
    return _generalizedParentEntities[d].find(entityIndex)->second;
  else
  {
    // entity may be a cell, in which case parent is also a cell (if there is a parent)
    if (d == _spaceDim)
    {
      auto foundCellEntry = _cells.find(entityIndex);
      if (foundCellEntry == _cells.end())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid cell index");
      }
      CellPtr cell = foundCellEntry->second;
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
        CellPtr cell = _cells.find(cellIndex)->second;
        if (cell->getParent() != Teuchos::null)
        {
          return make_pair(cell->getParent()->cellIndex(), _spaceDim);
        }
      }
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Entity generalized parent not found...");
//  return make_pair(-1,-1);
}

IndexType MeshTopology::getEntityIndex(unsigned d, const set<IndexType> &nodeSet) const
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
  vector<IndexType> sortedNodes(nodeSet.begin(),nodeSet.end());
  auto foundEntity = _knownEntities[d].find(sortedNodes);
  if (foundEntity != _knownEntities[d].end())
  {
    return foundEntity->second;
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
      foundEntity = _knownEntities[d].find(sortedEquivalentNodeVector);
      if (foundEntity != _knownEntities[d].end())
      {
        return foundEntity->second;
      }
    }
  }
  return -1;
}

IndexType MeshTopology::getEntityParent(unsigned d, IndexType entityIndex, unsigned parentOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(! entityHasParent(d, entityIndex), std::invalid_argument, "entity does not have parent");
  auto foundParentEntry = _parentEntities[d].find(entityIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(foundParentEntry == _parentEntities[d].end(), std::invalid_argument, "parent entity entry not found");
  return foundParentEntry->second[parentOrdinal].first;
}

CellTopoPtr MeshTopology::getEntityTopology(unsigned d, IndexType entityIndex) const
{
  if (d < _spaceDim)
  {
    for (map<Camellia::CellTopologyKey, RangeList<IndexType>>::const_iterator entryIt = _entityCellTopologyKeys[d].begin();
         entryIt != _entityCellTopologyKeys[d].end(); entryIt++)
    {
      if (entryIt->second.contains(entityIndex))
      {
        return CellTopology::cellTopology(entryIt->first);
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "entityIndex is out of bounds");
  }
  else
  {
    return getCell(entityIndex)->topology();
  }
}

vector<IndexType> MeshTopology::getEntityVertexIndices(unsigned d, IndexType entityIndex) const
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

set<IndexType> MeshTopology::getEntitiesForSide(IndexType sideEntityIndex, unsigned d) const
{
  unsigned sideDim = _spaceDim - 1;
  unsigned subEntityCount = getSubEntityCount(sideDim, sideEntityIndex, d);
  set<IndexType> subEntities;
  for (int subEntityOrdinal=0; subEntityOrdinal<subEntityCount; subEntityOrdinal++)
  {
    subEntities.insert(getSubEntityIndex(sideDim, sideEntityIndex, d, subEntityOrdinal));
  }
  return subEntities;
}

IndexType MeshTopology::getFaceEdgeIndex(IndexType faceIndex, unsigned int edgeOrdinalInFace) const
{
  return getSubEntityIndex(2, faceIndex, 1, edgeOrdinalInFace);
}

unsigned MeshTopology::getDimension() const
{
  return _spaceDim;
}

IndexType MeshTopology::getSubEntityCount(unsigned int d, IndexType entityIndex, unsigned int subEntityDim) const
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

IndexType MeshTopology::getSubEntityIndex(unsigned int d, IndexType entityIndex, unsigned int subEntityDim, unsigned int subEntityOrdinal) const
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
  set<IndexType> subEntityNodes;
  unsigned subEntityNodeCount = (subEntityDim > 0) ? entityTopo->getNodeCount(subEntityDim, subEntityOrdinal) : 1; // vertices are by definition just one node
  vector<IndexType> entityNodes = getEntityVertexIndices(d, entityIndex);

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
  IndexType subEntityIndex = getEntityIndex(subEntityDim, subEntityNodes);
  if (subEntityIndex == -1)
  {
    cout << "sub-entity not found with vertices:\n";
    printVertices(subEntityNodes);
    cout << "entity vertices:\n";
    set<IndexType> entityNodeSet(entityNodes.begin(),entityNodes.end());
    printVertices(entityNodeSet);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-entity not found");
  }
  return subEntityIndex;
}

void MeshTopology::getSubEntityIndices(unsigned d, IndexType entityIndex, unsigned subEntityDim, vector<IndexType> &subEntityIndices) const
{
  if (subEntityDim == d)
  {
    // entity is its own sub-entity:
    subEntityIndices.resize(1);
    subEntityIndices[0] = entityIndex;
  }
  else if (subEntityDim == 0)
  {
    // if interested in vertices, we know those:
    subEntityIndices = _canonicalEntityOrdering[d][entityIndex];
  }
  else
  {
    unsigned sideDim = getDimension() - 1;
    IndexType sideForEntity;
    if (d != sideDim)
    {
      TEUCHOS_TEST_FOR_EXCEPTION((entityIndex < 0) || (entityIndex > _sidesForEntities[d].size()), std::invalid_argument, "entityIndex is out of bounds");
      TEUCHOS_TEST_FOR_EXCEPTION(_sidesForEntities[d][entityIndex].size() == 0, std::invalid_argument, "No sides contain entity");
      sideForEntity = _sidesForEntities[d][entityIndex][0];
    }
    else
    {
      sideForEntity = entityIndex;
    }
    
    pair<IndexType,unsigned> cellEntry = getFirstCellForSide(sideForEntity);
    if (!isValidCellIndex(cellEntry.first))
    {
      cellEntry = getSecondCellForSide(sideForEntity);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellEntry.first), std::invalid_argument, "Internal error: cell found for side is not valid");
    CellPtr cell = getCell(cellEntry.first);
    
    CellTopoPtr cellTopo = cell->topology();
    unsigned sideOrdinal = cellEntry.second;
    unsigned entitySubcellOrdinalInCell = INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL;
    if (d == sideDim)
    {
      entitySubcellOrdinalInCell = sideOrdinal;
    }
    else
    {
      CellTopoPtr sideTopo = cellTopo->getSide(sideOrdinal);
      int subcellCount = sideTopo->getSubcellCount(d);
      for (int subcord=0; subcord<subcellCount; subcord++)
      {
        unsigned subcordInCell = CamelliaCellTools::subcellOrdinalMap(cellTopo, sideDim, sideOrdinal, d, subcord);
        if (cell->entityIndex(d, subcordInCell) == entityIndex)
        {
          entitySubcellOrdinalInCell = subcordInCell;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(entitySubcellOrdinalInCell == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL, std::invalid_argument, "entity not found in Cell");
    }
    CellTopoPtr subcellTopo = cellTopo->getSubcell(d, entitySubcellOrdinalInCell);
    int subsubcellCount = subcellTopo->getSubcellCount(subEntityDim);
    subEntityIndices.resize(subsubcellCount);
    for (int subsubcord=0; subsubcord<subsubcellCount; subsubcord++)
    {
      unsigned subsubcordInCell = CamelliaCellTools::subcellOrdinalMap(cellTopo, d, entitySubcellOrdinalInCell, subEntityDim, subsubcord);
      subEntityIndices[subsubcord] = cell->entityIndex(subEntityDim, subsubcordInCell);
    }
  }
}

const vector<double>& MeshTopology::getVertex(IndexType vertexIndex) const
{
  bool vertexOutOfBounds = (vertexIndex >= _vertices.size());
  TEUCHOS_TEST_FOR_EXCEPTION(vertexOutOfBounds, std::invalid_argument, "vertexIndex is out of bounds");
  return _vertices[vertexIndex];
}

bool MeshTopology::getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol) const
{
  auto foundVertexEntry = _vertexMap.find(vertex);
  if (foundVertexEntry != _vertexMap.end())
  {
    vertexIndex = foundVertexEntry->second;
    return true;
  }
  
  // if we don't have an exact match, we look for one that meets the tolerance.
  // (this is inefficient, and perhaps should be revisited.)
  
  vector<double> vertexForLowerBound;
  double magnitude = 0.0;
  double x_magnitude = (_spaceDim > 0) ? std::fabs(vertex[0]) : 0.0;
  for (int d=0; d<_spaceDim; d++)
  {
    double vertexForLowerBound_d;
    if (std::fabs(vertex[d]) < tol)
    {
      vertexForLowerBound_d = -tol;
    }
    else if (vertex[d] < 0.0)
    {
      vertexForLowerBound_d = vertex[d] * (1.0 + tol);
    }
    else
    {
      vertexForLowerBound_d = vertex[d] * (1.0 - tol);
    }
    vertexForLowerBound.push_back(vertexForLowerBound_d);
    magnitude += vertex[d] * vertex[d];
  }
  magnitude = std::sqrt(magnitude);

  auto lowerBoundIt = _vertexMap.lower_bound(vertexForLowerBound);
  long bestMatchIndex = -1;
  double bestMatchDistance = (magnitude > tol) ? tol * magnitude : tol;
  double xDist = 0; // xDist because vector<double> sorts according to the first entry: so we'll end up looking at
  // all the vertices that are near (x,...) in x...
  
  double xTol = (x_magnitude > tol) ? x_magnitude * tol : tol;
  while ((lowerBoundIt != _vertexMap.end()) && (xDist < xTol))
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
    vertexIndex = -1;
    return false;
  }
  else
  {
    vertexIndex = bestMatchIndex;
    return true;
  }
}

// Here, we assume that the initial coordinates provided are exactly equal (no round-off error) to the ones sought
vector<IndexType> MeshTopology::getVertexIndicesMatching(const vector<double> &vertexInitialCoordinates, double tol) const
{
  int numCoords = vertexInitialCoordinates.size();
  vector<double> vertexForLowerBound;
  for (int d=0; d<numCoords; d++)
  {
    vertexForLowerBound.push_back(vertexInitialCoordinates[d]-tol);
  }
  
  double xDist = 0; // xDist because vector<double> sorts according to the first entry: so we'll end up looking at
  // all the vertices that are near (x,...) in x...
  
  auto lowerBoundIt = _vertexMap.lower_bound(vertexInitialCoordinates);
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

IndexType MeshTopology::getVertexIndexAdding(const vector<double> &vertex, double tol)
{
  IndexType vertexIndex;
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
  _entityCellTopologyKeys[vertexDim][nodeTopo->getKey()].insert(vertexIndex);
  
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
      IndexType equivalentVertexIndex;
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
vector<IndexType> MeshTopology::getVertexIndices(const Camellia::FieldContainer<double> &vertices)
{
  double tol = 1e-14; // tolerance for vertex equality

  int numVertices = vertices.dimension(0);
  vector<IndexType> localToGlobalVertexIndex(numVertices);
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
map<unsigned, IndexType> MeshTopology::getVertexIndicesMap(const Camellia::FieldContainer<double> &vertices)
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

std::vector<IndexType> MeshTopology::getVertexIndicesForTime(double t) const
{
  // we take time to be the last dimension
  int d_time = getDimension() - 1;
  vector<IndexType> verticesThatMatch;
  int vertexCount = _vertices.size();
  for (IndexType vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
  {
    if (_vertices[vertexIndex][d_time] == t)
    {
      verticesThatMatch.push_back(vertexIndex);
    }
  }
  return verticesThatMatch;
}

vector<IndexType> MeshTopology::getChildEntities(unsigned int d, IndexType entityIndex) const
{
  vector<IndexType> childIndices;
  if (d==0) return childIndices;
  if (d==_spaceDim)
  {
    ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
    return getCell(entityIndex)->getChildIndices(thisPtr);
  }
  auto foundChildEntitiesEntry = _childEntities[d].find(entityIndex);
  if (foundChildEntitiesEntry == _childEntities[d].end()) return childIndices;
  vector< pair< RefinementPatternPtr, vector<IndexType> > > childEntries = foundChildEntitiesEntry->second;
  for (pair< RefinementPatternPtr, vector<IndexType> > childEntry : childEntries)
  {
    childIndices.insert(childIndices.end(),childEntry.second.begin(),childEntry.second.end());
  }
  return childIndices;
}

set<IndexType> MeshTopology::getChildEntitiesSet(unsigned int d, IndexType entityIndex) const
{
  set<IndexType> childIndices;
  if (d==0) return childIndices;
  auto foundChildEntitiesEntry = _childEntities[d].find(entityIndex);
  if (foundChildEntitiesEntry == _childEntities[d].end()) return childIndices;
  vector< pair< RefinementPatternPtr, vector<IndexType> > > childEntries = foundChildEntitiesEntry->second;
  for (pair< RefinementPatternPtr, vector<IndexType> > childEntry :  childEntries)
  {
    childIndices.insert(childEntry.second.begin(),childEntry.second.end());
  }
  return childIndices;
}

pair<IndexType, unsigned> MeshTopology::getConstrainingEntity(unsigned d, IndexType entityIndex) const
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
    auto foundEntry = _generalizedParentEntities[generalizedAncestorDim].find(generalizedAncestorEntityIndex);
    if (foundEntry != _generalizedParentEntities[generalizedAncestorDim].end())
    {
      pair< IndexType, unsigned > generalizedParent = foundEntry->second;
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

IndexType MeshTopology::getConstrainingEntityIndexOfLikeDimension(unsigned int d, IndexType entityIndex) const
{
  IndexType constrainingEntityIndex = entityIndex;

  if (d==0)   // one vertex can't constrain another...
  {
    return entityIndex;
  }

  // 3-9-16: I've found an example in which the below fails in a 2-irregular mesh
  // I think the following, simpler thing will work just fine.  (It does pass tests!)
  IndexType ancestorEntityIndex = entityIndex;
  while (entityHasParent(d, ancestorEntityIndex))
  {
    ancestorEntityIndex = getEntityParent(d, ancestorEntityIndex);
    if (getActiveCellCount(d, ancestorEntityIndex) > 0)
    {
      constrainingEntityIndex = ancestorEntityIndex;
    }
  }
  return constrainingEntityIndex;
}

// pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)
vector< pair<IndexType,unsigned> > MeshTopology::getConstrainingSideAncestry(IndexType sideEntityIndex) const
{
  // three possibilities: 1) compatible side, 2) side is parent, 3) side is child
  // 1) and 2) mean unconstrained.  3) means constrained (by parent)
  unsigned sideDim = _spaceDim - 1;
  vector< pair<IndexType, unsigned> > ancestry;
  if (_boundarySides.find(sideEntityIndex) != _boundarySides.end())
  {
    return ancestry; // sides on boundary are unconstrained...
  }

  vector< pair<IndexType,unsigned> > sideCellEntries = _activeCellsForEntities[sideDim][sideEntityIndex];
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
    auto parentIt = _parentEntities[sideDim].find(sideEntityIndex);
    while (parentIt != _parentEntities[sideDim].end())
    {
      vector< pair<IndexType, unsigned> > parents = parentIt->second;
      IndexType parentEntityIndex, refinementIndex;
      for (vector< pair<IndexType, unsigned> >::iterator entryIt = parents.begin(); entryIt != parents.end(); entryIt++)
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
  }
}

//RefinementBranch MeshTopology::getSideConstraintRefinementBranch(IndexType sideEntityIndex)
//{
//  // Returns a RefinementBranch that goes from the constraining side to the side indicated.
//  vector< pair<IndexType,unsigned> > constrainingSideAncestry = getConstrainingSideAncestry(sideEntityIndex);
//  pair< RefinementPattern*, unsigned > branchEntry;
//  unsigned sideDim = _spaceDim - 1;
//  IndexType previousChild = sideEntityIndex;
//  RefinementBranch refBranch;
//  for (vector< pair<IndexType,unsigned> >::iterator ancestorIt = constrainingSideAncestry.begin();
//       ancestorIt != constrainingSideAncestry.end(); ancestorIt++)
//  {
//    IndexType ancestorSideEntityIndex = ancestorIt->first;
//    unsigned refinementIndex = ancestorIt->second;
//    pair<RefinementPatternPtr, vector<IndexType> > children = _childEntities[sideDim][ancestorSideEntityIndex][refinementIndex];
//    branchEntry.first = children.first.get();
//    for (int i=0; i<children.second.size(); i++)
//    {
//      if (children.second[i]==previousChild)
//      {
//        branchEntry.second = i;
//        break;
//      }
//    }
//    refBranch.insert(refBranch.begin(), branchEntry);
//    previousChild = ancestorSideEntityIndex;
//  }
//  return refBranch;
//}

IndexType MeshTopology::getEntityParentForSide(unsigned d, IndexType entityIndex,
                                               IndexType parentSideEntityIndex) const
{
  // returns the entity index for the parent (which might be the entity itself) of entity (d,entityIndex) that is
  // a subcell of side parentSideEntityIndex

  // assuming valid input, three possibilities:
  // 1) parent side has entity as a subcell
  // 2) parent side has exactly one of entity's immediate parents as a subcell

  set<IndexType> entitiesForParentSide = getEntitiesForSide(parentSideEntityIndex, d);
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
  auto parentEntitiesEntry = _parentEntities[d].find(entityIndex);
  vector< pair<IndexType, unsigned> > entityParents = parentEntitiesEntry->second;
  //  cout << "parent entities of entity " << entityIndex << ": ";
  for (pair<IndexType, unsigned> parentEntry : entityParents)
  {
    IndexType parentEntityIndex = parentEntry.first;
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
//  return -1;
}

unsigned MeshTopology::getEntityParentCount(unsigned d, IndexType entityIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(d >= _parentEntities.size(), std::invalid_argument, "dimension is out of bounds");
  auto parentEntitiesEntry = _parentEntities[d].find(entityIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(parentEntitiesEntry == _parentEntities[d].end(), std::invalid_argument, "entityIndex not found in _parentEntities[d]");
  return parentEntitiesEntry->second.size();
}

// ! pairs are (cellIndex, sideOrdinal) where the sideOrdinal is a side that contains the entity
void MeshTopology::initializeTransformationFunction(MeshPtr mesh)
{
  if ((_cellIDsWithCurves.size() > 0) && (mesh != Teuchos::null))
  {
    // to avoid a circular reference, get a weak pointer to mesh
    MeshPtr meshWeakPtr = Teuchos::rcp(mesh.get(), false); // false: weak reference
    // mesh transformation function expects global ID type
    set<GlobalIndexType> cellIDsGlobal(_cellIDsWithCurves.begin(),_cellIDsWithCurves.end());
    _transformationFunction = Teuchos::rcp(new MeshTransformationFunction(meshWeakPtr, cellIDsGlobal));
  }
  else
  {
    _transformationFunction = Teuchos::null;
  }
}

bool MeshTopology::isBoundarySide(IndexType sideEntityIndex) const
{
  return _boundarySides.find(sideEntityIndex) != _boundarySides.end();
}

bool MeshTopology::isDistributed() const
{
  return (Comm() != Teuchos::null) && (Comm()->NumProc() > 1);
}

bool MeshTopology::isValidCellIndex(IndexType cellIndex) const
{
  return _validCells.contains(cellIndex);
//  return _cells.find(cellIndex) != _cells.end();
}

vector< IndexType > MeshTopology::getSidesContainingEntity(unsigned d, IndexType entityIndex) const
{
  unsigned sideDim = getDimension() - 1;
  if (d == sideDim) return {entityIndex};
  
  if (_sidesForEntities[d].size() <= entityIndex)
  {
    return {};
  }
  return _sidesForEntities[d][entityIndex];
}

vector<IndexType> MeshTopology::getSidesContainingEntities(unsigned d, const vector<IndexType> &entities) const
{
  set<IndexType> sidesSet;
  unsigned sideDim = getDimension() - 1;
  for (IndexType entityIndex : entities)
  {
    if (d == sideDim) sidesSet.insert(entityIndex);
    
    if (_sidesForEntities[d].size() > entityIndex)
    {
      sidesSet.insert(_sidesForEntities[d][entityIndex].begin(),_sidesForEntities[d][entityIndex].end());
    }
  }
  return vector<IndexType>(sidesSet.begin(),sidesSet.end());
}

unsigned MeshTopology::getSubEntityPermutation(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal) const
{
  if (subEntityDim==0) return 0;

  if (subEntityDim >= d)
  {
    cout << "subEntityDim cannot be greater than d!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subEntityDim cannot be greater than d!");
  }

  vector<IndexType> entityNodes = getEntityVertexIndices(d,entityIndex);
  CellTopoPtr topo = getEntityTopology(d, entityIndex);
  vector<IndexType> subEntityNodes;
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

IndexType MeshTopology::maxConstraint(unsigned d, IndexType entityIndex1, IndexType entityIndex2) const
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

IndexType MeshTopology::nextCellID() const
{
  return _nextCellIndex;
}

vector< ParametricCurvePtr > MeshTopology::parametricEdgesForCell(IndexType cellIndex, bool neglectCurves) const
{
  vector< ParametricCurvePtr > edges;
  CellPtr cell = getCell(cellIndex);
  
  vector<IndexType> vertices;
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
    auto foundEdgeEntry = _edgeToCurveMap.find(edge);
    if ( foundEdgeEntry != _edgeToCurveMap.end() )
    {
      edgeFxn = foundEdgeEntry->second;
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

void MeshTopology::printApproximateMemoryReport() const
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

void MeshTopology::printConstraintReport(unsigned d) const
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

void MeshTopology::printVertex(IndexType vertexIndex) const
{
  cout << "vertex " << vertexIndex << ": (";
  for (unsigned d=0; d<_spaceDim; d++)
  {
    cout << _vertices[vertexIndex][d];
    if (d != _spaceDim-1) cout << ",";
  }
  cout << ")\n";
}

void MeshTopology::printVertices(set<IndexType> vertexIndices) const
{
  for (IndexType vertexIndex : vertexIndices)
  {
    printVertex(vertexIndex);
  }
}

void MeshTopology::printEntityVertices(unsigned int d, IndexType entityIndex) const
{
  if (d==0)
  {
    printVertex(entityIndex);
    return;
  }
  vector<IndexType> entityVertices = _canonicalEntityOrdering[d][entityIndex];
  for (vector<IndexType>::iterator vertexIt=entityVertices.begin(); vertexIt !=entityVertices.end(); vertexIt++)
  {
    printVertex(*vertexIt);
  }
}

void MeshTopology::printAllEntities() const
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

  for (auto entry : _cells)
  {
    CellPtr cell = entry.second;
    cout << "Cell " << entry.first << ":\n";
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
        labelStream << subcord << ". ";
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
//        labelStream << " " << subcord << " nodes";
        labelStream << " " << cell->entityIndex(d, subcord) << " nodes";
        Camellia::print(labelStream.str(), cell->getEntityVertexIndices(d, subcord));
      }
    }
  }
  cout << "****************************      ";
  cout << "Refinement Hierarchy:";
  cout << "      ****************************\n";
  set<IndexType> levelCells = _rootCells;
  int level = 0;
  ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  while (levelCells.size() > 0)
  {
    ostringstream labelStream;
    labelStream << "level " << level;
    Camellia::print(labelStream.str(),levelCells);
    set<IndexType> nextLevelCells;
    for (IndexType cellIndex : levelCells)
    {
      if (isValidCellIndex(cellIndex))
      {
        CellPtr cell = getCell(cellIndex);
        auto childIndices = cell->getChildIndices(thisPtr);
        nextLevelCells.insert(childIndices.begin(), childIndices.end());
      }
    }
    level++;
    levelCells = nextLevelCells;
  }
}

Camellia::FieldContainer<double> MeshTopology::physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension) const
{
  CellPtr cell = getCell(cellIndex);
  unsigned vertexCount = cell->vertices().size();
  Camellia::FieldContainer<double> nodes(vertexCount, _spaceDim);
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

void MeshTopology::pruningHalo(std::set<GlobalIndexType> &haloCellIndices, const std::set<GlobalIndexType> &ownedCellIndices,
                               unsigned dimForNeighborRelation) const
{
  // TODO: Can we move this to MeshTopologyView?  I think we can, and should...
  
  // the cells passed in are the ones the user wants to include -- e.g. those owned by the MPI rank.
  // we keep more than that; we keep all ancestors and siblings of the cells, as well as all cells that share
  // dimForNeighborRelation-dimensional entities with the cells or their ancestors.

  this->cellHalo(haloCellIndices, ownedCellIndices, dimForNeighborRelation);
  
  // for now, manually prevent pruning cells with curved edges
  // (we don't support pruning these yet)
  // we add these and their ancestors
  for (GlobalIndexType cellID : _cellIDsWithCurves)
  {
    CellPtr cell = getCell(cellID);
    haloCellIndices.insert(cellID);
    while (cell->getParent() != Teuchos::null)
    {
      cell = cell->getParent();
      haloCellIndices.insert(cell->cellIndex());
    }
  }
}

void MeshTopology::pruneToInclude(Epetra_CommPtr Comm, const std::set<GlobalIndexType> &ownedCellIndices,
                                  unsigned dimForNeighborRelation)
{
  // the cells passed in are the ones the user wants to include -- e.g. those owned by the MPI rank.
  // we keep more than that; we keep all ancestors and siblings of the cells, as well as all cells that share
  // dimForNeighborRelation-dimensional entities with the cells or their ancestors.
  
  MeshTopologyViewPtr thisPtr = Teuchos::rcp(this,false);
  
  _pruningOrdinal++;
  
  _Comm = Comm;
  _ownedCellIndices = ownedCellIndices;
  
  set<GlobalIndexType> cellsToInclude;
  pruningHalo(cellsToInclude, ownedCellIndices, dimForNeighborRelation);

  // check whether any cells will be eliminated; if not, can skip the rebuild
  if (cellsToInclude.size() == _cells.size()) return;
  
  // now, collect all the entities that belong to the cells
  vector<set<IndexType>> entitiesToKeep(_spaceDim);
  for (GlobalIndexType cellID : cellsToInclude)
  {
    CellPtr cell = getCell(cellID);
    for (int d=0; d<_spaceDim; d++)
    {
      int subcellCount = cell->topology()->getSubcellCount(d);
      for (int scord=0; scord<subcellCount; scord++)
      {
        IndexType entityIndex = cell->entityIndex(d, scord);
        entitiesToKeep[d].insert(entityIndex);
      }
    }
  }
  
  // lookup table from the new, contiguous numbering, to the previous indices
  vector<vector<IndexType>> oldEntityIndices(_spaceDim);
  for (int d=0; d<_spaceDim; d++)
  {
    oldEntityIndices[d].insert(oldEntityIndices[d].begin(), entitiesToKeep[d].begin(), entitiesToKeep[d].end());
  }
  
  vector<map<IndexType,IndexType>> reverseLookup(_spaceDim); // from old to new
  
  for (int d=0; d<_spaceDim; d++)
  {
    int prunedCount = oldEntityIndices[d].size();
    for (int i=0; i<prunedCount; i++)
    {
      reverseLookup[d][oldEntityIndices[d][i]] = i;
    }
  }
  
  // now, the involved part: update all the lookup tables.
  // (not hard, but we need to make sure we get them all!)
  
  int vertexDim = 0;
  int sideDim = _spaceDim - 1;
  vector< vector<double> > prunedVertices(oldEntityIndices[vertexDim].size(),vector<double>(_spaceDim));
  int prunedVertexCount = oldEntityIndices[vertexDim].size();
  map<IndexType,IndexType>* reverseVertexLookup = &reverseLookup[vertexDim];
  map<IndexType,IndexType>* reverseSideLookup = &reverseLookup[sideDim]; // from old to new
  map< vector<double>, IndexType > prunedVertexMap;
  for (int i=0; i<prunedVertexCount; i++)
  {
    for (int d=0; d<_spaceDim; d++)
    {
      prunedVertices[i][d] = _vertices[oldEntityIndices[vertexDim][i]][d];
    }
    (*reverseVertexLookup)[oldEntityIndices[vertexDim][i]] = i;
    prunedVertexMap[prunedVertices[i]] = i;
  }
  
  vector<pair< pair<IndexType, unsigned>, pair<IndexType, unsigned> > > prunedCellsForSideEntities;
  int prunedSideCount = oldEntityIndices[sideDim].size();
  for (int prunedSideEntityIndex=0; prunedSideEntityIndex<prunedSideCount; prunedSideEntityIndex++)
  {
    IndexType oldSideEntityIndex = oldEntityIndices[sideDim][prunedSideEntityIndex];
    if (oldSideEntityIndex >= _cellsForSideEntities.size())
    {
      // no entries for this side.
      continue;
    }
    (*reverseSideLookup)[oldSideEntityIndex] = prunedSideEntityIndex;
    auto cellsForSideEntry = _cellsForSideEntities[oldSideEntityIndex];
    auto replacementSideEntry = [this, &cellsToInclude, oldSideEntityIndex, sideDim] (pair<IndexType, unsigned> existingEntry) -> pair<IndexType, unsigned>
    {
      IndexType cellID = existingEntry.first;
      if (cellsToInclude.find(cellID) != cellsToInclude.end())
      {
        return existingEntry;
      }
      else
      {
        // look for parents that match the side
        pair<IndexType, unsigned> replacementEntry = {-1,-1};
        if (cellID == -1) return replacementEntry;
        
        CellPtr cell = getCell(cellID);
        while (cell->getParent() != Teuchos::null)
        {
          cell = cell->getParent();
          unsigned sideOrdinal = cell->findSubcellOrdinal(sideDim, oldSideEntityIndex);
          if (sideOrdinal == INVALID_CELLTOPOLOGY_SIDE_ORDINAL) break; // parent does not share side
          // if we get here, parent *does* share side
          // is parent one of the cells we know about??
          if (cellsToInclude.find(cell->cellIndex()) != cellsToInclude.end())
          {
            replacementEntry = {cell->cellIndex(), sideOrdinal};
            break;
          }
        }
        return replacementEntry;
      }
    };

    cellsForSideEntry.first = replacementSideEntry(cellsForSideEntry.first);
    cellsForSideEntry.second = replacementSideEntry(cellsForSideEntry.second);
    
    bool firstEntryCleared = false, secondEntryCleared = false;
    if (cellsForSideEntry.first.first == -1)
    {
      firstEntryCleared = true;
    }
    if (cellsForSideEntry.second.first == -1)
    {
      secondEntryCleared = true;
    }
    
    if (firstEntryCleared && !secondEntryCleared)
    {
      // we've cleared the first entry, but not the second
      // the logic of _cellsForSideEntities requires that the first
      // entry be filled in first, so we flip them
      cellsForSideEntry = {cellsForSideEntry.second, cellsForSideEntry.first};
    }
    
    if (! (firstEntryCleared && secondEntryCleared) )
    {
      // if one of the entries remains, store in pruned container:
      if (prunedCellsForSideEntities.size() <= prunedSideEntityIndex)
      {
        prunedCellsForSideEntities.resize(prunedSideEntityIndex + 100, {{-1,-1},{-1,-1}});
      }
      prunedCellsForSideEntities[prunedSideEntityIndex] = cellsForSideEntry;
    }
  }
  
  for (auto entitySetEntry : _entitySets)
  {
    entitySetEntry.second->updateEntityIndices(reverseLookup);
  }
  
  vector< vector< vector<IndexType> > > prunedEntities(_spaceDim);
  
  vector< map< vector<IndexType>, IndexType > > prunedKnownEntities(_spaceDim);
  vector< vector< vector<IndexType> > > prunedCanonicalEntityOrdering(_spaceDim);
  vector< vector< vector< pair<IndexType, unsigned> > > > prunedActiveCellsForEntities(_spaceDim);
  vector< vector< vector<IndexType> > > prunedSidesForEntities(_spaceDim);
  vector< map< IndexType, vector< pair<IndexType, unsigned> > > > prunedParentEntities(_spaceDim);
  vector< map< IndexType, pair<IndexType, unsigned> > > prunedGeneralizedParentEntities(_spaceDim);
  vector< map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > > > prunedChildEntities(_spaceDim);
  vector< map< Camellia::CellTopologyKey, RangeList<IndexType> > > prunedEntityCellTopologyKeys(_spaceDim);
  map< pair<IndexType, IndexType>, ParametricCurvePtr > prunedEdgeToCurveMap;
  for (int d=0; d<_spaceDim; d++)
  {
    int prunedEntityCount = oldEntityIndices[d].size();
    prunedEntities[d].resize(prunedEntityCount);
    if (d > 0) prunedCanonicalEntityOrdering[d].resize(prunedEntityCount);
    prunedActiveCellsForEntities[d].resize(prunedEntityCount);
    prunedSidesForEntities[d].resize(prunedEntityCount);
    for (int prunedEntityIndex=0; prunedEntityIndex<prunedEntityCount; prunedEntityIndex++)
    {
      IndexType oldEntityIndex = oldEntityIndices[d][prunedEntityIndex];
      CellTopologyKey entityTopoKey = getEntityTopology(d, oldEntityIndex)->getKey();
      prunedEntityCellTopologyKeys[d][entityTopoKey].insert(prunedEntityIndex);
      int nodeCount = _entities[d][oldEntityIndex].size();
      prunedEntities[d][prunedEntityIndex].resize(nodeCount);
      
      if ((d==1) && (_edgeToCurveMap.size() > 0))
      {
        int edgeDim = 1;
        vector<IndexType> oldVertexIndices = getEntityVertexIndices(edgeDim, oldEntityIndex);
        int old_v0 = oldVertexIndices[0];
        int old_v1 = oldVertexIndices[1];
        pair<int,int> old_edge = {old_v0, old_v1};
        
        int new_v0 = reverseVertexLookup->find(old_v0)->second;
        int new_v1 = reverseVertexLookup->find(old_v1)->second;
        pair<int,int> new_edge = {new_v0, new_v1};
        
        if (_edgeToCurveMap.find(old_edge) != _edgeToCurveMap.end())
        {
          prunedEdgeToCurveMap[new_edge] = _edgeToCurveMap[old_edge];
        }
      }
      
      if (d > 0) prunedCanonicalEntityOrdering[d][prunedEntityIndex].resize(nodeCount);
      for (int nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++)
      {
        // first, update entities
        IndexType oldVertexIndex = _entities[d][oldEntityIndex][nodeOrdinal];
        IndexType newVertexIndex = (*reverseVertexLookup)[oldVertexIndex];
        prunedEntities[d][prunedEntityIndex][nodeOrdinal] = newVertexIndex;
        
        if (d == 0) continue; // no canonical ordering stored for vertices...
        // next, canonical entity ordering
        oldVertexIndex = _canonicalEntityOrdering[d][oldEntityIndex][nodeOrdinal];
        newVertexIndex = (*reverseVertexLookup)[oldVertexIndex];
        prunedCanonicalEntityOrdering[d][prunedEntityIndex][nodeOrdinal] = newVertexIndex;
      }
      prunedKnownEntities[d][prunedEntities[d][prunedEntityIndex]] = prunedEntityIndex;
      
      vector<pair<IndexType,unsigned>> oldActiveCellsForEntity = _activeCellsForEntities[d][oldEntityIndex];
      for (auto entry : oldActiveCellsForEntity)
      {
        // cell IDs haven't changed, but which cells are around have
        IndexType cellID = entry.first;
        if (cellsToInclude.find(cellID) != cellsToInclude.end())
        {
          prunedActiveCellsForEntities[d][prunedEntityIndex].push_back(entry);
        }
      }
      
      vector<IndexType> oldSidesForEntity = _sidesForEntities[d][oldEntityIndex];
      for (IndexType oldSideEntityIndex : oldSidesForEntity)
      {
        if (reverseSideLookup->find(oldSideEntityIndex) != reverseSideLookup->end())
        {
          IndexType prunedSideEntityIndex = (*reverseSideLookup)[oldSideEntityIndex];
          prunedSidesForEntities[d][prunedEntityIndex].push_back(prunedSideEntityIndex);
        }
      }
      
      if (_parentEntities[d].find(oldEntityIndex) != _parentEntities[d].end())
      {
        vector<pair<IndexType,unsigned>> oldParents = _parentEntities[d][oldEntityIndex];
        vector<pair<IndexType,unsigned>> newParents;
        for (pair<IndexType,unsigned> oldParentEntry : oldParents)
        {
          auto newParentLookup = reverseLookup[d].find(oldParentEntry.first);
          TEUCHOS_TEST_FOR_EXCEPTION(newParentLookup == reverseLookup[d].end(), std::invalid_argument, "reverseLookup does not contain parent entity");
          pair<IndexType,unsigned> parentEntry = {newParentLookup->second,oldParentEntry.second};
          newParents.push_back(parentEntry);
        }
        prunedParentEntities[d][prunedEntityIndex] = newParents;
      }
      if (_generalizedParentEntities[d].find(oldEntityIndex) != _generalizedParentEntities[d].end())
      {
        pair<IndexType, unsigned> oldEntry = _generalizedParentEntities[d][oldEntityIndex];
        unsigned parentDim = oldEntry.second;
        IndexType oldParentEntityIndex = oldEntry.first;
        if (reverseLookup[parentDim].find(oldParentEntityIndex) == reverseLookup[parentDim].end())
        {
          // this should mean that the geometric constraint involved in this relationship is not
          // one that we're concerned with; e.g., it lies on the far side of one of our ghost cells
          // (We have added all the entities that belong to the cells that could constrain our owned
          // cells.)
        }
        else
        {
          IndexType prunedParentEntityIndex = reverseLookup[parentDim][oldParentEntityIndex];
          prunedGeneralizedParentEntities[d][prunedEntityIndex] = {prunedParentEntityIndex,parentDim};
        }
        
//        if (reverseLookup[parentDim].find(oldParentEntityIndex) == reverseLookup[parentDim].end())
//        {
//          // print out some debugging info
//          cout << "During pruneToInclude, expected entry for " << CamelliaCellTools::entityTypeString(parentDim) << " " << oldParentEntityIndex;
//          cout << " is missing.\n";
//          cout << "(This is the generalized parent of " << CamelliaCellTools::entityTypeString(d) << " " << oldEntityIndex << ")\n";
//          cout << "dimension for neighbor relation: " << dimForNeighborRelation << endl;
//          
//          cout << "Cells that " << CamelliaCellTools::entityTypeString(d) << " " << oldEntityIndex;
//          cout << " participates in: ";
//          
//          vector< pair<IndexType,unsigned> > cellEntries = this->getActiveCellIndices(d, oldEntityIndex);
//          for (auto entry : cellEntries)
//          {
//            cout << entry.first << " ";
//          }
//          cout << endl;
//          
//          print("cellsToInclude", cellsToInclude);
//          print("ownedCellIndices", ownedCellIndices);
//          this->printAllEntities();
//        }
//        TEUCHOS_TEST_FOR_EXCEPTION(reverseLookup[parentDim].find(oldParentEntityIndex) == reverseLookup[parentDim].end(), std::invalid_argument,
//                                   "reverseLookup does not contain generalized parent entity");
      }
      if (_childEntities[d].find(oldEntityIndex) != _childEntities[d].end())
      {
        for (auto refinementEntry : _childEntities[d][oldEntityIndex])
        {
          RefinementPatternPtr refPattern = refinementEntry.first;
          vector<IndexType> oldChildEntityIndices = refinementEntry.second;
          
          vector<IndexType> newChildEntityIndices;
          for (IndexType oldChildEntityIndex : oldChildEntityIndices)
          {
            if (reverseLookup[d].find(oldChildEntityIndex) == reverseLookup[d].end())
            {
              // reverseLookup does not contain child entity
              // This can happen when the child entity isn't seen by the cells of interest
              
              // For now, anyway, we put a -1 here.  This should trigger exceptions if the child entity ever
              // gets used (which it should not be).
              newChildEntityIndices.push_back(-1);
            }
            else
            {
              newChildEntityIndices.push_back(reverseLookup[d][oldChildEntityIndex]);
            }
          }
          prunedChildEntities[d][prunedEntityIndex].push_back({refPattern,newChildEntityIndices});
        }
      }
    }
  }
  _vertices = prunedVertices;
  _vertexMap = prunedVertexMap;
  _cellsForSideEntities = prunedCellsForSideEntities;
  _edgeToCurveMap = prunedEdgeToCurveMap;
  _entities = prunedEntities;
  _knownEntities = prunedKnownEntities;
  _canonicalEntityOrdering = prunedCanonicalEntityOrdering;
  _activeCellsForEntities = prunedActiveCellsForEntities;
  _sidesForEntities = prunedSidesForEntities;
  _parentEntities = prunedParentEntities;
  _generalizedParentEntities = prunedGeneralizedParentEntities;
  _childEntities = prunedChildEntities;
  _entityCellTopologyKeys = prunedEntityCellTopologyKeys;
  
  set<IndexType> prunedBoundarySides;
  for (IndexType oldBoundarySideIndex : _boundarySides)
  {
    if (reverseSideLookup->find(oldBoundarySideIndex) != reverseSideLookup->end())
    {
      prunedBoundarySides.insert((*reverseSideLookup)[oldBoundarySideIndex]);
    }
  }
  _boundarySides = prunedBoundarySides;
  
  map<GlobalIndexType, CellPtr> prunedCells;
  set<IndexType> prunedActiveCells;
  RangeList<IndexType> prunedValidCells;
  for (GlobalIndexType cellID : cellsToInclude)
  {
    CellPtr cell = getCell(cellID);
    vector<IndexType> oldVertexIndices = cell->vertices();
    vector<IndexType> newVertexIndices;
    for (IndexType oldVertexIndex : oldVertexIndices)
    {
      newVertexIndices.push_back((*reverseVertexLookup)[oldVertexIndex]);
    }
    cell->setVertices(newVertexIndices);
    prunedCells[cellID] = cell;
    prunedValidCells.insert(cellID);
    if (!cell->isParent(thisPtr))
    {
      prunedActiveCells.insert(cellID);
    }
  }
  _cells = prunedCells;
  _activeCells = prunedActiveCells;
  _validCells = prunedValidCells;
  
  set<IndexType> prunedRootCells;
  set<IndexType> visitedCells;
  for (auto cellEntry : _cells)
  {
    GlobalIndexType cellID = cellEntry.first;
    while (visitedCells.find(cellID) == visitedCells.end())
    {
      visitedCells.insert(cellID);
      CellPtr cell = getCell(cellID);
      if (cell->getParent() != Teuchos::null)
      {
        cellID = cell->getParent()->cellIndex();
      }
      else
      {
        prunedRootCells.insert(cellID);
      }
    }
  }
  _rootCells = prunedRootCells;
  
  // things we haven't done yet:
  
  /*
   vector< PeriodicBCPtr > _periodicBCs;
   map<IndexType, set< pair<int, int> > > _periodicBCIndicesMatchingNode; // pair: first = index in _periodicBCs; second: 0 or 1, indicating first or second part of the identification matches.  IndexType is the vertex index.
   map< pair<IndexType, pair<int,int> >, IndexType > _equivalentNodeViaPeriodicBC;
   map<IndexType, IndexType> _canonicalVertexPeriodic; // key is a vertex *not* in _knownEntities; the value is the matching vertex in _knownEntities
   */
  
}

int MeshTopology::pruningOrdinal() const
{
  return _pruningOrdinal;
}

void MeshTopology::refineCell(IndexType cellIndex, RefinementPatternPtr refPattern, IndexType firstChildCellIndex)
{
  // TODO: worry about the case (currently unsupported in RefinementPattern) of children that do not share topology with the parent.  E.g. quad broken into triangles.  (3D has better examples.)

//  cout << "MeshTopology::refineCell(" << cellIndex << ", refPattern, " << firstChildCellIndex << ") -- _nextCellIndex = ";
//  cout << _nextCellIndex << "; _activeCellCount = " << _activeCellCount << "\n";
  
  // if we get request to refine a cell that we don't know about, we simply increment the _nextCellIndex and return
  // if we get a request to refine a cell whose first child has index less than _nextCellIndex, then we're being told about one that we already accounted for...
  if (firstChildCellIndex >= _nextCellIndex)
  {
    _nextCellIndex = firstChildCellIndex + refPattern->numChildren();
    _activeCellCount += refPattern->numChildren() - 1;
    _cellCount += refPattern->numChildren();
  }
  if (!isValidCellIndex(cellIndex))
  {
    return;
  }
  
  CellPtr cell = _cells[cellIndex];
  Camellia::FieldContainer<double> cellNodes(cell->vertices().size(), _spaceDim);

  for (int vertexIndex=0; vertexIndex < cellNodes.dimension(0); vertexIndex++)
  {
    for (int d=0; d<_spaceDim; d++)
    {
      cellNodes(vertexIndex,d) = _vertices[cell->vertices()[vertexIndex]][d];
    }
  }

  Camellia::FieldContainer<double> vertices = refPattern->verticesForRefinement(cellNodes);
  if (_transformationFunction.get())
  {
    bool changedVertices = _transformationFunction->mapRefCellPointsUsingExactGeometry(vertices, refPattern->verticesOnReferenceCell(), cellIndex);
    //    cout << "transformed vertices:\n" << vertices;
  }
  map<unsigned, IndexType> vertexOrdinalToVertexIndex = getVertexIndicesMap(vertices); // key: index in vertices; value: index in _vertices
  map<unsigned, GlobalIndexType> localToGlobalVertexIndex(vertexOrdinalToVertexIndex.begin(),vertexOrdinalToVertexIndex.end());

//  {
//    // DEBUGGING
//    cout << "MeshTopology: Refining cell " << cellIndex << ": ";
//    for (int childOrdinal=0; childOrdinal < refPattern->numChildren(); childOrdinal++)
//    {
//      cout << firstChildCellIndex + childOrdinal << " ";
//    }
//    cout << endl;
//  }
  
  // get the children, as vectors of vertex indices:
  vector< vector<GlobalIndexType> > childVerticesGlobalType = refPattern->children(localToGlobalVertexIndex);
  vector< vector<IndexType> > childVertices(childVerticesGlobalType.begin(),childVerticesGlobalType.end());

  int numChildren = childVertices.size();
  // this is where we assume all the children have same topology as parent:
  vector< CellTopoPtr > childTopos(numChildren,cell->topology());

  refineCellEntities(cell, refPattern);
  cell->setRefinementPattern(refPattern);

  bool newRefinement = (cell->children().size() == 0);
  if (newRefinement)
  {
    // cell is active; deactivate it before we add children
    deactivateCell(cell);
  }
  addChildren(firstChildCellIndex, cell, childTopos, childVertices);

  determineGeneralizedParentsForRefinement(cell, refPattern);

  if ((_edgeToCurveMap.size() > 0) && newRefinement)
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

  Camellia::FieldContainer<double> cellNodes(1,cell->vertices().size(), _spaceDim);

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
      Camellia::FieldContainer<double> refinedNodes = subcellRefPattern->refinedNodes(); // NOTE: refinedNodes implicitly assumes that all child topos are the same
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
      
      // to support distributed MeshTopology, we allow -1's to be filled in for some childEntities
      // but on refinement, we do need to replace these
      bool allChildEntitiesKnown = (_childEntities[d].find(parentIndex) != _childEntities[d].end());
      if (allChildEntitiesKnown)
      {
        int refinementOrdinal = 0; // will change if multiple parentage is allowed
        vector<IndexType>* childEntityIndices = &_childEntities[d][parentIndex][refinementOrdinal].second;
        for (IndexType childEntityIndex : *childEntityIndices)
        {
          if (childEntityIndex == -1)
          {
            allChildEntitiesKnown = false;
            break;
          }
        }
      }
      
      // if we ever allow multiple parentage, then we'll need to record things differently in both _childEntities and _parentEntities
      // (and the if statement just below will need to change in a corresponding way, indexed by the particular refPattern in question maybe
      if (!allChildEntitiesKnown)
      {
        vector<IndexType> childEntityIndices(childCount);
        for (unsigned childIndex=0; childIndex<childCount; childIndex++)
        {
          unsigned nodeCount = refinedNodes.dimension(1);
          Camellia::FieldContainer<double> nodesOnSubcell(nodeCount,d);
          for (int nodeIndex=0; nodeIndex<nodeCount; nodeIndex++)
          {
            for (int dimIndex=0; dimIndex<d; dimIndex++)
            {
              nodesOnSubcell(nodeIndex,dimIndex) = refinedNodes(childIndex,nodeIndex,dimIndex);
            }
          }
          //          cout << "nodesOnSubcell:\n" << nodesOnSubcell;
          Camellia::FieldContainer<double> nodesOnRefCell(nodeCount,_spaceDim);
          CamelliaCellTools::mapToReferenceSubcell(nodesOnRefCell, nodesOnSubcell, d, subcord, cellTopo);
          //          cout << "nodesOnRefCell:\n" << nodesOnRefCell;
          Camellia::FieldContainer<double> physicalNodes(1,nodeCount,_spaceDim);
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
          vector<IndexType> childEntityVertices = getVertexIndices(physicalNodes); // key: index in physicalNodes; value: index in _vertices

//          cout << "nodesOnRefCell:\n" << nodesOnRefCell;
//          cout << "physicalNodes:\n" << physicalNodes;
          
          unsigned entityPermutation;
          CellTopoPtr childTopo = cellTopo->getSubcell(d, subcord);
          IndexType childEntityIndex = addEntity(childTopo, childEntityVertices, entityPermutation);
          //          cout << "for d=" << d << ", entity index " << childEntityIndex << " is child of " << parentIndex << endl;
          if (childEntityIndex != parentIndex) // anisotropic and null refinements can leave the entity unrefined
          {
            _parentEntities[d][childEntityIndex] = vector< pair<IndexType,unsigned> >(1, make_pair(parentIndex,0)); // TODO: this is where we want to fill in a proper list of possible parents once we work through recipes
          }
          childEntityIndices[childIndex] = childEntityIndex;
          vector< pair<IndexType, unsigned> > parentActiveCells = _activeCellsForEntities[d][parentIndex];
          // TODO: ?? do something with parentActiveCells?  Seems like we just trailed off here...
        }
        _childEntities[d][parentIndex] = vector< pair<RefinementPatternPtr,vector<IndexType> > >(1, make_pair(subcellRefPattern, childEntityIndices) ); // TODO: this also needs to change when we work through recipes.  Note that the correct parent will vary here...  i.e. in the anisotropic case, the child we're ultimately interested in will have an anisotropic parent, and *its* parent would be the bigger guy referred to here.

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
  Camellia::FieldContainer<double> cellNodes(1,cell->vertices().size(), _spaceDim);

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
      Camellia::FieldContainer<double> refinedNodes = subcellRefPattern->refinedNodes(); // refinedNodes implicitly assumes that all child topos are the same
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

              IndexType equivalentNode = subcellEntityIndex;
              if (subcdim == 0)
              {
                if (_periodicBCIndicesMatchingNode.find(subcellEntityIndex) != _periodicBCIndicesMatchingNode.end())
                {
                  for (const auto &periodicBC : _periodicBCIndicesMatchingNode[subcellEntityIndex])
                  {
                    equivalentNode = _equivalentNodeViaPeriodicBC[make_pair(subcellEntityIndex, periodicBC)];
                  }
                }
              }
              
              // if this is a vertex that also belongs to the parent, then its parentage will already be handled...
              if ((subcdim==0) && ((parentVertexIndexSet.find(subcellEntityIndex) != parentVertexIndexSet.end() ) || (parentVertexIndexSet.find(equivalentNode) != parentVertexIndexSet.end() )))
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

const set<IndexType> &MeshTopology::getRootCellIndicesLocal() const
{
  return _rootCells;
}

set<IndexType> MeshTopology::getRootCellIndicesGlobal() const
{
  if (Comm() == Teuchos::null)
  {
    // replicated: _rootCells contains the global root cells
    return _rootCells;
  }
  
  /*
   If MeshTopology is distributed, then we use the owner of children
   to determine an owner for the parent.  Whichever rank owns the first
   child owns the parent.
   
   To determine whether we own a given root cell, we take its first child,
   then its first child, and so on, until we reach an unrefined cell.  If
   that cell is locally owned, then we own the root cell.
   */
  
  // which of my root cells do I own?
  ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  vector<GlobalIndexTypeToCast> ownedRootCells;
  for (IndexType rootCellIndex : _rootCells)
  {
    IndexType firstChildCellIndex = rootCellIndex;
    bool leafReached = false; // if we don't reach the leaf because some firstChildCellIndex isn't valid, then we can't possibly own the corresponding cell
    while (this->isValidCellIndex(firstChildCellIndex)) {
      CellPtr cell = getCell(firstChildCellIndex);
      if (cell->isParent(thisPtr))
      {
        firstChildCellIndex = cell->getChildIndices(thisPtr)[0];
      }
      else
      {
        leafReached = true;
        break;
      }
    }
    if (leafReached && (_ownedCellIndices.find(firstChildCellIndex) != _ownedCellIndices.end()))
    {
      ownedRootCells.push_back(rootCellIndex);
    }
  }
  
  GlobalIndexTypeToCast myOwnedRootCellCount = ownedRootCells.size();
  GlobalIndexTypeToCast globalRootCellCount = 0;
  Comm()->SumAll(&myOwnedRootCellCount, &globalRootCellCount, 1);
  
  GlobalIndexTypeToCast myEntryOffset = 0;
  Comm()->ScanSum(&myOwnedRootCellCount, &myEntryOffset, 1);
  myEntryOffset -= myOwnedRootCellCount;
  
  vector<GlobalIndexTypeToCast> allRootCellIDs(globalRootCellCount);
  for (GlobalIndexTypeToCast myEntryOrdinal=0; myEntryOrdinal<myOwnedRootCellCount; myEntryOrdinal++)
  {
    allRootCellIDs[myEntryOrdinal + myEntryOffset] = ownedRootCells[myEntryOrdinal];
  }
  vector<GlobalIndexTypeToCast> gatheredRootCellIDs(globalRootCellCount);
  Comm()->SumAll(&allRootCellIDs[0], &gatheredRootCellIDs[0], globalRootCellCount);
  
  set<IndexType> allRootSet(gatheredRootCellIDs.begin(),gatheredRootCellIDs.end());
  TEUCHOS_TEST_FOR_EXCEPTION(allRootSet.size() != globalRootCellCount, std::invalid_argument, "Internal error: some root cell indices appear to have been doubly claimed.");
  
  return allRootSet;
}

void MeshTopology::setEdgeToCurveMap(const map< pair<IndexType, IndexType>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh)
{
  TEUCHOS_TEST_FOR_EXCEPTION(isDistributed(), std::invalid_argument, "setEdgeToCurveMap() is not supported for distributed MeshTopology.");
  _edgeToCurveMap.clear();
  map< pair<IndexType, IndexType>, ParametricCurvePtr >::const_iterator edgeIt;
  _cellIDsWithCurves.clear();

  for (edgeIt = edgeToCurveMap.begin(); edgeIt != edgeToCurveMap.end(); edgeIt++)
  {
    addEdgeCurve(edgeIt->first, edgeIt->second);
  }
  initializeTransformationFunction(mesh);
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

Teuchos::RCP<MeshTransformationFunction> MeshTopology::transformationFunction() const
{
  return _transformationFunction;
}

void MeshTopology::unrefineCell(IndexType cellIndex, RefinementPatternPtr refPattern)
{
  // if we get a request to unrefine a cell that we don't know about, we simply return after updating
  // _activeCellCount and _cellCount.
  // for now, we don't worry about reclaiming cell IDs; we just count upward
  // (For reclaiming cell IDs, we might want just to do a global renumbering thing that
  // eliminates gaps, rather than reclaiming as we go, which would involve a fair amount of bookkeeping.)
  
  // Along similar lines, we do not delete the vertices, edges, etc. that belong only to the deleted cells.
  // Again, the book-keeping would be a bit much, since these are not maps, but contiguous arrays.  The lookups
  // are such that if this cell is refined again, the existing vertices and edges will be reused.  We can always
  // implement a method to clean up the extra entities, which would involve a renumbering.
  
//  cout << "unrefine cell " << cellIndex << endl;
  
  _activeCellCount -= (refPattern->numChildren() - 1);
  _cellCount       -= refPattern->numChildren();
  
  if (!isValidCellIndex(cellIndex))
  {
    return;
  }
  
  MeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  
  // The cleanest way, I think, is actually to record the cell info, delete the cell altogether, and recreate
  // through a call to addCell()
  CellPtr cell = _cells[cellIndex];
  
  auto cellVertices = cell->vertices();
  auto cellTopo = cell->topology();
  
  if (! cell->isParent(thisPtr))
  {
    // nothing to unrefine; just return
    return;
  }
  
  vector<GlobalIndexType> childIDs = cell->getChildIndices(thisPtr);
  
  for (auto childID : childIDs)
  {
    if (isValidCellIndex(childID))
    {
      auto childCell = getCell(childID);
      TEUCHOS_TEST_FOR_EXCEPTION(childCell->isParent(thisPtr), std::invalid_argument, "children must be unrefined before unrefine is called on parent");
    }
  }
  
  int sideDim = this->getDimension() - 1;
  
  childIDs.push_back(cellIndex); // also delete the parent cell
  
  for (GlobalIndexType cellIDToDelete : childIDs)
  {
    if (!isValidCellIndex(cellIDToDelete)) continue; // this child is not locally known
    CellPtr deletedCell = _cells[cellIDToDelete];
    if (_activeCells.find(cellIDToDelete) != _activeCells.end())
    {
      deactivateCell(deletedCell);
    }
    
    // _cellsForSideEntities needs to be updated not to refer to the deleted cell
    int sideCount = deletedCell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      auto sideEntityIndex = deletedCell->entityIndex(sideDim, sideOrdinal);
      
      auto sideCells = _cellsForSideEntities[sideEntityIndex];
      
      auto cell1 = sideCells.first;
      auto cell2 = sideCells.second;
      
      bool noCellsForSide = false;
      if (cell1.first == cellIDToDelete)
      {
        cell1 = make_pair(-1, -1);
        noCellsForSide = (cell2.first == -1);
      }
      else if (cell2.first == cellIDToDelete)
      {
        cell2 = make_pair(-1, -1);
        noCellsForSide = (cell1.first == -1);
      }
      
      if (!noCellsForSide)
      {
        _cellsForSideEntities[sideEntityIndex] = make_pair(cell1, cell2);
      }
      else
      {
        // because we number side entity indices contiguously, we don't have a convenient way
        // to delete this index, even though it's no longer used.  Worth noting that this means we
        // can't refine/unrefine indefinitely; we'll exhaust the index space.  If we don't do a complete
        // overhaul of MeshTopology (which I *am* thinking about, but which is probably not in the cards
        // for a while), we could consider a "rebuild" operation that renumbers things to allow deletion
        // of unused entity indices.
        _cellsForSideEntities[sideEntityIndex] = {{-1,-1}, {-1,-1}};
      }
    }
    
    _cells.erase(cellIDToDelete);
    _validCells.remove(cellIDToDelete);
  }
  
  // add the cell back in -- this will activate it, establish neighbor relationships, etc.
  this->addCell(cellIndex, cellTopo, cellVertices);
  
  // addCell does *not* rediscover parent-child relationships
  // if the original cell had a parent, we need to tell it about that parent again here
  if (cell->getParent() != Teuchos::null)
  {
    auto childIndices = cell->getParent()->getChildIndices(thisPtr);
    cell->getParent()->setChildren(childIndices); // this will call setParent() on each of the children
  }
}

void MeshTopology::verticesForCell(Camellia::FieldContainer<double>& vertices, GlobalIndexType cellID) const
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

MeshTopologyViewPtr MeshTopology::getView(const set<IndexType> &activeCells) const
{
  ConstMeshTopologyPtr thisPtr = Teuchos::rcp(this,false);
  return Teuchos::rcp( new MeshTopologyView(thisPtr, activeCells) );
}
