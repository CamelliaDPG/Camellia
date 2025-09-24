//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
//  MeshTools.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/9/14.
//
//

#include "MeshTools.h"

#include "GlobalDofAssignment.h"

#include "CamelliaCellTools.h"

#include "BasisCache.h"

#include "HDF5Exporter.h"

#include "MeshPartitionPolicy.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;

// time slice utilities assume a tensor product cell in the time dimension
// and further assume that if there are nodeCount vertices in the spatial element, then the nodes in space-time
// will be ordered such that nodes(i) and nodes(i+nodeCount) correspond to each other
vector< vector<double> > timeSliceForCell(Camellia::FieldContainer<double> &physicalCellNodes, double t)
{
  int nodeCount = physicalCellNodes.dimension(1) / 2;
  int spaceDim = physicalCellNodes.dimension(2) - 1; // # of true spatial dimensions
  int d_time = spaceDim; // the index for the time dimension
//  Camellia::FieldContainer<double> slice(1,nodeCount,spaceDim);
  vector< vector<double> > sliceVertices;
  for (int i=0; i<nodeCount; i++)
  {
    double t0 = physicalCellNodes(0,i,d_time);
    double t1 = physicalCellNodes(0,i+nodeCount,d_time);
    double dt = t1 - t0;
    // TODO: worry about curvilinear elements here--for now, we assume straight edges
    double w0 = 1.0 - (t - t0) / dt;
    double w1 = 1.0 - (t1 - t) / dt;
    vector<double> vertex(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      double x0 = physicalCellNodes(0,i,d);
      double x1 = physicalCellNodes(0,i+nodeCount,d);
      vertex[d] = x0 * w0 + x1 * w1;
    }
    sliceVertices.push_back(vertex);
  }
  return sliceVertices;
}

bool cellMatches(Camellia::FieldContainer<double> physicalNodes, double t)
{
  int nodeCount = physicalNodes.dimension(1);
  int spaceDim = physicalNodes.dimension(2) - 1; // # of true spatial dimensions
  int d_time = spaceDim; // the index for the time dimension

  double hasVerticesBelow = false, hasVerticesAbove = false;
  for (int node = 0; node < nodeCount; node++)
  {
    double t_node = physicalNodes(0,node,d_time);
    if (t_node <= t) hasVerticesBelow = true;
    if (t_node >= t) hasVerticesAbove = true;
  }

  return hasVerticesAbove && hasVerticesBelow;
}

CellTopoPtr getBottomTopology(MeshTopology* meshTopo, IndexType cellID)
{
  int spaceDim = meshTopo->getDimension() - 1;
  // determine cell topology:
  vector<IndexType> cellVertexIndices = meshTopo->getCell(cellID)->vertices();
  set<IndexType> bottomVertexIndices;
  int bottomNodeCount = cellVertexIndices.size() / 2;
  for (int i=0; i<bottomNodeCount; i++)
  {
    bottomVertexIndices.insert(cellVertexIndices[i]);
  }
  IndexType bottomEntityIndex = meshTopo->getEntityIndex(spaceDim, bottomVertexIndices);
  Camellia::CellTopologyKey bottomCellTopoKey = meshTopo->getEntityTopology(spaceDim, bottomEntityIndex)->getKey();
  CellTopoPtr cellTopo = CamelliaCellTools::cellTopoForKey(bottomCellTopoKey);
  return cellTopo;
}

void MeshTools::hRefineToMatchAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                         MeshPtr meshFrom, int fromBoundaryTag,
                                         MeshPtr meshTo)
{
  // raw ptrs to MeshTopo's:
  auto fromTopoRaw = meshFrom->getTopology()->baseMeshTopology();
  auto toTopoRaw   = meshTo->getTopology()->baseMeshTopology();
  
  ConstMeshTopologyPtr fromTopo = Teuchos::rcp(fromTopoRaw, false); // false: weak ptr
  ConstMeshTopologyPtr   toTopo = Teuchos::rcp(  toTopoRaw, false); // false: weak ptr
  
  bool didRefine = false;
  do // until we don't refine
  {
    didRefine = false;
    map<IndexType,IndexType> cellIDMapTemp;
    map<int,int> sideOrdinalMapTemp;
    mapCellIDsAtBoundary(cellIDMapTemp, sideOrdinalMapTemp, fromTopo, fromBoundaryTag, toTopo);
    
    // expected usage is for the "from" mesh to be a new time slab, everywhere coarser than the "to" mesh.
    // in this use case, we expect the "to" mesh to require no refinements.  But we do support more general cases.
    
    std::set<GlobalIndexType> fromCellsToRefine;
    std::set<GlobalIndexType>   toCellsToRefine;
    
    for (const auto & entry : cellIDMapTemp)
    {
      const auto & cellIDFrom = entry.first;
      const auto & cellIDTo   = entry.second;
      
      auto cellFrom = fromTopo->getCell(cellIDFrom);
      auto cellTo   =   toTopo->getCell(cellIDTo);
      
      if (cellFrom->isParent(fromTopo))
      {
        if (cellTo->isParent(toTopo))
        {
          // if we supported anisotropic refinements, this could be allowed.  But for now, this is an error.
          TEUCHOS_TEST_FOR_EXCEPTION(cellTo->isParent(toTopo), std::invalid_argument, "cellIDMap with one parent cell mapped to another parent cell");
        }
        toCellsToRefine.insert(cellIDTo);
      }
      else if (cellTo->isParent(toTopo))
      {
        if (cellFrom->isParent(fromTopo))
        {
          // if we supported anisotropic refinements, this could be allowed.  But for now, this is an error.
          TEUCHOS_TEST_FOR_EXCEPTION(cellFrom->isParent(fromTopo), std::invalid_argument, "cellIDMap with one parent cell mapped to another parent cell");
        }
        fromCellsToRefine.insert(cellIDFrom);
      }
    }
    
    if (fromCellsToRefine.size() > 0)
    {
      meshFrom->hRefine(fromCellsToRefine);
      didRefine = true;
    }
    if (toCellsToRefine.size() > 0)
    {
      meshTo->hRefine(toCellsToRefine);
      didRefine = true;
    }
    if (!didRefine)
    {
      // we'll exit, so the "temp" maps are the final ones
      cellIDMap      = cellIDMapTemp;
      sideOrdinalMap = sideOrdinalMapTemp;
    }
  } while (didRefine);
}

map<IndexType,IndexType> MeshTools::mapActiveCellIndices(MeshTopologyViewPtr meshTopoFrom, MeshTopologyViewPtr meshTopoTo)
{
  const set<IndexType>* activeCellIndices = &meshTopoFrom->getLocallyKnownActiveCellIndices();
  vector<vector<double>> centroid(activeCellIndices->size());
  Camellia::FieldContainer<double> centroidFC(activeCellIndices->size(),1,meshTopoFrom->getDimension());
  int cellOrdinal = 0;
  for (IndexType cellIndex : *activeCellIndices)
  {
    centroid[cellOrdinal++] = meshTopoFrom->getCellCentroid(cellIndex);
  }
  CamelliaCellTools::pointsFCFromVector(centroidFC, centroid);
  vector<IndexType> toIndices = meshTopoTo->cellIDsForPoints(centroidFC);
  cellOrdinal = 0;
  map<IndexType,IndexType> resultMap;
  for (IndexType cellIndex : *activeCellIndices)
  {
    resultMap[cellIndex] = toIndices[cellOrdinal++];
  }
  return resultMap;
}

void MeshTools::mapCellIDsAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                     ConstMeshTopologyPtr meshTopoFrom, int fromBoundaryTag,
                                     ConstMeshTopologyPtr meshTopoTo)
{
  cellIDMap.clear();
  sideOrdinalMap.clear();
  
  auto fromEntitySets = meshTopoFrom->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, fromBoundaryTag);
  
  int sideDim = meshTopoTo->getDimension() - 1;
  
  using std::set;
  using std::map;
  using std::vector;
  set<IndexType> fromSides;
  for (auto fromEntitySet : fromEntitySets)
  {
    set<IndexType> fromSidesForEntitySet = fromEntitySet->entitiesOfDimension(sideDim);
    for (const auto &fromSide : fromSidesForEntitySet)
    {
      if (meshTopoFrom->entityIsActive(sideDim, fromSide))
      {
        fromSides.insert(fromSide);
      }
    }
  }
  
  map<IndexType,IndexType> sideMap; // fromSide --> toSide
  
  for (auto fromSide : fromSides)
  {
    bool foundToMatch = false;
    while (!foundToMatch)
    {
      auto fromVertexIndices = meshTopoFrom->getEntityVertexIndices(sideDim, fromSide);
      
      vector<IndexType> toVertexIndices;
      for (auto fromVertexIndex : fromVertexIndices)
      {
        auto & fromVertex = meshTopoFrom->getVertex(fromVertexIndex);
        IndexType toVertexIndex;
        meshTopoTo->getVertexIndex(fromVertex, toVertexIndex);
        toVertexIndices.push_back(toVertexIndex);
      }
      
      auto toSides = meshTopoTo->getSidesContainingAllEntities(0, toVertexIndices);
      IndexType toSide = -1;
      // with perodic BCs and two elements in the periodic direction, we can have multiple matches due to vertex identification.
      // the following selects the true match: the one whose vertex indices match.  It also serves as verification that the matched
      // sides line up in terms of vertex ordering.
      for (const auto &toSideMatch : toSides)
      {
        auto vertexIndices = meshTopoTo->getEntityVertexIndices(sideDim, toSideMatch);
        int numVertices = vertexIndices.size();
        bool matchFailed = false;
        for (int viOrdinal=0; viOrdinal<numVertices; viOrdinal++)
        {
          if (vertexIndices[viOrdinal] != toVertexIndices[viOrdinal])
          {
            matchFailed = true;
            break;
          }
        }
        if (!matchFailed)
        {
          toSide = toSideMatch;
        }
      }
      
      if (toSides.size() > 0)
      {
        if (toSide == -1)
        {
          Camellia::print("toSides", toSides);
          meshTopoTo->printAllEntities();
        }
        TEUCHOS_TEST_FOR_EXCEPTION(toSide == -1, std::invalid_argument, "No side found that matches the vertices in the same order");
        
        sideMap[fromSide] = toSide;
        foundToMatch = true;
      }
      else
      {
        // no to side found.  Ascend the geometric hierarchy
        TEUCHOS_TEST_FOR_EXCEPTION(meshTopoFrom->entityHasParent(sideDim, fromSide), std::invalid_argument, "root-level sides must match; no toSide found for root-level fromSide");
        fromSide = meshTopoFrom->getEntityParent(sideDim, fromSide);
      }
    }
  }
  
  for (const auto &sideMapEntry : sideMap)
  {
    const auto & fromSide = sideMapEntry.first;
    const auto &   toSide = sideMapEntry.second;
    
    auto fromCellPair = meshTopoFrom->getFirstCellForSide(fromSide); // (cellID, side ordinal)
    const IndexType & fromCellID = fromCellPair.first;
    int fromSideOrdinal = fromCellPair.second;
    
    auto toCellPair = meshTopoTo->getFirstCellForSide(toSide); // (cellID, side ordinal)
    const IndexType & toCellID = toCellPair.first;
    int toSideOrdinal = toCellPair.second;
    
    cellIDMap[fromCellID] = toCellID;
    
    if (sideOrdinalMap.find(fromSideOrdinal) == sideOrdinalMap.end())
    {
      sideOrdinalMap[fromSideOrdinal] = toSideOrdinal;
    }
    else
    {
      // we do require that the side ordinal map is the same from one element to the next; this will generally be true for orthogonally-extruded meshes.
      // (Maybe there are exceptions when you have multiple cell topologies in the base mesh, but this is not our immediate use case.)
      TEUCHOS_TEST_FOR_EXCEPTION(sideOrdinalMap[fromSideOrdinal] != toSideOrdinal, std::invalid_argument, "Inconsistent side ordinal mapping; not supported");
    }
  }
}

MeshPtr MeshTools::timeSliceMesh(MeshPtr spaceTimeMesh, double t,
                                 map<GlobalIndexType, GlobalIndexType> &sliceCellIDToSpaceTimeCellID, int H1OrderForSlice)
{
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(spaceTimeMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "timeSliceMesh() called with spaceTimeMesh that appears to be pure MeshTopologyView.  This is not supported.");
  
  MeshTopologyViewPtr meshTopoLocal;
  
  if (! meshTopo->isDistributed())
  {
    meshTopoLocal = Teuchos::rcp(meshTopo,false);
  }
  else
  {
    // NOTE: for a large mesh, this could be an expensive operation!!
    // If timeSliceMesh() sees significant use, we should rewrite to have it operate on the original distributed mesh.
    meshTopoLocal = meshTopo->getGatheredCopy();
    meshTopo = dynamic_cast<MeshTopology*>(meshTopoLocal.get());
  }
  
  set<IndexType> activeCellIDsForTime;
  
  // for now, throw an exception if trying to do this with a distributed MeshTopology
  // We need to think through the construction of sliceTopo more carefully in that case
  bool isDistributedMeshTopo = meshTopoLocal->isDistributed();
  TEUCHOS_TEST_FOR_EXCEPTION(isDistributedMeshTopo, std::invalid_argument, "timeSliceMesh() does not yet supported distributed MeshTopology");
  
  set<IndexType> allActiveCellIDs = meshTopoLocal->getLocallyKnownActiveCellIndices();

  int spaceDim = meshTopoLocal->getDimension() - 1;  // # of true spatial dimensions

  MeshTopologyPtr sliceTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  set<IndexType> rootCellIDs = meshTopoLocal->getRootCellIndicesLocal();
  set<IndexType> cellIDsToCheck = rootCellIDs;
  for (IndexType rootCellID : rootCellIDs)
  {
    Camellia::FieldContainer<double> physicalNodes = meshTopoLocal->physicalCellNodesForCell(rootCellID, true);
    if (cellMatches(physicalNodes, t))   // cell and some subset of its descendents should be included in slice mesh
    {
      vector< vector< double > > sliceNodes = timeSliceForCell(physicalNodes, t);
      CellTopoPtr cellTopo = getBottomTopology(meshTopo, rootCellID);
      CellPtr sliceCell = sliceTopo->addCell(cellTopo, sliceNodes); // for consistency, this is only valid if run on every MPI rank.
      sliceCellIDToSpaceTimeCellID[sliceCell->cellIndex()] = rootCellID;
    }
  }
  
  MeshPartitionPolicyPtr partitionPolicy = MeshPartitionPolicy::inducedPartitionPolicy(Teuchos::null, spaceTimeMesh, sliceCellIDToSpaceTimeCellID);

  MeshPtr sliceMesh = Teuchos::rcp( new Mesh(sliceTopo, spaceTimeMesh->bilinearForm(), H1OrderForSlice, spaceDim,
                                             map<int,int>(), map<int,int>(), partitionPolicy) );

  // process refinements.  For now, we assume isotropic refinements, which means that each refinement in spacetime induces a refinement in the spatial slice
  set<IndexType> sliceCellIDsToCheckForRefinement = sliceTopo->getLocallyKnownActiveCellIndices();
  while (sliceCellIDsToCheckForRefinement.size() > 0)
  {
    set<IndexType>::iterator cellIt = sliceCellIDsToCheckForRefinement.begin();
    IndexType sliceCellID = *cellIt;
    sliceCellIDsToCheckForRefinement.erase(cellIt);

    CellPtr sliceCell = sliceTopo->getCell(sliceCellID);
    CellPtr spaceTimeCell = meshTopo->getCell(sliceCellIDToSpaceTimeCellID[sliceCellID]);
    if (spaceTimeCell->isParent(meshTopoLocal))
    {
      set<GlobalIndexType> cellsToRefine;
      cellsToRefine.insert(sliceCellID);
      sliceMesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPattern(sliceCell->topology()), false); // false: don't repartition and rebuild
      vector<IndexType> spaceTimeChildren = spaceTimeCell->getChildIndices(meshTopoLocal);
      for (int childOrdinal=0; childOrdinal<spaceTimeChildren.size(); childOrdinal++)
      {
        IndexType childID = spaceTimeChildren[childOrdinal];
        Camellia::FieldContainer<double> childNodes = meshTopo->physicalCellNodesForCell(childID, true);
        if (cellMatches(childNodes, t))
        {
          vector< vector<double> > childSlice = timeSliceForCell(childNodes, t);
          CellPtr childSliceCell = sliceTopo->findCellWithVertices(childSlice);
          sliceCellIDToSpaceTimeCellID[childSliceCell->cellIndex()] = childID;
          sliceCellIDsToCheckForRefinement.insert(childSliceCell->cellIndex());
        }
      }
    }
  }
  sliceMesh->repartitionAndRebuild(); // mostly to rebuild dof lookups

  return sliceMesh;
}

class SliceFunction : public TFunction<double>
{
  MeshPtr _spaceTimeMesh;
  map<GlobalIndexType, GlobalIndexType> _cellIDMap;
  TFunctionPtr<double> _spaceTimeFunction;
  double _t;
public:
  SliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap,
                TFunctionPtr<double> spaceTimeFunction, double t) : TFunction<double>("SliceFunction", spaceTimeFunction->rank())
  {
    _spaceTimeMesh = spaceTimeMesh;
    _cellIDMap = cellIDMap;
    _spaceTimeFunction = spaceTimeFunction;
    _t = t;
  }
  using TFunction<double>::values;
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr sliceBasisCache)
  {
    vector<GlobalIndexType> sliceCellIDs = sliceBasisCache->cellIDs();

    Teuchos::Array<int> dim;
    values.dimensions(dim);
    dim[0] = 1; // one cell
    std::vector<int> offset(dim.size());

    for (int cellOrdinal = 0; cellOrdinal < sliceCellIDs.size(); cellOrdinal++)
    {
      offset[0] = cellOrdinal;
      int enumeration = SerialDenseWrapper::getEnumeration(offset,values);
      Camellia::FieldContainer<double>valuesForCell(dim,&values[enumeration]);
      GlobalIndexType sliceCellID = sliceCellIDs[cellOrdinal];
      int numPoints = sliceBasisCache->getPhysicalCubaturePointsFC().dimension(1);
      int spaceDim = sliceBasisCache->getPhysicalCubaturePointsFC().dimension(2);
      Camellia::FieldContainer<double> spaceTimePhysicalPoints(1,numPoints,spaceDim+1);
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          spaceTimePhysicalPoints(0,ptOrdinal,d) = sliceBasisCache->getPhysicalCubaturePointsFC()(cellOrdinal,ptOrdinal,d);
        }
        spaceTimePhysicalPoints(0,ptOrdinal,spaceDim) = _t;
      }

      GlobalIndexType cellID = _cellIDMap[sliceCellID];
      BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForCell(_spaceTimeMesh, cellID);

      Camellia::FieldContainer<double> spaceTimeRefPoints(1,numPoints,spaceDim+1);
      CamelliaCellTools::mapToReferenceFrame(spaceTimeRefPoints, spaceTimePhysicalPoints, _spaceTimeMesh->getTopology(), cellID, spaceTimeBasisCache->cubatureDegree());
      spaceTimeRefPoints.resize(numPoints,spaceDim+1);
      spaceTimeBasisCache->setRefCellPointsFC(spaceTimeRefPoints);
      _spaceTimeFunction->values(valuesForCell, spaceTimeBasisCache);
    }
  }
};

void MeshTools::timeSliceExport(std::string dirPath, MeshPtr mesh, TFunctionPtr<double> spaceTimeFunction, std::vector<double> tValues, std::string functionName)
{
  // user is responsible for ensuring that tValues all generate the same slice.  It's a bit of a burden, but there it is...
  map<GlobalIndexType, GlobalIndexType> cellIDMap;
  int spatialH1Order = mesh->globalDofAssignment()->getInitialH1Order()[0];
  MeshPtr meshSlice =  timeSliceMesh(mesh, tValues[0], cellIDMap, spatialH1Order);

  HDF5Exporter exporter(meshSlice,dirPath);

  for (int i=0; i<tValues.size(); i++)
  {
    TFunctionPtr<double> sliceFunction = Teuchos::rcp( new SliceFunction(mesh,cellIDMap,spaceTimeFunction,tValues[i]) );
    exporter.exportFunction(sliceFunction, functionName, tValues[i]);
  }
}

TFunctionPtr<double> MeshTools::timeSliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap, TFunctionPtr<double> spaceTimeFunction, double t)
{
  TFunctionPtr<double> timeSliceFunction = Teuchos::rcp(new SliceFunction(spaceTimeMesh, cellIDMap, spaceTimeFunction, t) );
  if (spaceTimeFunction->boundaryValueOnly())
    timeSliceFunction = TFunction<double>::restrictToCellBoundary(timeSliceFunction);
  return timeSliceFunction;
}

std::vector<double> MeshTools::verticesFor1DIntervalMesh(const double &x0, const double &xWidth, const int &numElements)
{
  std::vector<double> vertices(numElements+1);
  for (int vOrdinal=0; vOrdinal<=numElements; vOrdinal++)
  {
    vertices[vOrdinal] = x0 + double(xWidth * vOrdinal) / numElements;
  }
  return vertices;
}
