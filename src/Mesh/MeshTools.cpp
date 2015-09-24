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

// #include "EpetraExt_ConfigDefs.h"
// #ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
// #endif

#include "MeshPartitionPolicy.h"

using namespace Intrepid;
using namespace Camellia;

// time slice utilities assume a tensor product cell in the time dimension
// and further assume that if there are nodeCount vertices in the spatial element, then the nodes in space-time
// will be ordered such that nodes(i) and nodes(i+nodeCount) correspond to each other
vector< vector<double> > timeSliceForCell(FieldContainer<double> &physicalCellNodes, double t)
{
  int nodeCount = physicalCellNodes.dimension(1) / 2;
  int spaceDim = physicalCellNodes.dimension(2) - 1; // # of true spatial dimensions
  int d_time = spaceDim; // the index for the time dimension
//  FieldContainer<double> slice(1,nodeCount,spaceDim);
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

bool cellMatches(FieldContainer<double> physicalNodes, double t)
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

MeshPtr MeshTools::timeSliceMesh(MeshPtr spaceTimeMesh, double t,
                                 map<GlobalIndexType, GlobalIndexType> &sliceCellIDToSpaceTimeCellID, int H1OrderForSlice)
{
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(spaceTimeMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "timeSliceMesh() called with spaceTimeMesh that appears to be pure MeshTopologyView.  This is not supported.");
  set<IndexType> cellIDsToCheck = meshTopo->getRootCellIndices();
  set<IndexType> activeCellIDsForTime;

  set<IndexType> allActiveCellIDs = meshTopo->getActiveCellIndices();

  int spaceDim = meshTopo->getDimension() - 1;  // # of true spatial dimensions

  MeshTopologyPtr sliceTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  set<IndexType> rootCellIDs = meshTopo->getRootCellIndices();
  for (set<IndexType>::iterator rootCellIt = rootCellIDs.begin(); rootCellIt != rootCellIDs.end(); rootCellIt++)
  {
    IndexType rootCellID = *rootCellIt;
    FieldContainer<double> physicalNodes = spaceTimeMesh->physicalCellNodesForCell(rootCellID);
    if (cellMatches(physicalNodes, t))   // cell and some subset of its descendents should be included in slice mesh
    {
      vector< vector< double > > sliceNodes = timeSliceForCell(physicalNodes, t);
      CellTopoPtr cellTopo = getBottomTopology(meshTopo, rootCellID);
      CellPtr sliceCell = sliceTopo->addCell(cellTopo, sliceNodes);
      sliceCellIDToSpaceTimeCellID[sliceCell->cellIndex()] = rootCellID;
    }
  }

  MeshPtr sliceMesh = Teuchos::rcp( new Mesh(sliceTopo, spaceTimeMesh->bilinearForm(), H1OrderForSlice, spaceDim) );

  // process refinements.  For now, we assume isotropic refinements, which means that each refinement in spacetime induces a refinement in the spatial slice
  set<IndexType> sliceCellIDsToCheckForRefinement = sliceTopo->getActiveCellIndices();
  while (sliceCellIDsToCheckForRefinement.size() > 0)
  {
    set<IndexType>::iterator cellIt = sliceCellIDsToCheckForRefinement.begin();
    IndexType sliceCellID = *cellIt;
    sliceCellIDsToCheckForRefinement.erase(cellIt);

    CellPtr sliceCell = sliceTopo->getCell(sliceCellID);
    CellPtr spaceTimeCell = meshTopo->getCell(sliceCellIDToSpaceTimeCellID[sliceCellID]);
    if (spaceTimeCell->isParent(spaceTimeMesh->getTopology()))
    {
      set<GlobalIndexType> cellsToRefine;
      cellsToRefine.insert(sliceCellID);
      sliceMesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPattern(sliceCell->topology()));
      vector<IndexType> spaceTimeChildren = spaceTimeCell->getChildIndices(spaceTimeMesh->getTopology());
      for (int childOrdinal=0; childOrdinal<spaceTimeChildren.size(); childOrdinal++)
      {
        IndexType childID = spaceTimeChildren[childOrdinal];
        FieldContainer<double> childNodes = meshTopo->physicalCellNodesForCell(childID);
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

  MeshPartitionPolicyPtr partitionPolicy = MeshPartitionPolicy::inducedPartitionPolicy(sliceMesh, spaceTimeMesh, sliceCellIDToSpaceTimeCellID);

  sliceMesh->setPartitionPolicy(partitionPolicy);

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
                TFunctionPtr<double> spaceTimeFunction, double t) : TFunction<double>(spaceTimeFunction->rank())
  {
    _spaceTimeMesh = spaceTimeMesh;
    _cellIDMap = cellIDMap;
    _spaceTimeFunction = spaceTimeFunction;
    _t = t;
  }
  void values(FieldContainer<double> &values, BasisCachePtr sliceBasisCache)
  {
    vector<GlobalIndexType> sliceCellIDs = sliceBasisCache->cellIDs();

    Teuchos::Array<int> dim;
    values.dimensions(dim);
    dim[0] = 1; // one cell
    Teuchos::Array<int> offset(dim.size());

    for (int cellOrdinal = 0; cellOrdinal < sliceCellIDs.size(); cellOrdinal++)
    {
      offset[0] = cellOrdinal;
      int enumeration = values.getEnumeration(offset);
      FieldContainer<double>valuesForCell(dim,&values[enumeration]);
      GlobalIndexType sliceCellID = sliceCellIDs[cellOrdinal];
      int numPoints = sliceBasisCache->getPhysicalCubaturePoints().dimension(1);
      int spaceDim = sliceBasisCache->getPhysicalCubaturePoints().dimension(2);
      FieldContainer<double> spaceTimePhysicalPoints(1,numPoints,spaceDim+1);
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          spaceTimePhysicalPoints(0,ptOrdinal,d) = sliceBasisCache->getPhysicalCubaturePoints()(cellOrdinal,ptOrdinal,d);
        }
        spaceTimePhysicalPoints(0,ptOrdinal,spaceDim) = _t;
      }

      GlobalIndexType cellID = _cellIDMap[sliceCellID];
      BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForCell(_spaceTimeMesh, cellID);

      FieldContainer<double> spaceTimeRefPoints(1,numPoints,spaceDim+1);
      CamelliaCellTools::mapToReferenceFrame(spaceTimeRefPoints, spaceTimePhysicalPoints, _spaceTimeMesh->getTopology(), cellID, spaceTimeBasisCache->cubatureDegree());
      spaceTimeRefPoints.resize(numPoints,spaceDim+1);
      spaceTimeBasisCache->setRefCellPoints(spaceTimeRefPoints);
      _spaceTimeFunction->values(valuesForCell, spaceTimeBasisCache);
    }
  }
};

void MeshTools::timeSliceExport(std::string dirPath, MeshPtr mesh, TFunctionPtr<double> spaceTimeFunction, std::vector<double> tValues, std::string functionName)
{
  // user is responsible for ensuring that tValues all generate the same slice.  It's a bit of a burden, but there it is...
// #ifdef HAVE_EPETRAEXT_HDF5
  map<GlobalIndexType, GlobalIndexType> cellIDMap;
  int spatialH1Order = mesh->globalDofAssignment()->getInitialH1Order()[0];
  MeshPtr meshSlice =  timeSliceMesh(mesh, tValues[0], cellIDMap, spatialH1Order);

//  cout << "At time " << t << ", slice has " << meshSlice->numActiveElements() << " active elements.\n";
  HDF5Exporter exporter(meshSlice,dirPath);

  for (int i=0; i<tValues.size(); i++)
  {
    TFunctionPtr<double> sliceFunction = Teuchos::rcp( new SliceFunction(mesh,cellIDMap,spaceTimeFunction,tValues[i]) );
    exporter.exportFunction(sliceFunction, functionName, tValues[i]);
  }
// #else
//   cout << "timeSliceExport requires Trilinos/Epetra to be built with HDF5 support.\n";
// #endif
}

TFunctionPtr<double> MeshTools::timeSliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap, TFunctionPtr<double> spaceTimeFunction, double t)
{
  TFunctionPtr<double> timeSliceFunction = Teuchos::rcp(new SliceFunction(spaceTimeMesh, cellIDMap, spaceTimeFunction, t) );
  if (spaceTimeFunction->boundaryValueOnly())
    timeSliceFunction = TFunction<double>::restrictToCellBoundary(timeSliceFunction);
  return timeSliceFunction;
}

