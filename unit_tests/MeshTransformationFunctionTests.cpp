//
//  MeshTransformationFunctionTests
//  Camellia
//
//  Created by Nate Roberts on 7/2/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CamelliaTestingHelpers.h"
#include "MeshFactory.h"
#include "MeshTransformationFunction.h"
#include "SpaceTimeHeatFormulation.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
  void testTimeCoordinatesOfCell(double t0, double t1, MeshPtr spaceTimeMesh, IndexType cellIndex, Teuchos::FancyOStream &out, bool &success)
  {
    FieldContainer<double> physicalCellNodes = spaceTimeMesh->physicalCellNodesForCell(cellIndex);
    CellPtr spaceTimeCell = spaceTimeMesh->getTopology()->getCell(cellIndex);
    CellTopoPtr spaceTimeTopo = spaceTimeCell->topology();
    int spaceDim = spaceTimeTopo->getDimension() - 1;
    
    unsigned sideTime0 = spaceTimeCell->topology()->getTemporalSideOrdinal(0);
    unsigned sideTime1 = spaceTimeTopo->getTemporalSideOrdinal(1);
    CellTopoPtr spaceTopo = spaceTimeTopo->getSubcell(spaceDim, sideTime0);
    unsigned spatialNodeCount = spaceTopo->getNodeCount();
    double tol = 1e-15;
    for (int node=0; node<spatialNodeCount; node++)
    {
      int spaceTimeNodeTime0 = spaceTimeTopo->getNodeMap(spaceDim, sideTime0, node);
      int spaceTimeNodeTime1 = spaceTimeTopo->getNodeMap(spaceDim, sideTime1, node);
      TEST_FLOATING_EQUALITY(t0, physicalCellNodes(0,spaceTimeNodeTime0,spaceDim), tol);
      TEST_FLOATING_EQUALITY(t1, physicalCellNodes(0,spaceTimeNodeTime1,spaceDim), tol);
    }
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(spaceTimeMesh, cellIndex);
    
    FieldContainer<double> physicalCellNodesBasisCache = basisCache->getPhysicalCellNodes();
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(physicalCellNodes, physicalCellNodesBasisCache, 1e-15);
    
    FieldContainer<double> refCellNodes(spaceTimeTopo->getNodeCount(),spaceTimeTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(refCellNodes, spaceTimeTopo);
    basisCache->setRefCellPoints(refCellNodes);
    FieldContainer<double> mappedPhysicalCellNodes = basisCache->getPhysicalCubaturePoints();
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(physicalCellNodes, mappedPhysicalCellNodes, 1e-15);
  }
  
  TEUCHOS_UNIT_TEST( MeshTransformationFunction, SpaceTimeCellGetsCorrectTimeCoordinates)
  {
    MeshTopologyPtr unitQuadMeshTopo = MeshFactory::rectilinearMeshTopology({1.0,1.0}, {1,1});
    int spaceDim = unitQuadMeshTopo->getDimension();
    double t0 = 1.0, t1 = 2.0;
    int H1OrderSpace = 2, H1OrderTime = 1, delta_k = 1;
    double epsilon = 1e-2;
    bool useConformingTraces = false;
    SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
    MeshPtr spaceTimeMesh = MeshFactory::spaceTimeMesh(unitQuadMeshTopo, t0, t1, form.bf(), H1OrderSpace, H1OrderTime, delta_k);
    
    map<pair<GlobalIndexType,GlobalIndexType>,ParametricCurvePtr> edgeToCurveMap;

    IndexType cellIndex = 0;
    CellPtr spatialCell = unitQuadMeshTopo->getCell(cellIndex);
    vector<ParametricCurvePtr> edgeFxns = unitQuadMeshTopo->parametricEdgesForCell(cellIndex, true);
    
    CellPtr spaceTimeCell = spaceTimeMesh->getTopology()->getCell(cellIndex);
    unsigned sideTime0 = spaceTimeCell->topology()->getTemporalSideOrdinal(0);
    unsigned sideTime1 = spaceTimeCell->topology()->getTemporalSideOrdinal(1);

    int edgeDim = 1;
    int edgeCount = spaceTimeCell->topology()->getSubcell(spaceDim, sideTime0)->getSubcellCount(edgeDim);
    for (int edgeOrdinal = 0; edgeOrdinal < edgeCount; edgeOrdinal++) // ordinal in spatial topo
    {
      unsigned edgeOrdinalSpaceTime = CamelliaCellTools::subcellOrdinalMap(spaceTimeCell->topology(), spaceDim, sideTime0,
                                                                           edgeDim, edgeOrdinal);
      vector<IndexType> edgeNodes = spaceTimeCell->getEntityVertexIndices(edgeDim, edgeOrdinalSpaceTime);
      edgeToCurveMap[{edgeNodes[0],edgeNodes[1]}] = edgeFxns[edgeOrdinal];
    }
    spaceTimeMesh->setEdgeToCurveMap(edgeToCurveMap);

    testTimeCoordinatesOfCell(t0, t1, spaceTimeMesh, cellIndex, out, success);
    
    spaceTimeMesh->hRefine(set<GlobalIndexType>({cellIndex}));
    set<GlobalIndexType> bottomCells, topCells;
    
    vector< pair<GlobalIndexType, unsigned> > childrenSide0 = spaceTimeCell->childrenForSide(sideTime0);
    vector< pair<GlobalIndexType, unsigned> > childrenSide1 = spaceTimeCell->childrenForSide(sideTime1);
    for (pair<GlobalIndexType, unsigned> childEntry : childrenSide0)
    {
      bottomCells.insert(childEntry.first);
    }
    for (pair<GlobalIndexType, unsigned> childEntry : childrenSide1)
    {
      topCells.insert(childEntry.first);
    }
    
    for (GlobalIndexType bottomCellIndex : bottomCells)
    {
      double t0_bottom = t0;
      double t1_bottom = (t1 + t0) / 2.0;
      testTimeCoordinatesOfCell(t0_bottom, t1_bottom, spaceTimeMesh, bottomCellIndex, out, success);
    }
    
    for (GlobalIndexType topCellIndex : topCells)
    {
      double t0_top = (t1 + t0) / 2.0;
      double t1_top = t1;
      testTimeCoordinatesOfCell(t0_top, t1_top, spaceTimeMesh, topCellIndex, out, success);
    }
    
//    FieldContainer<double> physicalCellNodes = spaceTimeMesh->physicalCellNodesForCell(cellIndex);
//    CellTopoPtr spaceTimeTopo = spaceTimeCell->topology();
//    unsigned sideTime1 = spaceTimeTopo->getTemporalSideOrdinal(1);
//    CellTopoPtr spaceTopo = spaceTimeTopo->getSubcell(spaceDim, sideTime0);
//    unsigned spatialNodeCount = spaceTopo->getNodeCount();
//    double tol = 1e-15;
//    for (int node=0; node<spatialNodeCount; node++)
//    {
//      int spaceTimeNodeTime0 = spaceTimeTopo->getNodeMap(spaceDim, sideTime0, node);
//      int spaceTimeNodeTime1 = spaceTimeTopo->getNodeMap(spaceDim, sideTime1, node);
//      TEST_FLOATING_EQUALITY(t0, physicalCellNodes(0,spaceTimeNodeTime0,spaceDim), tol);
//      TEST_FLOATING_EQUALITY(t1, physicalCellNodes(0,spaceTimeNodeTime1,spaceDim), tol);
//    }
//    
//    BasisCachePtr basisCache = BasisCache::basisCacheForCell(spaceTimeMesh, cellIndex);
//    
//    FieldContainer<double> physicalCellNodesBasisCache = basisCache->getPhysicalCellNodes();
//    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(physicalCellNodes, physicalCellNodesBasisCache, 1e-15);
//    
//    FieldContainer<double> refCellNodes(spaceTimeTopo->getNodeCount(),spaceTimeTopo->getDimension());
//    CamelliaCellTools::refCellNodesForTopology(refCellNodes, spaceTimeTopo);
//    basisCache->setRefCellPoints(refCellNodes);
//    FieldContainer<double> mappedPhysicalCellNodes = basisCache->getPhysicalCubaturePoints();
//    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(physicalCellNodes, mappedPhysicalCellNodes, 1e-15);
  }
} // namespace
