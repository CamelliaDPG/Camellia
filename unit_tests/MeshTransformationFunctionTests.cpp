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
    
    FieldContainer<double> physicalCellNodes = spaceTimeMesh->physicalCellNodesForCell(cellIndex);
    CellTopoPtr spaceTimeTopo = spaceTimeCell->topology();
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
} // namespace
