//
//  CamelliaCellToolsTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Intrepid_FieldContainer.hpp"

#include "Intrepid_CellTools.hpp"

#include "Shards_CellTopology.hpp"

#include "SerialDenseWrapper.h"

#include "CellTopology.h"

#include "CamelliaCellTools.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
vector< CellTopoPtr > getShardsTopologies()
{
  vector< CellTopoPtr > shardsTopologies;

  shardsTopologies.push_back(CellTopology::point());
  shardsTopologies.push_back(CellTopology::line());
  shardsTopologies.push_back(CellTopology::quad());
  shardsTopologies.push_back(CellTopology::triangle());
  shardsTopologies.push_back(CellTopology::hexahedron());
  //  shardsTopologies.push_back(CellTopology::tetrahedron()); // tetrahedron not yet supported by permutation
  return shardsTopologies;
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, GetReferenceSideNormal_Space)
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr simpleTopo = shardsTopologies[topoOrdinal];
    shards::CellTopology shardsTopo = simpleTopo->getShardsTopology();

    int spaceDim = simpleTopo->getDimension();
    if (spaceDim == 0) continue; // skip point topology

    int sideCount = simpleTopo->getSideCount();

    FieldContainer<double> sideNormal(spaceDim);
    FieldContainer<double> expectedSideNormal(spaceDim);

    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      if (spaceDim == 1)
      {
        expectedSideNormal[0] = (sideOrdinal==0) ? -1 : 1;
      }
      else
      {
        CellTools<double>::getReferenceSideNormal(expectedSideNormal, sideOrdinal, shardsTopo);
      }

      CamelliaCellTools::getReferenceSideNormal(sideNormal, sideOrdinal, simpleTopo);
      TEST_COMPARE_FLOATING_ARRAYS(expectedSideNormal, sideNormal, 1e-15);
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, GetReferenceSideNormal_SpaceTime)
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr simpleTopo = shardsTopologies[topoOrdinal];
    shards::CellTopology spaceTopo = simpleTopo->getShardsTopology();

    int tensorialDegree = 1;
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);

    int spaceDim = spaceTimeTopo->getDimension();

    FieldContainer<double> sideNormal(spaceDim);
    FieldContainer<double> expectedSideNormal(spaceDim);

    int sideCount = spaceTimeTopo->getSideCount();
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      if (spaceDim == 1)
      {
        expectedSideNormal[0] = (sideOrdinal==0) ? -1 : 1;
      }
      else if (spaceDim == 2)
      {
        // Line_2 x Line_2.
        if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
        {
          // temporal direction is tangent to this side.
          int spatialNode = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
          expectedSideNormal(0) = (spatialNode == 0) ? -1 : 1;
          expectedSideNormal(1) = 0.0;
        }
        else
        {
          // spatial direction is tangent to this side.
          int temporalNode = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
          expectedSideNormal(0) = 0;
          expectedSideNormal(1) = (temporalNode == 0) ? -1 : 1;
        }
      }
      else
      {
        if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
        {
          unsigned spatialSideOrdinal = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
          FieldContainer<double> spaceSideNormal(spaceDim-1);
          CellTools<double>::getReferenceSideNormal(spaceSideNormal, spatialSideOrdinal, spaceTopo);
          for (int d=0; d<spaceDim-1; d++)
          {
            expectedSideNormal[d] = spaceSideNormal[d];
          }
          expectedSideNormal[spaceDim-1] = 0;
        }
        else
        {
          expectedSideNormal.initialize(0);
          unsigned temporalSideOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
          if (temporalSideOrdinal == 0)
          {
            expectedSideNormal[spaceDim - 1] = -1;
          }
          else
          {
            expectedSideNormal[spaceDim - 1] =  1;
          }
        }
      }
      CamelliaCellTools::getReferenceSideNormal(sideNormal, sideOrdinal, spaceTimeTopo);
      TEST_COMPARE_FLOATING_ARRAYS(expectedSideNormal, sideNormal, 1e-15);
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, MapToReferenceSubcell_ShardsAndCamelliaAgree)
{
  // for shards topologies supported by Intrepid::CellTools<double>::mapToReferenceSubcell(), test that
  // CamelliaCellTools::mapToReferenceSubcell() agrees with the Intrepid implementation.
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr simpleTopo = shardsTopologies[topoOrdinal];
    shards::CellTopology shardsTopo = simpleTopo->getShardsTopology();

    if (shardsTopo.getDimension() == 0) continue; // don't bother testing point topology

    int maxSubcellDim = min((int)shardsTopo.getDimension()-1,2);
    for (int subcellDim = 1; subcellDim <= maxSubcellDim; subcellDim++)
    {
      for (unsigned scord=0; scord < shardsTopo.getSubcellCount(subcellDim); scord++)
      {
        shards::CellTopology shardsSubcellTopo = shardsTopo.getCellTopologyData(subcellDim, scord);
        FieldContainer<double> refSubcellNodes(shardsSubcellTopo.getVertexCount(), subcellDim);
        CamelliaCellTools::refCellNodesForTopology(refSubcellNodes, shardsSubcellTopo);

        FieldContainer<double> intrepidParentNodes(shardsSubcellTopo.getVertexCount(),shardsTopo.getDimension());
        CellTools<double>::mapToReferenceSubcell(intrepidParentNodes, refSubcellNodes, subcellDim, scord, shardsTopo);

        FieldContainer<double> camelliaParentNodes(shardsSubcellTopo.getVertexCount(),shardsTopo.getDimension());
        CamelliaCellTools::mapToReferenceSubcell(camelliaParentNodes, refSubcellNodes, subcellDim, scord, simpleTopo);

        TEST_COMPARE_FLOATING_ARRAYS(intrepidParentNodes, camelliaParentNodes, 1e-15);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, MapToReferenceSubcell_NodeOrderingAgrees)
{
  // check that mapping subcell reference nodes to parent topology agrees with the
  // node ordering reported by parent for the subcell topology
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr simpleTopo = shardsTopologies[topoOrdinal];
    shards::CellTopology shardsTopo = simpleTopo->getShardsTopology();

    if (shardsTopo.getDimension() == 0) continue; // don't bother testing point topology

    int maxSubcellDim = min((int)shardsTopo.getDimension()-1,2);
    for (int subcellDim = 1; subcellDim <= maxSubcellDim; subcellDim++)
    {
      for (unsigned scord=0; scord < shardsTopo.getSubcellCount(subcellDim); scord++)
      {
        shards::CellTopology shardsSubcellTopo = shardsTopo.getCellTopologyData(subcellDim, scord);
        FieldContainer<double> refSubcellNodes(shardsSubcellTopo.getVertexCount(), subcellDim);
        CamelliaCellTools::refCellNodesForTopology(refSubcellNodes, shardsSubcellTopo);

        FieldContainer<double> nodesInParentActual(shardsSubcellTopo.getVertexCount(),shardsTopo.getDimension());
        CamelliaCellTools::mapToReferenceSubcell(nodesInParentActual, refSubcellNodes, subcellDim, scord, simpleTopo);

        FieldContainer<double> refParentNodes(shardsTopo.getNodeCount(),shardsTopo.getDimension());
        CamelliaCellTools::refCellNodesForTopology(refParentNodes, shardsTopo);
        FieldContainer<double> nodesInParentExpected(shardsSubcellTopo.getVertexCount(),shardsTopo.getDimension());
        for (int subcellNode=0; subcellNode<shardsSubcellTopo.getNodeCount(); subcellNode++)
        {
          int parentNode = simpleTopo->getNodeMap(subcellDim, scord, subcellNode);
          for (int d=0; d<simpleTopo->getDimension(); d++)
          {
            nodesInParentExpected(subcellNode,d) = refParentNodes(parentNode,d);
          }
        }

        TEST_COMPARE_FLOATING_ARRAYS(nodesInParentExpected, nodesInParentActual, 1e-15);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, MapToReferenceSubcell_Line_2_x_Line_2 )
{
  // in this test, we look at a particular space-time topology and check that points get mapped as we expect
  // (contrast with MapToReferenceSubcell_SpaceTime, in which we look at a bunch of topologies, but only check
  //  consistency with the node mapping)
  CellTopoPtr line = CellTopology::line();
  int tensorialDegree = 1;
  CellTopoPtr line_line = CellTopology::cellTopology(line->getShardsTopology(), tensorialDegree);

  FieldContainer<double> sidePoints(2,1); // the points we'll map
  sidePoints(0,0) = -0.5;
  sidePoints(1,0) =  0.5;

  FieldContainer<double> volumePoints(2,2);
  FieldContainer<double> expectedVolumePoints(2,2);

  FieldContainer<double> lineRefNodes(2,1);
  CamelliaCellTools::refCellNodesForTopology(lineRefNodes, line);

  TEST_EQUALITY(4, line_line->getSideCount());

  for (int sideOrdinal = 0; sideOrdinal < line_line->getSideCount(); sideOrdinal++)
  {
    out << "Testing side " << sideOrdinal << endl;
    if (line_line->sideIsSpatial(sideOrdinal))
    {
      // sideIsSpatial: the spatial coordinate is fixed
      int spatialNodeOrdinal = line_line->getSpatialComponentSideOrdinal(sideOrdinal);
      expectedVolumePoints(0,0) = lineRefNodes(spatialNodeOrdinal,0);
      expectedVolumePoints(0,1) = sidePoints(0,0);
      expectedVolumePoints(1,0) = lineRefNodes(spatialNodeOrdinal,0);
      expectedVolumePoints(1,1) = sidePoints(1,0);
    }
    else
    {
      int temporalNode = line_line->getTemporalComponentSideOrdinal(sideOrdinal);
      expectedVolumePoints(0,0) = sidePoints(0,0);
      expectedVolumePoints(0,1) = lineRefNodes(temporalNode,0);
      expectedVolumePoints(1,0) = sidePoints(1,0);
      expectedVolumePoints(1,1) = lineRefNodes(temporalNode,0);
    }

    CamelliaCellTools::mapToReferenceSubcell(volumePoints, sidePoints, line->getDimension(), sideOrdinal, line_line);

    SerialDenseWrapper::roundZeros(volumePoints, 1e-15);
    SerialDenseWrapper::roundZeros(expectedVolumePoints, 1e-15);

    out << "expectedVolumePoints:\n" << expectedVolumePoints;
    out << "volumePoints:\n" << volumePoints;

    TEST_COMPARE_FLOATING_ARRAYS(expectedVolumePoints, volumePoints, 1e-15);

  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, MapToReferenceSubcell_SpaceTime )
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr spaceTopo = shardsTopologies[topoOrdinal];
    int tensorialDegree = 1;
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), tensorialDegree);

    FieldContainer<double> parentRefNodes(spaceTimeTopo->getVertexCount(),spaceTimeTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(parentRefNodes, spaceTimeTopo);

    int maxSubcellDim = spaceTimeTopo->getDimension();
    for (int subcellDim = 0; subcellDim <= maxSubcellDim; subcellDim++)
    {
      for (unsigned scord=0; scord < spaceTimeTopo->getSubcellCount(subcellDim); scord++)
      {
        CellTopoPtr subcellTopo = spaceTimeTopo->getSubcell(subcellDim, scord);
        FieldContainer<double> refSubcellNodes(subcellTopo->getVertexCount(), subcellDim);
        CamelliaCellTools::refCellNodesForTopology(refSubcellNodes, subcellTopo);

        FieldContainer<double> actualParentNodes(subcellTopo->getVertexCount(),spaceTimeTopo->getDimension());
        CamelliaCellTools::mapToReferenceSubcell(actualParentNodes, refSubcellNodes, subcellDim, scord, spaceTimeTopo);

        FieldContainer<double> expectedParentNodes(subcellTopo->getVertexCount(),spaceTimeTopo->getDimension());

        int nodeCount = subcellTopo->getNodeCount();
        for (int scNode=0; scNode<nodeCount; scNode++)
        {
          int parentNode = spaceTimeTopo->getNodeMap(subcellDim, scord, scNode);
          for (int d=0; d<spaceTimeTopo->getDimension(); d++)
          {
            expectedParentNodes(scNode,d) = parentRefNodes(parentNode,d);
          }
        }

        SerialDenseWrapper::roundZeros(expectedParentNodes, 1e-15);
        SerialDenseWrapper::roundZeros(actualParentNodes, 1e-15);

        TEST_COMPARE_FLOATING_ARRAYS(expectedParentNodes, actualParentNodes, 1e-15);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, MapToPhysicalFrame)
{
  // particularly important to try it out on some topologies defined in terms of tensor products
  // main use case is with tensorial degree equal to 1 (for space-time), but might be worth trying with tensorial degree 2 and 3, too

  // to begin, just a very simple test that *nodes* are mapped appropriately

  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int tensorialDegree=0; tensorialDegree<2; tensorialDegree++)
  {
    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
    {
      CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];

      if (shardsTopo->getDimension() == 0) continue; // don't bother testing point topology

      CellTopoPtr topo = CellTopology::cellTopology(shardsTopo->getShardsTopology(), tensorialDegree);

      FieldContainer<double> refNodes(topo->getNodeCount(),topo->getDimension());

      CamelliaCellTools::refCellNodesForTopology(refNodes, topo);
      FieldContainer<double> physicalNodes(1,topo->getNodeCount(),topo->getDimension());

      // to make the physical cell, multiply node by 1/2/3 in x/y/z and add 1/4/9
      for (int node=0; node < topo->getNodeCount(); node++)
      {
        for (int d=0; d<topo->getDimension(); d++)
        {
          physicalNodes(0,node,d) = refNodes(node,d) * d + d * d;
//            physicalNodes(0,node,d) = refNodes(node,d);
        }
      }
//        cout << "WARNING: in CamelliaCellToolsTests, temporarily replaced the physical nodes with the reference nodes (i.e. just testing the identity mapping right now).\n";

      FieldContainer<double> mappedNodes(1,topo->getNodeCount(),topo->getDimension());
      CamelliaCellTools::mapToPhysicalFrame(mappedNodes, refNodes, physicalNodes, topo);

      TEST_COMPARE_FLOATING_ARRAYS(physicalNodes, mappedNodes, 1e-15);

    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, SetJacobianForSimpleShardsTopologies )
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr topo = shardsTopologies[topoOrdinal];

    int spaceDim = topo->getDimension();

    if (spaceDim == 0) continue; // don't bother testing point topology

    FieldContainer<double> refCellNodes(topo->getNodeCount(),spaceDim);

    CamelliaCellTools::refCellNodesForTopology(refCellNodes, topo);

    FieldContainer<double> cellNodes = refCellNodes;
    cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
    int numPoints = refCellNodes.dimension(0);
    FieldContainer<double> jacobianCamellia(1,numPoints,spaceDim,spaceDim);
    FieldContainer<double> jacobianShards(1,numPoints,spaceDim,spaceDim);
    CamelliaCellTools::setJacobian(jacobianCamellia, refCellNodes, cellNodes, topo);
    Intrepid::CellTools<double>::setJacobian(jacobianShards, refCellNodes, cellNodes, topo->getShardsTopology());

    TEST_COMPARE_FLOATING_ARRAYS(jacobianShards, jacobianCamellia, 1e-15);

    for (int i=0; i<cellNodes.size(); i++)
    {
      cellNodes[i] /= 2.0;
    }
    CamelliaCellTools::setJacobian(jacobianCamellia, refCellNodes, cellNodes, topo);
    Intrepid::CellTools<double>::setJacobian(jacobianShards, refCellNodes, cellNodes, topo->getShardsTopology());

    TEST_COMPARE_FLOATING_ARRAYS(jacobianShards, jacobianCamellia, 1e-15);
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, SetJacobianForSpaceTimeTopologies )
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  // Two simple tests for each shards topology:
  //  1. identity in both space and time.
  //  2. doubling coordinates in space, tripling in time.

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr spaceTopo = shardsTopologies[topoOrdinal];

    int spaceDim = spaceTopo->getDimension();
    if (spaceDim == 0) continue; // don't bother testing point topology

    int tensorialDegree = 1;
    CellTopoPtr topo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), tensorialDegree);

    CellTopoPtr timeTopo = CellTopology::line();
    int timeDim = timeTopo->getDimension();

    int spaceTimeDim = topo->getDimension();

    FieldContainer<double> refCellNodesSpace(spaceTopo->getNodeCount(),spaceDim);
    FieldContainer<double> refCellNodesTime(timeTopo->getNodeCount(),timeDim);
    FieldContainer<double> refCellNodes(topo->getNodeCount(),spaceTimeDim);

    CamelliaCellTools::refCellNodesForTopology(refCellNodesSpace, spaceTopo);
    CamelliaCellTools::refCellNodesForTopology(refCellNodesTime, timeTopo);
    CamelliaCellTools::refCellNodesForTopology(refCellNodes, topo);

    FieldContainer<double> cellNodesSpace = refCellNodesSpace;
    FieldContainer<double> cellNodesTime = refCellNodesTime;
    FieldContainer<double> cellNodesSpaceTime = refCellNodes;
    cellNodesSpace.resize(1,cellNodesSpace.dimension(0),cellNodesSpace.dimension(1));
    cellNodesTime.resize(1,cellNodesTime.dimension(0),cellNodesTime.dimension(1));
    cellNodesSpaceTime.resize(1,cellNodesSpaceTime.dimension(0),cellNodesSpaceTime.dimension(1));
    int numPointsSpace = refCellNodesSpace.dimension(0);
    int numPointsTime = refCellNodesTime.dimension(0);
    int numPointsSpaceTime = refCellNodes.dimension(0);
    FieldContainer<double> jacobianTime(1,numPointsTime,timeDim,timeDim);
    FieldContainer<double> jacobianSpace(1,numPointsSpace,spaceDim,spaceDim);
    FieldContainer<double> jacobianSpaceTime(1,numPointsSpaceTime,spaceTimeDim,spaceTimeDim);

    CamelliaCellTools::setJacobian(jacobianTime, refCellNodesTime, cellNodesTime, timeTopo);
    CamelliaCellTools::setJacobian(jacobianSpace, refCellNodesSpace, cellNodesSpace, spaceTopo);
    CamelliaCellTools::setJacobian(jacobianSpaceTime, refCellNodes, cellNodesSpaceTime, topo);

    // combine the Jacobians:
    FieldContainer<double> expectedJacobian(1,numPointsSpaceTime,spaceTimeDim,spaceTimeDim);
    for (int spaceNode=0; spaceNode<spaceTopo->getNodeCount(); spaceNode++)
    {
      for (int timeNode=0; timeNode<timeTopo->getNodeCount(); timeNode++)
      {
        vector<unsigned> nodes(2);
        nodes[0] = spaceNode;
        nodes[1] = timeNode;
        int spaceTimeNode = topo->getNodeFromTensorialComponentNodes(nodes);
        for (int d1=0; d1<spaceDim; d1++)
        {
          for (int d2=0; d2<spaceDim; d2++)
          {
            expectedJacobian(0,spaceTimeNode,d1,d2) = jacobianSpace(0,spaceNode,d1,d2);
          }
        }
        for (int d1=0; d1<timeDim; d1++)
        {
          for (int d2=0; d2<timeDim; d2++)
          {
            expectedJacobian(0,spaceTimeNode,spaceDim + d1, spaceDim + d2) = jacobianTime(0,timeNode,d1,d2);
          }
        }
      }
    }

    TEST_COMPARE_FLOATING_ARRAYS(jacobianSpaceTime, expectedJacobian, 1e-15);

    for (int i=0; i<cellNodesSpace.size(); i++)
    {
      cellNodesSpace[i] *= 2.0;
    }

    for (int i=0; i<cellNodesTime.size(); i++)
    {
      cellNodesTime[i] *= 3.0;
    }

    cellNodesSpaceTime.resize(topo->getNodeCount(), topo->getDimension());
    vector< FieldContainer<double> > tensorComponentNodes;
    tensorComponentNodes.push_back(cellNodesSpace);
    tensorComponentNodes.push_back(cellNodesTime);
    tensorComponentNodes[0].resize(spaceTopo->getNodeCount(), spaceTopo->getDimension());
    tensorComponentNodes[1].resize(timeTopo->getNodeCount(), timeTopo->getDimension());
    topo->initializeNodes(tensorComponentNodes, cellNodesSpaceTime);
    cellNodesSpaceTime.resize(1, topo->getNodeCount(), topo->getDimension());

    CamelliaCellTools::setJacobian(jacobianTime, refCellNodesTime, cellNodesTime, timeTopo);
    CamelliaCellTools::setJacobian(jacobianSpace, refCellNodesSpace, cellNodesSpace, spaceTopo);
    CamelliaCellTools::setJacobian(jacobianSpaceTime, refCellNodes, cellNodesSpaceTime, topo);

    // combine the Jacobians:
    for (int spaceNode=0; spaceNode<spaceTopo->getNodeCount(); spaceNode++)
    {
      for (int timeNode=0; timeNode<timeTopo->getNodeCount(); timeNode++)
      {
        vector<unsigned> nodes(2);
        nodes[0] = spaceNode;
        nodes[1] = timeNode;

        int spaceTimeNode = topo->getNodeFromTensorialComponentNodes(nodes);
        for (int d1_space=0; d1_space<spaceDim; d1_space++)   // d1: which spatial component we're taking derivative of
        {
          for (int d2_space=0; d2_space<spaceDim; d2_space++)   // d2: which spatial direction we take the derivative in
          {
            expectedJacobian(0,spaceTimeNode,d1_space,d2_space) = jacobianSpace(0,spaceNode,d1_space,d2_space);
          }
        }
        for (int d1=0; d1<timeDim; d1++)
        {
          for (int d2=0; d2<timeDim; d2++)
          {
            expectedJacobian(0,spaceTimeNode,spaceDim + d1, spaceDim + d2) = jacobianTime(0,timeNode,d1,d2);
          }
        }
      }
    }

//      cout << "expectedJacobian:\n" << expectedJacobian;
//      cout << "jacobianSpaceTime:\n" << jacobianSpaceTime;

    TEST_COMPARE_FLOATING_ARRAYS(jacobianSpaceTime, expectedJacobian, 1e-15);
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, PermutationMatchingOrder_Space ) // tensorial degree 0 Camellia CellTopology
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr camelliaTopo = shardsTopologies[topoOrdinal];

    int spaceDim = camelliaTopo->getDimension();
    if (spaceDim == 0) continue; // don't bother testing point topology

    int permutationCount = camelliaTopo->getNodePermutationCount(); // Camellia CellTopology defines permutations for 3D entities, while shards does not, so using Camellia's permutation count will provide a more rigorous test

    int nodeCount = camelliaTopo->getNodeCount();

    for (int permutation=0; permutation<permutationCount; permutation++)
    {
      vector<unsigned> fromOrder(nodeCount);
      vector<unsigned> toOrder(nodeCount);

      for (unsigned node=0; node<nodeCount; node++)
      {
        unsigned permutedNode = camelliaTopo->getNodePermutation(permutation, node);
        fromOrder[node] = node;
        toOrder[permutedNode] = node;
      }

      unsigned permutationMatchingOrder = CamelliaCellTools::permutationMatchingOrder(camelliaTopo, fromOrder, toOrder);

      TEST_EQUALITY(permutation, permutationMatchingOrder);
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, PermutationMatchingOrder_SpaceTime_Slow ) // tensorial degree 1 Camellia CellTopology
{
  int tensorialDegree = 1;

  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr camelliaTopo = CellTopology::cellTopology(shardsTopologies[topoOrdinal]->getShardsTopology(), tensorialDegree);

    int spaceDim = camelliaTopo->getDimension();
    if (spaceDim == 0) continue; // don't bother testing point topology

    int permutationCount = camelliaTopo->getNodePermutationCount(); // Camellia CellTopology defines permutations for 3D entities, while shards does not, so using Camellia's permutation count will provide a more rigorous test

    int nodeCount = camelliaTopo->getNodeCount();

    for (int permutation=0; permutation<permutationCount; permutation++)
    {
      vector<unsigned> fromOrder(nodeCount);
      vector<unsigned> toOrder(nodeCount);

      for (unsigned node=0; node<nodeCount; node++)
      {
        unsigned permutedNode = camelliaTopo->getNodePermutation(permutation, node);
        fromOrder[node] = node;
        toOrder[permutedNode] = node;
      }

      unsigned permutationMatchingOrder = CamelliaCellTools::permutationMatchingOrder(camelliaTopo, fromOrder, toOrder);

      TEST_EQUALITY(permutation, permutationMatchingOrder);
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, PermutedReferenceCellPoints_Space_Slow ) // tensorial degree 0 Camellia CellTopology
{
  // to begin, just a very simple test that *nodes* are permuted appropriately

  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr camelliaTopo = shardsTopologies[topoOrdinal];

    int spaceDim = camelliaTopo->getDimension();
    if (spaceDim == 0) continue; // don't bother testing point topology

    shards::CellTopology shardsTopo = camelliaTopo->getShardsTopology();

    FieldContainer<double> refPoints(camelliaTopo->getNodeCount(),camelliaTopo->getDimension());
    FieldContainer<double> topoNodesPermuted(camelliaTopo->getNodeCount(),camelliaTopo->getDimension());

    int permutationCount = camelliaTopo->getNodePermutationCount(); // Camellia CellTopology defines permutations for 3D entities, while shards does not, so using Camellia's permutation count will provide a more rigorous test

    CamelliaCellTools::refCellNodesForTopology(refPoints, camelliaTopo);

    FieldContainer<double> permutedRefPoints(camelliaTopo->getNodeCount(),camelliaTopo->getDimension());

    for (int permutation=0; permutation<permutationCount; permutation++)
    {
      CamelliaCellTools::refCellNodesForTopology(topoNodesPermuted, camelliaTopo, permutation);

      CamelliaCellTools::permutedReferenceCellPoints(camelliaTopo, permutation, refPoints, permutedRefPoints);

      // expect permutedRefPoints = topoNodesPermuted

      for (int nodeOrdinal=0; nodeOrdinal < topoNodesPermuted.dimension(0); nodeOrdinal++)
      {
        for (int d=0; d<topoNodesPermuted.dimension(1); d++)
        {
          TEST_FLOATING_EQUALITY(topoNodesPermuted(nodeOrdinal,d), permutedRefPoints(nodeOrdinal,d), 1e-15);
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, PermutedReferenceCellPoints_SpaceTime_Slow ) // tensorial degree 1 Camellia CellTopology
{
  // to begin, just a very simple test that *nodes* are permuted appropriately

  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  int tensorialDegree = 1;

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr camelliaTopo = CellTopology::cellTopology(shardsTopologies[topoOrdinal]->getShardsTopology(), tensorialDegree);

    int spaceDim = camelliaTopo->getDimension();
    if (spaceDim == 0) continue; // don't bother testing point topology

    shards::CellTopology shardsTopo = camelliaTopo->getShardsTopology();

    FieldContainer<double> refPoints(camelliaTopo->getNodeCount(),camelliaTopo->getDimension());
    FieldContainer<double> topoNodesPermuted(camelliaTopo->getNodeCount(),camelliaTopo->getDimension());

    int permutationCount = camelliaTopo->getNodePermutationCount(); // Camellia CellTopology defines permutations for 3D entities, while shards does not, so using Camellia's permutation count will provide a more rigorous test

    CamelliaCellTools::refCellNodesForTopology(refPoints, camelliaTopo);

    FieldContainer<double> permutedRefPoints(camelliaTopo->getNodeCount(),camelliaTopo->getDimension());

    for (int permutation=0; permutation<permutationCount; permutation++)
    {
      CamelliaCellTools::refCellNodesForTopology(topoNodesPermuted, camelliaTopo, permutation);

      CamelliaCellTools::permutedReferenceCellPoints(camelliaTopo, permutation, refPoints, permutedRefPoints);

      // expect permutedRefPoints = topoNodesPermuted

      for (int nodeOrdinal=0; nodeOrdinal < topoNodesPermuted.dimension(0); nodeOrdinal++)
      {
        for (int d=0; d<topoNodesPermuted.dimension(1); d++)
        {
          TEST_FLOATING_EQUALITY(topoNodesPermuted(nodeOrdinal,d), permutedRefPoints(nodeOrdinal,d), 1e-15);
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, RefCellPointsForTopology )
{
  // just check that the version that takes a Camellia CellTopology matches
  // the one that takes a shards CellTopology

  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr topo = shardsTopologies[topoOrdinal];

    if (topo->getDimension() == 0) continue; // don't bother testing point topology

    FieldContainer<double> refCellNodesShards(topo->getNodeCount(),topo->getDimension());
    FieldContainer<double> refCellNodesCamellia(topo->getNodeCount(),topo->getDimension());
    int permutationCount;
    if (topo->getDimension() <= 2)
    {
      permutationCount = topo->getNodePermutationCount();
    }
    else
    {
      permutationCount = 1; // shards doesn't provide permutations for 3D objects
    }

    for (int permutation=0; permutation<permutationCount; permutation++)
    {
      CamelliaCellTools::refCellNodesForTopology(refCellNodesShards, topo->getShardsTopology(), permutation);
      CamelliaCellTools::refCellNodesForTopology(refCellNodesCamellia, topo, permutation);

      TEST_COMPARE_FLOATING_ARRAYS(refCellNodesShards, refCellNodesCamellia, 1e-15);

      if (!success)
      {
        cout << "Test failure (set breakpoint here) \n";
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, RefCellPointsForTensorTopology )
{
  // just check that the version that takes a Camellia CellTopology of tensorial degree 1 matches:
  // shardsTopo refNodes + t = -1
  // shardsTopo refNodes + t =  1
  // in that order...

  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];

    int tensorialDegree = 1;
    CellTopoPtr topo = CellTopology::cellTopology(shardsTopo->getShardsTopology(), tensorialDegree);

    FieldContainer<double> refCellNodesShards(shardsTopo->getNodeCount(),shardsTopo->getDimension());
    FieldContainer<double> refCellNodesCamellia(topo->getNodeCount(),topo->getDimension());

    int permutation=0;
    CamelliaCellTools::refCellNodesForTopology(refCellNodesShards, shardsTopo, permutation);
    CamelliaCellTools::refCellNodesForTopology(refCellNodesCamellia, topo, permutation);

    FieldContainer<double> refCellNodesExpected(topo->getNodeCount(),topo->getDimension());

    if (shardsTopo->getDimension() == 0)
    {
      // special case: point x line = line
      CellTopoPtr line = CellTopology::line();
      CamelliaCellTools::refCellNodesForTopology(refCellNodesExpected, line);
    }
    else
    {
      for (int node=0; node<topo->getNodeCount(); node++)
      {
        int shardsNode = node % shardsTopo->getNodeCount();
        for (int d=0; d<topo->getDimension(); d++)
        {
          if (d < shardsTopo->getDimension())
            refCellNodesExpected(node,d) = refCellNodesShards(shardsNode,d);
          else if (shardsNode == node) refCellNodesExpected(node,d) = -1;
          else refCellNodesExpected(node,d) = 1;
        }
      }
    }

    TEST_COMPARE_FLOATING_ARRAYS(refCellNodesExpected, refCellNodesCamellia, 1e-15);

    if (!success)
    {
      cout << "Test failure (set breakpoint here) \n";
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, SubcellOrdinalMap_Space )
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr spaceTopo = shardsTopologies[topoOrdinal];

    int spaceDim = spaceTopo->getDimension();
    if (spaceDim == 0) continue; // don't bother testing point topology

    for (int scDim = 0; scDim <= spaceDim; scDim++)
    {
      int scCount = spaceTopo->getSubcellCount(scDim);
      for (int scord=0; scord < scCount; scord++)
      {
        CellTopoPtr scTopo = spaceTopo->getSubcell(scDim, scord);
        for (int sscDim = 0; sscDim <= scTopo->getDimension(); sscDim++)
        {
          int sscCount = scTopo->getSubcellCount(sscDim);
          for (int sscord=0; sscord < sscCount; sscord++)
          {
            unsigned sscord_parent = CamelliaCellTools::subcellOrdinalMap(spaceTopo, scDim, scord, sscDim, sscord);
            int sscNodeCount = scTopo->getNodeCount(sscDim, sscord);
            vector<unsigned> subsubcellNodesInParent(sscNodeCount);
            vector<unsigned> subsubcellNodesInSubcellInParent(sscNodeCount);
            for (unsigned sscNodeOrdinal=0; sscNodeOrdinal<sscNodeCount; sscNodeOrdinal++)
            {
              // first, map node from subsubcell to spaceTopo
              unsigned subsubcellNode_parent = spaceTopo->getNodeMap(sscDim, sscord_parent, sscNodeOrdinal);
              subsubcellNodesInParent[sscNodeOrdinal] = subsubcellNode_parent;
              // next, map same node from subsubcell to subcell
              // then, map from subcell to spaceTopo and check that they match
              unsigned subcellNode = scTopo->getNodeMap(sscDim, sscord, sscNodeOrdinal);
              unsigned subcellNode_parent = spaceTopo->getNodeMap(scDim, scord, subcellNode);
              subsubcellNodesInSubcellInParent[sscNodeOrdinal] = subcellNode_parent;
            }
            // we require that the two node sets are the same, not that they come in the same order
            // (parent and subcell may have differing ideas about the sub-subcell orientation)
            std::sort(subsubcellNodesInParent.begin(), subsubcellNodesInParent.end());
            std::sort(subsubcellNodesInSubcellInParent.begin(), subsubcellNodesInSubcellInParent.end());
            TEST_COMPARE_ARRAYS(subsubcellNodesInParent, subsubcellNodesInSubcellInParent);
          }
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( CamelliaCellTools, SubcellOrdinalMap_SpaceTime )
{
  std::vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

  bool havePrintedFirstFailureInfo = false;

  int tensorialDegree = 1;
  for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
  {
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(shardsTopologies[topoOrdinal]->getShardsTopology(), tensorialDegree);

    int spaceDim = spaceTimeTopo->getDimension();

    for (int scDim = 0; scDim <= spaceDim; scDim++)
    {
      int scCount = spaceTimeTopo->getSubcellCount(scDim);
      for (int scord=0; scord < scCount; scord++)
      {
        CellTopoPtr scTopo = spaceTimeTopo->getSubcell(scDim, scord);
        for (int sscDim = 0; sscDim <= scTopo->getDimension(); sscDim++)
        {
          int sscCount = scTopo->getSubcellCount(sscDim);
          for (int sscord=0; sscord < sscCount; sscord++)
          {
            unsigned sscord_parent = CamelliaCellTools::subcellOrdinalMap(spaceTimeTopo, scDim, scord, sscDim, sscord);
            int sscNodeCount = scTopo->getNodeCount(sscDim, sscord);
            vector<unsigned> subsubcellNodesInParent(sscNodeCount);
            vector<unsigned> subsubcellNodesInSubcellInParent(sscNodeCount);
            for (unsigned sscNodeOrdinal=0; sscNodeOrdinal<sscNodeCount; sscNodeOrdinal++)
            {
              // first, map node from subsubcell to spaceTopo
              unsigned subsubcellNode_parent = spaceTimeTopo->getNodeMap(sscDim, sscord_parent, sscNodeOrdinal);
              subsubcellNodesInParent[sscNodeOrdinal] = subsubcellNode_parent;
              // next, map same node from subsubcell to subcell
              // then, map from subcell to spaceTopo and check that they match
              unsigned subcellNode = scTopo->getNodeMap(sscDim, sscord, sscNodeOrdinal);
              unsigned subcellNode_parent = spaceTimeTopo->getNodeMap(scDim, scord, subcellNode);
              subsubcellNodesInSubcellInParent[sscNodeOrdinal] = subcellNode_parent;
            }
            // we require that the two node sets are the same, not that they come in the same order
            // (parent and subcell may have differing ideas about the sub-subcell orientation)
            std::sort(subsubcellNodesInParent.begin(), subsubcellNodesInParent.end());
            std::sort(subsubcellNodesInSubcellInParent.begin(), subsubcellNodesInSubcellInParent.end());
            TEST_COMPARE_ARRAYS(subsubcellNodesInParent, subsubcellNodesInSubcellInParent);

            if ((success == false) && !havePrintedFirstFailureInfo)
            {
              cout << "First failed test here.\n";
              cout << "spaceTimeTopo = " << spaceTimeTopo->getName() << "; subcell " << scTopo->getName() << " ordinal " << scord;
              cout << "; sscDim " << sscDim << ", sscord " << sscord << endl;
              havePrintedFirstFailureInfo = true;
            }
          }
        }
      }
    }
  }
}
} // namespace
