//
//  RefinementPatternTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/10/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "CamelliaCellTools.h"

#include "RefinementPattern.h"

using namespace Camellia;
using namespace Intrepid;

namespace {
  vector< CellTopoPtr > getShardsTopologies() {
    vector< CellTopoPtr > shardsTopologies;
    
    shardsTopologies.push_back(CellTopology::point());
    shardsTopologies.push_back(CellTopology::line());
    shardsTopologies.push_back(CellTopology::quad());
    shardsTopologies.push_back(CellTopology::triangle());
    shardsTopologies.push_back(CellTopology::hexahedron());
    //  shardsTopologies.push_back(CellTopology::tetrahedron()); // tetrahedron not yet supported by permutation
    return shardsTopologies;
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapSubcellFromChildToParent_CommonVertices )
  {
    // test checks that regular refinement patterns do the right thing mapping from each child vertex to the parent,
    // in the case of vertices that are shared by parent.
    
    vector< CellTopoPtr > shardsTopologies = getShardsTopologies();
    
    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];
      if (shardsTopo->getDimension() == 0) continue; // skip the point topology
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(shardsTopo);
      
      FieldContainer<double> childVertices = refPattern->refinedNodes();
      FieldContainer<double> parentVertices(shardsTopo->getVertexCount(),shardsTopo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(parentVertices, shardsTopo);
      
      double tol = 1e-15; // for vertex coordinate equality
      
      unsigned vertexDim = 0;
      for (int childOrdinal=0; childOrdinal<childVertices.dimension(0); childOrdinal++) {
        CellTopoPtr childTopo = refPattern->childTopology(childOrdinal);
        for (int childVertexOrdinal = 0; childVertexOrdinal < childTopo->getVertexCount(); childVertexOrdinal++) {
          int parentVertexOrdinal = -1;
          // see if this vertex is present in parent:
          for (int vertexOrdinal=0; vertexOrdinal<shardsTopo->getVertexCount(); vertexOrdinal++) {
            bool matches = true;
            for (int d=0; d < shardsTopo->getDimension(); d++) {
              if (abs(childVertices(childOrdinal,childVertexOrdinal,d)-parentVertices(vertexOrdinal,d)) > tol) {
                matches = false;
              }
            }
            if (matches) {
              parentVertexOrdinal = vertexOrdinal;
              break; // we've found our match
            }
          }
          if (parentVertexOrdinal == -1) continue; // no match in parent: we don't test this vertex
          pair<unsigned,unsigned> vertexAncestor = refPattern->mapSubcellFromChildToParent(childOrdinal, vertexDim, childVertexOrdinal);
          
          TEST_EQUALITY(vertexAncestor.first, vertexDim);
          TEST_EQUALITY(vertexAncestor.second, parentVertexOrdinal);
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, MapSubcellFromChildToParent_LineMiddleVertex)
  {
    // for the case of the line refinement pattern, tests the vertex we don't test in MapSubcellFromChildToParent_CommonVertices
    
    CellTopoPtr shardsTopo = CellTopology::line();
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(shardsTopo);
    
    FieldContainer<double> childVertices = refPattern->refinedNodes();
    FieldContainer<double> parentVertices(shardsTopo->getVertexCount(),shardsTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(parentVertices, shardsTopo);
    
    double tol = 1e-15; // for vertex coordinate equality
    
    unsigned vertexDim = 0, lineDim = 1;
    unsigned parentLineOrdinal = 0;
    for (int childOrdinal=0; childOrdinal<childVertices.dimension(0); childOrdinal++) {
      CellTopoPtr childTopo = refPattern->childTopology(childOrdinal);
      for (int childVertexOrdinal = 0; childVertexOrdinal < childTopo->getVertexCount(); childVertexOrdinal++) {
        int parentVertexOrdinal = -1;
        // see if this vertex is present in parent:
        for (int vertexOrdinal=0; vertexOrdinal<shardsTopo->getVertexCount(); vertexOrdinal++) {
          bool matches = true;
          for (int d=0; d < shardsTopo->getDimension(); d++) {
            if (abs(childVertices(childOrdinal,childVertexOrdinal,d)-parentVertices(vertexOrdinal,d)) > tol) {
              matches = false;
            }
          }
          if (matches) {
            parentVertexOrdinal = vertexOrdinal;
            break; // we've found our match
          }
        }
        if (parentVertexOrdinal != -1) continue; // found match in parent: this is a vertex we test above
        pair<unsigned,unsigned> vertexAncestor = refPattern->mapSubcellFromChildToParent(childOrdinal, vertexDim, childVertexOrdinal);
        
        TEST_EQUALITY(vertexAncestor.first, lineDim);
        TEST_EQUALITY(vertexAncestor.second, parentLineOrdinal);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, MapRefCellPointsToAncestor_ChildNodeIdentity )
  {
    // test takes regular refinement patterns and checks that RefinementPattern::mapRefCellPointsToAncestor
    // does the right thing when we use the reference cell points on child cell in a one-level refinement branch

    vector< CellTopoPtrLegacy > cellTopos;
    cellTopos.push_back( Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Line<2> >()) ) );
    cellTopos.push_back( Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()) ) );
    cellTopos.push_back( Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Triangle<3> >()) ) );
    cellTopos.push_back( Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Hexahedron<8> >()) ) );
    
    for (int topoOrdinal=0; topoOrdinal<cellTopos.size(); topoOrdinal++) {
      CellTopoPtrLegacy cellTopo = cellTopos[topoOrdinal];
      
      int nodeCount = cellTopo->getNodeCount();
      int spaceDim = cellTopo->getDimension();
      
      FieldContainer<double> refCellNodes(nodeCount,spaceDim);
      
      CamelliaCellTools::refCellNodesForTopology(refCellNodes, *cellTopo);
      
      FieldContainer<double> childRefCellNodes = refCellNodes;
      
      RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(cellTopo->getKey());
      
      int childCount = regularRefinement->numChildren();
      
      for (int childOrdinal=0; childOrdinal<childCount; childOrdinal++) {
        RefinementBranch refBranch;
        refBranch.push_back(make_pair(regularRefinement.get(),childOrdinal));
        
        FieldContainer<double> expectedPoints(nodeCount,spaceDim);
        for (int nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++) {
          for (int d=0; d<spaceDim; d++) {
            expectedPoints(nodeOrdinal,d) = regularRefinement->refinedNodes()(childOrdinal,nodeOrdinal,d);
          }
        }
        FieldContainer<double> actualPoints(nodeCount,spaceDim);
        RefinementPattern::mapRefCellPointsToAncestor(refBranch, childRefCellNodes, actualPoints);
        
        TEST_COMPARE_FLOATING_ARRAYS(expectedPoints, actualPoints, 1e-15);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, SpaceTimeTopology )
  {
    // tests refinement patterns for space-time topologies.
    int tensorialDegree = 1;
    vector<CellTopoPtr> shardsTopologies = getShardsTopologies();
    for (int topoOrdinal=0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      shards::CellTopology shardsTopo = shardsTopologies[topoOrdinal]->getShardsTopology();
      CellTopoPtr cellTopo = CellTopology::cellTopology(shardsTopo, tensorialDegree);
      
      FieldContainer<double> spaceRefinedNodes;
      RefinementPatternPtr refPatternSpace = RefinementPattern::regularRefinementPattern(shardsTopo.getKey());
      spaceRefinedNodes = refPatternSpace->refinedNodes();
      
      RefinementPatternPtr refPatternTime = RefinementPattern::regularRefinementPattern(CellTopology::line());
      FieldContainer<double> timeRefinedNodes = refPatternTime->refinedNodes();
      
      RefinementPatternPtr refPatternSpaceTime = RefinementPattern::regularRefinementPattern(cellTopo);
      FieldContainer<double> spaceTimeRefinedNodes = refPatternSpaceTime->refinedNodes();
      
      int numChildrenSpace = spaceRefinedNodes.dimension(0);
      int numChildrenTime = timeRefinedNodes.dimension(0);
      int numChildrenSpaceTime = spaceTimeRefinedNodes.dimension(0);
      
      TEST_EQUALITY(numChildrenSpace * numChildrenTime, numChildrenSpaceTime);
      
      int numNodesSpace = spaceRefinedNodes.dimension(1);
      int numNodesTime = timeRefinedNodes.dimension(1);
      int numNodesSpaceTime = spaceTimeRefinedNodes.dimension(1);
      
      TEST_EQUALITY(numNodesSpace * numNodesTime, numNodesSpaceTime);
      
      int dSpace = spaceRefinedNodes.dimension(2);
      int dTime = timeRefinedNodes.dimension(2);
      int dSpaceTime = spaceTimeRefinedNodes.dimension(2);
      
      TEST_EQUALITY(dSpace + dTime, dSpaceTime);
      
      FieldContainer<double> expectedRefinedNodes(numChildrenSpaceTime,numNodesSpaceTime,dSpaceTime);
      
      int spaceTimeChildOrdinal = 0;
      for (int timeChildOrdinal=0; timeChildOrdinal<numChildrenTime; timeChildOrdinal++) {
        for (int spaceChildOrdinal=0; spaceChildOrdinal<numChildrenSpace; spaceChildOrdinal++, spaceTimeChildOrdinal++) {
          int spaceTimeNodeOrdinal=0;
          for (int timeNodeOrdinal=0; timeNodeOrdinal<numNodesTime; timeNodeOrdinal++) {
            for (int spaceNodeOrdinal=0; spaceNodeOrdinal<numNodesSpace; spaceNodeOrdinal++, spaceTimeNodeOrdinal++) {
              for (int d_space=0; d_space<dSpace; d_space++) {
                expectedRefinedNodes(spaceTimeChildOrdinal,spaceTimeNodeOrdinal,d_space) = spaceRefinedNodes(spaceChildOrdinal,spaceNodeOrdinal,d_space);
              }
              for (int d_time=0; d_time<dTime; d_time++) {
                expectedRefinedNodes(spaceTimeChildOrdinal,spaceTimeNodeOrdinal,d_time + dSpace) = timeRefinedNodes(timeChildOrdinal,timeNodeOrdinal,d_time);
              }
            }
          }
        }
      }
      TEST_COMPARE_FLOATING_ARRAYS(spaceTimeRefinedNodes, expectedRefinedNodes, 1e-15);
    }
  }
} // namespace
