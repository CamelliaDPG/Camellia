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

#include "Shards_CellTopology.hpp"

#include "CamelliaCellTools.h"

#include "RefinementPattern.h"

namespace {
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
} // namespace