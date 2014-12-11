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

#include "Shards_CellTopology.hpp"

#include "CamelliaCellTools.h"

namespace {
  TEUCHOS_UNIT_TEST( CamelliaCellTools, MapToPhysicalFrame) {
    // TODO: implement this test
    // particularly important to try it out on some topologies defined in terms of tensor products
    // main use case is with tensorial degree equal to 1 (for space-time), but might be worth trying with tensorial degree 2 and 3, too
    
  }
  
  TEUCHOS_UNIT_TEST( CamelliaCellTools, PermutedReferenceCellPoints )
  {
    // to begin, just a very simple test that *nodes* are permuted appropriately
    
    shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );

    FieldContainer<double> refPoints(quad_4.getNodeCount(),quad_4.getDimension());
    FieldContainer<double> quadNodesPermuted(quad_4.getNodeCount(),quad_4.getDimension());
    
    int permutationCount = quad_4.getNodePermutationCount();
    
    CamelliaCellTools::refCellNodesForTopology(refPoints, quad_4);
    
    FieldContainer<double> permutedRefPoints(quad_4.getNodeCount(),quad_4.getDimension());
    
    for (int permutation=0; permutation<permutationCount; permutation++) {
      CamelliaCellTools::refCellNodesForTopology(quadNodesPermuted, quad_4, permutation);
      
      CamelliaCellTools::permutedReferenceCellPoints(quad_4, permutation, refPoints, permutedRefPoints);
      
      // expect permutedRefPoints = quadNodesPermuted
      
      for (int nodeOrdinal=0; nodeOrdinal < quadNodesPermuted.dimension(0); nodeOrdinal++) {
        for (int d=0; d<quadNodesPermuted.dimension(1); d++) {
          TEST_FLOATING_EQUALITY(quadNodesPermuted(nodeOrdinal,d), permutedRefPoints(nodeOrdinal,d), 1e-15);
        }
      }
    }
  }
} // namespace