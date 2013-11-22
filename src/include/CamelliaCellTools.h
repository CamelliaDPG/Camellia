//
//  CamelliaCellTools.h
//  Camellia-debug
//
//  Created by Nate Roberts on 11/21/13.
//
//

#ifndef Camellia_debug_CamelliaCellTools_h
#define Camellia_debug_CamelliaCellTools_h

#include "Shards_CellTopology.hpp"

class CamelliaCellTools {
public:
  static void refCellNodesForTopology(FieldContainer<double> &cellNodes, const shards::CellTopology &cellTopo, unsigned permutation = 0) { // 0 permutation is the identity
    if (cellNodes.dimension(0) != cellTopo.getNodeCount()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
    }
    if (cellNodes.dimension(1) != cellTopo.getDimension()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
    }
    unsigned cellKey = cellTopo.getKey();
    switch (cellKey) {
      case shards::Line<2>::key:
        cellNodes(0,0) = -1;
        cellNodes(1,0) =  1;
        break;
      case shards::Triangle<3>::key:
        cellNodes(0,0) =  0;
        cellNodes(0,1) =  0;
        
        cellNodes(1,0) =  1;
        cellNodes(1,1) =  0;
        
        cellNodes(2,0) =  0;
        cellNodes(2,1) =  1;
      case shards::Quadrilateral<4>::key:
        cellNodes(0,0) = -1.0; // x1
        cellNodes(0,1) = -1.0; // y1
        cellNodes(1,0) = 1.0;
        cellNodes(1,1) = -1.0;
        cellNodes(2,0) = 1.0;
        cellNodes(2,1) = 1.0;
        cellNodes(3,0) = -1.0;
        cellNodes(3,1) = 1.0;
        break;
      case shards::Hexahedron<8>::key:
        cellNodes(0,0) = -1;
        cellNodes(0,1) = -1;
        cellNodes(0,2) = -1;
        
        cellNodes(1,0) = 1;
        cellNodes(1,1) = -1;
        cellNodes(1,2) = -1;
        
        cellNodes(2,0) = 1;
        cellNodes(2,1) = 1;
        cellNodes(2,2) = -1;
        
        cellNodes(3,0) = -1;
        cellNodes(3,1) = 1;
        cellNodes(3,2) = -1;
        
        cellNodes(4,0) = -1;
        cellNodes(4,1) = -1;
        cellNodes(4,2) = 1;
        
        cellNodes(5,0) = 1;
        cellNodes(5,1) = -1;
        cellNodes(5,2) = 1;
        
        cellNodes(6,0) = 1;
        cellNodes(6,1) = 1;
        cellNodes(6,2) = 1;
        
        cellNodes(7,0) = -1;
        cellNodes(7,1) = 1;
        cellNodes(7,2) = 1;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled CellTopology.");
    }
    
    if ( permutation != 0 ) {
      FieldContainer<double> cellNodesCopy = cellNodes;
      unsigned nodeCount = cellNodes.dimension(0);
      unsigned spaceDim = cellNodes.dimension(1);
      for (int n = 0; n<nodeCount; n++) {
        int n_permuted = cellTopo.getNodePermutation(permutation, n);
        for (int d = 0; d<spaceDim; d++) {
          cellNodes(n,d) = cellNodesCopy(n_permuted,d);
        }
      }
    }
  }
  
  static unsigned matchingVolumePermutationForSidePermutation( const shards::CellTopology &volumeTopo, unsigned sideIndex, unsigned sidePermutation) {
    // brute force search for a volume permutation that will make side line up according to sidePermutation
    int d = volumeTopo.getDimension();
    shards::CellTopology sideTopo = volumeTopo.getCellTopologyData(d-1, sideIndex);
    
    unsigned sideNodeCount = sideTopo.getNodeCount();
    unsigned volumePermutationCount = volumeTopo.getNodePermutationCount();
    for (unsigned volumePermutation=0; volumePermutation<volumePermutationCount; volumePermutation++) {
      bool matches = true;
      for (unsigned sideNodeIndex = 0; sideNodeIndex < sideNodeCount; sideNodeIndex++) {
        unsigned volumeNodeIndex = volumeTopo.getNodeMap(d-1, sideIndex, sideNodeIndex);
        unsigned permutedSideNodeIndex = sideTopo.getNodePermutation(sidePermutation, sideNodeIndex);
        unsigned permutedVolumeNodeIndex = volumeTopo.getNodeMap(d-1,sideIndex,permutedSideNodeIndex);
        if ( permutedVolumeNodeIndex != volumeTopo.getNodePermutation(volumePermutation, volumeNodeIndex) ) {
          matches = false;
          break;
        }
      }
      if (matches) {
        return volumePermutation;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching permutation found");
    return volumePermutationCount; // an impossible (out of bounds) answer: this line just to satisfy compilers that warn about missing return values.
  }
};

#endif
