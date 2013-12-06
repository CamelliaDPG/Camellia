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
  
  static unsigned permutationMatchingOrder( const shards::CellTopology &cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder) {
    if (cellTopo.getDimension() == 0) {
      return 0;
    }
    unsigned permutationCount = cellTopo.getNodePermutationCount();
    unsigned nodeCount = fromOrder.size();
    for (unsigned permutation=0; permutation<permutationCount; permutation++) {
      bool matches = true;
      for (unsigned fromIndex=0; fromIndex<nodeCount; fromIndex++) {
        unsigned toIndex = cellTopo.getNodePermutation(permutation, fromIndex);
        if (fromOrder[fromIndex] != toOrder[toIndex]) {
          matches = false;
          break;
        }
      }
      if (matches) return permutation;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching permutation found");
    return permutationCount; // an impossible (out of bounds) answer: this line just to satisfy compilers that warn about missing return values.
  }
  
  static unsigned matchingVolumePermutationForSidePermutation( const shards::CellTopology &volumeTopo, unsigned sideIndex, unsigned sidePermutation) {
    // brute force search for a volume permutation that will make side line up according to sidePermutation
    // (I believe this works for cases when the volume topology has permutations defined (not actually true even for Hexahedron<8>),
    //  but I'm not sure that this is actually legitimately useful--certainly it isn't the right thing in the case for which I originally
    //  concocted it.) -- NVR 11/25/13
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
  
  static unsigned subcellOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord) {
    typedef unsigned CellTopoKey;
    typedef unsigned SubcellOrdinal;
    typedef unsigned SubcellDimension;
    typedef unsigned SubSubcellOrdinal;
    typedef unsigned SubSubcellDimension;
    typedef unsigned SubSubcellOrdinalInCellTopo;
    typedef pair< SubcellDimension, SubcellOrdinal > SubcellIdentifier;    // dim, ord in cellTopo
    typedef pair< SubSubcellDimension, SubSubcellOrdinal > SubSubcellIdentifier; // dim, ord in subcell
    typedef map< SubcellIdentifier, map< SubSubcellIdentifier, SubSubcellOrdinalInCellTopo > > OrdinalMap;
    static map< CellTopoKey, OrdinalMap > ordinalMaps;
    
    CellTopoKey key = cellTopo.getKey();
    if (ordinalMaps.find(key) == ordinalMaps.end()) {
      // then we construct the map for this cellTopo
      OrdinalMap ordinalMap;
      unsigned sideDim = cellTopo.getDimension() - 1;
      typedef unsigned NodeOrdinal;
      map< set<NodeOrdinal>, SubcellIdentifier > subcellMap; // given set of nodes in cellTopo, what subcell is it?)
      
      for (unsigned d=1; d<=sideDim; d++) { // only things of dimension >= 1 will have subcells
        unsigned subcellCount = cellTopo.getSubcellCount(d);
        for (unsigned subcellOrdinal=0; subcellOrdinal<subcellCount; subcellOrdinal++) {
          
          set<NodeOrdinal> nodes;
          unsigned nodeCount = cellTopo.getNodeCount(d, subcellOrdinal);
          for (NodeOrdinal subcNode=0; subcNode<nodeCount; subcNode++) {
            nodes.insert(cellTopo.getNodeMap(d, subcellOrdinal, subcNode));
          }
          SubcellIdentifier subcell = make_pair(d, subcellOrdinal);
          subcellMap[nodes] = subcell;
          
          shards::CellTopology subcellTopo = cellTopo.getCellTopologyData(d, subcellOrdinal);
          // now, go over all the subsubcells, and look them up...
          for (unsigned subsubcellDim=0; subsubcellDim<d; subsubcellDim++) {
            unsigned subsubcellCount = subcellTopo.getSubcellCount(subsubcellDim);
            for (unsigned subsubcellOrdinal=0; subsubcellOrdinal<subsubcellCount; subsubcellOrdinal++) {
              SubSubcellIdentifier subsubcell = make_pair(subsubcellDim,subsubcellOrdinal);
              if (subsubcellDim==0) { // treat vertices separately
                ordinalMap[subcell][subsubcell] = cellTopo.getNodeMap(subcell.first, subcell.second, subsubcellOrdinal);
                continue;
              }
              unsigned nodeCount = subcellTopo.getNodeCount(subsubcellDim, subsubcellOrdinal);
              set<NodeOrdinal> subcellNodes; // NodeOrdinals index into cellTopo, though!
              for (NodeOrdinal subsubcNode=0; subsubcNode<nodeCount; subsubcNode++) {
                NodeOrdinal subcNode = subcellTopo.getNodeMap(subsubcellDim, subsubcellOrdinal, subsubcNode);
                NodeOrdinal node = cellTopo.getNodeMap(d, subcellOrdinal, subcNode);
                subcellNodes.insert(node);
              }

              SubcellIdentifier subsubcellInCellTopo = subcellMap[subcellNodes];
              ordinalMap[ subcell ][ subsubcell ] = subsubcellInCellTopo.second;
              cout << "ordinalMap( (" << subcell.first << "," << subcell.second << "), (" << subsubcell.first << "," << subsubcell.second << ")) ";
              cout << " ---> " << subsubcellInCellTopo.second << endl;
            }
          }
        }
      }
      ordinalMaps[key] = ordinalMap;
    }
    SubcellIdentifier subcell = make_pair(subcdim, subcord);
    SubSubcellIdentifier subsubcell = make_pair(subsubcdim, subsubcord);
    return ordinalMaps[key][subcell][subsubcell];
  }
};

#endif
