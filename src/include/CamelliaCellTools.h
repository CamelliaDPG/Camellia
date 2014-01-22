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

#include "Mesh.h"
#include "MeshTopology.h"

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
        break;
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
//              cout << "ordinalMap( (" << subcell.first << "," << subcell.second << "), (" << subsubcell.first << "," << subsubcell.second << ") ) ";
//              cout << " ---> " << subsubcellInCellTopo.second << endl;
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
  
  // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
  static void mapToReferenceFrameInitGuess(       FieldContainer<double>  &        refPoints,
                                           const FieldContainer<double>  &        initGuess,
                                           const FieldContainer<double>  &        physPoints,
                                           MeshPtr mesh, int cellID)
  {
    ElementPtr elem = mesh->getElement(cellID);
    int spaceDim  = elem->elementType()->cellTopoPtr->getDimension();
    int numPoints;
    int numCells=1;
    
    // Temp arrays for Newton iterates and Jacobians. Resize according to rank of ref. point array
    FieldContainer<double> xOld;
    FieldContainer<double> xTem;
    FieldContainer<double> error;
    FieldContainer<double> cellCenter(spaceDim);
    
    // Default: map (C,P,D) array of physical pt. sets to (C,P,D) array. Requires (C,P,D) temp arrays and (C,P,D,D) Jacobians.
    numPoints = physPoints.dimension(1);
    xOld.resize(numCells, numPoints, spaceDim);
    xTem.resize(numCells, numPoints, spaceDim);
    error.resize(numCells,numPoints);
    // Set initial guess to xOld
    for(int c = 0; c < numCells; c++){
      for(int p = 0; p < numPoints; p++){
        for(int d = 0; d < spaceDim; d++){
          xOld(c, p, d) = initGuess(c, p, d);
        }// d
      }// p
    }// c
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    // Newton method to solve the equation F(refPoints) - physPoints = 0:
    // refPoints = xOld - DF^{-1}(xOld)*(F(xOld) - physPoints) = xOld + DF^{-1}(xOld)*(physPoints - F(xOld))
    for(int iter = 0; iter < INTREPID_MAX_NEWTON; ++iter) {
      
      // compute Jacobians at the old iterates and their inverses.
      xOld.resize(numPoints,spaceDim); // BasisCache expects (P,D) sizing...
      basisCache->setRefCellPoints(xOld);
      xOld.resize(numCells,numPoints,spaceDim);
      
      // The Newton step.
      xTem = basisCache->getPhysicalCubaturePoints();                    // xTem <- F(xOld)
      RealSpaceTools<double>::subtract( xTem, physPoints, xTem );        // xTem <- physPoints - F(xOld)
      RealSpaceTools<double>::matvec( refPoints, basisCache->getJacobianInv(), xTem);        // refPoints <- DF^{-1}( physPoints - F(xOld) )
      RealSpaceTools<double>::add( refPoints, xOld );                    // refPoints <- DF^{-1}( physPoints - F(xOld) ) + xOld
      
      // l2 error (Euclidean distance) between old and new iterates: |xOld - xNew|
      RealSpaceTools<double>::subtract( xTem, xOld, refPoints );
      RealSpaceTools<double>::vectorNorm( error, xTem, NORM_TWO );
      
      // Average L2 error for a multiple sets of physical points: error is rank-2 (C,P) array
      double totalError;
      FieldContainer<double> cellWiseError(numCells);
      // error(C,P) -> cellWiseError(P)
      RealSpaceTools<double>::vectorNorm( cellWiseError, error, NORM_ONE );
      totalError = RealSpaceTools<double>::vectorNorm( cellWiseError, NORM_ONE );
      
      // Stopping criterion:
      if (totalError < INTREPID_TOL) {
        break;
      }
      else if ( iter > INTREPID_MAX_NEWTON) {
        INTREPID_VALIDATE(std::cout << " CamelliaCellTools::mapToReferenceFrameInitGuess failed to converge to desired tolerance within "
                          << INTREPID_MAX_NEWTON  << " iterations\n" );
        break;
      }
      
      // initialize next Newton step
      xOld = refPoints;
    } // for(iter)
  }
  
  // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
  static void mapToReferenceFrame(          FieldContainer<double>      &        refPoints,
                                  const FieldContainer<double>      &        physPoints,
                                  MeshPtr mesh, int cellID)
  {
    ElementPtr elem = mesh->getElement(cellID);
    shards::CellTopology cellTopo = *(elem->elementType()->cellTopoPtr);
    int spaceDim  = cellTopo.getDimension();
    int numPoints;
    int numCells;
    
    // Define initial guesses to be  the Cell centers of the reference cell topology
    FieldContainer<double> cellCenter(spaceDim);
    switch( cellTopo.getKey() ){
        // Standard Base topologies (number of cellWorkset = number of vertices)
      case shards::Line<2>::key:
        cellCenter(0) = 0.0;    break;
        
      case shards::Triangle<3>::key:
      case shards::Triangle<6>::key:
        cellCenter(0) = 1./3.;    cellCenter(1) = 1./3.;  break;
        
      case shards::Quadrilateral<4>::key:
      case shards::Quadrilateral<9>::key:
        cellCenter(0) = 0.0;      cellCenter(1) = 0.0;    break;
        
      case shards::Tetrahedron<4>::key:
      case shards::Tetrahedron<10>::key:
      case shards::Tetrahedron<11>::key:
        cellCenter(0) = 1./6.;    cellCenter(1) =  1./6.;    cellCenter(2) =  1./6.;  break;
        
      case shards::Hexahedron<8>::key:
      case shards::Hexahedron<27>::key:
        cellCenter(0) = 0.0;      cellCenter(1) =  0.0;       cellCenter(2) =  0.0;   break;
        
      case shards::Wedge<6>::key:
      case shards::Wedge<18>::key:
        cellCenter(0) = 1./3.;    cellCenter(1) =  1./3.;     cellCenter(2) = 0.0;    break;
        
        // These extended topologies are not used for mapping purposes
      case shards::Quadrilateral<8>::key:
      case shards::Hexahedron<20>::key:
      case shards::Wedge<15>::key:
        TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                                   ">>> ERROR (CamelliaCellTools::mapToReferenceFrame): Cell topology not supported. ");
        break;
        
        // Base and Extended Line, Beam and Shell topologies
      case shards::Line<3>::key:
      case shards::Beam<2>::key:
      case shards::Beam<3>::key:
      case shards::ShellLine<2>::key:
      case shards::ShellLine<3>::key:
      case shards::ShellTriangle<3>::key:
      case shards::ShellTriangle<6>::key:
      case shards::ShellQuadrilateral<4>::key:
      case shards::ShellQuadrilateral<8>::key:
      case shards::ShellQuadrilateral<9>::key:
        TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                                   ">>> ERROR (CamelliaCellTools::mapToReferenceFrame): Cell topology not supported. ");
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                                   ">>> ERROR (CamelliaCellTools::mapToReferenceFrame): Cell topology not supported.");
    }// switch key
    
    // Resize initial guess depending on the rank of the physical points array
    FieldContainer<double> initGuess;
    
    // Default: map (C,P,D) array of physical pt. sets to (C,P,D) array. Requires (C,P,D) initial guess.
    numPoints = physPoints.dimension(1);
    numCells = 1;
    initGuess.resize(numCells, numPoints, spaceDim);
    // Set initial guess:
    for(int c = 0; c < numCells; c++){
      for(int p = 0; p < numPoints; p++){
        for(int d = 0; d < spaceDim; d++){
          initGuess(c, p, d) = cellCenter(d);
        }// d
      }// p
    }// c
    
    // Call method with initial guess
    mapToReferenceFrameInitGuess(refPoints, initGuess, physPoints, mesh, cellID);
  }
};

#endif


