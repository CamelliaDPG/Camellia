//
//  CamelliaCellTools.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 6/6/14.
//
//

#include "CamelliaCellTools.h"
#include "BasisCache.h"

#include "MeshTransformationFunction.h"

CellTopoPtrLegacy CamelliaCellTools::cellTopoForKey(unsigned key) {
  static CellTopoPtrLegacy node, line, triangle, quad, tet, hex;
  
  switch (key) {
    case shards::Node::key:
      if (node.get()==NULL) {
        node = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Node >() ));
      }
      return node;
      break;
    case shards::Line<2>::key:
      if (line.get()==NULL) {
        line = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Line<2> >() ));
      }
      return line;
      break;
    case shards::Triangle<3>::key:
      if (triangle.get()==NULL) {
        triangle = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Triangle<3> >() ));
      }
      return triangle;
      break;
    case shards::Quadrilateral<4>::key:
      if (quad.get()==NULL) {
        quad = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >() ));
      }
      return quad;
      break;
    case shards::Tetrahedron<4>::key:
      if (tet.get()==NULL) {
        tet = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Tetrahedron<4> >() ));
      }
      return tet;
      break;
    case shards::Hexahedron<8>::key:
      if (hex.get()==NULL) {
        hex = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Hexahedron<8> >() ));
      }
      return hex;
      break;
    default:
      cout << "Unhandled CellTopology.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled CellTopology.");
  }
}

string CamelliaCellTools::entityTypeString(unsigned entityDimension) { // vertex, edge, face, solid, hypersolid
  switch (entityDimension) {
    case 0:
      return "vertex";
    case 1:
      return "edge";
    case 2:
      return "face";
    case 3:
      return "solid";
    case 4:
      return "hypersolid";
    default:
      return "unknown entity type";
  }
}

int CamelliaCellTools::getSideCount(const shards::CellTopology &cellTopo) {
  // unlike shards itself, defines vertices as sides for Line topo
  return (cellTopo.getDimension() > 1) ? cellTopo.getSideCount() : cellTopo.getVertexCount();
}

void CamelliaCellTools::refCellNodesForTopology(FieldContainer<double> &cellNodes, const shards::CellTopology &cellTopo, unsigned permutation) { // 0 permutation is the identity
  if (cellNodes.dimension(0) != cellTopo.getNodeCount()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
  }
  if (cellNodes.dimension(1) != cellTopo.getDimension()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
  }
  unsigned cellKey = cellTopo.getKey();
  switch (cellKey) {
    case shards::Node::key:
      break;
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
      cout << "Unhandled CellTopology.\n";
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

void CamelliaCellTools::refCellNodesForTopology(FieldContainer<double> &cellNodes, CellTopoPtr cellTopo, unsigned permutation) {
  if ((cellNodes.dimension(0) != cellTopo->getNodeCount()) && (cellTopo->getDimension() != 0) ) { // shards and Camellia disagree on the node count for points (0 vs. 1), so we accept either as dimensions for cellNodes, which is a size 0 container in any case...
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
  }
  if (cellNodes.dimension(1) != cellTopo->getDimension()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
  }
  
  if (cellTopo->getTensorialDegree() == 0) {
    // then the other variant of refCellNodesForTopology will do the trick
    refCellNodesForTopology(cellNodes, cellTopo->getShardsTopology(), permutation);
    return;
  }
  
  shards::CellTopology shardsTopology = cellTopo->getShardsTopology();
  FieldContainer<double> shardsCellNodes(shardsTopology.getNodeCount(), shardsTopology.getDimension());
  
  refCellNodesForTopology(shardsCellNodes, shardsTopology);
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refCellNodesForTopology does not yet support tensorial degree > 0.");
}

unsigned CamelliaCellTools::permutationMatchingOrder( CellTopoPtr cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder) {
  if (cellTopo->getTensorialDegree() == 0) {
    return permutationMatchingOrder(cellTopo->getShardsTopology(), fromOrder, toOrder);
  } else {
    cout << "CamelliaCellTools::permutationMatchingOrder() does not yet support tensorial degree > 0.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::permutationMatchingOrder() does not yet support tensorial degree > 0.");
  }
}

unsigned CamelliaCellTools::permutationMatchingOrder( const shards::CellTopology &cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder) {
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
  cout << "No matching permutation found.\n";
  Camellia::print("fromOrder", fromOrder);
  Camellia::print("toOrder", toOrder);
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching permutation found");
  return permutationCount; // an impossible (out of bounds) answer: this line just to satisfy compilers that warn about missing return values.
}

unsigned CamelliaCellTools::permutationComposition( const shards::CellTopology &cellTopo, unsigned a_permutation, unsigned b_permutation ) {
  // returns the permutation ordinal for a composed with b -- the lookup table is determined in a fairly brute force way (treating CellTopo as a black box), but we just do this once per topology.
  
  typedef unsigned CellTopoKey;
  typedef unsigned Permutation;
  typedef pair<Permutation, Permutation> PermutationPair;
  static map< CellTopoKey, map< PermutationPair, Permutation > > compositionMap;
  
  if (cellTopo.getKey() == shards::Node::key) {
    if ((a_permutation==0) && (b_permutation==0)) {
      return 0;
    }
  }
  
  if (compositionMap.find(cellTopo.getKey()) == compositionMap.end()) { // build lookup table
    int permCount = cellTopo.getNodePermutationCount();
    int nodeCount = cellTopo.getNodeCount();
    vector<unsigned> identityOrder;
    for (unsigned node=0; node<nodeCount; node++) {
      identityOrder.push_back(node);
    }
    for (int i=0; i<permCount; i++) {
      for (int j=0; j<permCount; j++) {
        vector<unsigned> composedOrder(nodeCount);
        PermutationPair ijPair = make_pair(i,j);
        for (unsigned node=0; node<nodeCount; node++) {
          unsigned j_of_node = cellTopo.getNodePermutation(j, node);
          unsigned i_of_j_of_node = cellTopo.getNodePermutation(i, j_of_node);
          composedOrder[node] = i_of_j_of_node;
        }
        compositionMap[cellTopo.getKey()][ijPair] = permutationMatchingOrder(cellTopo, identityOrder, composedOrder);
      }
    }
  }
  PermutationPair abPair = make_pair(a_permutation, b_permutation);
  
  if (compositionMap[cellTopo.getKey()].find(abPair) == compositionMap[cellTopo.getKey()].end()) {
    cout << "Permutation pair not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Permutation pair not found");
  }
  return compositionMap[cellTopo.getKey()][abPair];
}

unsigned CamelliaCellTools::permutationInverse( CellTopoPtr cellTopo, unsigned permutation ) {
  if (cellTopo->getTensorialDegree() == 0) {
    return permutationInverse(cellTopo->getShardsTopology(), permutation);
  } else {
    cout << "CamelliaCellTools::permutationInverse() does not yet support tensorial degree > 0.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::permutationInverse() does not yet support tensorial degree > 0.");
  }
}

unsigned CamelliaCellTools::permutationInverse( const shards::CellTopology &cellTopo, unsigned permutation ) {
  // returns the permutation ordinal for the inverse of this permutation -- the lookup table is determined in a fairly brute force way (treating CellTopo as a black box), but we just do this once per topology.  (CellTopology lets you execute an inverse, but doesn't give any way to determine the ordinal of the inverse.)
  
  typedef unsigned CellTopoKey;
  typedef unsigned Permutation;
  static map< CellTopoKey, map< Permutation, Permutation > > inverseMap;
  
  if (permutation==0) return 0;  // identity
  
  if (inverseMap.find(cellTopo.getKey()) == inverseMap.end()) { // build lookup table
    int permCount = cellTopo.getNodePermutationCount();
    int nodeCount = cellTopo.getNodeCount();
    vector<unsigned> identityOrder;
    for (unsigned node=0; node<nodeCount; node++) {
      identityOrder.push_back(node);
    }
    for (int i=0; i<permCount; i++) {
      vector<unsigned> inverseOrder(nodeCount);
      for (unsigned node=0; node<nodeCount; node++) {
        unsigned i_inverse_of_node = cellTopo.getNodePermutationInverse(i, node);
        inverseOrder[node] = i_inverse_of_node;
      }
      inverseMap[cellTopo.getKey()][i] = permutationMatchingOrder(cellTopo, identityOrder, inverseOrder);
    }
    if (cellTopo.getKey() == shards::Node::key) {
      // for consistency of interface, we treat this a bit differently than shards -- we support the identity permutation
      // (we also consider a Node to have one node, index 0)
      inverseMap[cellTopo.getKey()][0] = 0;
    }
  }
  
  if (inverseMap[cellTopo.getKey()].find(permutation) == inverseMap[cellTopo.getKey()].end()) {
    cout << "Permutation not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Permutation not found");
  }
  
  //    // debugging line added because I suspect that in our tests we aren't running into a case where we have a permutation which is not its own inverse...
  //    if (inverseMap[cellTopo.getKey()][permutation] != permutation) {
  //      cout << "permutation encountered which is not its own inverse.\n";
  //    }
  
  return inverseMap[cellTopo.getKey()][permutation];
}

void CamelliaCellTools::permutedReferenceCellPoints(const shards::CellTopology &cellTopo, unsigned int permutation,
                                                    const FieldContainer<double> &refPoints, FieldContainer<double> &permutedPoints) {
  FieldContainer<double> permutedNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(permutedNodes, cellTopo, permutation);
  
  permutedNodes.resize(1,permutedNodes.dimension(0), permutedNodes.dimension(1));
  int whichCell = 0;
  CellTools<double>::mapToPhysicalFrame(permutedPoints,refPoints,permutedNodes,cellTopo, whichCell);
}

unsigned CamelliaCellTools::subcellOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord) {
  if (cellTopo->getTensorialDegree() == 0) {
    return subcellOrdinalMap(cellTopo->getShardsTopology(), subcdim, subcord, subsubcdim, subsubcord);
  } else {
    cout << "CamelliaCellTools::subcellOrdinalMap() does not yet support tensorial degree > 0.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::subcellOrdinalMap() does not yet support tensorial degree > 0.");
  }
}

// this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
unsigned CamelliaCellTools::subcellOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord) {
  // maps from a subcell's ordering of its subcells (the sub-subcells) to the cell topology's ordering of those subcells.
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
  
  if (subsubcdim==subcdim) {
    if (subsubcord==0) { // i.e. the "subsubcell" is really just the subcell
      return subcord;
    } else {
      cout << "request for subsubcell of the same dimension as subcell, but with subsubcord > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "request for subsubcell of the same dimension as subcell, but with subsubcord > 0.");
    }
  }
  
  if (subcdim==cellTopo.getDimension()) {
    if (subcord==0) { // i.e. the subcell is the cell itself
      return subsubcord;
    } else {
      cout << "request for subcell of the same dimension as cell, but with subsubcord > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "request for subcell of the same dimension as cell, but with subsubcord > 0.");
    }
  }
  
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
  if (ordinalMaps[key][subcell].find(subsubcell) != ordinalMaps[key][subcell].end()) {
    return ordinalMaps[key][subcell][subsubcell];
  } else {
    cout << "For topology " << cellTopo.getName() << " and subcell " << subcord << " of dim " << subcdim;
    cout << ", subsubcell " << subsubcord << " of dim " << subsubcdim << " not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subsubcell not found");
    return -1; // NOT FOUND
  }
}

unsigned CamelliaCellTools::subcellReverseOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell) {
  // looks for the ordinal of a sub-sub-cell in the subcell
  const shards::CellTopology subcellTopo = cellTopo.getCellTopologyData(subcdim, subcord);
  int subsubcCount = subcellTopo.getSubcellCount(subsubcdim);
  //    cout << "For cellTopo " << cellTopo.getName() << ", subcell dim " << subcdim << ", ordinal " << subcord;
  //    cout << ", and subsubcdim " << subsubcdim << ":\n";
  for (int subsubcOrdinal = 0; subsubcOrdinal < subsubcCount; subsubcOrdinal++) {
    unsigned mapped_subsubcOrdinal = subcellOrdinalMap(cellTopo, subcdim, subcord, subsubcdim, subsubcOrdinal);
    //      cout << "subsubcOrdinal " << subsubcOrdinal << " --> subcord " << mapped_subsubcOrdinal << endl;
    if (mapped_subsubcOrdinal == subsubcordInCell) {
      return subsubcOrdinal;
    }
  }
  cout << "ERROR: subcell " << subsubcordInCell << " not found in subcellReverseOrdinalMap.\n";
  cout << "For topology " << cellTopo.getName() << ", looking for subcell of dimension " << subsubcdim << " with ordinal " << subsubcordInCell << " in cell.\n";
  cout << "Looking in subcell of dimension " << subcdim << " with ordinal " << subcord << ".\n";
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subcell not found in subcellReverseOrdinalMap.");
  return -1; // NOT FOUND
}

// copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
void CamelliaCellTools::mapToReferenceFrameInitGuess(       FieldContainer<double>  &        refPoints,
                                                     const FieldContainer<double>  &        initGuess,
                                                     const FieldContainer<double>  &        physPoints,
                                                     MeshTopologyPtr meshTopo, IndexType cellID, int cubatureDegree)
{
  CellPtr cell = meshTopo->getCell(cellID);
  int spaceDim  = meshTopo->getSpaceDim();
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
  
  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(*cell->topology(), cubatureDegree);
  
  if (meshTopo->transformationFunction().get() != NULL) {
    FunctionPtr transformFunction = meshTopo->transformationFunction();
    basisCache->setTransformationFunction(transformFunction, true);
  }
  std::vector<GlobalIndexType> cellIDs;
  cellIDs.push_back(cellID);
  bool includeCellDimension = true;
  basisCache->setPhysicalCellNodes(meshTopo->physicalCellNodesForCell(cellID, includeCellDimension), cellIDs, false);
  
//  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
  
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
void CamelliaCellTools::mapToReferenceFrame(          FieldContainer<double>      &        refPoints,
                                            const FieldContainer<double>      &        physPoints,
                                            MeshTopologyPtr meshTopo, IndexType cellID, int cubatureDegree)
{
  CellPtr cell = meshTopo->getCell(cellID);
  CellTopoPtrLegacy cellTopo = cell->topology();
  int spaceDim  = cellTopo->getDimension();
  int numPoints;
  int numCells;
  
  // Define initial guesses to be  the Cell centers of the reference cell topology
  FieldContainer<double> cellCenter(spaceDim);
  switch( cellTopo->getKey() ){
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
  mapToReferenceFrameInitGuess(refPoints, initGuess, physPoints, meshTopo, cellID, cubatureDegree);
}

void CamelliaCellTools::mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
                                              const FieldContainer<double> &paramPoints,
                                              const int                     subcellDim,
                                              const int                     subcellOrd,
                                              CellTopoPtr                  &parentCell) {
  if (parentCell->getTensorialDegree() == 0) {
    mapToReferenceSubcell(refSubcellPoints, paramPoints, subcellDim, subcellOrd, parentCell->getShardsTopology());
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::mapToReferenceSubcell does not yet support tensorial degree > 0.");
  }
}

void CamelliaCellTools::mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
                                              const FieldContainer<double> &paramPoints,
                                              const int                     subcellDim,
                                              const int                     subcellOrd,
                                              const shards::CellTopology   &parentCell) {
  // for cells that Intrepid's CellTools supports, we just use that
  int cellDim = parentCell.getDimension();
  if ((subcellDim > 0) && ((cellDim == 2) || (cellDim == 3)) ) {
    CellTools<double>::mapToReferenceSubcell(refSubcellPoints, paramPoints, subcellDim, subcellOrd, parentCell);
  } else if (subcellDim == 0) {
    FieldContainer<double> refCellNodes(parentCell.getNodeCount(),cellDim);
    refCellNodesForTopology(refCellNodes,parentCell);
    // neglect paramPoints argument here; assume that refSubcellPoints is appropriately sized
    int numPoints = refSubcellPoints.dimension(0);
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      for (int d=0; d<cellDim; d++) {
        refSubcellPoints(ptIndex,d) = refCellNodes(subcellOrd,d);
      }
    }
  } else {
    // TODO: add support for 4D elements.
    cout << "CamelliaCellTools::mapToReferenceSubcell -- unsupported arguments.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::mapToReferenceSubcell -- unsupported arguments.");
  }
}