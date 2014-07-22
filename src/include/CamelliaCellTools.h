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

#include "CamelliaDebugUtility.h"

class CamelliaCellTools {
public:
  static CellTopoPtr cellTopoForKey(unsigned key);
  
  static int getSideCount(const shards::CellTopology &cellTopo); // unlike shards itself, defines vertices as sides for Line topo
  
  static void refCellNodesForTopology(FieldContainer<double> &cellNodes, const shards::CellTopology &cellTopo, unsigned permutation = 0); // 0 permutation is the identity
  
  static unsigned permutationMatchingOrder( const shards::CellTopology &cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder);
  
  static unsigned permutationComposition( const shards::CellTopology &cellTopo, unsigned a_permutation, unsigned b_permutation );
  
  static unsigned permutationInverse( const shards::CellTopology &cellTopo, unsigned permutation );
  
  // this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
  static unsigned subcellOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord);
  
  static unsigned subcellReverseOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell);
  
  // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
  static void mapToReferenceFrameInitGuess(       FieldContainer<double>  &        refPoints,
                                           const FieldContainer<double>  &        initGuess,
                                           const FieldContainer<double>  &        physPoints,
                                           MeshPtr mesh, int cellID);
  
  // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
  static void mapToReferenceFrame(          FieldContainer<double>      &        refPoints,
                                  const FieldContainer<double>      &        physPoints,
                                  MeshPtr mesh, int cellID);
  
  static void mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
                                    const FieldContainer<double> &paramPoints,
                                    const int                     subcellDim,
                                    const int                     subcellOrd,
                                    const shards::CellTopology   &parentCell);
  
  static string entityTypeString(unsigned entityDimension); // vertex, edge, face, solid, hypersolid
};


#endif