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

#include "CellTopology.h"

#include "CamelliaDebugUtility.h"

class CamelliaCellTools {
public:
  static CellTopoPtr cellTopoForKey(Camellia::CellTopologyKey key);
  
  static CellTopoPtrLegacy cellTopoForKey(unsigned key);
  
  static int getSideCount(const shards::CellTopology &cellTopo); // unlike shards itself, defines vertices as sides for Line topo
  
  static void refCellNodesForTopology(FieldContainer<double> &cellNodes, const shards::CellTopology &cellTopo, unsigned permutation = 0); // 0 permutation is the identity

  static void refCellNodesForTopology(FieldContainer<double> &cellNodes, CellTopoPtr cellTopo, unsigned permutation = 0); // 0 permutation is the identity

  static unsigned permutationMatchingOrder( CellTopoPtr cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder);

  static unsigned permutationMatchingOrder( const shards::CellTopology &cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder);

  static unsigned permutationComposition( CellTopoPtr cellTopo, unsigned a_permutation, unsigned b_permutation );
  
  static unsigned permutationComposition( const shards::CellTopology &cellTopo, unsigned a_permutation, unsigned b_permutation );

  static unsigned permutationInverse( CellTopoPtr cellTopo, unsigned permutation );

  static unsigned permutationInverse( const shards::CellTopology &cellTopo, unsigned permutation );
  
  //! Take refPoints on reference cell, take as physical nodes the specified permutation of the reference cell points.  Permuted points are then the physical points mapped.
  static void permutedReferenceCellPoints(const shards::CellTopology &cellTopo, unsigned permutation, const FieldContainer<double> &refPoints, FieldContainer<double> &permutedPoints);

  //! Take refPoints on reference cell, take as physical nodes the specified permutation of the reference cell points.  Permuted points are then the physical points mapped.
  static void permutedReferenceCellPoints(CellTopoPtr cellTopo, unsigned permutation, const FieldContainer<double> &refPoints, FieldContainer<double> &permutedPoints);
  
  //! Computes the Jacobian matrix DF of the reference-to-physical frame map
  static void setJacobian (FieldContainer<double> &jacobian, const FieldContainer<double> &points, const FieldContainer<double> &cellWorkset, CellTopoPtr cellTopo, const int &whichCell=-1);
  
  // this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
  static unsigned subcellOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord);
  
  // this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
  static unsigned subcellOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord);
  
  static unsigned subcellReverseOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell);
  
  static unsigned subcellReverseOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell);
  
  // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
  static void mapToReferenceFrameInitGuess(       FieldContainer<double>  &        refPoints,
                                           const FieldContainer<double>  &        initGuess,
                                           const FieldContainer<double>  &        physPoints,
                                           MeshTopologyPtr meshTopo, IndexType cellID, int cubatureDegree);

  // ! calls Intrepid's CellTools<double> when cellTopo is a non-tensorial topology
  static void mapToPhysicalFrame(FieldContainer<double>       &         physPoints,
                                 const FieldContainer<double> &         refPoints,
                                 const FieldContainer<double> &         cellWorkset,
                                 CellTopoPtr                            cellTopo,
                                 const int                    &         whichCell = -1);
  
  // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
  static void mapToReferenceFrame(      FieldContainer<double>      &        refPoints,
                                  const FieldContainer<double>      &        physPoints,
                                  MeshTopologyPtr meshTopo, IndexType cellID, int cubatureDegree);
  
  static void mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
                                    const FieldContainer<double> &paramPoints,
                                    const int                     subcellDim,
                                    const int                     subcellOrd,
                                    const shards::CellTopology   &parentCell);
  
  static void mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
                                    const FieldContainer<double> &paramPoints,
                                    const int                     subcellDim,
                                    const int                     subcellOrd,
                                    CellTopoPtr                   parentCell);
  
  static string entityTypeString(unsigned entityDimension); // vertex, edge, face, solid, hypersolid
};


#endif