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
  
  /** \brief Generalization of Intrepid::CellTools method; computes subcell parameterizations for subcells up to dimension d-1 for a parent cell of dimension d. (documentation below copied from that in Intrepid::CellTools).  The intent is to support any Camellia::CellTopology that has a reference cell (as of this writing, all supported topologies have reference cells).  The implementation does make the assumption that the mapping from (d-1)-dimensional reference space to d-dimensional reference space is affine.  (Intrepid's implementation is limited to 1D and 2D subcells of 2D and 3D cells.)
   
   Returns array with the coefficients of the parametrization maps for the edges or faces
   of a reference cell topology.
   
   Defines orientation-preserving parametrizations of reference edges and faces of cell
   topologies with reference cells.
   
   Given an edge {V0, V1} of some reference cell, its parametrization is a mapping from
   [-1,1] onto the edge. Parametrization of a triangular face {V0,V1,V2} is mapping from
   the standard 2-simplex {(0,0,0), (1,0,0), (0,1,0)}, embedded in 3D onto that face.
   Parametrization of a quadrilateral face {V0,V1,V2,V3} is mapping from the standard
   2-cube {(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)}, embedded in 3D, onto that face.
   
   This method computes the coefficients of edge and face parametrization maps and stores
   them in static arrays owned by CellTools<Scalar>::getSubcellParametrization method.
   All mappings are affine and orientation-preserving, i.e., they preserve the tangent
   and normal directions implied by the vertex order of the edge or the face relative to
   the reference cell:
   
   \li     the tangent on [-1,1] from -1 in the direction of 1 is mapped to a tangent on edge {V0,V1}
   from V0 in the direction of V1  (the forward direction of the edge determined by its
   start and end vertices)
   
   \li     the normal in the direction of (0,0,1) to the standard 2-simplex {(0,0,0),(1,0,0),(0,1,0)}
   and the standard 2-cube {(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)} is mapped to a normal
   on {V0,V1,V2} and {V0,V1,V2,V3}, determined according to the right-hand rule
   (see http://mathworld.wolfram.com/Right-HandRule.html for definition of right-hand rule
   and Section \ref Section sec_cell_topology_subcell_map for further details).
   
   Because faces of all reference cells supported in Intrepid are affine images of either
   the standard 2-simplex or the standard 2-cube, the coordinate functions of the respective
   parmetrization maps are linear polynomials in the parameter variables (u,v), i.e., they
   are of the form \c F_i(u,v)=C_0(i)+C_1(i)u+C_2(i)v;  \c 0<=i<3 (face parametrizations
   are supported only for 3D cells, thus parametrization maps have 3 coordinate functions).
   As a result, application of these maps is independent of the face type which is convenient
   for cells such as Wedge or Pyramid that have both types of faces. Also, coefficients of
   coordinate functions for all faces can be stored together in the same array.
   
   \param  subcellDim        [in]  - dimension of subcells whose parametrization map is returned
   \param  parentCell        [in]  - topology of the reference cell owning the subcells
   
   \return FieldContainer<double> with the coefficients of the parametrization map for all subcells
   of the specified dimension.
   */
  static const FieldContainer<double>& getSubcellParametrization(const int subcellDim, CellTopoPtr parentCell);
  
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