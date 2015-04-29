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

namespace Camellia {
  class CamelliaCellTools {
  public:
    static CellTopoPtr cellTopoForKey(CellTopologyKey key);

    static CellTopoPtrLegacy cellTopoForKey(unsigned key);

    /** \brief  Variation on Intrepid FunctionSpaceTools computeEdgeMeasure and computeFaceMeasure;
     intended to support Camellia CellTopology generally, though unlike computeEdgeMeasure we here
     assume that the subcell over which we integrate is of dimension (d-1) for a d-dimensional parent cell.
     Note that this initial implementation does require orthogonality of the transformations in space and
     time; it is assumed that \f$dF_i/dxi_j = 0\f$ if \f$i\f$ is a spatial dimension and \f$j\f$ is a temporal one, or vice
     versa.

     Returns the weighted integration measures \a <b>outVals</b> with dimensions
     (C,P) used for the computation of side integrals, based on the provided
     cell Jacobian array \a <b>inJac</b> with dimensions (C,P,D,D) and the
     provided integration weights \a <b>inWeights</b> with dimensions (P).

     Returns a rank-2 array (C, P) array such that
     \f[
     \mbox{outVals}(c,p)   =
     \left\|\frac{\partial\Phi_c(\widehat{x}_p)}{\partial u}\times
     \frac{\partial\Phi_c(\widehat{x}_p)}{\partial v}\right\|\omega_{p} \,,
     \f]
     where:
     \li      \f$\{(\widehat{x}_p,\omega_p)\}\f$ is a cubature rule defined on \b reference
     \b side \f$\widehat{\mathcal{F}}\f$, with ordinal \e whichSide relative to the specified parent reference cell;
     \li      \f$ \Phi_c : R \mapsto \mathcal{F} \f$ is parameterization of the physical face
     corresponding to \f$\widehat{\mathcal{F}}\f$; see Section \ref sec_cell_topology_subcell_map.

     \warning
     The user is responsible for providing input arrays with consistent data: the Jacobians
     in \a <b>inJac</b> should be evaluated at integration points on the <b>reference face</b>
     corresponding to the weights in \a <b>inWeights</b>.

     \remark
     Cubature rules on reference faces are defined by a two-step process:
     \li     A cubature rule is defined on the parametrization domain \e R of the face
     (\e R is the standard 2-simplex {(0,0),(1,0),(0,1)} or the standard 2-cube [-1,1] X [-1,1]).
     \li     The points are mapped to a reference face using Intrepid::CellTools::mapToReferenceSubcell

     \remark
     See Intrepid::CellTools::setJacobian for computation of \e DF and
     Intrepid::CellTools::setJacobianDet for computation of its determinant.

     \code
     C - num. integration domains                     dim0 in all input containers
     P - num. integration points                      dim1 in all input containers
     D - spatial dimension                            dim2 and dim3 in Jacobian container
     \endcode

     \param  outVals     [out] - Output array with weighted face measures.
     \param  inJac        [in] - Input array containing cell Jacobians.
     \param  inWeights    [in] - Input integration weights.
     \param  whichSide    [in] - Index of the side subcell relative to the parent cell; defines the domain of integration.
     \param  parentCell   [in] - Parent cell topology.
     */
    static void computeSideMeasure(Intrepid::FieldContainer<double> & weightedMeasure, const Intrepid::FieldContainer<double> &cellJacobian, const Intrepid::FieldContainer<double> &cubWeights,
                            int sideOrdinal, CellTopoPtr parentCell);

    static void getUnitSideNormals(Intrepid::FieldContainer<double> &unitSideNormals, int sideOrdinal, const Intrepid::FieldContainer<double> &inCellJacobian, CellTopoPtr parentCell);

    /** \brief  Computes constant normal vectors to sides of reference cells.  Generalizes Intrepid's CellTools<double>'s version of the same method to apply to Camellia CellTopology parent cells (the present implementation calls Intrepid's method for some of the base shards topologies).  This allows treatment of some cells that are not 2D or 3D (1D and 4D in particular are supported).  The documentation below is largely copied from Intrepid's documentation.

     A side is defined as a subcell of dimension one less than that of its parent cell.
     Therefore, sides of 2D cells are 1-subcells (edges) and sides of 3D cells
     are 2-subcells (faces).

     Returns rank-1 array with dimension (D), D >= 1 such that
     \f[
     {refSideNormal}(*) = \hat{\bf n}_i =
     \left\{\begin{array}{rl}
     \displaystyle
     \left({\partial\hat{\Phi}_i(t)\over\partial t}\right)^{\perp}
     & \mbox{for 2D parent cells} \\[2ex]
     \displaystyle
     {\partial\hat{\Phi}_{i}\over\partial u} \times
     {\partial\hat{\Phi}_{i}\over\partial v}   & \mbox{for 3D parent cells}
     \end{array}\right.
     \f]
     where \f$ (u_1,u_2)^\perp = (u_2, -u_1)\f$, and \f$\hat{\Phi}_i: R \mapsto \hat{\mathcal S}_i\f$
     is the parametrization map of the specified reference side \f$\hat{\mathcal S}_i\f$ given by
     \f[
     \hat{\Phi}_i(u,v) =
     \left\{\begin{array}{rl}
     (\hat{x}(t),\hat{y}(t))                   & \mbox{for 2D parent cells} \\[1ex]
     (\hat{x}(u,v),\hat{y}(u,v),\hat{z}(u,v))  & \mbox{for 3D parent cells}
     \end{array}\right.

     \f]
     For sides of 2D cells \e R=[-1,1] and for sides of 3D cells
     \f[
     R = \left\{\begin{array}{rl}
     \{(0,0),(1,0),(0,1)\}   & \mbox{if $\hat{\mathcal S}_i$ is Triangle} \\[1ex]
     [-1,1]\times [-1,1] & \mbox{if $\hat{\mathcal S}_i$ is Quadrilateral} \,.
     \end{array}\right.
     \f]
     For 3D cells the length of computed side normals is proportional to side area:
     \f[
     |\hat{\bf n}_i | = \left\{\begin{array}{rl}
     2 \mbox{Area}(\hat{\mathcal F}_i) & \mbox{if $\hat{\mathcal F}_i$  is Triangle} \\[1ex]
     \mbox{Area}(\hat{\mathcal F}_i) & \mbox{if $\hat{\mathcal F}_i$ is Quadrilateral} \,.
     \end{array}\right.
     \f]
     For 2D cells the length of computed side normals is proportional to side length:
     \f[
     |\hat{\bf n}_i | = {1\over 2} |\hat{\mathcal F}_i |\,.
     \f]
     Because the sides of all reference cells are always affine images of \e R ,
     the coordinate functions \f$\hat{x},\hat{y},\hat{z}\f$ of the parametrization maps
     are linear and the side normal is a constant vector.

     \param  refSideNormal     [out] - rank-1 array (D) with (constant) side normal
     \param  sideOrd           [in]  - ordinal of the side whose normal is computed
     \param  parentCell        [in]  - cell topology of the parent reference cell
     */
    static void getReferenceSideNormal(Intrepid::FieldContainer<double> &refSideNormal, int sideOrdinal, CellTopoPtr parentCell);

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

     \return Intrepid::FieldContainer<double> with the coefficients of the parametrization map for all subcells
     of the specified dimension.
     */
    static const Intrepid::FieldContainer<double>& getSubcellParametrization(const int subcellDim, CellTopoPtr parentCell);

    static void refCellNodesForTopology(Intrepid::FieldContainer<double> &cellNodes, const shards::CellTopology &cellTopo, unsigned permutation = 0); // 0 permutation is the identity

    static void refCellNodesForTopology(Intrepid::FieldContainer<double> &cellNodes, CellTopoPtr cellTopo, unsigned permutation = 0); // 0 permutation is the identity

    static void refCellNodesForTopology(std::vector< vector<double> > &cellNodes, CellTopoPtr cellTopo, unsigned permutation = 0); // 0 permutation is the identity

    static void pointsVectorFromFC(std::vector< vector<double> > &pointsVector, const Intrepid::FieldContainer<double> &pointsFC);

    static void pointsFCFromVector(Intrepid::FieldContainer<double> &pointsFC, const std::vector< vector<double> > &pointsVector);

    static unsigned permutationMatchingOrder( CellTopoPtr cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder);

    static unsigned permutationMatchingOrder( const shards::CellTopology &cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder);

    static unsigned permutationComposition( CellTopoPtr cellTopo, unsigned a_permutation, unsigned b_permutation );

    static unsigned permutationComposition( const shards::CellTopology &cellTopo, unsigned a_permutation, unsigned b_permutation );

    static unsigned permutationInverse( CellTopoPtr cellTopo, unsigned permutation );

    static unsigned permutationInverse( const shards::CellTopology &cellTopo, unsigned permutation );

    //! Take refPoints on reference cell, take as physical nodes the specified permutation of the reference cell points.  Permuted points are then the physical points mapped.
    static void permutedReferenceCellPoints(const shards::CellTopology &cellTopo, unsigned permutation, const Intrepid::FieldContainer<double> &refPoints, Intrepid::FieldContainer<double> &permutedPoints);

    //! Take refPoints on reference cell, take as physical nodes the specified permutation of the reference cell points.  Permuted points are then the physical points mapped.
    static void permutedReferenceCellPoints(CellTopoPtr cellTopo, unsigned permutation, const Intrepid::FieldContainer<double> &refPoints, Intrepid::FieldContainer<double> &permutedPoints);

    //! Computes the Jacobian matrix DF of the reference-to-physical frame map
    static void setJacobian (Intrepid::FieldContainer<double> &jacobian, const Intrepid::FieldContainer<double> &points, const Intrepid::FieldContainer<double> &cellWorkset, CellTopoPtr cellTopo, const int &whichCell=-1);

    // this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
    static unsigned subcellOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord);

    // this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
    static unsigned subcellOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord);

    static unsigned subcellReverseOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell);

    static unsigned subcellReverseOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell);

    static void getTensorPoints(Intrepid::FieldContainer<double>& tensorPoints, const Intrepid::FieldContainer<double> & spatialPoints,
                                const Intrepid::FieldContainer<double> & temporalPoints);
    
    // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
    static void mapToReferenceFrameInitGuess(      Intrepid::FieldContainer<double>  &        refPoints,
                                             const Intrepid::FieldContainer<double>  &        initGuess,
                                             const Intrepid::FieldContainer<double>  &        physPoints,
                                             MeshTopologyPtr meshTopo, IndexType cellID, int cubatureDegree);
    
    // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry and/or space-time elements
    static void mapToReferenceFrameInitGuess(      Intrepid::FieldContainer<double>  &        refPoints,
                                             const Intrepid::FieldContainer<double>  &        initGuess,
                                             const Intrepid::FieldContainer<double>  &        physPoints,
                                             BasisCachePtr basisCache);

    // ! calls Intrepid's CellTools<double> when cellTopo is a non-tensorial topology
    static void mapToPhysicalFrame(Intrepid::FieldContainer<double>       &         physPoints,
                                   const Intrepid::FieldContainer<double> &         refPoints,
                                   const Intrepid::FieldContainer<double> &         cellWorkset,
                                   CellTopoPtr                            cellTopo,
                                   const int                    &         whichCell = -1);

    // copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
    static void mapToReferenceFrame(      Intrepid::FieldContainer<double>      &        refPoints,
                                    const Intrepid::FieldContainer<double>      &        physPoints,
                                    MeshTopologyPtr meshTopo, IndexType cellID, int cubatureDegree);

    static void mapToReferenceSubcell(Intrepid::FieldContainer<double>       &refSubcellPoints,
                                      const Intrepid::FieldContainer<double> &paramPoints,
                                      const int                     subcellDim,
                                      const int                     subcellOrd,
                                      const shards::CellTopology   &parentCell);

    static void mapToReferenceSubcell(Intrepid::FieldContainer<double>       &refSubcellPoints,
                                      const Intrepid::FieldContainer<double> &paramPoints,
                                      const int                     subcellDim,
                                      const int                     subcellOrd,
                                      CellTopoPtr                   parentCell);

    static string entityTypeString(unsigned entityDimension); // vertex, edge, face, solid, hypersolid
  };
}


#endif
