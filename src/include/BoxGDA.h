#ifndef Camellia_BoxGDA_h
#define Camellia_BoxGDA_h

//
//  BoxGDA.h
//  Camellia
//
//  GlobalDofAssignment for meshes on a BoxMeshTopology, for trial spaces made of L2 fields and L2 fluxes.
//  See notes/anisotropic-refinement-design.md.
//
//  Degrees of freedom: each field variable has its volume basis on every active cell; each flux variable has one
//  tensor-Legendre polynomial per trace patch of the BoxForest skeleton (minimum rule: a single-valued flux that is a
//  polynomial on every cell face).  A cell's side coefficients are the L2 projection of its patch's polynomial onto
//  the side basis, which is exact.
//
//  Ownership: a rank owns the field dofs of its cells and the flux dofs of the patches whose first face belongs to
//  one of its cells.  Global numbers are contiguous per rank.  Because the topology is replicated, every rank
//  computes the whole numbering without communication.
//
//  The flux space on a side may be any space spanned by products of Legendre polynomials whose multi-indices form a
//  downward-closed set: the full tensor-product space Q_p, or Camellia's serendipity spaces.  Such a space is
//  invariant under the axis-aligned affine maps between a patch and the faces in it, which is what makes the patch
//  polynomial representable on every face.  The index set is found numerically from the side basis.
//
//  Polynomial order may vary from cell to cell (p-refinement).  A patch's flux has the lowest order among the cells
//  whose faces make it up (the minimum rule); a higher-order cell's side basis contains it.
//
//  Limitations: no continuous traces.
//

#include <map>
#include <vector>

#include "BasisCache.h"
#include "BoxMeshTopology.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "RefinementObserver.h"

namespace Camellia
{
  class BoxGDA : public GlobalDofAssignment
  {
    //! map from one basis (a field's volume basis, or a flux's basis on one side) to global dofs
    struct BasisMap
    {
      std::vector<int> localDofIndices;           // in the cell's trial ordering
      std::vector<GlobalIndexType> globalIndices;
      std::vector<double> weights;                // (local x global), column-major; empty means identity
      std::vector<double> sideMass;               // the side basis's mass matrix, when it has more functions than the patch's
    };
    struct CellMap
    {
      std::vector<GlobalIndexType> globalIndices; // all global dofs seen by the cell, no repeats
      std::vector<std::vector<std::pair<int,double>>> rows; // per local trial dof: (position in globalIndices, weight)
      std::map<std::pair<int,int>, BasisMap> basisMaps;     // key: (varID, sideOrdinal), -1 for volume
    };

    //! Multi-indices (one Legendre degree per tangential direction) spanning a flux space on a side
    using IndexSet = std::vector<std::vector<int>>;
    //! What the dof layout needs to know about one polynomial order: H^1 order per direction (all equal unless the
    //! order was raised in some directions only; see pRefineInDirections())
    struct OrderData
    {
      ElementTypePtr elemType;
      int fieldDofs = 0;                          // per cell
      std::vector<int> fieldOffsets;
      // [normal direction][flux variable]: the cell's side basis on sides normal to that direction, as an index set
      std::vector<std::vector<IndexSet>> sideIndexSets;
      BasisCachePtr basisCache, parentBasisCache;
    };
    //! The flux space of a patch: for each flux variable, the intersection of the index sets of its faces' cells (the
    //! minimum rule), which lies in each of their side bases
    struct PatchSpace
    {
      std::vector<IndexSet> indexSets;            // per flux variable
      std::vector<int> degrees, offsets;          // per flux variable: largest index entry; offset within the patch's dofs
      int dofs = 0;
    };
    BoxMeshTopology* _boxTopology;
    std::vector<VarPtr> _fieldVars, _fluxVars;
    std::map<std::vector<int>, OrderData> _orderData;     // key: H^1 order per direction
    std::vector<PatchSpace> _patchSpaces;                 // distinct patch spaces
    std::map<std::vector<IndexSet>, int> _patchSpaceIndex;
    std::vector<int> _patchSpaceForPatch;                 // index into _patchSpaces, per patch
    std::set<std::pair<std::vector<int>,int>> _verifiedFluxSpaces; // (cell order, patch space) pairs checked in cellMap()
    std::map<GlobalIndexType, std::vector<int>> _directionalIncrements; // order raised per direction, beyond the isotropic p-refinements
    std::map<std::vector<int>, ElementTypePtr> _directionalElementTypes;
    bool _allowPruning = false;

    std::vector<GlobalIndexType> _fieldDofOffsetForCell;  // indexed by cell ID; -1 for inactive cells
    std::vector<GlobalIndexType> _fluxDofOffsetForPatch;
    std::vector<GlobalIndexType> _partitionDofOffsets;    // size numPartitions + 1
    GlobalIndexType _myFieldDofCount = 0, _myFluxDofCount = 0;
    std::map<GlobalIndexType, CellMap> _cellMaps;

    //! The order data for the cell's order, computed (from that cell) the first time the order is seen
    const OrderData &orderDataForCell(GlobalIndexType cellID);
    //! H^1 order per direction
    std::vector<int> cellOrder(GlobalIndexType cellID);
    const PatchSpace &patchSpace(int patchOrdinal) const { return _patchSpaces[_patchSpaceForPatch[patchOrdinal]]; }
    const CellMap &cellMap(GlobalIndexType cellID);
    //! products of Legendre polynomials, on the patch's bounding box, for the multi-indices of the index set
    void patchBasisValues(const BoxForest::Patch &patch, const IndexSet &indexSet, int degree, const double *x, std::vector<double> &values) const;
    //! after the cells' orders changed: projects registered solutions' coefficients from the old element types
    void projectSolutionsOntoNewOrders(const std::map<GlobalIndexType, ElementTypePtr> &oldTypes);
  public:
    BoxGDA(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
           std::vector<int> initialH1OrderTrial, int testOrderEnhancement);

    GlobalDofAssignmentPtr deepCopy() override;
    void didChangePartitionPolicy() override;
    //! children take their parent's polynomial order
    void didHRefine(const std::set<GlobalIndexType> &parentCellIDs) override;
    //! true if the cell topology has one tensorial component per direction, so that orders can differ by direction
    bool supportsDirectionalOrders() const;
    //! the isotropic order (GlobalDofAssignment's) plus any raised by pRefineInDirections(); one entry per direction
    //! when the latter are present
    std::vector<int> getH1Order(GlobalIndexType cellID) override;
    ElementTypePtr elementType(GlobalIndexType cellID) override;
    //! the partition's cells with the given element type (GlobalDofAssignment's version only knows isotropic types)
    std::vector<GlobalIndexType> cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr) override;
    /**
     \brief  Raises cells' orders by the given amount in each direction (anisotropic p-refinement), and projects the
             registered solutions onto the new bases.  Increments the same in every direction are applied as isotropic
             p-refinements.  On meshes whose cells are not tensor products of lines (2D quadrilaterals, 3D hexahedra),
             orders cannot differ by direction: each cell is raised by its largest increment in every direction.
     Does not repartition or rebuild; call Mesh::repartitionAndRebuild() afterwards.  Must be called on every rank with
     the same increments.
     */
    static void pRefineInDirections(MeshPtr mesh, const std::map<GlobalIndexType, std::vector<int>> &increments);
    //! raises the cells' orders, and projects registered solutions onto the new bases (exactly: the spaces are nested)
    void didPRefine(const std::set<GlobalIndexType> &cellIDs, int deltaP) override;

    GlobalIndexType globalDofCount() override;
    std::set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber) override;
    bool isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const override;
    PartitionIndexType partitionForGlobalDofIndex(GlobalIndexType globalDofIndex) override;
    GlobalIndexType numPartitionOwnedGlobalFieldIndices() override;
    GlobalIndexType numPartitionOwnedGlobalFluxIndices() override;
    GlobalIndexType numPartitionOwnedGlobalTraceIndices() override;
    IndexType localDofCount() override;

    using GlobalDofAssignment::interpretLocalData;
    void interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localData,
                            Camellia::FieldContainer<double> &globalData,
                            Camellia::FieldContainer<GlobalIndexType> &globalDofIndices) override;
    void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal,
                                         const Camellia::FieldContainer<double> &basisCoefficients,
                                         Camellia::FieldContainer<double> &globalCoefficients,
                                         Camellia::FieldContainer<GlobalIndexType> &globalDofIndices) override;
    void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients,
                                     const TVectorPtr<double> globalCoefficients) override;
    std::set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) override;
    std::set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal,
                                                              bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime) override;
    std::vector<GlobalIndexType> globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID) override;

    //! Fields: the parent's polynomial restricted to the child.  Fluxes: on child sides that lie on a side of the parent,
    //! the parent's side polynomial restricted to the child side; zero on child sides interior to the parent.
    bool projectParentCoefficientsOntoChildren(GlobalIndexType parentCellID, const Camellia::FieldContainer<double> &parentCoefficients,
                                               const std::vector<GlobalIndexType> &childIDs,
                                               std::vector<Camellia::FieldContainer<double>> &childCoefficients) override;
    void rebuildLookups() override;
    void setAllowMeshTopologyPruning(bool allowMeshTopologyPruning) override;
    bool allowMeshTopologyPruning() override;

    //! The number of trace patches (flux supports) in the mesh.
    IndexType patchCount() const;

    /**
     \brief  The box-mesh counterpart of MeshTools::hRefineToMatchAtBoundary(), for two box meshes with the same root
             cells in every direction but one (e.g. successive time slabs).

     Refines both meshes until each active cell of meshFrom on its lower boundary in the given direction has the same
     extent, in the other directions, as the active cell of meshTo across from it on meshTo's upper boundary.  A cell
     is bisected only in the directions in which it is coarser than its counterpart, and raised to its counterpart's
     order if that is higher (with the halo of orderIncrementsWithHalo()).  On return, cellIDMap maps meshFrom's boundary cells to meshTo's, and
     sideOrdinalMap has the single entry (side ordinal in meshFrom, side ordinal in meshTo).  MPI-collective.
     */
    static void hRefineToMatchAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                         MeshPtr meshFrom, MeshPtr meshTo, int direction);

    /**
     \brief  Meshes for geometric multigrid on a box mesh, coarsest first, ending with fineMesh itself.

     As GMGSolver::meshesForMultigrid() does for other meshes: the root cells at polynomial order kCoarse, then one
     more refinement generation per level at order kCoarse up to the fine mesh's cells, then the fine mesh.  Each
     coarse mesh has its own BoxMeshTopology (a truncated copy of the fine one) and is partitioned so that a coarse
     cell lives on the rank that owns the fine cell at its lower corner.
     */
    static std::vector<MeshPtr> meshesForMultigrid(MeshPtr fineMesh, int kCoarse, int delta_k);

    /**
     \brief  The prolongation from a coarser box mesh (larger cells and/or lower polynomial order) to this GDA's mesh.

     Exact: the coarse space is a subspace of the fine one wherever both are defined.  A fine cell's field
     coefficients are those of the coarse cell's polynomial; a fine patch's flux coefficients are those of the coarse
     patch's polynomial, where the fine patch lies on the coarse mesh's skeleton, and zero where it lies inside a coarse
     cell (the coarse space has no flux there).  Rows follow fineMap, columns coarseMap; any rows beyond the dof count
     (Lagrange constraints) map one to one.
     */
    TMatrixPtr<double> prolongationOperator(MeshPtr coarseMesh, TMapPtr coarseMap, TMapPtr fineMap);

    /**
     \brief  For each of the given active cells, the directions in which to bisect it: those in which f is least
             resolved on the cell.  MPI-collective; each rank evaluates f on the cells it owns.

     On each cell, f is sampled at tensor Gauss points and expanded in tensor Legendre polynomials.  eta_k is the L2
     norm of the part of the expansion in the highest modes of direction k (the top two for p >= 3, the top one
     below that); direction k is selected if
     eta_k >= threshold * max_j eta_j, over the allowed directions (bit k of allowedDirections).  A threshold of 0
     selects every allowed direction; a threshold near 1, only the worst.

     On piecewise-constant cells (p = 0) there are no high modes; there eta_k is the largest jump of f's cell mean
     across the cell's faces normal to direction k.
     This is a cheap indicator of anisotropy based on the solution's own high modes.  It is not the DPG indicator
     based on local residual reduction.
     */
    static std::map<GlobalIndexType,unsigned> anisotropicRefinementMasks(MeshPtr mesh, FunctionPtr f, const std::vector<GlobalIndexType> &cellIDs,
                                                                         double threshold, unsigned allowedDirections = ~0u);

    //! The h- and p-refinements chosen by hpRefinements()
    struct HPRefinements
    {
      std::map<GlobalIndexType,unsigned> hMasks;   // cells to bisect, with their directions
      std::map<GlobalIndexType,int> pIncrements;   // cells to raise isotropically, with the number of orders to add
      std::map<GlobalIndexType, std::vector<int>> pIncrementsInDirections; // cells to raise direction by direction
    };
    /**
     \brief  hp-refinement: turns the directions in which marked cells should be refined into h- and p-refinements.
     Each marked cell is bisected in those of its directions in which half its width is at least minWidths[k] (0: no
     limit).  A marked cell that may be bisected in none of them is raised by one order instead, up to maxH1Order.
     With anisotropicP (on meshes whose cells allow orders that differ by direction; see pRefineInDirections()), the
     choice is made direction by direction instead: each chosen direction in which the cell may be bisected is
     bisected, and each other chosen direction is raised by one order (up to maxH1Order), so a cell may be bisected in
     some directions and raised in others.
     Then cells are raised wherever needed, transitively, for face neighbors' orders to differ by at most one (in
     each direction, with anisotropicP).
     (Children take their parent's order, so the rule holds after the bisections too.)  The topology and the cells'
     orders are replicated, so every rank computes the same result; the masks must be the same on every rank (as
     anisotropicRefinementMasks() returns them).  No communication.
     */
    static HPRefinements hpRefinements(MeshPtr mesh, const std::map<GlobalIndexType,unsigned> &directionMasks,
                                       const std::vector<double> &minWidths, int maxH1Order, bool anisotropicP = false);
    //! The order increments that raise the given cells to (at least) the given orders, together with those that raise
    //! other cells, transitively, so that face neighbors' orders differ by at most one.  No communication.
    static std::map<GlobalIndexType,int> orderIncrementsWithHalo(MeshPtr mesh, std::map<GlobalIndexType,int> targetOrders);
    //! As orderIncrementsWithHalo(), direction by direction: raises cells to the given orders per direction, and others so
    //! that in each direction face neighbors' orders differ by at most one.  No communication.
    static std::map<GlobalIndexType, std::vector<int>> orderIncrementsWithHaloInDirections(MeshPtr mesh,
                                                                                            std::map<GlobalIndexType, std::vector<int>> targetOrders);
    //! The given cells that hpRefinements() would leave alone if marked with the given directions: at the minimum width
    //! in every one of them, and at maxH1Order (in every one of them, with anisotropicP).  No communication.
    static std::set<GlobalIndexType> hpSaturatedCells(MeshPtr mesh, const std::map<GlobalIndexType,unsigned> &directionMasks,
                                                      const std::vector<double> &minWidths, int maxH1Order, bool anisotropicP = false);
    //! An active cell as a location-based record: its box (in the integer coordinates of the root grid) and its order.
    struct CellRecord
    {
      BoxForest::Box box;
      std::vector<int> H1Orders; // per direction
    };
    //! The active cells of a box mesh, as records that can be applied to another mesh with the same root grid.
    static std::vector<CellRecord> cellRecords(MeshPtr mesh);
    /**
     \brief  Refines a box mesh until each recorded box is a union of its active cells, and raises each active cell inside
             a recorded box to at least that box's order (with the halo of orderIncrementsWithHalo()).
     Applied to a mesh that has the recorded mesh's root grid (e.g. the next time slab, whose t root is again root 0) and
     is no finer, it reproduces the recorded mesh.  A cell is bisected only in the directions in which it is coarser
     than a recorded box it overlaps.  No communication beyond the refinements' own.
     */
    static void refineToContain(MeshPtr mesh, const std::vector<CellRecord> &records);
    //! Applies hpRefinements(): the p-refinements, then the h-refinements (whose children inherit the new orders).
    static void hpRefine(MeshPtr mesh, const HPRefinements &refinements, bool repartitionAndRebuild = true);

    //! For each given cell, every direction except those normal to its sides carrying the tag: the directions in which
    //! refinement improves the resolution of boundary data on those sides.
    static std::map<GlobalIndexType,unsigned> masksTangentialToTaggedSides(MeshPtr mesh, const std::string &tagName, int tagID,
                                                                           const std::vector<GlobalIndexType> &cellIDs);

    /**
     \brief  The coefficients, in the side basis of (toCellID, toSide) of toMesh, of the flux variable's function given by
             fromCoefficients in the side basis of (fromCellID, fromSide) of fromMesh: an L2 projection, exact when the
             to-side space contains the from-side space (e.g. the next time slab's cell across the interface, at an
             order at least as high).  The two sides must coincide physically in their tangential extents.
     */
    static Camellia::FieldContainer<double> transferSideCoefficients(MeshPtr fromMesh, GlobalIndexType fromCellID, int fromSide,
                                                                     MeshPtr toMesh, GlobalIndexType toCellID, int toSide,
                                                                     int fromVarID, int toVarID, const Camellia::FieldContainer<double> &fromCoefficients);
    //! A Mesh on a box topology, with a BoxGDA.
    static MeshPtr mesh(Teuchos::RCP<BoxMeshTopology> topology, BFPtr bf, int H1Order, int delta_k,
                        MeshPartitionPolicyPtr partitionPolicy = Teuchos::null, Teuchos_CommPtr Comm = Teuchos::null,
                        std::map<int,int> trialOrderEnhancements = std::map<int,int>(),
                        std::map<int,int> testOrderEnhancements = std::map<int,int>());
  };

  /**
   \brief  Keeps a configuration-space mesh at least as fine as a phase-space mesh on a BoxMeshTopology.

   When phase-space cells are bisected in configuration directions, the active configuration cells under them are
   refined (regular refinement, so in every configuration direction) until they are at least as fine as the
   phase-space children in each direction, and one-irregularity of the configuration mesh is then enforced.
   Bisection in velocity directions alone leaves the configuration mesh untouched.
   */
  class BoxConfigurationMeshUpdater : public RefinementObserver
  {
    MeshPtr _phaseSpaceMesh;       // weak
    MeshPtr _configurationMesh;    // weak
    BoxMeshTopology* _boxTopology;
  public:
    BoxConfigurationMeshUpdater(MeshPtr phaseSpaceMesh, MeshPtr configurationMesh);
    void didHRefineInDirections(const std::map<GlobalIndexType,unsigned> &cellDirectionMasks, bool didRepartitionAndRebuild) override;
    static void registerForUpdates(MeshPtr phaseSpaceMesh, MeshPtr configurationMesh);
  };
}

#endif
