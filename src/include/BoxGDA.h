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

    //! What the dof layout needs to know about one polynomial order (H^1 order of the trial space)
    struct OrderData
    {
      ElementTypePtr elemType;
      int fieldDofs = 0, fluxDofs = 0;            // per cell, per patch
      std::vector<int> fieldOffsets, fluxOffsets, fluxDegrees;
      // For each flux variable, the multi-indices (one Legendre degree per tangential direction) of its patch basis
      std::vector<std::vector<std::vector<int>>> fluxIndexSets;
      BasisCachePtr basisCache, parentBasisCache;
    };
    BoxMeshTopology* _boxTopology;
    std::vector<VarPtr> _fieldVars, _fluxVars;
    std::map<int, OrderData> _orderData;                  // key: H^1 order
    std::vector<int> _patchOrder;                         // H^1 order of each patch: the lowest of its faces' cells
    std::set<std::pair<int,int>> _verifiedFluxSpaces;     // (cell order, patch order) pairs checked in cellMap()
    bool _allowPruning = false;

    std::vector<GlobalIndexType> _fieldDofOffsetForCell;  // indexed by cell ID; -1 for inactive cells
    std::vector<GlobalIndexType> _fluxDofOffsetForPatch;
    std::vector<GlobalIndexType> _partitionDofOffsets;    // size numPartitions + 1
    GlobalIndexType _myFieldDofCount = 0, _myFluxDofCount = 0;
    std::map<GlobalIndexType, CellMap> _cellMaps;

    //! The order data for the cell's order, computed (from that cell) the first time the order is seen
    const OrderData &orderDataForCell(GlobalIndexType cellID);
    const OrderData &orderData(int H1Order) const;
    int cellOrder(GlobalIndexType cellID);
    const CellMap &cellMap(GlobalIndexType cellID);
    void patchBasisValues(const BoxForest::Patch &patch, const OrderData &patchData, int fluxOrdinal, const double *x, std::vector<double> &values) const;
  public:
    BoxGDA(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
           std::vector<int> initialH1OrderTrial, int testOrderEnhancement);

    GlobalDofAssignmentPtr deepCopy() override;
    void didChangePartitionPolicy() override;
    //! children take their parent's polynomial order
    void didHRefine(const std::set<GlobalIndexType> &parentCellIDs) override;
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
      std::map<GlobalIndexType,int> pIncrements;   // cells to raise, with the number of orders to add
    };
    /**
     \brief  hp-refinement: turns the directions in which marked cells should be refined into h- and p-refinements.
     Each marked cell is bisected in those of its directions in which half its width is at least minWidths[k] (0: no
     limit).  A marked cell that may be bisected in none of them is raised by one order instead, up to maxH1Order.
     Then cells are raised wherever needed, transitively, for face neighbours' orders to differ by at most one.
     (Children take their parent's order, so the rule holds after the bisections too.)  The topology and the cells'
     orders are replicated, so every rank computes the same result; the masks must be the same on every rank (as
     anisotropicRefinementMasks() returns them).  No communication.
     */
    static HPRefinements hpRefinements(MeshPtr mesh, const std::map<GlobalIndexType,unsigned> &directionMasks,
                                       const std::vector<double> &minWidths, int maxH1Order);
    //! The order increments that raise the given cells to (at least) the given orders, together with those that raise
    //! other cells, transitively, so that face neighbours' orders differ by at most one.  No communication.
    static std::map<GlobalIndexType,int> orderIncrementsWithHalo(MeshPtr mesh, std::map<GlobalIndexType,int> targetOrders);
    //! Applies hpRefinements(): the p-refinements, then the h-refinements (whose children inherit the new orders).
    static void hpRefine(MeshPtr mesh, const HPRefinements &refinements, bool repartitionAndRebuild = true);

    //! For each given cell, every direction except those normal to its sides carrying the tag: the directions in which
    //! refinement improves the resolution of boundary data on those sides.
    static std::map<GlobalIndexType,unsigned> masksTangentialToTaggedSides(MeshPtr mesh, const std::string &tagName, int tagID,
                                                                           const std::vector<GlobalIndexType> &cellIDs);

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
