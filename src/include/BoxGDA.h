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
//  Limitations: uniform polynomial order (no p-refinement); no continuous traces.
//

#include <map>
#include <vector>

#include "BasisCache.h"
#include "BoxMeshTopology.h"
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
    };
    struct CellMap
    {
      std::vector<GlobalIndexType> globalIndices; // all global dofs seen by the cell, no repeats
      std::vector<std::vector<std::pair<int,double>>> rows; // per local trial dof: (position in globalIndices, weight)
      std::map<std::pair<int,int>, BasisMap> basisMaps;     // key: (varID, sideOrdinal), -1 for volume
    };

    BoxMeshTopology* _boxTopology;
    std::vector<VarPtr> _fieldVars, _fluxVars;
    std::vector<int> _fieldOffsets, _fluxOffsets, _fluxDegrees;
    int _fieldDofsPerCell = 0, _fluxDofsPerPatch = 0;
    bool _allowPruning = false;

    std::vector<GlobalIndexType> _fieldDofOffsetForCell;  // indexed by cell ID; -1 for inactive cells
    std::vector<GlobalIndexType> _fluxDofOffsetForPatch;
    std::vector<GlobalIndexType> _partitionDofOffsets;    // size numPartitions + 1
    GlobalIndexType _myFieldDofCount = 0, _myFluxDofCount = 0;
    std::map<GlobalIndexType, CellMap> _cellMaps;
    BasisCachePtr _basisCache, _parentBasisCache;

    void initializeVariables();
    const CellMap &cellMap(GlobalIndexType cellID);
    void patchBasisValues(const BoxForest::Patch &patch, int degree, const double *x, std::vector<double> &values) const;
  public:
    BoxGDA(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
           std::vector<int> initialH1OrderTrial, int testOrderEnhancement);

    GlobalDofAssignmentPtr deepCopy() override;
    void didChangePartitionPolicy() override;
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
     is bisected only in the directions in which it is coarser than its counterpart (plus whatever the phase-space
     rule for configuration directions requires).  On return, cellIDMap maps meshFrom's boundary cells to meshTo's, and
     sideOrdinalMap has the single entry (side ordinal in meshFrom, side ordinal in meshTo).  MPI-collective.
     */
    static void hRefineToMatchAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                         MeshPtr meshFrom, MeshPtr meshTo, int direction);

    //! A Mesh on a box topology, with a BoxGDA.
    static MeshPtr mesh(Teuchos::RCP<BoxMeshTopology> topology, BFPtr bf, int H1Order, int delta_k,
                        MeshPartitionPolicyPtr partitionPolicy = Teuchos::null, Teuchos_CommPtr Comm = Teuchos::null,
                        std::map<int,int> trialOrderEnhancements = std::map<int,int>(),
                        std::map<int,int> testOrderEnhancements = std::map<int,int>());
  };

  /**
   \brief  Keeps a configuration-space mesh at least as fine as a phase-space mesh on a BoxMeshTopology.

   When phase-space cells are bisected in the configuration directions (which must be all of them together, so that
   a phase-space cell's projection stays a cell of the configuration mesh's refinement tree), the configuration cells
   under them that are not yet that fine are refined (regular refinement), and one-irregularity of the configuration
   mesh is then enforced.  Bisection in velocity directions alone leaves the configuration mesh untouched.
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
