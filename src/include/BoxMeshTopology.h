#ifndef Camellia_BoxMeshTopology_h
#define Camellia_BoxMeshTopology_h

//
//  BoxMeshTopology.h
//  Camellia
//
//  A MeshTopologyView backed by a BoxForest, so that Mesh, Solution and friends can run on box meshes: hypercube cells
//  of any dimension, bisected in any subset of directions.  See notes/anisotropic-refinement-design.md.
//
//  There is no base MeshTopology and there are no numbered entities: a cell knows its box, and everything else is
//  computed.  The topology is replicated on every MPI rank (Comm() is null); cells are distributed by the Mesh's
//  partition policy as usual.  Cell IDs are BoxForest node indices.
//
//  Use with BoxGDA (BoxGDA.h) through BoxMeshTopology::mesh().
//

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "BoxForest.h"
#include "Cell.h"
#include "CellTopology.h"
#include "MeshTopologyView.h"
#include "PhaseSpaceTopology.h"
#include "TypeDefs.h"

namespace Camellia
{
  class BoxMeshTopology;

  //! Cell subclass used in BoxMeshTopology.
  class BoxCell : public Cell
  {
    const BoxMeshTopology* _boxTopology;
  public:
    BoxCell(CellTopoPtr cellTopo, const std::vector<IndexType> &vertices, const std::vector<std::vector<unsigned>> &subcellPermutations,
            IndexType cellIndex, const BoxMeshTopology* boxTopology);

    //! Sides have no entity numbers; returns an index unique to (cell, side) for sides, and throws for other dimensions.
    IndexType entityIndex(unsigned subcdim, unsigned subcord) override;
    std::vector<IndexType> entitiesOnNeighborInterfaces(unsigned dimensionOfInterest, bool peersOnly, ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    //! The active descendants of this cell that touch the given side, with their side ordinals (the same ordinal).
    std::vector<std::pair<IndexType,unsigned>> getDescendantsForSide(int sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly = true) override;
    //! All active cells sharing part of a side with this cell.
    std::set<GlobalIndexType> getActiveNeighborIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    //! A side can have several neighbours; these return the first active one (and the opposite side ordinal), or -1 on the boundary.
    Teuchos::RCP<Cell> getNeighbor(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    std::pair<GlobalIndexType,unsigned> getNeighborInfo(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    bool isBoundary(unsigned sideOrdinal) override;
    //! Box meshes have no RefinementPatterns; throws.
    RefinementBranch refinementBranch(IndexType ancestralCellID = -1) override;
  };

  class BoxMeshTopology : public MeshTopologyView, public PhaseSpaceTopology
  {
  public:
    //! (lower, upper) physical coordinates of a cell in each direction
    using Extents = std::vector<std::pair<double,double>>;
    //! Selects boundary sides: direction orthogonal to the side, whether it is the upper side, and the cell's extents.
    using SideSelector = std::function<bool(int direction, bool upper, const Extents &cellExtents)>;
  private:
    BoxForest _forest;
    std::vector<std::vector<double>> _rootVertices;
    int _dim;
    CellTopoPtr _cellTopo;
    Camellia::FieldContainer<double> _refNodes;
    std::vector<int> _sideForFace; // BoxForest side (2 * direction + upper) -> side ordinal of _cellTopo
    std::vector<int> _faceForSide;
    std::vector<CellPtr> _cells;   // indexed by BoxForest node
    mutable BoxForest::Skeleton _skeleton;
    mutable bool _skeletonValid = false;
    mutable std::map<IndexType,std::vector<double>> _vertexCache;
    std::map<std::pair<std::string,int>, std::vector<SideSelector>> _sideTags;
    MeshTopologyViewPtr _configurationTopology; // when this is a phase-space mesh
    int _configurationDimensionCount = 0;
    mutable std::map<IndexType,IndexType> _configurationCellForCell;

    CellPtr createCell(IndexType cellIndex);
    //! The configuration cell containing the projection (configuration extents) that is smallest, or at level maxLevel
    CellPtr configurationCellContaining(const Extents &projection, int maxLevel) const;
  public:
    /**
     \param rootVertices  for each direction, the vertex coordinates of the Cartesian root grid
     Cells are quads in 2D, hexahedra in 3D, and Line x ... x Line tensor topologies in higher dimensions.
     */
    BoxMeshTopology(const std::vector<std::vector<double>> &rootVertices);

    // box-specific interface
    const BoxForest &forest() const { return _forest; }
    const BoxForest::Skeleton &skeleton() const;
    CellTopoPtr cellTopology() const { return _cellTopo; }
    int sideOrdinalForFace(int boxForestSide) const { return _sideForFace[boxForestSide]; }
    int faceForSideOrdinal(int sideOrdinal) const { return _faceForSide[sideOrdinal]; }
    double coordinate(int direction, BoxForest::Coord c) const;
    Extents cellExtents(IndexType cellIndex) const;
    //! true if the side lies on the boundary of the domain
    bool sideIsOnBoundary(IndexType cellIndex, unsigned sideOrdinal) const;
    //! active cells sharing part of the side, with positive measure
    std::vector<IndexType> sideNeighbors(IndexType cellIndex, unsigned sideOrdinal) const;

    //! Bisects an active cell in the directions set in directionMask.  Returns the child cell IDs.
    //! (To refine a cell that belongs to a Mesh, call Mesh::hRefine(); it calls this.)
    std::vector<IndexType> refineCell(IndexType cellIndex, unsigned directionMask);
    //! Bisects cells so that face neighbours' levels differ by at most one in each tangential direction.
    //! Returns the cells that were refined, each with its direction mask.
    std::vector<std::pair<IndexType,unsigned>> cellsToRefineForBalance(int maxLevelDifference = 1) const;

    //! The number of refinement generations in the mesh (0 for an unrefined mesh).
    int refinementDepth() const;
    //! A new topology with the same root cells, tags and refinements, except that cells at generation maxDepth are
    //! left unrefined.  (maxDepth >= refinementDepth() gives a full copy.)  Cell IDs in the copy differ from those here
    //! in general; cells correspond by geometry.  Used to build multigrid hierarchies.  The copy has no configuration topology.
    Teuchos::RCP<BoxMeshTopology> copyTruncatedAtDepth(int maxDepth) const;

    //! Tags the boundary sides picked by the selector.  Tags with name DIRICHLET_SET_TAG_NAME are used by BC::addDirichlet(var, tagID, f).
    void applyTag(const std::string &tagName, int tagID, SideSelector selector);

    /**
     \brief  Declares this to be a phase-space mesh whose first configurationDimensionCount directions are those of
             the given configuration-space mesh topology (which must have the same root cells in those directions).

     The configuration mesh must be kept at least as fine as this mesh (see BoxConfigurationMeshUpdater), so that the
     projection of each phase-space cell onto configuration space is tiled by active configuration cells.  A
     phase-space cell may be at different refinement levels in different configuration directions.
     */
    void setConfigurationTopology(MeshTopologyViewPtr configurationTopology, int configurationDimensionCount);
    MeshTopologyViewPtr configurationTopology() const { return _configurationTopology; }
    int configurationDimensionCount() const { return _configurationDimensionCount; }

    // PhaseSpaceTopology interface
    int phaseSpaceVelocityDimensionCount() const override;
    void phaseSpaceVelocityExtents(double &v0, double &v1, IndexType phaseSpaceCellID, int velocityDimension) const override;
    IndexType phaseSpaceConfigurationCellID(IndexType phaseSpaceCellID) const override;
    MeshTopologyViewPtr phaseSpaceConfigurationTopology() const override { return _configurationTopology; }
    std::vector<IndexType> phaseSpaceActiveConfigurationCells(IndexType phaseSpaceCellID) const override;
    //! (lower, upper) in each configuration direction of a cell of the configuration topology (an axis-aligned box)
    Extents configurationCellExtents(IndexType configurationCellID) const;

    // MeshTopologyView interface
    long long approximateMemoryFootprint() const override;
    std::vector<IndexType> cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const override;
    IndexType cellCount() const override;
    IndexType activeCellCount() const override;
    Teuchos_CommPtr Comm() const override;
    Teuchos::RCP<MeshTopology> deepCopy() const override;
    const std::set<IndexType> &getLocallyKnownActiveCellIndices() const override;
    const std::set<IndexType> &getMyActiveCellIndices() const override;
    const MeshTopology* baseMeshTopology() const override;
    CellPtr getCell(IndexType cellIndex) const override;
    std::vector<double> getCellCentroid(IndexType cellIndex) const override;
    bool getCellSideParities(IndexType cellIndex, std::vector<int> &parities) const override;
    std::vector<unsigned> getSidesMatchingTag(IndexType cellIndex, const std::string &tagName, int tagID) const override;
    unsigned getDimension() const override;
    std::vector<IndexType> refineCellInDirections(IndexType cellIndex, unsigned directionMask) override;
    //! cellsToRefineForBalance(), as a map
    std::map<IndexType,unsigned> cellsToRefineForRegularity() const override;
    const std::set<IndexType> &getRootCellIndicesLocal() const override;
    bool isDistributed() const override;
    bool isParent(IndexType cellIndex) const override;
    bool isValidCellIndex(IndexType cellIndex) const override;
    const std::vector<double>& getVertex(IndexType vertexIndex) const override;
    Camellia::FieldContainer<double> physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension = false) const override;
    void verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const override;
    MeshTopologyViewPtr getView(const std::set<IndexType> &activeCellIndices) const override;
    MeshTopologyPtr getGatheredCopy() const override;
    MeshTopologyPtr getGatheredCopy(const std::set<IndexType> &cellsToInclude) const override;
    const std::vector< Teuchos::RCP<PeriodicBC> > & getPeriodicBCs() const override;
  };
}

#endif
