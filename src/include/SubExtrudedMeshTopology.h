//
//  SubExtrudedMeshTopology.h
//  Camellia
//
//  Created by Roberts, Nathan V on 4/7/21.
//

#ifndef SubExtrudedMeshTopology_h
#define SubExtrudedMeshTopology_h

#include "Camellia.h"
#include "Camellia_FieldContainerConversion.hpp"
#include "ExtrudedMeshTopology.h"
#include "MeshFactory.h"
#include "MeshTopology.h"
#include "RefinementObserver.h"

namespace Camellia {
  //! A MeshTopology subclass that uses an ExtrudedMeshTopology to produce a lower-dimensional mesh, defined as the finest cells in physical MeshTopo times finest cells in each specified extrusion dimension.
  class SubExtrudedMeshTopology : public MeshTopologyView, public RefinementObserver
  {
    static constexpr int BIT_COUNT_PHYSICAL_CELL = sizeof(IndexType) * 4 - 2; // allow physical cell index to occupy half the bits in the composite index, leaving 4 bits for a side ordinal
    static constexpr int BIT_COUNT_SIDE_ORDINAL  = 4; // allows up to to 16 sides
    
    Teuchos::RCP<ExtrudedMeshTopology> _extrudedMeshTopology;
    int _extrusionDimensionToInclude;
    
    MeshTopologyPtr _physicalMeshTopology;
    MeshTopologyPtr _extrusionMeshTopology;
    
    mutable std::set<IndexType> _locallyKnownActiveCellIndices; // mutable because we need to rebuild sometimes in a const context
    
    std::vector<CellPtr> _cells;
  public:
    SubExtrudedMeshTopology(Teuchos::RCP<ExtrudedMeshTopology> extrudedMeshTopology, int extrusionDimensionToInclude);

    static void compositeIndexFromComponentIndices(IndexType &compositeIndex, const std::vector<IndexType> &componentIndices,
                                                   const std::vector<IndexType> &componentCounts);

//    static void componentIndicesFromCompositeIndex(std::vector<IndexType> &componentIndices, const std::vector<IndexType> &componentCounts,
//                                                   const IndexType &compositeIndex);
    
    static void compositeIndexFromComponentIndices(IndexType &compositeIndex, const IndexType &physicalComponent, const IndexType &extrusionComponent,
                                                   const IndexType &physicalCount);
    
//    static void componentIndicesFromCompositeIndex(IndexType &physicalComponent, IndexType &extrusionComponent,
//                                                   const IndexType &physicalCount, const IndexType &compositeIndex);
    
    static void componentCellIndices(IndexType &physicalComponent, IndexType &extrusionComponent, const IndexType &compositeIndex);
    static IndexType getCompositeCellIndex(const IndexType &physicalComponent, const IndexType &extrusionComponent);
    
    static IndexType getCompositeSideIndex(const IndexType &compositeCellIndex, const IndexType &sideOrdinal);
    static void componentCellIndexAndSideOrdinal(IndexType &compositeCellIndex, IndexType &sideOrdinal, const IndexType &compositeSideIndex);
    
    static pair<GlobalIndexType, unsigned> getNeighborInfo(const IndexType &physicalCellIndex, const IndexType &extrusionCellIndex,
                                                           const unsigned &sideOrdinal,
                                                           MeshTopologyPtr physicalMeshTopo, MeshTopologyPtr extrusionMeshTopo);
    
    virtual std::vector<IndexType> cellIDsWithCentroids(const std::vector<std::vector<double>> &centroids, double tol=1e-14) const override;
    
    virtual std::vector<IndexType> cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const override;
    
    virtual IndexType cellCount() const override;
    
    // ! Returns the global active cell count.
    virtual IndexType activeCellCount() const override;
    
    // ! If the base MeshTopology is distributed, returns the Comm object used.  Otherwise, returns Teuchos::null, which is meant to indicate that the MeshTopology is replicated on every MPI rank on which it is used.
    virtual Epetra_CommPtr Comm() const override;
    
    // ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
    virtual Teuchos::RCP<MeshTopology> deepCopy() const override;
    
    // RefinementObserver interface (we observe refinements in the phase space mesh):
    using RefinementObserver::didHRefine;
    void didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern,
                    bool didRepartitionAndRebuild) override;
    
    // RefinementObserver interface:
    using RefinementObserver::didHUnrefine;
    void didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild) override;
    
    virtual bool entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent) const override;
    
    virtual bool entityIsGeneralizedAncestor(unsigned ancestorDimension, IndexType ancestor,
                                             unsigned descendentDimension, IndexType descendent) const override;
    
    virtual IndexType getActiveCellCount(unsigned d, IndexType entityIndex) const override;
    virtual const std::set<IndexType> &getLocallyKnownActiveCellIndices() const override;
    
    virtual std::set<IndexType> getGatheredActiveCellsForTime(double t) const override;
    virtual std::set<IndexType> getLocallyKnownSidesForTime(double t) const override;
    virtual const std::set<IndexType> &getMyActiveCellIndices() const override;
    virtual std::set<IndexType> getActiveCellIndicesForAncestorsOfMyCellsInBaseMeshTopology() const override;
    virtual std::vector< std::pair<IndexType,unsigned> > getActiveCellIndices(unsigned d, IndexType entityIndex) const override;
    
    virtual const MeshTopology* baseMeshTopology() const override;
    
    virtual CellPtr getCell(IndexType cellIndex) const override;
    
    virtual std::vector<double> getCellCentroid(IndexType cellIndex) const override;
    
    virtual std::set< std::pair<IndexType, unsigned> > getCellsContainingEntity(unsigned d, IndexType entityIndex) const override;
    
    virtual std::set< std::pair<IndexType, unsigned> > getCellsContainingEntities(unsigned d, const std::vector<IndexType> &entities) const override;
    
    virtual std::set< std::pair<IndexType, unsigned> > getCellsContainingSides(const std::vector<IndexType> &sideEntityIndices) const override;
    
    virtual std::vector<IndexType> getSidesContainingEntities(unsigned d, const std::vector<IndexType> &entities) const override;
    
    std::vector< std::pair<IndexType,unsigned> > getCellInfoForSide(IndexType sideEntityIndex) const;
    
    virtual std::vector<IndexType> getCellsForSide(IndexType sideEntityIndex) const override;

    virtual std::pair<IndexType, unsigned> getConstrainingEntity(unsigned d, IndexType entityIndex) const override;
    virtual IndexType getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex) const override;
    virtual std::vector< std::pair<IndexType,unsigned> > getConstrainingSideAncestry(IndexType sideEntityIndex) const override;
    
    virtual unsigned getDimension() const override;
    
    virtual std::vector<IndexType> getEntityVertexIndices(unsigned d, IndexType entityIndex) const override;
    
    virtual const std::set<IndexType> &getRootCellIndicesLocal() const override;
    
    virtual std::vector< IndexType > getSidesContainingEntity(unsigned d, IndexType entityIndex) const override;
    
    virtual bool isDistributed() const override;
    
    virtual bool isParent(IndexType cellIndex) const override;
    
    virtual bool isValidCellIndex(IndexType cellIndex) const override;
    
    virtual const std::vector<double>& getVertex(IndexType vertexIndex) const override;
    
    virtual bool getVertexIndex(const std::vector<double> &vertex, IndexType &vertexIndex, double tol=1e-14) const override;
    
    virtual std::vector<IndexType> getVertexIndicesMatching(const std::vector<double> &vertexInitialCoordinates, double tol=1e-14) const override;

    virtual Camellia::FieldContainer<double> physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension = false) const override;
    
    virtual Teuchos::RCP<MeshTransformationFunction> transformationFunction() const override;
    
    virtual void setGlobalDofAssignment(GlobalDofAssignment* gda) override;
    
    virtual void verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const override;
    
    virtual MeshTopologyViewPtr getView(const std::set<IndexType> &activeCellIndices) const override;
    
    //! AllGather MeshTopology info, and create a new non-distributed copy on each rank.  May be expensive, particularly in terms of memory cost of the gathered object.
    virtual MeshTopologyPtr getGatheredCopy() const override;
    
    //! AllGather MeshTopology info, including only the cells indicated and their ancestors, and create a new non-distributed copy on each rank.
    virtual MeshTopologyPtr getGatheredCopy(const std::set<IndexType> &cellsToInclude) const override;
    
    virtual const std::vector< Teuchos::RCP<PeriodicBC> > & getPeriodicBCs() const override;
    
    static Teuchos::RCP<SubExtrudedMeshTopology> subExtrudedMeshTopology(MeshTopologyViewPtr extrudedMeshTopo, int extrusionDimensionNumber);
  };

  IndexType packSideEntityIndexForSubExtrudedMeshTopology(const IndexType &physicalCellIndex, const IndexType &extrusionCellIndex,
                                                          const unsigned &sideOrdinal,
                                                          MeshTopologyPtr physicalMeshTopo, MeshTopologyPtr extrusionMeshTopo);

  void unpackSideEntityIndexForSubExtrudedMeshTopology(IndexType &compositeCellIndex,
                                                       IndexType &sideOrdinal,
                                                       const IndexType &compositeSideIndex,
                                                       MeshTopologyPtr physicalMeshTopo,
                                                       MeshTopologyPtr extrusionMeshTopo);
}

#endif /* SubExtrudedMeshTopology_h */
