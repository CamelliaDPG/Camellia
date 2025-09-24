// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MeshTopologyView.h
//  Camellia
//
//  Created by Nate Roberts on 6/23/15.
//
//

//! MeshTopologyView: a class that defines a minimal interface for MeshTopology objects used by Mesh, GlobalDofAssignment, and
//! subclasses of GlobalDofAssignment.

/*!
 \author Nathan V. Roberts, ALCF.
 
 \date Last modified on 23-June-2015.
 */


#ifndef Camellia_MeshTopologyView_h
#define Camellia_MeshTopologyView_h

#include "TypeDefs.h"

#include "Camellia_HDF5.hpp"
#include "Camellia_FieldContainer.hpp"

#include <set>
#include <vector>

namespace Camellia {
  
  class MeshTransformationFunction;
  class PeriodicBC;

  class MeshTopologyView
  {
    ConstMeshTopologyPtr _meshTopo; // null when subclass constructor is used
    std::set<IndexType> _allKnownCells; // empty when subclass constructor is used
    mutable std::set<GlobalIndexType> _ownedCellIndices; // depends on base MeshTopology's _ownedCellIndices.
    mutable int _ownedCellIndicesPruningOrdinal = -1; // what the pruningOrdinal was when _ownedCellIndices was last determined.  If _meshTopo->pruningOrdinal is different, we need to rebuild.
    
    IndexType _globalCellCount;
    IndexType _globalActiveCellCount;
    
    void buildLookups(); // _rootCellIndices and _ancestralCells
  protected:
    std::set<IndexType> _activeCells;
    std::set<IndexType> _rootCells; // filled during construction when meshTopoPtr is not null; otherwise responsibility belongs to subclass.
    GlobalDofAssignment* _gda = NULL; // for cubature degree lookups
    
    std::vector<IndexType> getActiveCellsForSide(IndexType sideEntityIndex) const;
  public:
    // ! Constructor for use by MeshTopology and any other subclasses
    MeshTopologyView();
    
    // ! Constructor that defines a view in terms of an existing MeshTopology and a set of cells selected to be active.
    MeshTopologyView(ConstMeshTopologyPtr meshTopoPtr, const std::set<IndexType> &activeCellIDs);
    
    // ! Destructor
    virtual ~MeshTopologyView() {}
    
    // ! This method only gets within a factor of 2 or so, but can give a rough estimate
    virtual long long approximateMemoryFootprint() const;
    
    virtual std::vector<IndexType> cellIDsWithCentroids(const std::vector<std::vector<double>> &centroids, double tol=1e-14) const;
    virtual std::vector<IndexType> cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const;
    virtual IndexType cellCount() const;
    
    // ! Returns the global active cell count.
    virtual IndexType activeCellCount() const;
    
    template<typename GlobalIndexContainer>
    void cellHalo(GlobalIndexContainer &haloCellIndices, const std::set<GlobalIndexType> &cellIndices,
                  unsigned dimForNeighborRelation) const;
    
    // ! If the base MeshTopology is distributed, returns the Comm object used.  Otherwise, returns Teuchos::null, which is meant to indicate that the MeshTopology is replicated on every MPI rank on which it is used.
    virtual Teuchos_CommPtr Comm() const;
    
    // ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
    virtual Teuchos::RCP<MeshTopology> deepCopy() const;
    
    virtual bool entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent) const;
    virtual bool entityIsGeneralizedAncestor(unsigned ancestorDimension, IndexType ancestor,
                                             unsigned descendentDimension, IndexType descendent) const;
    
    virtual bool entityMatchesTag(unsigned d, IndexType entityIndex, const std::pair<std::string, int> &tag) const;
    
    virtual IndexType getActiveCellCount(unsigned d, IndexType entityIndex) const;
    virtual const std::set<IndexType> &getLocallyKnownActiveCellIndices() const;
    
    virtual std::set<IndexType> getGatheredActiveCellsForTime(double t) const;
    virtual std::set<IndexType> getLocallyKnownSidesForTime(double t) const;
    virtual const std::set<IndexType> &getMyActiveCellIndices() const;
    virtual std::set<IndexType> getActiveCellIndicesForAncestorsOfMyCellsInBaseMeshTopology() const;
    virtual std::vector< std::pair<IndexType,unsigned> > getActiveCellIndices(unsigned d, IndexType entityIndex) const; // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
    
    virtual const MeshTopology* baseMeshTopology() const;
    
    virtual CellPtr getCell(IndexType cellIndex) const;
    virtual std::vector<double> getCellCentroid(IndexType cellIndex) const;
    //! Each entry in tagEntitySets is understood to specify a tag.  If the searched-for entity is contained in some of those tags, then it is required that the matching cell contains a side that matches each such tag.
    virtual std::set< std::pair<IndexType, unsigned> >
    getCellsContainingEntity(unsigned d, IndexType entityIndex,
                             std::vector< std::pair<std::string, int> > tags = std::vector< std::pair<std::string, int> >()) const;
    //! If the searched-for entity is contained in some of the tags, then it is required that the matching cell contains a side that matches each such tag.
    virtual std::set< std::pair<IndexType, unsigned> >
    getCellsContainingEntities(unsigned d, const std::vector<IndexType> &entities,
                               std::vector< std::pair<std::string, int> > tags = std::vector< std::pair<std::string, int> >()) const;
    
    virtual std::set< std::pair<IndexType, unsigned> > getCellsContainingSides(const std::vector<IndexType> &sideEntityIndices) const;
    virtual std::vector<IndexType> 
    getSidesContainingEntities(unsigned d, const std::vector<IndexType> &entities,
                               std::vector< std::pair<std::string, int> > tags = std::vector< std::pair<std::string, int> >()) const;
    virtual std::vector<IndexType> getCellsForSide(IndexType sideEntityIndex) const;

    virtual std::pair<IndexType, unsigned> getConstrainingEntity(unsigned d, IndexType entityIndex) const;
    virtual IndexType getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex) const;
    virtual std::vector< std::pair<IndexType,unsigned> > getConstrainingSideAncestry(IndexType sideEntityIndex) const;
    
    virtual unsigned getDimension() const;
    
    virtual std::vector<IndexType> getEntityVertexIndices(unsigned d, IndexType entityIndex) const;
    
    virtual const std::set<IndexType> &getRootCellIndicesLocal() const;
    
    virtual std::vector< IndexType > 
    getSidesContainingEntity(unsigned d, IndexType entityIndex,
                             std::vector< std::pair<std::string, int> > tags = std::vector< std::pair<std::string, int> >()) const;
    
    virtual bool isDistributed() const;
    
    virtual bool isParent(IndexType cellIndex) const;
    
    virtual bool isValidCellIndex(IndexType cellIndex) const;
    
    virtual const std::vector<double>& getVertex(IndexType vertexIndex) const;
    
    virtual bool getVertexIndex(const std::vector<double> &vertex, IndexType &vertexIndex, double tol=1e-14) const;
    
    virtual std::vector<IndexType> getVertexIndicesMatching(const std::vector<double> &vertexInitialCoordinates, double tol=1e-14) const;

    virtual Camellia::FieldContainer<double> physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension = false) const;
    
    virtual Teuchos::RCP<MeshTransformationFunction> transformationFunction() const;
    
    virtual void setGlobalDofAssignment(GlobalDofAssignment* gda); // for cubature degree lookups
    
    virtual void verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const;
    
    virtual MeshTopologyViewPtr getView(const std::set<IndexType> &activeCellIndices) const;
    
    //! AllGather MeshTopology info, and create a new non-distributed copy on each rank.  May be expensive, particularly in terms of memory cost of the gathered object.
    virtual MeshTopologyPtr getGatheredCopy() const;
    
    //! AllGather MeshTopology info, including only the cells indicated and their ancestors, and create a new non-distributed copy on each rank.
    virtual MeshTopologyPtr getGatheredCopy(const std::set<IndexType> &cellsToInclude) const;
    
    virtual const std::vector< Teuchos::RCP<PeriodicBC> > & getPeriodicBCs() const;
    
    void printAllEntitiesInBaseMeshTopology() const;
    
    void printActiveCellAncestors() const;
    void printCellAncestors(IndexType cellIndex) const;
    
    double totalTimeComputingCellHalos() const;
    
    // I/O support (uses HDF5 methods, below, to implement)
    static MeshTopologyViewPtr readFromFile(Teuchos_CommPtr Comm, std::string filename);
    void writeToFile(std::string filename) const;
    
    // HDF5 support
    static MeshTopologyViewPtr readFromHDF5(Teuchos_CommPtr comm, HDF5 &hdf5);
    void writeToHDF5(Teuchos_CommPtr comm, HDF5 &hdf5) const;
  };

}
#endif
