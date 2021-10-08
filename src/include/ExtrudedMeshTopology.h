// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  ExtrudedMeshTopology.h
//  Camellia
//
//  Created by Roberts, Nathan V on 3/13/21.
//

#ifndef ExtrudedMeshTopology_h
#define ExtrudedMeshTopology_h

#include "Camellia.h"
#include "Camellia_FieldContainerConversion.hpp"
#include "MeshFactory.h"
#include "MeshTopology.h"

#include "Intrepid2_Data.hpp"
#include "Intrepid2_TensorArgumentIterator.hpp"
#include "Intrepid2_TensorData.hpp"

namespace Camellia {
/**\brief MeshTopology extruded in additional dimensions.  Intended especially for Boltzmann discretizations; may also be useful for space-time.
 
 This initial implementation continues to use the base class's entity indexing, refinement, and neighbor logic.  In higher dimensions, this may be fairly
 inefficient, especially in memory footprint.  Our immediate purpose here is not performance, but functionality: we want a mechanism for relating the
 lower-dimensional mesh to cells in the higher-dimensional mesh, for purposes of taking slices and integrating in the velocity dimensions.
 
 Once we have some fully functioning 1D3V drivers, we can assess performance and determine whether (and when) it's worth revisiting this class.  In
 particular, it's worth noting that for our intended (non-conforming) Boltzmann discretization we do not require any logic relating lower-dimensional
 neighbor relationships (no vertex/edge neighbors; we only need full side neighbors).  Taking advantage of that fact would result in improved performance.
 */
  class ExtrudedMeshTopology : public MeshTopology {
    using CellDataHost = Intrepid2::Data<IndexType,Kokkos::HostSpace>;
    using CellTensorDataHost = Intrepid2::TensorData<IndexType,Kokkos::HostSpace>;
    
    //! The MeshTopology that is extruded.
    MeshTopologyPtr _lowerDimensionalMeshTopology;
    
    //! In each extrusion dimension, a 1D mesh topology that includes elements that are at least as fine as any element in the ExtrudedMeshTopology
    std::vector<MeshTopologyPtr> _extrusionMeshTopologies;
    
    //! First entry belongs to _lowerDimensionalMeshTopology, after that, one entry per extrusion MeshTopology; each has a map from cell indices in component to a set of cell indices in the ExtrudedMeshTopology.
    std::vector<std::map<IndexType,std::set<IndexType>> > _componentToPhaseMaps;
    
    //! keys are phase space cell indices; values are vectors of cell indices, first the cell index in _lowerDimensionalMeshTopology, then the indices in the 1D extrusion topologies.
    std::map<IndexType,std::vector<IndexType> > _phaseToComponentMap;
    
    std::vector< Teuchos::RCP<RefinementObserver> > _refinementObservers;
    
    std::map<RefinementPatternKey, RefinementPatternPtr> _phaseToPhysicalRefinementMap;
    
    //! extract the refinement in physical space from a refinement of phase space.
    RefinementPatternPtr physicalRefinement(RefinementPatternPtr phaseRefinement);
  public:
    /** \brief Constructor.
    
     \param lowerDimensionalMeshTopology     [in] - the MeshTopology that is being extruded
     \param startCoords                                           [in] - in each extrusion dimension, starting coordinate
     \param extents                                                    [in] - in each extrusion dimension, the extent of the domain
     \param initialSubdivisions                         [in] - in each extrusion dimension, the number of subdivisions of the domain in that direction
     
     If there are periodic BCs, these should be specified (even for higher dimensions, if applicable) in lowerDimensionalMeshTopology.
    */
    ExtrudedMeshTopology(MeshTopologyPtr lowerDimensionalMeshTopology,
                         std::vector<double> startCoords,
                         std::vector<double> extents,
                         std::vector<int> initialSubdivisions);
    
    virtual ~ExtrudedMeshTopology();

    IndexType addCell(IndexType cellIndex, CellTopoPtr cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex = -1) override;
    
    //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
    void addVertex(const std::vector<double>& vertex) override;

    //! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs and the like
    Teuchos::RCP<MeshTopology> deepCopy() const override;

    bool cellContainsPoint(GlobalIndexType cellID, const std::vector<double> &point, int cubatureDegree) const override;
    
    IndexType extrusionCellIDForPhaseSpaceCellID(const IndexType &phaseSpaceCellID, const int &extrusionDimension) const;
    
    //! returns the 1D extrusionMeshTopology (the extrusionDimension is numbered from 0, and is independent of the number of physical/base dimensions).
    MeshTopologyPtr extrusionMeshTopology(int extrusionDimension) const;
    
    //! sets v0 and v1 to be the coordinates in the specified extrusion dimension of the phase space cell corresponding to phaseSpaceCellID.
    void extrusionVerticesForPhaseSpaceCell(double &v0, double &v1, const IndexType &phaseSpaceCellID, int extrusionDimension) const;
    
    //! computes the measure in the specified extrusion dimensions, starting with the first extrusion dimension.  Involves map lookups; consider caching the result if performance is critical.
    double extrusionMeasureForPhaseSpaceCell(const IndexType &phaseSpaceCellID, int extrusionDimensionToExclude=-1) const;
    
    int numExtrusionDimensions() const;
    
    const std::set<IndexType> & phaseSpaceCellIDsForPhysicalCell(const IndexType &componentCellID) const;
    
    MeshTopologyPtr physicalMeshTopology() const;
    
    IndexType physicalCellIDForPhaseSpaceCellID(const IndexType &phaseSpaceCellID) const;
    
    //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
    Camellia::FieldContainer<double> physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension = false) const override;
    
    //! At present, calls base class's implementation and then didHRefine() for any directly-registered RefinementObservers.  In the future, may do something more sophisticated involving tensor structures.
    void refineCell(IndexType cellIndex, RefinementPatternPtr refPattern, IndexType firstChildCellIndex) override;
    
    void registerRefinementObserver(Teuchos::RCP<RefinementObserver> observer);
    
    //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
    void unrefineCell(IndexType cellIndex, RefinementPatternPtr refPattern) override;
    
    //! At present, simply calls base class's implementation.  In the future, may do something more sophisticated involving tensor structures.
    void verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const override;
  };

  //! When a phase space mesh is constructed, a PhysicalMeshUpdater should be created to go with it.  Typical use case employs the static registerForUpdates method.
  class PhysicalMeshUpdater : public RefinementObserver
  {
    MeshPtr _phaseSpaceMesh;
    MeshPtr _physicalMesh;
    
    ExtrudedMeshTopology* _extrudedMeshTopology; // phase space MeshTopology
  public:
    PhysicalMeshUpdater(MeshPtr phaseSpaceMesh, MeshPtr physicalMesh);

    using RefinementObserver::didHRefine;
    void didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern,
                    bool didRepartitionAndRebuild) override;
    
    static void registerForUpdates(MeshPtr phaseSpaceMesh, MeshPtr physicalMesh);
  };
}

#endif /* ExtrudedMeshTopology_h */
