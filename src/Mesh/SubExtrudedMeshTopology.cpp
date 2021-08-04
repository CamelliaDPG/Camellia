//
//  SubExtrudedMeshTopology.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/1/21.
//

#include "SubExtrudedMeshTopology.h"

#include "SubExtrudedCell.h"

namespace Camellia
{
  IndexType packSideEntityIndexForSubExtrudedMeshTopology(const IndexType &physicalCellIndex, const IndexType &extrusionCellIndex,
                                                          const unsigned &sideOrdinal,
                                                          MeshTopologyPtr physicalMeshTopo, MeshTopologyPtr extrusionMeshTopo)
  {
    // we define the sideEntityIndex as (compositeCellIndex, sideOrdinal)
    // Two cells may share a side; we ensure uniqueness by specifying that the smaller compositeCellIndex is the one we use.
    
    IndexType compositeCellIndex = SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellIndex, extrusionCellIndex);
    
    using std::pair;
    pair<GlobalIndexType,unsigned> neighborInfo = SubExtrudedMeshTopology::getNeighborInfo(physicalCellIndex, extrusionCellIndex, sideOrdinal, physicalMeshTopo, extrusionMeshTopo);

    unsigned sideOrdinalToPack = sideOrdinal;
    if ((neighborInfo.first != -1) && (neighborInfo.first < compositeCellIndex))
    {
      // then neighbor is the one to use in the sideEntityIndex
      compositeCellIndex = neighborInfo.first;
      sideOrdinalToPack  = neighborInfo.second;
    }
    
    IndexType sideEntityIndex = SubExtrudedMeshTopology::getCompositeSideIndex(compositeCellIndex, sideOrdinalToPack);
    return sideEntityIndex;
  }

  void unpackSideEntityIndexForSubExtrudedMeshTopology(IndexType &compositeCellIndex,
                                                       IndexType &sideOrdinal,
                                                       const IndexType &compositeSideIndex,
                                                       MeshTopologyPtr physicalMeshTopo,
                                                       MeshTopologyPtr extrusionMeshTopo)
  {
    SubExtrudedMeshTopology::componentCellIndexAndSideOrdinal(compositeCellIndex, sideOrdinal, compositeSideIndex);
  }

  SubExtrudedMeshTopology::SubExtrudedMeshTopology(Teuchos::RCP<ExtrudedMeshTopology> extrudedMeshTopology, int extrusionDimensionToInclude)
  :
  MeshTopologyView()
  {
    _extrudedMeshTopology = extrudedMeshTopology;
    _extrusionDimensionToInclude = extrusionDimensionToInclude;
    Teuchos::RCP<RefinementObserver> thisPtr = Teuchos::rcp(this,false);
    extrudedMeshTopology->registerRefinementObserver(thisPtr);
    
    _physicalMeshTopology = extrudedMeshTopology->physicalMeshTopology();
    _extrusionMeshTopology = extrudedMeshTopology->extrusionMeshTopology(extrusionDimensionToInclude);
    
    TEUCHOS_TEST_FOR_EXCEPTION(_physicalMeshTopology->pruningOrdinal() != -1, std::invalid_argument, "Component MeshTopologies must not be pruned");
    TEUCHOS_TEST_FOR_EXCEPTION(_extrusionMeshTopology->pruningOrdinal() != -1, std::invalid_argument, "Component MeshTopologies must not be pruned");
  }

  void SubExtrudedMeshTopology::compositeIndexFromComponentIndices(IndexType &compositeIndex, const std::vector<IndexType> &componentIndices,
                                                                   const std::vector<IndexType> &componentCounts)
  {
    compositeIndex = 0;
    for (int d=componentIndices.size()-1; d>0; d--)
    {
      compositeIndex += componentIndices[d];
      compositeIndex *= componentCounts[d];
    }
    compositeIndex += componentIndices[0];
  }

//  void SubExtrudedMeshTopology::componentIndicesFromCompositeIndex(std::vector<IndexType> &componentIndices, const std::vector<IndexType> &componentCounts,
//                                                                   const IndexType &compositeIndex)
//  {
//    IndexType remainingCompositeIndex = compositeIndex;
//    for (int d=0; d<componentIndices.size(); d++)
//    {
//      componentIndices[d] = remainingCompositeIndex % componentCounts[d];
//      remainingCompositeIndex /= componentCounts[d];
//    }
//  }
  
  void SubExtrudedMeshTopology::compositeIndexFromComponentIndices(IndexType &compositeIndex, const IndexType &physicalComponent, const IndexType &extrusionComponent,
                                                                   const IndexType &physicalCount)
  {
    compositeIndex = physicalComponent + extrusionComponent * physicalCount;
  }
  
//  void SubExtrudedMeshTopology::componentIndicesFromCompositeIndex(IndexType &physicalComponent, IndexType &extrusionComponent,
//                                                                   const IndexType &physicalCount, const IndexType &compositeIndex)
//  {
//    physicalComponent  = compositeIndex % physicalCount;
//    extrusionComponent = compositeIndex / physicalCount;
//  }

  void SubExtrudedMeshTopology::componentCellIndices(IndexType &physicalComponent, IndexType &extrusionComponent, const IndexType &compositeIndex)
  {
    extrusionComponent = compositeIndex >> BIT_COUNT_PHYSICAL_CELL;
    physicalComponent  = compositeIndex - (extrusionComponent << BIT_COUNT_PHYSICAL_CELL);
  }

  IndexType SubExtrudedMeshTopology::getCompositeCellIndex(const IndexType &physicalComponent, const IndexType &extrusionComponent)
  {
    return (extrusionComponent << BIT_COUNT_PHYSICAL_CELL) + physicalComponent;
  }


  IndexType SubExtrudedMeshTopology::getCompositeSideIndex(const IndexType &compositeCellIndex, const IndexType &sideOrdinal)
  {
    return (compositeCellIndex << BIT_COUNT_SIDE_ORDINAL) + sideOrdinal;
  }

  void SubExtrudedMeshTopology::componentCellIndexAndSideOrdinal(IndexType &compositeCellIndex, IndexType &sideOrdinal, const IndexType &compositeSideIndex)
  {
    compositeCellIndex = compositeSideIndex >> BIT_COUNT_SIDE_ORDINAL;
    sideOrdinal = compositeSideIndex - (compositeCellIndex << BIT_COUNT_SIDE_ORDINAL);
  }
  
  pair<GlobalIndexType, unsigned> SubExtrudedMeshTopology::getNeighborInfo(const IndexType &physicalCellIndex, const IndexType &extrusionCellIndex,
                                                                           const unsigned &sideOrdinal,
                                                                           MeshTopologyPtr physicalMeshTopo, MeshTopologyPtr extrusionMeshTopo)
  {
    /** \brief  Mapping from this CellTopology's side ordinal of dimension d-1 to the corresponding
     *          node ordinal in the Line_2 topology; that is, if
     *              this = (shardsTopo x Line_2 x Line_2 ...) x Line_2,
     *          the mapping returns the corresponding node ordinal in Line_2
     *          Note that the sideOrdinal must be one for which sideIsSpatial() returns false.
     *  \param  thisSideOrdinal      [in]  - sideOrdinal in this cell topology.
     */
//      unsigned getTemporalComponentSideOrdinal(unsigned thisSideOrdinal);
    /** \brief  Mapping from this CellTopology's side ordinal of dimension d-1 to the corresponding
     *          side ordinal in one of the tensorial component nodes; that is, if
     *              this = (shardsTopo x Line_2 x Line_2 ...) x Line_2,
     *          the mapping returns the corresponding side ordinal of dimension d-2 in
     *              (shardsTopo x Line_2 x Line_2 ...)
     *          Note that the sideOrdinal must be one for which sideIsSpatial() returns true.
     *  \param  thisSideOrdinal      [in]  - sideOrdinal in this cell topology.
     */
//      unsigned getSpatialComponentSideOrdinal(unsigned thisSideOrdinal);
    
    auto cellTopo = CellTopology::cellTopology(physicalMeshTopo->getCell(physicalCellIndex)->topology(), 1);
    using std::pair;
    using std::make_pair;
    if (cellTopo->sideIsSpatial(sideOrdinal))
    {
      // then the side is physical x Line
      // get the physical neighbor on the corresponding physical side
      unsigned spatialSide = cellTopo->getSpatialComponentSideOrdinal(sideOrdinal);
      pair<GlobalIndexType, unsigned> physicalNeighborInfo = physicalMeshTopo->getCell(physicalCellIndex)->getNeighborInfo(spatialSide, physicalMeshTopo);
      const auto & physicalNeighborCellIndex = physicalNeighborInfo.first;
      if (physicalNeighborCellIndex != -1)
      {
        auto extrudedNeighborTopo = CellTopology::cellTopology(physicalMeshTopo->getCell(physicalNeighborCellIndex)->topology(),1);
        const auto & mySideOrdinalInPhysicalNeighbor = physicalNeighborInfo.second;
        
        const unsigned mySideOrdinalInExtrudedNeighbor = extrudedNeighborTopo->getExtrudedSideOrdinal(mySideOrdinalInPhysicalNeighbor);
        
        IndexType extrudedNeighborCellIndex = getCompositeCellIndex(physicalNeighborCellIndex, extrusionCellIndex);
        
        return make_pair(extrudedNeighborCellIndex,mySideOrdinalInExtrudedNeighbor);
      }
    }
    else // side is not spatial
    {
      unsigned temporalSide = cellTopo->getTemporalComponentSideOrdinal(sideOrdinal);
      
      pair<GlobalIndexType, unsigned> extrusionNeighborInfo = extrusionMeshTopo->getCell(extrusionCellIndex)->getNeighborInfo(temporalSide, extrusionMeshTopo);
      const auto & extrusionNeighborCellIndex = extrusionNeighborInfo.first;
      if (extrusionNeighborCellIndex != -1)
      {
        auto physicalCellTopo = physicalMeshTopo->getCell(physicalCellIndex)->topology();
        
        auto extrudedNeighborTopo = CellTopology::cellTopology(physicalCellTopo,1);
        const auto & mySideOrdinalInExtrusionNeighbor = extrusionNeighborInfo.second; // 0 or 1
        
        const unsigned mySideOrdinalInExtrudedNeighbor = extrudedNeighborTopo->getTemporalSideOrdinal(mySideOrdinalInExtrusionNeighbor);
        
        IndexType extrudedNeighborCellIndex = getCompositeCellIndex(physicalCellIndex, extrusionNeighborCellIndex);
        
        return make_pair(extrudedNeighborCellIndex,mySideOrdinalInExtrudedNeighbor);
      }
    }
    // default:
    return make_pair(-1,-1);
  }
  
  std::vector<IndexType> SubExtrudedMeshTopology::cellIDsWithCentroids(const std::vector<std::vector<double>> &centroids, double tol) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  std::vector<IndexType> SubExtrudedMeshTopology::cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  IndexType SubExtrudedMeshTopology::cellCount() const
  {
    return _physicalMeshTopology->cellCount() * _extrusionMeshTopology->cellCount();
  }
  
  // ! Returns the global active cell count.
  IndexType SubExtrudedMeshTopology::activeCellCount() const
  {
    return _physicalMeshTopology->activeCellCount() * _extrusionMeshTopology->activeCellCount();
  }
  
  // ! If the base MeshTopology is distributed, returns the Comm object used.  Otherwise, returns Teuchos::null, which is meant to indicate that the MeshTopology is replicated on every MPI rank on which it is used.
  Epetra_CommPtr SubExtrudedMeshTopology::Comm() const
  {
    return Teuchos::null;
  }
  
  // ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
  Teuchos::RCP<MeshTopology> SubExtrudedMeshTopology::deepCopy() const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  // RefinementObserver interface (we observe refinements in the phase space mesh):
  void SubExtrudedMeshTopology::didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern,
                                           bool didRepartitionAndRebuild)
  {
    if (this->_gda != NULL)
    {
      std::set<GlobalIndexType> refinedSubExtrudedMeshTopologyCellIDs;
      
      std::set<GlobalIndexType> possiblyRefinedPhysicalCellIndices;
      std::set<GlobalIndexType> possiblyRefinedExtrusionCellIndices;
      
      auto thisPtr = Teuchos::rcp(this,false);
      
      for (auto phaseSpaceCellID : cellIDs)
      {
        const IndexType physicalCellIndex  = _extrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
        const IndexType extrusionCellIndex = _extrudedMeshTopology->extrusionCellIDForPhaseSpaceCellID(phaseSpaceCellID, _extrusionDimensionToInclude);
        
        // have we previously processed a refinement of this particular (physicalCellIndex, extrusionCellIndex) pair?
        bool newRefinement = false;
        IndexType compositeCellIndex = SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellIndex, extrusionCellIndex);
        auto compositeCell = this->getCell(compositeCellIndex);
        auto childIndices = compositeCell->getChildIndices(thisPtr);
        for (const auto & childCellIndex : childIndices)
        {
          if (this->_gda->partitionForCellID(childCellIndex) == -1)
          {
            newRefinement = true;
            break;
          }
        }
        if (newRefinement)
        {
          refinedSubExtrudedMeshTopologyCellIDs.insert(compositeCellIndex);
          possiblyRefinedPhysicalCellIndices.insert(physicalCellIndex);
          possiblyRefinedExtrusionCellIndices.insert(extrusionCellIndex);
        }
      }
      
      // since physical, extrusion MeshTopology are replicated on each MPI rank, *all* active indices are locally known:
      const auto & physicalActiveCellIDs = _physicalMeshTopology->getLocallyKnownActiveCellIndices();
      const auto & extrusionActiveCellIDs = _extrusionMeshTopology->getLocallyKnownActiveCellIndices();
      
      for (const auto & physicalCellIndex : possiblyRefinedPhysicalCellIndices)
      {
        CellPtr physicalCell = _physicalMeshTopology->getCell(physicalCellIndex);
        if (physicalCell->isParent(_physicalMeshTopology))
        {
          // if cell is newly refined, then we need to add cells for each active extrusion cell
          for (const auto & extrusionCellIndex : extrusionActiveCellIDs)
          {
            IndexType compositeParentIndex = SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellIndex, extrusionCellIndex);
            // if compositeParentIndex corresponds to an active cell as far as _gda is concerned, then we need to let _gda know about refinement
            if (this->_gda->partitionForCellID(compositeParentIndex) != -1)
            {
              refinedSubExtrudedMeshTopologyCellIDs.insert(compositeParentIndex);
            }
          }
        }
      }
      
      for (const auto & extrusionCellIndex : possiblyRefinedExtrusionCellIndices)
      {
        CellPtr extrusionCell = _extrusionMeshTopology->getCell(extrusionCellIndex);
        if (extrusionCell->isParent(_extrusionMeshTopology))
        {
          // if cell is newly refined, then we need to add cells for each active physical cell
          for (const auto & physicalCellIndex : physicalActiveCellIDs)
          {
            IndexType compositeParentIndex = SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellIndex, extrusionCellIndex);
            // if compositeParentIndex corresponds to an active cell as far as _gda is concerned, then we need to let _gda know about refinement
            if (this->_gda->partitionForCellID(compositeParentIndex) != -1)
            {
              refinedSubExtrudedMeshTopologyCellIDs.insert(compositeParentIndex);
            }
          }
        }
      }
      
      this->_gda->didHRefine(refinedSubExtrudedMeshTopologyCellIDs);
      
      // get mesh to repartition, if phase space mesh was repartitioned.
      if (didRepartitionAndRebuild)
      {
        this->_gda->getMesh()->repartitionAndRebuild();
      }
    }
  }
  
  // RefinementObserver interface:
  void SubExtrudedMeshTopology::didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
  {
    // Our rule is that each component is everywhere at least as fine as the phase space mesh.
    // I'm not sure whether this means we can afford to ignore unrefinements.  I think it might.
  }
  
  bool SubExtrudedMeshTopology::entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  bool SubExtrudedMeshTopology::entityIsGeneralizedAncestor(unsigned ancestorDimension, IndexType ancestor,
                                                                    unsigned descendentDimension, IndexType descendent) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  IndexType SubExtrudedMeshTopology::getActiveCellCount(unsigned d, IndexType entityIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
    
  const std::set<IndexType> &SubExtrudedMeshTopology::getLocallyKnownActiveCellIndices() const
  {
    // all cells are (by the non-pruned assumption) locally known
    
    // we don't get notified when a component is refined, but we do assume that the components do *not* ever get unrefined, so the cell index sets only increase in size; we can determine whether we need to rebuild our active set based on active cell counts
    
    IndexType currentActiveCellCount = _physicalMeshTopology->getLocallyKnownActiveCellIndices().size() * _extrusionMeshTopology->getLocallyKnownActiveCellIndices().size();
    
    if (currentActiveCellCount != _locallyKnownActiveCellIndices.size())
    {
      // rebuild
      _locallyKnownActiveCellIndices.clear();
            
      auto & physicalActiveCellIndices  = _physicalMeshTopology->getLocallyKnownActiveCellIndices();
      auto & extrusionActiveCellIndices = _extrusionMeshTopology->getLocallyKnownActiveCellIndices();
      
      for (const auto & extrusionCellID : extrusionActiveCellIndices)
      {
        for (const auto & physicalCellID : physicalActiveCellIndices)
        {
          IndexType compositeCellID = getCompositeCellIndex(physicalCellID, extrusionCellID);
          
          _locallyKnownActiveCellIndices.insert(compositeCellID);
        }
      }
    }
    
    return _locallyKnownActiveCellIndices;
  }
  
  std::set<IndexType> SubExtrudedMeshTopology::getGatheredActiveCellsForTime(double t) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
    
  std::set<IndexType> SubExtrudedMeshTopology::getLocallyKnownSidesForTime(double t) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
    
  const std::set<IndexType> &SubExtrudedMeshTopology::getMyActiveCellIndices() const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
    
  std::set<IndexType> SubExtrudedMeshTopology::getActiveCellIndicesForAncestorsOfMyCellsInBaseMeshTopology() const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
    
  std::vector< std::pair<IndexType,unsigned> > SubExtrudedMeshTopology::getActiveCellIndices(unsigned d, IndexType entityIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  } // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
  
  const MeshTopology* SubExtrudedMeshTopology::baseMeshTopology() const
  {
    // no base MeshTopology
    return NULL;
  }
  
  CellPtr SubExtrudedMeshTopology::getCell(IndexType cellIndex) const
  {
    IndexType physicalCellIndex, extrusionCellIndex;
    componentCellIndices(physicalCellIndex, extrusionCellIndex, cellIndex);
    
    return Teuchos::rcp(new SubExtrudedCell(_physicalMeshTopology, physicalCellIndex,
                                            _extrusionMeshTopology, extrusionCellIndex));
  }
  
  std::vector<double> SubExtrudedMeshTopology::getCellCentroid(IndexType cellIndex) const
  {
    IndexType physicalCellIndex, extrusionCellIndex;
    componentCellIndices(physicalCellIndex, extrusionCellIndex, cellIndex);
    
    using std::vector;
    vector<double> physicalCentroid  = _physicalMeshTopology->getCellCentroid(physicalCellIndex);
    vector<double> extrusionCentroid = _extrusionMeshTopology->getCellCentroid(extrusionCellIndex);
    
    physicalCentroid.push_back(extrusionCentroid[0]);
    return physicalCentroid;
  }
  
  std::set< std::pair<IndexType, unsigned> > SubExtrudedMeshTopology::getCellsContainingEntity(unsigned d, IndexType entityIndex) const
  {
    std::vector<IndexType> entityIndices { entityIndex };
    return getCellsContainingEntities(d, entityIndices);
  }
  
  std::set< std::pair<IndexType, unsigned> > SubExtrudedMeshTopology::getCellsContainingEntities(unsigned d, const std::vector<IndexType> &entities) const
  {
    const int sideDim = this->getDimension() - 1;
    
    TEUCHOS_TEST_FOR_EXCEPTION(d < sideDim, std::invalid_argument, "SubExtrudedMeshTopology only supports entity indexing for side entities (and, trivially, for volumes)");
    
    std::set< std::pair<IndexType, unsigned> > cellInfoSet;
    if (d > sideDim)
    {
      // volume
      for (const auto & cellIndex : entities)
      {
        std::pair<IndexType, unsigned> cellInfo {cellIndex, 0};
        cellInfoSet.insert(cellInfo);
      }
    }
    else // side
    {
      for (const auto & sideEntityIndex : entities)
      {
        auto cellInfoForSide = getCellInfoForSide(sideEntityIndex);
        cellInfoSet.insert(cellInfoForSide.begin(), cellInfoForSide.end());
      }
    }
    
    return cellInfoSet;
  }
  
  std::set< std::pair<IndexType, unsigned> > SubExtrudedMeshTopology::getCellsContainingSides(const std::vector<IndexType> &sideEntityIndices) const
  {
    const int sideDim = this->getDimension() - 1;
    return getCellsContainingEntities(sideDim, sideEntityIndices);
  }
  
  std::vector<IndexType> SubExtrudedMeshTopology::getSidesContainingEntities(unsigned d, const std::vector<IndexType> &entities) const
  {
    const int sideDim = this->getDimension() - 1;
    
    TEUCHOS_TEST_FOR_EXCEPTION(d != sideDim, std::invalid_argument, "SubExtrudedMeshTopology only supports entity indexing for side entities");
    return entities;
  }
  
  std::vector< std::pair<IndexType,unsigned> > SubExtrudedMeshTopology::getCellInfoForSide(IndexType sideEntityIndex) const
  {
    IndexType compositeCellIndex, sideOrdinal;
    unpackSideEntityIndexForSubExtrudedMeshTopology(compositeCellIndex, sideOrdinal, sideEntityIndex,
                                                    _physicalMeshTopology, _extrusionMeshTopology);
    
    IndexType physicalCellIndex, extrusionCellIndex;
    componentCellIndices(physicalCellIndex, extrusionCellIndex, compositeCellIndex);
    
    auto cellTopo = CellTopology::cellTopology(_physicalMeshTopology->getCell(physicalCellIndex)->topology(),1);
    
    auto neighborInfo = getNeighborInfo(physicalCellIndex, extrusionCellIndex, sideOrdinal, _physicalMeshTopology, _extrusionMeshTopology);
    
    using std::vector;
    using std::pair;
    std::vector< pair<IndexType,unsigned> > cellInfoForSide { {compositeCellIndex,unsigned(sideOrdinal)} };
    if (neighborInfo.first != -1)
    {
      cellInfoForSide.push_back(neighborInfo);
    }
    return cellInfoForSide;
  }
  
  std::vector<IndexType> SubExtrudedMeshTopology::getCellsForSide(IndexType sideEntityIndex) const
  {
    auto cellInfo = getCellInfoForSide(sideEntityIndex);
    
    std::vector<IndexType> cellsForSide(cellInfo.size());
    for (int i=0; i<cellsForSide.size(); i++)
    {
      cellsForSide[i] = cellInfo[i].first;
    }
    return cellsForSide;
  }

  std::pair<IndexType, unsigned> SubExtrudedMeshTopology::getConstrainingEntity(unsigned d, IndexType entityIndex) const
  {
    // the MeshTopology is a tensor product of MeshTopologies; there are no hanging nodes, no nontrivial constraining entities
    return {entityIndex, d};
  }
  IndexType SubExtrudedMeshTopology::getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  std::vector< std::pair<IndexType,unsigned> > SubExtrudedMeshTopology::getConstrainingSideAncestry(IndexType sideEntityIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  unsigned SubExtrudedMeshTopology::getDimension() const
  {
    return _physicalMeshTopology->getDimension() + 1;
  }
  
  std::vector<IndexType> SubExtrudedMeshTopology::getEntityVertexIndices(unsigned d, IndexType entityIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  const std::set<IndexType> & SubExtrudedMeshTopology::getRootCellIndicesLocal() const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  std::vector< IndexType > SubExtrudedMeshTopology::getSidesContainingEntity(unsigned d, IndexType entityIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  bool SubExtrudedMeshTopology::isDistributed() const
  {
    return false;
  }
  
  bool SubExtrudedMeshTopology::isParent(IndexType cellIndex) const
  {
    IndexType physicalCellIndex, extrusionCellIndex;
    componentCellIndices(physicalCellIndex, extrusionCellIndex, cellIndex);
    
    return _physicalMeshTopology->isParent(physicalCellIndex) || _extrusionMeshTopology->isParent(extrusionCellIndex);
  }
  
  bool SubExtrudedMeshTopology::isValidCellIndex(IndexType cellIndex) const
  {
    IndexType physicalCellIndex, extrusionCellIndex;
    componentCellIndices(physicalCellIndex, extrusionCellIndex, cellIndex);
    
    return _physicalMeshTopology->isValidCellIndex(physicalCellIndex) && _extrusionMeshTopology->isValidCellIndex(extrusionCellIndex);
  }
  
  const std::vector<double>& SubExtrudedMeshTopology::getVertex(IndexType vertexIndex) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  bool SubExtrudedMeshTopology::getVertexIndex(const std::vector<double> &vertex, IndexType &vertexIndex, double tol) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  std::vector<IndexType> SubExtrudedMeshTopology::getVertexIndicesMatching(const std::vector<double> &vertexInitialCoordinates, double tol) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }

  Camellia::FieldContainer<double> SubExtrudedMeshTopology::physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension) const
  {
    IndexType physicalCellIndex, extrusionCellIndex;
    componentCellIndices(physicalCellIndex, extrusionCellIndex, cellIndex);
    
    auto physicalNodesFC  = _physicalMeshTopology->physicalCellNodesForCell(physicalCellIndex);
    auto extrusionNodesFC = _extrusionMeshTopology->physicalCellNodesForCell(extrusionCellIndex);
    
    // make TensorPoints object from physicalNodes, extrusionNodes; convert to FieldContainer.
    // (We can worry about efficiency later!)
    auto physicalNodesPoints  = FieldContainerConversion<double>::toTensorPoints(physicalNodesFC);
    auto extrusionNodesPoints = FieldContainerConversion<double>::toTensorPoints(extrusionNodesFC);
    
    using ViewType = Intrepid2::ScalarView<double,Camellia::DeviceType>;
    Kokkos::Array<ViewType,2> tensorComponents {physicalNodesPoints.getTensorComponent(0), extrusionNodesPoints.getTensorComponent(0)};
    Intrepid2::TensorPoints<double,Camellia::DeviceType> tensorPoints(tensorComponents);
    
    auto tensorPointsFC = FieldContainerConversion<double>::fromTensorPoints(tensorPoints);
    if (includeCellDimension)
    {
      tensorPointsFC.resize(1,tensorPointsFC.dimension(0),tensorPointsFC.dimension(1));
    }
    return tensorPointsFC;
  }
  
  Teuchos::RCP<MeshTransformationFunction> SubExtrudedMeshTopology::transformationFunction() const
  {
    // transformation function unsupported by SubExtrudedMeshTopology (we could eventually do something that allows physical mesh to be curvilinear, but still orthogonally extruded)
    return Teuchos::null;
  }
  
  void SubExtrudedMeshTopology::setGlobalDofAssignment(GlobalDofAssignment* gda)
  {
    this->_gda = gda;
    
    // Base class uses gda for quadrature in the context of cellIDsWithCentroids().
    // We may want to do something similar if we need cellIDsWithCentroids AND we are working with more interesting physical geometry (higher dimensional, non-axis-aligned).
    
    // We also use _gda for repartitioning after components have repartitioned
  }
  
  void SubExtrudedMeshTopology::verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  MeshTopologyViewPtr SubExtrudedMeshTopology::getView(const std::set<IndexType> &activeCellIndices) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  //! AllGather MeshTopology info, and create a new non-distributed copy on each rank.  May be expensive, particularly in terms of memory cost of the gathered object.
  MeshTopologyPtr SubExtrudedMeshTopology::getGatheredCopy() const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  //! AllGather MeshTopology info, including only the cells indicated and their ancestors, and create a new non-distributed copy on each rank.
  MeshTopologyPtr SubExtrudedMeshTopology::getGatheredCopy(const std::set<IndexType> &cellsToInclude) const
  {
    // we start by overriding all virtual methods in MeshTopologyView with an implementation that simply throws an exception; we let the tests help us figure out what methods we actually need to implement…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unimplemented method");
  }
  
  const std::vector< Teuchos::RCP<PeriodicBC> > & SubExtrudedMeshTopology::getPeriodicBCs() const
  {
    // physical MeshTopology should have PeriodicBCs for all dimensions:
    return _physicalMeshTopology->getPeriodicBCs();
  }

  Teuchos::RCP<SubExtrudedMeshTopology> SubExtrudedMeshTopology::subExtrudedMeshTopology(MeshTopologyViewPtr meshTopo, int extrusionDimensionNumber)
  {
    // meshTopo must be a ExtrudedMeshTopology
    ExtrudedMeshTopology* extrudedMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(meshTopo.get());
    if (extrudedMeshTopo == NULL)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only ExtrudedMeshTopology meshes are supported by PhaseSpaceVisualization.");
    }
    Teuchos::RCP<ExtrudedMeshTopology> extrudedMeshTopoRCP = Teuchos::rcp(extrudedMeshTopo,false); // weak pointer
    auto subExtrudedMeshTopo = Teuchos::rcp( new SubExtrudedMeshTopology(extrudedMeshTopoRCP, extrusionDimensionNumber) );
    return subExtrudedMeshTopo;
  }
} // namespace Camellia
