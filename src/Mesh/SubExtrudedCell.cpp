// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  SubExtrudedCell.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 4/15/21.
//

#include "SubExtrudedCell.h"

namespace Camellia
{
  SubExtrudedCell::SubExtrudedCell(MeshTopologyPtr physicalMeshTopo,  const IndexType &physicalCellIndex,
                                   MeshTopologyPtr extrusionMeshTopo, const IndexType &extrusionCellIndex)
  :
  Cell(CellTopology::cellTopology(physicalMeshTopo->getCell(physicalCellIndex)->topology(), 1),
       vector<IndexType>(), vector< vector< unsigned > >(),
       SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellIndex,extrusionCellIndex),
       NULL)
  {
    _physicalMeshTopo  =  physicalMeshTopo;
    _extrusionMeshTopo = extrusionMeshTopo;
    
    _physicalCellIndex  = physicalCellIndex;
    _extrusionCellIndex = extrusionCellIndex;
  }
  
  IndexType SubExtrudedCell::entityIndex(unsigned subcdim, unsigned subcord)
  {
    unsigned sideDim = _physicalMeshTopo->getDimension();
    if (subcdim == sideDim)
    {
      return packSideEntityIndexForSubExtrudedMeshTopology(_physicalCellIndex, _extrusionCellIndex, subcord, _physicalMeshTopo, _extrusionMeshTopo);
    }
    else if (subcdim == sideDim + 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(subcord != 0, std::invalid_argument, "if subcdim matches cell dimension, subcord must be 0");
      return this->cellIndex();
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "SubExtrudedCell does not support subcell entity indexing, except for side entities (and, trivially, volume entities)");
    }
  }
  
  vector<IndexType> SubExtrudedCell::entitiesOnNeighborInterfaces(unsigned dimensionOfInterest, bool peersOnly, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
  {
    // peersOnly can be ignored; SubExtrudedMeshTopology does not support hanging nodes
    // (technically perhaps one could call this on a parent cell and expect descendant sides, but we do not support this)
    
    TEUCHOS_TEST_FOR_EXCEPTION(_physicalMeshTopo->getCell(_physicalCellIndex)->isParent(_physicalMeshTopo), std::invalid_argument, "SubExtrudedCell::entitiesOnNeighborInterfaces() is not  supported for parent cells");
    TEUCHOS_TEST_FOR_EXCEPTION(_extrusionMeshTopo->getCell(_extrusionCellIndex)->isParent(_extrusionMeshTopo), std::invalid_argument, "SubExtrudedCell::entitiesOnNeighborInterfaces() is not  supported for parent cells");
    
    unsigned sideDim = _physicalMeshTopo->getDimension();
    TEUCHOS_TEST_FOR_EXCEPTION(dimensionOfInterest < sideDim, std::invalid_argument, "SubExtrudedCell does not support subcell entity indexing, except for side entities (and, trivially, volume entities)");
    
    if (dimensionOfInterest > sideDim)
    {
      // dimension for neighbor relation is the volume dimension; by definition, this means there are no neighbors: this cell is the only one that has its volume topology.
      return vector<IndexType>();
    }

    vector<IndexType> sidesOfInterest;
    int sideCount = topology()->getSideCount();
    for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      if ( !isBoundary(sideOrdinal) )
      {
        IndexType sideEntityIndex = entityIndex(sideDim, sideOrdinal);
        sidesOfInterest.push_back(sideEntityIndex);
      }
    }
    
    return sidesOfInterest;
  }
  
  const vector< Teuchos::RCP< Cell > > & SubExtrudedCell::children()
  {
    if (_children.size() == 0)
    {
      vector<IndexType> childIndices = getChildIndices(Teuchos::null);
      _children = vector< CellPtr >(childIndices.size());
      for (int i=0; i<childIndices.size(); i++)
      {
        IndexType childPhysicalCellIndex, childExtrusionCellIndex;
        SubExtrudedMeshTopology::componentCellIndices(childPhysicalCellIndex, childExtrusionCellIndex, childIndices[i]);
        _children[i] = Teuchos::rcp(new SubExtrudedCell(_physicalMeshTopo, childPhysicalCellIndex, _extrusionMeshTopo, childExtrusionCellIndex) );
      }
    }
    return _children;
  }
  
  vector<IndexType> SubExtrudedCell::getChildIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
  {
    CellPtr physicalCell  = _physicalMeshTopo->getCell(_physicalCellIndex);
    CellPtr extrusionCell = _extrusionMeshTopo->getCell(_extrusionCellIndex);
    
    const bool physicalCellIsParent  = physicalCell->isParent(_physicalMeshTopo);
    const bool extrusionCellIsParent = extrusionCell->isParent(_extrusionMeshTopo);
    
    if (!physicalCellIsParent && !extrusionCellIsParent)
    {
      // no children
      return vector<IndexType>();
    }
    
    vector<IndexType> physicalChildIndices  = physicalCellIsParent  ? physicalCell->getChildIndices(_physicalMeshTopo)   : vector<IndexType>{_physicalCellIndex};
    vector<IndexType> extrusionChildIndices = extrusionCellIsParent ? extrusionCell->getChildIndices(_extrusionMeshTopo) : vector<IndexType>{_extrusionCellIndex};
    
    vector<IndexType> childIndices;
    for (const auto &extrusionChildIndex : extrusionChildIndices)
    {
      for (const auto &physicalChildIndex : physicalChildIndices)
      {
        IndexType compositeIndex = SubExtrudedMeshTopology::getCompositeCellIndex(physicalChildIndex, extrusionChildIndex);
        childIndices.push_back(compositeIndex);
      }
    }
    return childIndices;
  }
  
  Teuchos::RCP<Cell> SubExtrudedCell::getNeighbor(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
  {
    const auto neighborCellIndex = this->getNeighborInfo(sideOrdinal, meshTopoViewForCellValidity).first;
    if (neighborCellIndex == -1) return Teuchos::null;
    else
    {
      IndexType physicalCellIndex, extrusionCellIndex;
      SubExtrudedMeshTopology::componentCellIndices(physicalCellIndex, extrusionCellIndex, neighborCellIndex);
      return Teuchos::rcp(new SubExtrudedCell(_physicalMeshTopo,  physicalCellIndex,
                                              _extrusionMeshTopo, extrusionCellIndex) );
    }
  }
  
  pair<GlobalIndexType, unsigned> SubExtrudedCell::getNeighborInfo(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
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
    
    auto cellTopo = this->topology();
    using std::pair;
    using std::make_pair;
    if (cellTopo->sideIsSpatial(sideOrdinal))
    {
      // then the side is physical x Line
      // get the physical neighbor on the corresponding physical side
      unsigned spatialSide = cellTopo->getSpatialComponentSideOrdinal(sideOrdinal);
      pair<GlobalIndexType, unsigned> physicalNeighborInfo = _physicalMeshTopo->getCell(_physicalCellIndex)->getNeighborInfo(spatialSide, _physicalMeshTopo);
      const auto & physicalNeighborCellIndex = physicalNeighborInfo.first;
      if (physicalNeighborCellIndex != -1)
      {
        auto physicalNeighborTopo = _physicalMeshTopo->getCell(physicalNeighborCellIndex)->topology();
        auto extrudedNeighborTopo = CellTopology::cellTopology(physicalNeighborTopo,1);
        const auto & mySideOrdinalInPhysicalNeighbor = physicalNeighborInfo.second;
        
        const unsigned mySideOrdinalInExtrudedNeighbor = extrudedNeighborTopo->getExtrudedSubcellOrdinal(physicalNeighborTopo->getDimension()-1, mySideOrdinalInPhysicalNeighbor);
        
        IndexType extrudedNeighborCellIndex = SubExtrudedMeshTopology::getCompositeCellIndex(physicalNeighborCellIndex, _extrusionCellIndex);
        
        return make_pair(extrudedNeighborCellIndex,mySideOrdinalInExtrudedNeighbor);
      }
    }
    else // side is not spatial
    {
      unsigned temporalSide = cellTopo->getTemporalComponentSideOrdinal(sideOrdinal);
      
      pair<GlobalIndexType, unsigned> extrusionNeighborInfo = _extrusionMeshTopo->getCell(_extrusionCellIndex)->getNeighborInfo(temporalSide, _extrusionMeshTopo);
      const auto & extrusionNeighborCellIndex = extrusionNeighborInfo.first;
      if (extrusionNeighborCellIndex != -1)
      {
        auto physicalCellTopo = _physicalMeshTopo->getCell(_physicalCellIndex)->topology();
        
        auto extrudedNeighborTopo = CellTopology::cellTopology(physicalCellTopo,1);
        const auto & mySideOrdinalInExtrusionNeighbor = extrusionNeighborInfo.second; // 0 or 1
        
        const unsigned mySideOrdinalInExtrudedNeighbor = extrudedNeighborTopo->getTemporalSideOrdinal(mySideOrdinalInExtrusionNeighbor);
        
        IndexType extrudedNeighborCellIndex = SubExtrudedMeshTopology::getCompositeCellIndex(_physicalCellIndex, extrusionNeighborCellIndex);
        
        return make_pair(extrudedNeighborCellIndex,mySideOrdinalInExtrudedNeighbor);
      }
    }
    // default:
    return make_pair(-1,-1);
  }
  
  bool SubExtrudedCell::isBoundary(unsigned sideOrdinal)
  {
    auto neighborInfo = getNeighborInfo(sideOrdinal, Teuchos::null);
    return neighborInfo.first == -1;
  }
  
  IndexType SubExtrudedCell::extrusionCellIndex() const
  {
    return _extrusionCellIndex;
  }
  
  IndexType SubExtrudedCell::physicalCellIndex() const
  {
    return _physicalCellIndex;
  }
  
  RefinementBranch SubExtrudedCell::refinementBranch(IndexType ancestralCellID)
  {
    // at some point, we could define a tensorial refinement, which would make implementing this here simpler…
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method; use refinementBranch() on component cells…");
  }
} // namespace Camellia
