//
//  SubExtrudedCell.h
//  Camellia
//
//  Created by Roberts, Nathan V on 4/7/21.
//

#ifndef SubExtrudedCell_h
#define SubExtrudedCell_h

#include "SubExtrudedMeshTopology.h"

namespace Camellia {
  //! cell subclass used in SubExtrudedMeshTopology
  class SubExtrudedCell : public Cell
  {
    MeshTopologyPtr _physicalMeshTopo;
    MeshTopologyPtr _extrusionMeshTopo;
    
    IndexType _physicalCellIndex;
    IndexType _extrusionCellIndex;
    
    vector< CellPtr > _children;
  public:
    SubExtrudedCell(MeshTopologyPtr physicalMeshTopo,  const IndexType &physicalCellIndex,
                    MeshTopologyPtr extrusionMeshTopo, const IndexType &extrusionCellIndex);
    
    IndexType entityIndex(unsigned subcdim, unsigned subcord) override;
    
    vector<IndexType> entitiesOnNeighborInterfaces(unsigned dimensionOfInterest, bool peersOnly, ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    
    const vector< Teuchos::RCP< Cell > > &children() override;
    
    vector<IndexType> getChildIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    
    Teuchos::RCP<Cell> getNeighbor(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    
    pair<GlobalIndexType, unsigned> getNeighborInfo(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity) override;
    
    bool isBoundary(unsigned sideOrdinal) override;
    
    IndexType extrusionCellIndex() const;
    
    IndexType physicalCellIndex() const;
    
    RefinementBranch refinementBranch(IndexType ancestralCellID = -1) override;
  };
}

#endif /* SubExtrudedCell_h */
