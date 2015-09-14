/*
 //@HEADER
 // *************************************************************************
 //
 //                     Camellia EntitySet:
 //
 // Defines a set of entities relative to a MeshTopology.  The MeshTopology,
 // which will soon change to a distributed data structure, has the 
 // responsibility of updating the set when entities migrate and/or get
 // relabeled.  When an entity is refined, in general its children should
 // also be added to the the entity set.
 //
 // *************************************************************************
 //@HEADER
 */

#ifndef Camellia_EntitySet_h
#define Camellia_EntitySet_h

#include "TypeDefs.h"

namespace Camellia {
  class EntitySet
  {
    std::vector<std::set<IndexType>> _entities; // index in outer vector: dimension of entity.
    EntityHandle _handle;
  public:
    EntitySet(EntityHandle thisHandle);
    
    EntityHandle getHandle() const;
    
    void addEntity(int d, IndexType entityIndex);
    bool containsEntity(int d, IndexType entityIndex) const;
    void removeEntity(int d, IndexType entityIndex);
    
    // ! Returns the cellIDs from the provided set that contain at least one entity in the set, according to the provided MeshTopologyView.
    std::set<IndexType> cellIDsThatMatch(MeshTopologyViewPtr meshTopo, const std::set<IndexType> &cellIDs) const;
    
    // ! Returns the subcell ordinals of the given dimension for entities that belong to the indicated cell.
    std::vector<unsigned> subcellOrdinals(MeshTopologyViewPtr meshTopo, IndexType cellID, unsigned d) const;
    
    // ! Returns the subcell ordinals of the given dimension for entities that belong to the indicated cell.
    std::vector<unsigned> subcellOrdinalsOnSide(MeshTopologyViewPtr meshTopo, IndexType cellID, unsigned sideOrdinal, unsigned d) const;
  };
}

#endif
