// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  InducedMeshPartitionPolicy.h
//  Camellia
//
//  Created by Nate Roberts on 9/24/15.
//
//

#ifndef Camellia_InducedMeshPartitionPolicy_h
#define Camellia_InducedMeshPartitionPolicy_h

#include "MeshPartitionPolicy.h"
#include "RefinementObserver.h"

namespace Camellia
{
  class InducedMeshPartitionPolicy : public MeshPartitionPolicy, public RefinementObserver
  {
    // (note that as presently implemented the induced partition policy will break if either mesh is refined, since the cellID map will change...)
    // TODO: implement didHRefine, didHUnrefine.  (These are empty stubs right now.)
    
    map<GlobalIndexType, GlobalIndexType> _cellIDMap; // keys are this mesh's cellIDs; values are otherMesh's
    MeshPtr _otherMesh;
    MeshPtr _thisMesh;
  public:
    InducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh);
    InducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap);
    
    // ! Destructor will deregister from any meshes that this has registered with...
    ~InducedMeshPartitionPolicy();
    
    // ! Suitable for setting up a partition policy passed in during Mesh construction.
    InducedMeshPartitionPolicy(MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap);
    
    using RefinementObserver::didHRefine;
    void didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool didRepartitionAndRebuild);
    using RefinementObserver::didHUnrefine;
    void didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild);
    
    void partitionMesh(Mesh *mesh, PartitionIndexType numPartitions);
    
    // ! Returns an induced mesh partition policy with the identity map as the initial cellIDMap
    static Teuchos::RCP<InducedMeshPartitionPolicy> inducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh);
    
    // ! Returns an induced mesh partition policy with the specified initial cellIDMap (keys are cellIDs that belong to thisMesh; values belong to otherMesh)
    static Teuchos::RCP<InducedMeshPartitionPolicy> inducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap);
  };
}
#endif
