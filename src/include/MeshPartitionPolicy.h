//
//  MeshPartitionPolicy.h
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_MeshPartitionPolicy_h
#define Camellia_MeshPartitionPolicy_h

#include "Mesh.h"
#include "TypeDefs.h"

namespace Camellia
{
class MeshPartitionPolicy
{
  Epetra_CommPtr _Comm;
  Teuchos_CommPtr _TeuchosComm; // lazily initialized from _Comm
public:
  MeshPartitionPolicy(Epetra_CommPtr Comm);
  
  virtual ~MeshPartitionPolicy() {}
  virtual void partitionMesh(Mesh *mesh, PartitionIndexType numPartitions);
  virtual Epetra_CommPtr& Comm();
  virtual Teuchos_CommPtr& TeuchosComm();

  static MeshPartitionPolicyPtr standardPartitionPolicy(Epetra_CommPtr Comm); // aims to balance across all MPI ranks; present implementation uses Zoltan
//  static MeshPartitionPolicyPtr oneRankPartitionPolicy(int rank=0); // all cells belong to the rank specified
  static MeshPartitionPolicyPtr inducedPartitionPolicy(MeshPtr inducedMesh, MeshPtr inducingMesh); // for two meshes that have the same cell indices, uses inducingMesh to define partitioning
  static MeshPartitionPolicyPtr inducedPartitionPolicy(MeshPtr inducedMesh, MeshPtr inducingMesh, const std::map<GlobalIndexType,GlobalIndexType> &cellIDMap);
};
}

#endif
