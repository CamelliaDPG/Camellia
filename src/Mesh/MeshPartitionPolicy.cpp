//
//  MeshPartitionPolicy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>

#include "MeshPartitionPolicy.h"

#include "GlobalDofAssignment.h"
#include "InducedMeshPartitionPolicy.h"
#include "MeshTools.h"
#include "MPIWrapper.h"
#include "ZoltanMeshPartitionPolicy.h"

using namespace Intrepid;
using namespace Camellia;
using namespace std;

MeshPartitionPolicy::MeshPartitionPolicy(Epetra_CommPtr Comm) : _Comm(Comm) {
  TEUCHOS_TEST_FOR_EXCEPTION(Comm == Teuchos::null, std::invalid_argument, "Comm may not be null!");
}

Epetra_CommPtr& MeshPartitionPolicy::Comm()
{
  return _Comm;
}

void MeshPartitionPolicy::partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
{
  // default simply divides the active cells into equally-sized partitions, in the order listed in activeCells…
  MeshTopologyViewPtr meshTopology = mesh->getTopology();
  int numActiveCells = meshTopology->getActiveCellIndices().size(); // leaf nodes
  FieldContainer<GlobalIndexType> partitionedActiveCells(numPartitions,numActiveCells);

  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition
  int chunkSize = numActiveCells / numPartitions;
  int remainder = numActiveCells % numPartitions;
  IndexType activeCellIndex = 0;
  vector<GlobalIndexType> activeCellIDs;
  set<IndexType> cellIDSet = meshTopology->getActiveCellIndices();
  activeCellIDs.insert(activeCellIDs.begin(),cellIDSet.begin(),cellIDSet.end());
  for (int i=0; i<numPartitions; i++)
  {
    int chunkSizeWithRemainder = (i < remainder) ? chunkSize + 1 : chunkSize;
    for (int j=0; j<chunkSizeWithRemainder; j++)
    {
      partitionedActiveCells(i,j) = activeCellIDs[activeCellIndex];
      activeCellIndex++;
    }
  }
  mesh->globalDofAssignment()->setPartitions(partitionedActiveCells);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh)
{
  return InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(thisMesh, otherMesh);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType,GlobalIndexType> &cellIDMap)
{
  return InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(thisMesh, otherMesh, cellIDMap);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::standardPartitionPolicy(Epetra_CommPtr Comm)
{
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy(Comm) );
  return partitionPolicy;
}

Teuchos_CommPtr& MeshPartitionPolicy::TeuchosComm()
{
  if (_TeuchosComm == Teuchos::null)
  {
#ifdef HAVE_MPI
    Epetra_MpiComm* mpiComm = dynamic_cast<Epetra_MpiComm*>(_Comm.get());
    if (mpiComm == NULL)
    {
      // serial communicator
      _TeuchosComm = MPIWrapper::TeuchosCommSerial();
    }
    else
    {
      if (mpiComm->GetMpiComm() == MPI_COMM_WORLD)
      {
        _TeuchosComm = MPIWrapper::TeuchosCommWorld();
      }
      else
      {
        _TeuchosComm = Teuchos::rcp( new Teuchos::MpiComm<int> (mpiComm->GetMpiComm()) );
      }
    }
#else
  // if we don't have MPI, then it can only be a serial communicator
    _TeuchosComm = MPIWrapper::TeuchosSerialComm();
#endif
  }
  return _TeuchosComm;
}

//class OneRankPartitionPolicy : public MeshPartitionPolicy
//{
//  int _rankNumber;
//public:
//  OneRankPartitionPolicy(int rankNumber) : MeshPartitionPolicy(Teuchos::rcp(new Epetra_SerialComm()))
//  {
//    _rankNumber = rankNumber;
//  }
//  void partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
//  {
//    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
//    vector< set<GlobalIndexType> > partitions(numPartitions);
//    partitions[_rankNumber] = activeCellIDs;
//    mesh->globalDofAssignment()->setPartitions(partitions);
//  }
//};
//
//MeshPartitionPolicyPtr MeshPartitionPolicy::oneRankPartitionPolicy(int rankNumber)
//{
//  return Teuchos::rcp( new OneRankPartitionPolicy(rankNumber) );
//}