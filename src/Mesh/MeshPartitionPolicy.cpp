//
//  MeshPartitionPolicy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>

#include "MeshPartitionPolicy.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "GlobalDofAssignment.h"

using namespace Intrepid;
using namespace Camellia;

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

MeshPartitionPolicyPtr MeshPartitionPolicy::standardPartitionPolicy()
{
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy() );
  return partitionPolicy;
}

class OneRankPartitionPolicy : public MeshPartitionPolicy
{
  int _rankNumber;
public:
  OneRankPartitionPolicy(int rankNumber)
  {
    _rankNumber = rankNumber;
  }
  void partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
  {
    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
    vector< set<GlobalIndexType> > partitions(numPartitions);
    partitions[_rankNumber] = activeCellIDs;
    mesh->globalDofAssignment()->setPartitions(partitions);
  }
};

MeshPartitionPolicyPtr MeshPartitionPolicy::oneRankPartitionPolicy(int rankNumber)
{
  return Teuchos::rcp( new OneRankPartitionPolicy(rankNumber) );
}