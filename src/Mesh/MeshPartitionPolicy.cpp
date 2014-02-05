//
//  MeshPartitionPolicy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>

#include "MeshPartitionPolicy.h"

void MeshPartitionPolicy::partitionMesh(MeshTopology *meshTopology, PartitionIndexType numPartitions, FieldContainer<GlobalIndexType> &partitionedActiveCells) {
  // default simply divides the active cells into equally-sized partitions, in the order listed in activeCells…
  TEUCHOS_TEST_FOR_EXCEPTION(numPartitions != partitionedActiveCells.dimension(0), std::invalid_argument,
                     "numPartitions must match the first dimension of partitionedActiveCells");
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveCells = meshTopology->activeCellCount(); // leaf nodes
  TEUCHOS_TEST_FOR_EXCEPTION(numActiveCells > maxPartitionSize, std::invalid_argument,
                     "second dimension of partitionedActiveCells must be at least as large as the number of active cells.");
  
  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition
  int chunkSize = numActiveCells / numPartitions;
  int remainder = numActiveCells % numPartitions;
  IndexType activeCellIndex = 0;
  vector<GlobalIndexType> activeCellIDs;
  set<IndexType> cellIDSet = meshTopology->getActiveCellIndices();
  activeCellIDs.insert(activeCellIDs.begin(),cellIDSet.begin(),cellIDSet.end());
  for (int i=0; i<numPartitions; i++) {
    int chunkSizeWithRemainder = (i < remainder) ? chunkSize + 1 : chunkSize;
    for (int j=0; j<chunkSizeWithRemainder; j++) {
      partitionedActiveCells(i,j) = activeCellIDs[activeCellIndex];
      activeCellIndex++;
    }
  }
}