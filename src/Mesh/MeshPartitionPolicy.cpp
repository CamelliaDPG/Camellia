//
//  MeshPartitionPolicy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>

#include "MeshPartitionPolicy.h"

void MeshPartitionPolicy::partitionMesh(Mesh &mesh, int numPartitions, FieldContainer<int> &partitionedActiveCells) {
  // TODO: make the default just a single partition (i.e. amounts to a "serial" partition),
  //       and split the current partition policy into a new class, MeshContiguousCellOrderingPartitionPolicy
  //       (On second thought, I'm not sure of this.  We can achieve the serial partition by simply setting numPartitions=1.)
  // default simply divides the active cells into equally-sized partitions, in the order listed in activeCells…
  TEST_FOR_EXCEPTION(numPartitions != partitionedActiveCells.dimension(0), std::invalid_argument,
                     "numPartitions must match the first dimension of partitionedActiveCells");
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveElements = mesh.activeElements().size();
  TEST_FOR_EXCEPTION(numActiveElements > maxPartitionSize, std::invalid_argument,
                     "second dimension of partitionedActiveCells must be at least as large as the number of active cells.");
  
  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition
  int chunkSize = numActiveElements / numPartitions;
  int remainder = numActiveElements % numPartitions;
  int activeCellIndex = 0;
  for (int i=0; i<numPartitions; i++) {
    int chunkSizeWithRemainder = (i < remainder) ? chunkSize + 1 : chunkSize;
    for (int j=0; j<chunkSizeWithRemainder; j++) {
      partitionedActiveCells(i,j) = mesh.activeElements()[activeCellIndex]->cellID();
      activeCellIndex++;
    }
  }
}