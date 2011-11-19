//
//  MeshPartitionPolicy.h
//  Camellia
//
//  Created by Nathan Roberts on 11/18/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_MeshPartitionPolicy_h
#define Camellia_MeshPartitionPolicy_h

// Intrepid includes
#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;

class Mesh;

class MeshPartitionPolicy {
  virtual void partitionMesh(Mesh &mesh, int numPartitions, FieldContainer<int> &partitionedActiveCells);
};

#include "Mesh.h"

#endif
