//
//  GDAMaximumRule2D.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMaximumRule2D.h"

GDAMaximumRule2D::GDAMaximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy)
: GlobalDofAssignment(meshTopology,varFactory,dofOrderingFactory,partitionPolicy)
{
  
}