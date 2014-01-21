//
//  GlobalDofAssignment.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/20/14.
//
//

#include "GlobalDofAssignment.h"

// subclasses:
#include "GDAMaximumRule2D.h"

GlobalDofAssignment::GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                         DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy) {
  _meshTopology = meshTopology;
  _varFactory = varFactory;
  _dofOrderingFactory = dofOrderingFactory;
  _partitionPolicy = partitionPolicy;
}


GlobalDofAssignmentPtr maximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactory dofOrderingFactory, MeshPartitionPolicy partitionPolicy) {
  return Teuchos::rcp( new GDAMaximumRule2D(meshTopology,varFactory,dofOrderingFactory,partitionPolicy) );
}

GlobalDofAssignmentPtr minumumRule(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactory dofOrderingFactory, MeshPartitionPolicy partitionPolicy) {
  
  
}