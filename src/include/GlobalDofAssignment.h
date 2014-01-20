//
//  GlobalDofAssignment.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/20/14.
//
//

#ifndef __Camellia_debug__GlobalDofAssignment__
#define __Camellia_debug__GlobalDofAssignment__

#include <iostream>

#include "MeshTopology.h"
#include "DofOrderingFactory.h"
#include "VarFactory.h"
#include "MeshPartitionPolicy.h"
#include "ElementType.h"

class GlobalDofAssignment {
  MeshTopologyPtr _meshTopology;
  VarFactory _varFactory;
  DofOrderingFactory _dofOrderingFactory;
  Teuchos::RCP< MeshPartitionPolicy > _partitionPolicy;
public:
  GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactory dofOrderingFactory, MeshPartitionPolicy partitionPolicy);
  
  void didHRefine(set<int> &parentCellIDs);
  void didPRefine(set<int> &cellIDs);
  void didHUnrefine(set<int> &parentCellIDs);
  void didPUnrefine(set<int> &cellIDs);
  
  virtual ElementTypePtr elementType(unsigned cellID);
  
  virtual unsigned globalDofCount();
  virtual unsigned localDofCount(); // local to the MPI node
  
};

#endif /* defined(__Camellia_debug__GlobalDofAssignment__) */
