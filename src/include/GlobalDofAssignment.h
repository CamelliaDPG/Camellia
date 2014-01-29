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

class GlobalDofAssignment;
typedef Teuchos::RCP<GlobalDofAssignment> GlobalDofAssignmentPtr;

class GlobalDofAssignment {
protected:
  MeshTopologyPtr _meshTopology;
  VarFactory _varFactory;
  DofOrderingFactoryPtr _dofOrderingFactory;
  MeshPartitionPolicyPtr _partitionPolicy;
  unsigned _initialH1OrderTrial;
  unsigned _testOrderEnhancement;
  
  map<unsigned, unsigned> _cellH1Orders;
  
  unsigned _numPartitions;
public:
  GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                      MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  
  // after calling any of these, must call rebuildLookups
  virtual void didHRefine(const set<int> &parentCellIDs) = 0;
  virtual void didPRefine(const set<int> &cellIDs, int deltaP) = 0;
  virtual void didHUnrefine(const set<int> &parentCellIDs) = 0;
  
  virtual void didChangePartitionPolicy() = 0;
  
  virtual ElementTypePtr elementType(unsigned cellID) = 0;
  
  virtual unsigned globalDofCount() = 0;
  virtual unsigned localDofCount() = 0; // local to the MPI node
  
  virtual void rebuildLookups() = 0;
  
  void setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy );
  
  static GlobalDofAssignmentPtr maximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                              MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  static GlobalDofAssignmentPtr minumumRule(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                            DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                            unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
};

#endif /* defined(__Camellia_debug__GlobalDofAssignment__) */
