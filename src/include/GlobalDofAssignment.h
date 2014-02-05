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

#include "IndexType.h"

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
  
  map<GlobalIndexType, unsigned> _cellH1Orders;
  
  unsigned _numPartitions;
public:
  GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                      MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  
  // after calling any of these, must call rebuildLookups
  virtual void didHRefine(const set<GlobalIndexType> &parentCellIDs) = 0;
  virtual void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) = 0;
  virtual void didHUnrefine(const set<GlobalIndexType> &parentCellIDs) = 0;
  
  virtual void didChangePartitionPolicy() = 0; // called by superclass after setPartitionPolicy() is invoked
  
  virtual ElementTypePtr elementType(GlobalIndexType cellID) = 0;
  
  virtual GlobalIndexType globalDofCount() = 0;
  
  virtual void interpretLocalDofs(GlobalIndexType cellID, const FieldContainer<double> &localDofs, FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  
  virtual IndexType localDofCount() = 0; // local to the MPI node
  
  virtual void rebuildLookups() = 0;
  
  void setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy );
  
  // static constructors:
  static GlobalDofAssignmentPtr maximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                              MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  static GlobalDofAssignmentPtr minumumRule(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                            DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                            unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
};

#endif /* defined(__Camellia_debug__GlobalDofAssignment__) */
