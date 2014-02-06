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

#include "ElementTypeFactory.h"

#include "Epetra_Vector.h"

#include "IndexType.h"

class GlobalDofAssignment;
typedef Teuchos::RCP<GlobalDofAssignment> GlobalDofAssignmentPtr;

class GlobalDofAssignment {
  GlobalIndexType _activeCellOffset; // among active cells, an offset to allow the current partition to identify unique cell indices
protected:
  map< GlobalIndexType, vector<int> > _cellSideParitiesForCellID;
  
  ElementTypeFactory _elementTypeFactory;
  
  MeshTopologyPtr _meshTopology;
  VarFactory _varFactory;
  DofOrderingFactoryPtr _dofOrderingFactory;
  MeshPartitionPolicyPtr _partitionPolicy;
  unsigned _initialH1OrderTrial;
  unsigned _testOrderEnhancement;
  
  map<GlobalIndexType, unsigned> _cellH1Orders;
  map<GlobalIndexType, ElementTypePtr> _elementTypeForCell; // keys are cellIDs
  
  vector< vector< GlobalIndexType > > _partitions; // GlobalIndexType: cellIDs
  map<GlobalIndexType, IndexType> _partitionForCellID;
  
  unsigned _numPartitions;
  
  void assignInitialElementType( GlobalIndexType cellID ); // this is the "natural" element type, before side modifications for constraints (when using maximum rule)
  void determineActiveElements();
public:
  GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                      MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);

  GlobalIndexType activeCellOffset();
  
  // after calling any of these, must call rebuildLookups
  virtual void didHRefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  virtual void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP); // subclasses should call super
  virtual void didHUnrefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  
  virtual void didChangePartitionPolicy() = 0; // called by superclass after setPartitionPolicy() is invoked
  
  virtual ElementTypePtr elementType(GlobalIndexType cellID) = 0;
  
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber) = 0;
  
  virtual void interpretLocalDofs(GlobalIndexType cellID, const FieldContainer<double> &localDofs, FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  virtual void interpretGlobalDofs(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs) = 0;
  
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
