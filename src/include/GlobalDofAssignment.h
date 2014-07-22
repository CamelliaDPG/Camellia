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
#include "DofInterpreter.h"

class GlobalDofAssignment;
typedef Teuchos::RCP<GlobalDofAssignment> GlobalDofAssignmentPtr;

class GlobalDofAssignment : public DofInterpreter {
  GlobalIndexType _activeCellOffset; // among active cells, an offset to allow the current partition to identify unique cell indices
protected:
  map< GlobalIndexType, vector<int> > _cellSideParitiesForCellID;
  
  ElementTypeFactory _elementTypeFactory;
  bool _enforceConformityLocally; // whether the local DofOrdering should e.g. identify vertex dofs belonging to trace bases on sides -- currently true for max rule, false for min rule.  (Min rule will still enforce conformity, but this does not depend on it being enforced locally).  Set in base class constructor
  
  MeshPtr _mesh;
  MeshTopologyPtr _meshTopology;
  VarFactory _varFactory;
  DofOrderingFactoryPtr _dofOrderingFactory;
  MeshPartitionPolicyPtr _partitionPolicy;
  unsigned _initialH1OrderTrial;
  unsigned _testOrderEnhancement;
  
  map<GlobalIndexType, unsigned> _cellH1Orders;
  map<GlobalIndexType, ElementTypePtr> _elementTypeForCell; // keys are cellIDs
  
  vector< map< ElementType*, vector<GlobalIndexType> > > _cellIDsForElementType; // divided by partition
  
  vector< set< GlobalIndexType > > _partitions; // GlobalIndexType: cellIDs
  map<GlobalIndexType, IndexType> _partitionForCellID;
  
  unsigned _numPartitions;
  
  vector< Solution* > _registeredSolutions; // solutions that should be modified upon refinement (by subclasses--maximum rule has to worry about cell side upgrades, whereas minimum rule does not, so there's not a great way to do this in the abstract superclass.)
  
  void assignInitialElementType( GlobalIndexType cellID ); // this is the "natural" element type, before side modifications for constraints (when using maximum rule)
  void assignParities( GlobalIndexType cellID );
  
  virtual void rebuildLookups() = 0;
public:
  GlobalDofAssignment(MeshPtr mesh, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                      MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement,
                      bool enforceConformityLocally);

  GlobalIndexType activeCellOffset();
  
  virtual GlobalIndexType cellID(ElementTypePtr elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber);
  virtual vector<GlobalIndexType> cellIDsOfElementType(unsigned partitionNumber, ElementTypePtr elemTypePtr);
  const set< GlobalIndexType > &cellsInPartition(PartitionIndexType partitionNumber);
  FieldContainer<double> cellSideParitiesForCell( GlobalIndexType cellID );
  
  // after calling any of these, must call rebuildLookups
  virtual void didHRefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  virtual void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP); // subclasses should call super
  virtual void didHUnrefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  
  virtual void didChangePartitionPolicy() = 0; // called by superclass after setPartitionPolicy() is invoked
  
  virtual ElementTypePtr elementType(GlobalIndexType cellID) = 0;
  virtual vector< ElementTypePtr > elementTypes(PartitionIndexType partitionNumber);
  
  DofOrderingFactoryPtr getDofOrderingFactory();
  ElementTypeFactory & getElementTypeFactory();
  
  virtual int getH1Order(GlobalIndexType cellID) = 0;
  int getInitialH1Order();
  
  bool getPartitions(FieldContainer<GlobalIndexType> &partitions);
  PartitionIndexType getPartitionCount();

  int getTestOrderEnrichment();
  
  virtual GlobalIndexType globalCellIndex(GlobalIndexType cellID);
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber) = 0;
  virtual set<GlobalIndexType> partitionOwnedGlobalFieldIndices() = 0;
  virtual set<GlobalIndexType> partitionOwnedGlobalFluxIndices() = 0;
  virtual set<GlobalIndexType> partitionOwnedGlobalTraceIndices() = 0;
  
  virtual void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localData, FieldContainer<double> &globalData,
                                  FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  void interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients);
  virtual void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                               FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  virtual void interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients) = 0;

  virtual set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) = 0;
  
  virtual IndexType localDofCount() = 0; // local to the MPI node
  
  PartitionIndexType partitionForCellID( GlobalIndexType cellID );
  virtual IndexType partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber = -1); // partitionNumber == -1 means use MPI rank as partitionNumber
  
  virtual PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) = 0;
  
  void repartitionAndMigrate();

  void registerSolution(Solution* solution);
  vector<Solution *> getRegisteredSolutions();
  void unregisterSolution(Solution* solution);
  
  void setPartitions(FieldContainer<GlobalIndexType> &partitions);
  void setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy );
  
  // static constructors:
  static GlobalDofAssignmentPtr maximumRule2D(MeshPtr mesh, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                              MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  static GlobalDofAssignmentPtr minimumRule(MeshPtr mesh, VarFactory varFactory,
                                            DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                            unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
};

#endif /* defined(__Camellia_debug__GlobalDofAssignment__) */
