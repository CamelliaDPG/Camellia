//
//  GlobalDofAssignment.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/20/14.
//
//

#ifndef __Camellia_debug__GlobalDofAssignment__
#define __Camellia_debug__GlobalDofAssignment__

#include "TypeDefs.h"

#include <iostream>

#include "MeshTopologyView.h"
#include "DofOrderingFactory.h"
#include "VarFactory.h"
#include "MeshPartitionPolicy.h"
#include "ElementType.h"

#include "ElementTypeFactory.h"

#include "Epetra_Vector.h"

#include "DofInterpreter.h"

#include "Epetra_Map.h"

namespace Camellia
{
class GlobalDofAssignment : public DofInterpreter
{
  GlobalIndexType _activeCellOffset; // among active cells, an offset to allow the current partition to identify unique cell indices
protected:
  map< GlobalIndexType, vector<int> > _cellSideParitiesForCellID;

  ElementTypeFactory _elementTypeFactory;
  bool _enforceConformityLocally; // whether the local DofOrdering should e.g. identify vertex dofs belonging to trace bases on sides -- currently true for max rule, false for min rule.  (Min rule will still enforce conformity, but this does not depend on it being enforced locally).  Set in base class constructor

  MeshPtr _mesh;
  MeshTopologyViewPtr _meshTopology;
  VarFactoryPtr _varFactory;
  DofOrderingFactoryPtr _dofOrderingFactory;
  MeshPartitionPolicyPtr _partitionPolicy;
  std::vector<int> _initialH1OrderTrial;
  int _testOrderEnhancement;

  std::map<GlobalIndexType, std::vector<int> > _cellH1Orders;
  std::map<GlobalIndexType, ElementTypePtr> _elementTypeForCell; // keys are cellIDs

  vector< map< ElementType*, vector<GlobalIndexType> > > _cellIDsForElementType; // divided by partition

  vector< set< GlobalIndexType > > _partitions; // GlobalIndexType: cellIDs
  map<GlobalIndexType, IndexType> _partitionForCellID;

  Teuchos::RCP<Epetra_Map> _activeCellMap;
  MapPtr _activeCellMap2;

  unsigned _numPartitions;

  vector< TSolutionPtr<double> > _registeredSolutions; // solutions that should be modified upon refinement (by subclasses--maximum rule has to worry about cell side upgrades, whereas minimum rule does not, so there's not a great way to do this in the abstract superclass.)

  void assignInitialElementType( GlobalIndexType cellID ); // this is the "natural" element type, before side modifications for constraints (when using maximum rule)
  void assignParities( GlobalIndexType cellID );

  void constructActiveCellMap();
  void constructActiveCellMap2();

  void projectParentCoefficientsOntoUnsetChildren();
  virtual void rebuildLookups() = 0;

  // private constructor for subclass's implementation of deepCopy()
  GlobalDofAssignment( GlobalDofAssignment& otherGDA );
public:
  GlobalDofAssignment(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                      MeshPartitionPolicyPtr partitionPolicy, std::vector<int> initialH1OrderTrial,
                      int testOrderEnhancement, bool enforceConformityLocally);
  virtual ~GlobalDofAssignment() {}

  GlobalIndexType activeCellOffset();
  Teuchos::RCP<Epetra_Map> getActiveCellMap();
  MapPtr getActiveCellMap2();

  // ! copies
  virtual GlobalDofAssignmentPtr deepCopy() = 0;

  virtual GlobalIndexType cellID(ElementTypePtr elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber);
  virtual vector<GlobalIndexType> cellIDsOfElementType(unsigned partitionNumber, ElementTypePtr elemTypePtr);
  const set< GlobalIndexType > &cellsInPartition(PartitionIndexType partitionNumber);
  Intrepid::FieldContainer<double> cellSideParitiesForCell( GlobalIndexType cellID );

  // after calling any of these, must call rebuildLookups
  virtual void didHRefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  virtual void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP); // subclasses should call super
  virtual void didHUnrefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super

  virtual void didChangePartitionPolicy() = 0; // called by superclass after setPartitionPolicy() is invoked

  virtual ElementTypePtr elementType(GlobalIndexType cellID) = 0;
  virtual vector< ElementTypePtr > elementTypes(PartitionIndexType partitionNumber);

  DofOrderingFactoryPtr getDofOrderingFactory();
  ElementTypeFactory & getElementTypeFactory();

  virtual int getCubatureDegree(GlobalIndexType cellID);

  virtual std::vector<int> getH1Order(GlobalIndexType cellID);
  std::vector<int> getInitialH1Order();
  
  MeshPtr getMesh();
  MeshTopologyViewPtr getMeshTopology();

  bool getPartitions(Intrepid::FieldContainer<GlobalIndexType> &partitions);
  PartitionIndexType getPartitionCount();

  int getTestOrderEnrichment();

  virtual GlobalIndexType globalCellIndex(GlobalIndexType cellID);
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber) = 0;
  virtual set<GlobalIndexType> partitionOwnedIndicesForVariables(set<int> varIDs) = 0;
  virtual set<GlobalIndexType> partitionOwnedGlobalFieldIndices() = 0;
  virtual set<GlobalIndexType> partitionOwnedGlobalFluxIndices() = 0;
  virtual set<GlobalIndexType> partitionOwnedGlobalTraceIndices() = 0;

  virtual void interpretLocalData(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localData, Intrepid::FieldContainer<double> &globalData,
                                  Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  void interpretLocalCoefficients(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients);
  template <typename Scalar>
  void interpretLocalCoefficients(GlobalIndexType cellID, const Intrepid::FieldContainer<Scalar> &localCoefficients, TVectorPtr<Scalar> globalCoefficients);
  virtual void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Intrepid::FieldContainer<double> &basisCoefficients,
      Intrepid::FieldContainer<double> &globalCoefficients, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  virtual void interpretGlobalCoefficients(GlobalIndexType cellID, Intrepid::FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients) = 0;
  // template <typename Scalar>
  //   virtual void interpretGlobalCoefficients(GlobalIndexType cellID, Intrepid::FieldContainer<Scalar> &localCoefficients, const TVectorPtr<Scalar> globalCoefficients) = 0;

  virtual set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) = 0;

  virtual IndexType localDofCount() = 0; // local to the MPI node

  // ! method for setting mesh and meshTopology after a deep copy of GDA.  Doesn't rebuild anything!!
  void setMeshAndMeshTopology(MeshPtr mesh);

  PartitionIndexType partitionForCellID( GlobalIndexType cellID );
  virtual IndexType partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber = -1); // partitionNumber == -1 means use MPI rank as partitionNumber

  virtual PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) = 0;

  void repartitionAndMigrate();

  void registerSolution(TSolutionPtr<double> solution);
  vector<TSolutionPtr<double>> getRegisteredSolutions();
  void unregisterSolution(TSolutionPtr<double> solution);

  void setPartitions(std::vector< std::set<IndexType> > &partitions);
  void setPartitions(Intrepid::FieldContainer<GlobalIndexType> &partitions);
  void setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy );

  // static constructors:
  static GlobalDofAssignmentPtr maximumRule2D(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory,
      MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  static GlobalDofAssignmentPtr minimumRule(MeshPtr mesh, VarFactoryPtr varFactory,
      DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
      unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
};
}

#endif /* defined(__Camellia_debug__GlobalDofAssignment__) */
