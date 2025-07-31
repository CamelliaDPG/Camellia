// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

  bool _allowMeshTopologyPruning = true; // GDAMaximumRule2D sets to false, since its global dof numbering is not distributed...
  
  ElementTypeFactory _elementTypeFactory;
  bool _enforceConformityLocally; // whether the local DofOrdering should e.g. identify vertex dofs belonging to trace bases on sides -- currently true for max rule, false for min rule.  (Min rule will still enforce conformity, but this does not depend on it being enforced locally).  Set in base class constructor

  MeshPtr _mesh;
  MeshTopologyViewPtr _meshTopology;
  VarFactoryPtr _varFactory;
  DofOrderingFactoryPtr _dofOrderingFactory;
  MeshPartitionPolicyPtr _partitionPolicy;
  std::vector<int> _initialH1OrderTrial;
  
  int _minH1Order;
  int _maxH1Order;
  
  int _minContinuityDimension; // 0 for vertex continuity; (d-1) for "face" continuity
  int _testOrderEnhancement;

//  std::map<GlobalIndexType, std::vector<int> > _cellH1Orders;
//  std::map<GlobalIndexType, ElementTypePtr> _elementTypeForCell; // keys are cellIDs
  using tag = pair<std::string,int>;
  std::map<GlobalIndexType, int> _cellPRefinements; // key: cellID; value: deltaP.  0 assumed for cells not present.
  std::map<pair<CellTopologyKey,int>, ElementTypePtr> _elementTypesForCellTopology; // key.second: deltaP (may be negative)
  std::map<std::tuple<CellTopologyKey,int,std::set<tag>,std::map<int,std::set<tag>>>, ElementTypePtr> _elementTypesForKeyAndTags; // key<2>: volume tags; key<3>: side tags
  
  vector< set< GlobalIndexType > > _partitions; // GlobalIndexType: cellIDs
  map<GlobalIndexType, IndexType> _partitionForCellID;

  unsigned _numPartitions;

  vector< TSolutionPtr<double> > _registeredSolutions; // solutions that should be modified upon refinement (by subclasses--maximum rule has to worry about cell side upgrades, whereas minimum rule does not, so there's not a great way to do this in the abstract superclass.)

  // ! Returns the key for the specified cell, suitable for lookups in _elementTypesForCellTopology
  pair<CellTopologyKey,int> getElementTypeLookupKey(GlobalIndexType cellID);
  
  vector<int> getH1OrderForPRefinement(int deltaP);
  
  // for subclasses to let super know about changes to the p-refinement degree...
  void setPRefinementDegree(GlobalIndexType cellID, int deltaP);

  // private constructor for subclass's implementation of deepCopy()
  GlobalDofAssignment( GlobalDofAssignment& otherGDA );
public:
  GlobalDofAssignment(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                      MeshPartitionPolicyPtr partitionPolicy, std::vector<int> initialH1OrderTrial,
                      int testOrderEnhancement, bool enforceConformityLocally);
  virtual ~GlobalDofAssignment() {}

  GlobalIndexType activeCellOffset();

  // ! copies
  virtual GlobalDofAssignmentPtr deepCopy() = 0;

  void assignParities( GlobalIndexType cellID );
  
//  virtual GlobalIndexType cellID(ElementTypePtr elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber);
  virtual vector<GlobalIndexType> cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr);
  const set< GlobalIndexType > &cellsInPartition(PartitionIndexType partitionNumber) const;
  
  Camellia::FieldContainer<double> cellSideParitiesForCell( GlobalIndexType cellID );
  int sideParity(GlobalIndexType cellID, unsigned int sideOrdinal);

  // after calling any of these, must call rebuildLookups
  virtual void didHRefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  virtual void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP); // subclasses should call super
  virtual void willHUnrefine(const set<GlobalIndexType> &parentCellIDs); // subclasses should call super
  virtual void didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild);

  virtual void didChangePartitionPolicy() = 0; // called by superclass after setPartitionPolicy() is invoked

  virtual ElementTypePtr elementType(GlobalIndexType cellID);
  virtual vector< ElementTypePtr > elementTypes(PartitionIndexType partitionNumber);
//  virtual void setElementType(GlobalIndexType cellID, ElementTypePtr elem);

  ElementTypePtr getElementTypeForKey(pair<CellTopologyKey,int> key);
  pair<CellTopologyKey,int> getElementTypeLookupKey(CellTopoPtr cellTopo, int deltaP)
  {
    return {cellTopo->getKey(),deltaP};
  }
  
  ElementTypePtr getElementTypeForKeyAndTags(pair<CellTopologyKey,int> key, const std::set< pair<std::string,int> > &tagsForVolume,
                                             const std::map<int, std::set< pair<std::string,int> > > &tagsForSide);
    
  // gets *some* element type; useful when it's known or expected that the mesh will be of uniform type.  Note that this can fail for distributed meshes, if there are ranks that don't have any cells.  (Does *not* do any MPI communication.)  But this should be safe for any mesh where allowPruning = false.
  ElementTypePtr getSampleElementType();
  
  map< pair<CellTopologyKey,int>, set<GlobalIndexType> > getElementTypePartitionForCells(const set<GlobalIndexType> &cellIDs);
  
  DofOrderingFactoryPtr getDofOrderingFactory();
  ElementTypeFactory & getElementTypeFactory();
  
  virtual int getCubatureDegree(GlobalIndexType cellID);

  virtual std::vector<int> getH1Order(GlobalIndexType cellID);
  std::vector<int> getInitialH1Order();
  
  MeshPtr getMesh();
  MeshTopologyViewPtr getMeshTopology();

  bool getPartitions(Camellia::FieldContainer<GlobalIndexType> &partitions);
  PartitionIndexType getPartitionCount();

  const map<GlobalIndexType,int>& getCellPRefinements() const;
  virtual void setCellPRefinements(const map<GlobalIndexType,int>& pRefinements);
  
  int getPRefinementDegree(GlobalIndexType cellID);
  int getTestOrderEnrichment();
  
  //! assumes isotropic in p.  Returns the maximum H^1 order for an element.
  int getMaxH1Order() const;
  
  //! assumes isotropic in p.  Returns the minimum H^1 order for an element.
  int getMinH1Order() const;

  virtual GlobalIndexType globalCellIndex(GlobalIndexType cellID);
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber) = 0;
  
  virtual GlobalIndexType numPartitionOwnedGlobalFieldIndices() = 0;
  virtual GlobalIndexType numPartitionOwnedGlobalFluxIndices() = 0;
  virtual GlobalIndexType numPartitionOwnedGlobalTraceIndices() = 0;

  using DofInterpreter::interpretLocalData;
  virtual void interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localData,
                                  Camellia::FieldContainer<double> &globalData,
                                  Camellia::FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  
  using DofInterpreter::interpretLocalCoefficients;
  void interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localCoefficients,
                                  Epetra_MultiVector &globalCoefficients, int columnOrdinal);
  
  template <typename Scalar>
  void interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localCoefficients,
                                  TVectorPtr<Scalar> globalCoefficients, int columnOrdinal);
  virtual void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal,
                                               const Camellia::FieldContainer<double> &basisCoefficients,
                                               Camellia::FieldContainer<double> &globalCoefficients,
                                               Camellia::FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  virtual void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients,
                                           const Epetra_MultiVector &globalCoefficients) = 0;
  // template <typename Scalar>
  //   virtual void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<Scalar> &localCoefficients, const TVectorPtr<Scalar> globalCoefficients) = 0;

  virtual set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) = 0;

  // ! Returns the global dof indices, in the same order as the basis ordinals, for a discontinuous variable.
  // ! For minimum-rule meshes, may throw an exception if invoked with a continuous variable's ID as argument.
  virtual vector<GlobalIndexType> globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID) = 0;
  
  virtual IndexType localDofCount() = 0; // local to the MPI node

  // ! Returns the smallest dimension along which continuity will be enforced.  GlobalDofAssignment's implementation
  // ! assumes that the function spaces for the bases defined on cells determine this (e.g. H^1-conforming basis --> 0).
  // ! MPI-collective method.
  virtual void determineMinimumSubcellDimensionForContinuityEnforcement();
  
  // ! Returns the smallest dimension along which continuity will be enforced.  GlobalDofAssignment's implementation
  // ! assumes that the function spaces for the bases defined on cells determine this (e.g. H^1-conforming basis --> 0).
  virtual int minimumSubcellDimensionForContinuityEnforcement() const;
  
  // ! method for setting mesh and meshTopology after a deep copy of GDA.  Doesn't rebuild anything!!
  void setMeshAndMeshTopology(MeshPtr mesh);

  PartitionIndexType partitionForCellID( GlobalIndexType cellID );
  virtual IndexType partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber = -1); // partitionNumber == -1 means use MPI rank as partitionNumber

  MeshPartitionPolicyPtr getPartitionPolicy();
  
  // project the children of the specified cells onto their parents in any registered solutions -- one use case is immediately prior to unrefinement
  void projectChildCoefficientsOntoParents(const std::set<GlobalIndexType> &parentCellIDs);
  
  // project parent values onto any child elements that do not have coefficients specified in any registered solutions -- one use case is immediately after refinement
  void projectParentCoefficientsOntoUnsetChildren();
  
  virtual void rebuildLookups() = 0;
  
  void repartitionAndMigrate();

  void registerSolution(TSolutionPtr<double> solution);
  vector<TSolutionPtr<double>> getRegisteredSolutions();
  void unregisterSolution(TSolutionPtr<double> solution);

  virtual void setAllowMeshTopologyPruning(bool allowMeshTopologyPruning) = 0;
  virtual bool allowMeshTopologyPruning() = 0;
  
  void setPartitions(std::vector< std::set<IndexType> > &partitions, bool rebuildLookups=true);
  void setPartitions(Camellia::FieldContainer<GlobalIndexType> &partitions, bool rebuildLookups=true);
  void setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy, bool repartitionAndMigrate=true );

  // static constructors:
  static GlobalDofAssignmentPtr maximumRule2D(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory,
      MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  static GlobalDofAssignmentPtr minimumRule(MeshPtr mesh, VarFactoryPtr varFactory,
      DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
      unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
};
}

#endif /* defined(__Camellia_debug__GlobalDofAssignment__) */
