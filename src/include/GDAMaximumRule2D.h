//
//  GDAMaximumRule2D.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#ifndef __Camellia_debug__GDAMaximumRule2D__
#define __Camellia_debug__GDAMaximumRule2D__

#include <iostream>

#include "GlobalDofAssignment.h"
#include "ElementTypeFactory.h"

class Solution;

class GDAMaximumRule2D : public GlobalDofAssignment {
  // much of this code copied from Mesh    
  
  // keep track of upgrades to the sides of cells since the last rebuild:
  // (used to remap solution coefficients)
  map< GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > _cellSideUpgrades; // cellID --> (oldType, newType)
  
  map< pair<GlobalIndexType,IndexType>, pair<GlobalIndexType,IndexType> > _dofPairingIndex; // key/values are (cellID,localDofIndex)
  // note that the FieldContainer for cellSideParities has dimensions (numCellsForType,numSidesForType),
  // and that the values are 1.0 or -1.0.  These are weights to account for the fact that fluxes are defined in
  // terms of an outward normal, and thus one cell's idea about the flux is the negative of its neighbor's.
  // We decide parity by cellID: the neighbor with the lower cellID gets +1, the higher gets -1.
  
  // call buildTypeLookups to rebuild the elementType data structures:
  vector< map< ElementType*, vector<GlobalIndexType> > > _cellIDsForElementType;
  map< ElementType*, map<IndexType, GlobalIndexType> > _globalCellIndexToCellID;
  vector< vector< ElementTypePtr > > _elementTypesForPartition;
  vector< ElementTypePtr > _elementTypeList;
  
  map<GlobalIndexType, PartitionIndexType> _partitionForGlobalDofIndex;
  map<GlobalIndexType, IndexType> _partitionLocalIndexForGlobalDofIndex;
  vector< map< ElementType*, FieldContainer<double> > > _partitionedPhysicalCellNodesForElementType;
  vector< map< ElementType*, FieldContainer<double> > > _partitionedCellSideParitiesForElementType;
  map< ElementType*, FieldContainer<double> > _physicalCellNodesForElementType; // for uniform mesh, just a single entry..
  vector< set<GlobalIndexType> > _partitionedGlobalDofIndices;
  map<GlobalIndexType, IndexType> _partitionLocalCellIndices; // keys are cellIDs; index is relative to both MPI node and ElementType
  map<GlobalIndexType, IndexType> _globalCellIndices; // keys are cellIDs; index is relative to ElementType
  
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType> _localToGlobalMap; // pair<cellID, localDofIndex> --> globalDofIndex
  
  vector< Solution* > _registeredSolutions; // solutions that should be modified upon refinement
  
  map<unsigned, GlobalIndexType> getGlobalVertexIDs(const FieldContainer<double> &vertexCoordinates);

  void addDofPairing(GlobalIndexType cellID1, IndexType dofIndex1, GlobalIndexType cellID2, IndexType dofIndex2);
  void buildTypeLookups();
  void buildLocalToGlobalMap();
  void determineDofPairings();
  void determinePartitionDofIndices();

  void getMultiBasisOrdering(DofOrderingPtr &originalNonParentOrdering, CellPtr parent, unsigned sideOrdinal, unsigned parentSideOrdinalInNeighbor, CellPtr nonParent);
  void matchNeighbor(GlobalIndexType cellID, int sideOrdinal);
  map< int, BasisPtr > multiBasisUpgradeMap(CellPtr parent, unsigned sideOrdinal, unsigned bigNeighborPolyOrder);
  
  void verticesForCells(FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs);
  void verticesForCell(FieldContainer<double>& vertices, GlobalIndexType cellID);
  
  bool _enforceMBFluxContinuity;
  
  GlobalIndexType _numGlobalDofs;
public:
  GDAMaximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                   unsigned initialH1OrderTrial, unsigned testOrderEnhancement, bool enforceMBFluxContinuity = false);
    
  FieldContainer<double> & cellSideParities( ElementTypePtr elemTypePtr );
  FieldContainer<double> cellSideParitiesForCell( GlobalIndexType cellID );
  
  vector< GlobalIndexType > cellsInPartition(PartitionIndexType partitionNumber);
  
  void didHRefine(const set<GlobalIndexType> &parentCellIDs);
  void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP);
  void didHUnrefine(const set<GlobalIndexType> &parentCellIDs);
  
  void didChangePartitionPolicy();
  
  ElementTypePtr elementType(GlobalIndexType cellID);
  vector< Teuchos::RCP< ElementType > > elementTypes(PartitionIndexType partitionNumber);
  
  DofOrderingFactoryPtr getDofOrderingFactory();
  ElementTypeFactory & getElementTypeFactory();
  
  GlobalIndexType cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber);
  vector<GlobalIndexType> cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr);
  
  GlobalIndexType globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex);
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber);

  GlobalIndexType globalDofCount();
  void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs, FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs);
  IndexType localDofCount(); // local to the MPI node
  
  PartitionIndexType getPartitionCount();
  
  IndexType partitionLocalCellIndex(GlobalIndexType cellID);
  GlobalIndexType globalCellIndex(GlobalIndexType cellID);
  
  PartitionIndexType partitionForCellID( GlobalIndexType cellID );
  PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex );
  GlobalIndexType partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex );
  
  FieldContainer<double> & physicalCellNodes( ElementTypePtr elemTypePtr );
  FieldContainer<double> & physicalCellNodesGlobal( ElementTypePtr elemTypePtr );
  
  void rebuildLookups();
  void registerSolution(Solution* solution);
  void unregisterSolution(Solution* solution);
  
  // used in some tests:
  int cellPolyOrder(GlobalIndexType cellID);
  void setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly);
  const map< pair<GlobalIndexType,IndexType>, GlobalIndexType>& getLocalToGlobalMap() { // pair<cellID, localDofIndex> --> globalDofIndex
    return _localToGlobalMap;
  }
  
  // used by MultiBasis:
  static int neighborChildPermutation(int childIndex, int numChildrenInSide);
  static IndexType neighborDofPermutation(IndexType dofIndex, IndexType numDofsForSide);
};

#endif /* defined(__Camellia_debug__GDAMaximumRule2D__) */
