//
//  GDAMaximumRule2D.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#ifndef __Camellia_debug__GDAMaximumRule2D__
#define __Camellia_debug__GDAMaximumRule2D__

#include "TypeDefs.h"

#include <iostream>

#include "GlobalDofAssignment.h"
#include "ElementTypeFactory.h"

namespace Camellia
{
class GDAMaximumRule2D : public GlobalDofAssignment
{
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
  map< ElementType*, map<IndexType, GlobalIndexType> > _globalCellIndexToCellID;
  vector< vector< ElementTypePtr > > _elementTypesForPartition;
  vector< ElementTypePtr > _elementTypeList;

  map<GlobalIndexType, PartitionIndexType> _partitionForGlobalDofIndex;
  map<GlobalIndexType, IndexType> _partitionLocalIndexForGlobalDofIndex;
  vector< map< ElementType*, Intrepid::FieldContainer<double> > > _partitionedPhysicalCellNodesForElementType;
  vector< map< ElementType*, Intrepid::FieldContainer<double> > > _partitionedCellSideParitiesForElementType;
  map< ElementType*, Intrepid::FieldContainer<double> > _physicalCellNodesForElementType; // for uniform mesh, just a single entry..
  vector< set<GlobalIndexType> > _partitionedGlobalDofIndices;
  map<GlobalIndexType, IndexType> _partitionLocalCellIndices; // keys are cellIDs; index is relative to both MPI node and ElementType
  map<GlobalIndexType, IndexType> _globalCellIndices; // keys are cellIDs; index is relative to ElementType

  map< pair<GlobalIndexType,IndexType>, GlobalIndexType> _localToGlobalMap; // pair<cellID, localDofIndex> --> globalDofIndex

  map<unsigned, GlobalIndexType> getGlobalVertexIDs(const Intrepid::FieldContainer<double> &vertexCoordinates);

  void addDofPairing(GlobalIndexType cellID1, IndexType dofIndex1, GlobalIndexType cellID2, IndexType dofIndex2);
  void buildTypeLookups();
  void buildLocalToGlobalMap();
  void determineDofPairings();
  void determinePartitionDofIndices();

  void getMultiBasisOrdering(DofOrderingPtr &originalNonParentOrdering, CellPtr parent, unsigned sideOrdinal, unsigned parentSideOrdinalInNeighbor, CellPtr nonParent);
  void matchNeighbor(GlobalIndexType cellID, int sideOrdinal);
  map< int, BasisPtr > multiBasisUpgradeMap(CellPtr parent, unsigned sideOrdinal, unsigned bigNeighborPolyOrder);

  void verticesForCells(Intrepid::FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs);
  void verticesForCell(Intrepid::FieldContainer<double>& vertices, GlobalIndexType cellID);

  bool _enforceMBFluxContinuity;

  GlobalIndexType _numGlobalDofs;
public:
  GDAMaximumRule2D(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                   unsigned initialH1OrderTrial, unsigned testOrderEnhancement, bool enforceMBFluxContinuity = false);

  //  GlobalIndexType cellID(ElementTypePtr elemType, IndexType cellIndex, PartitionIndexType partitionNumber);
  Intrepid::FieldContainer<double> & cellSideParities( ElementTypePtr elemTypePtr );

  GlobalDofAssignmentPtr deepCopy();

  void didHRefine(const set<GlobalIndexType> &parentCellIDs);
  void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP);
  void didHUnrefine(const set<GlobalIndexType> &parentCellIDs);

  void didChangePartitionPolicy();

  ElementTypePtr elementType(GlobalIndexType cellID);
  vector< ElementTypePtr > elementTypes(PartitionIndexType partitionNumber);

  bool enforceConformityLocally()
  {
    return true;
  }

  // ! get the global dof indices corresponding to the specified cellID/varID/sideOrdinal.  GDAMinimumRule's implementation overrides to return only "fittable" dof indices, as required by CondensedDofInterpreter.
  std::set<GlobalIndexType> getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal);
  
  // ! Returns the global dof indices, in the same order as the basis ordinals, for a discontinuous variable.
  // ! For minimum-rule meshes, may throw an exception if invoked with a continuous variable's ID as argument.
  std::vector<GlobalIndexType> globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID);
  
  std::vector<int> getH1Order(GlobalIndexType cellID);

  GlobalIndexType globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex);
  std::vector<GlobalIndexType> globalDofIndices(GlobalIndexType cellID, const std::vector<IndexType> &localDofIndices);
  set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID);
  //!! Returns the global dof indices for the indicated subcell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal);
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber);

  GlobalIndexType globalDofCount();
  void interpretLocalData(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localDofs, Intrepid::FieldContainer<double> &globalDofs, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Intrepid::FieldContainer<double> &basisCoefficients,
                                       Intrepid::FieldContainer<double> &globalCoefficients, Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretGlobalCoefficients(GlobalIndexType cellID, Intrepid::FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients);
  IndexType localDofCount(); // local to the MPI node

  IndexType partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber = -1); // partitionNumber == -1 means use MPI rank as partitionNumber
  GlobalIndexType globalCellIndex(GlobalIndexType cellID);

  GlobalIndexType numPartitionOwnedGlobalFieldIndices();
  GlobalIndexType numPartitionOwnedGlobalFluxIndices();
  GlobalIndexType numPartitionOwnedGlobalTraceIndices();
  
  PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex );
  GlobalIndexType partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex );

  set<GlobalIndexType> partitionOwnedGlobalFieldIndices();
  set<GlobalIndexType> partitionOwnedGlobalFluxIndices();
  set<GlobalIndexType> partitionOwnedGlobalTraceIndices();
  set<GlobalIndexType> partitionOwnedIndicesForVariables(set<int> varIDs);
  
  Intrepid::FieldContainer<double> & physicalCellNodes( ElementTypePtr elemTypePtr );
  Intrepid::FieldContainer<double> & physicalCellNodesGlobal( ElementTypePtr elemTypePtr );

  void rebuildLookups();

  // used in some tests:
  int cellPolyOrder(GlobalIndexType cellID);
  void setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly);
  const map< pair<GlobalIndexType,IndexType>, GlobalIndexType>& getLocalToGlobalMap()   // pair<cellID, localDofIndex> --> globalDofIndex
  {
    return _localToGlobalMap;
  }

  // used by MultiBasis:
  static int neighborChildPermutation(int childIndex, int numChildrenInSide);
  static IndexType neighborDofPermutation(IndexType dofIndex, IndexType numDofsForSide);
};
}

#endif /* defined(__Camellia_debug__GDAMaximumRule2D__) */
