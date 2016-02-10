//
//  GDAMinimumRule.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#ifndef __Camellia_debug__GDAMinimumRule__
#define __Camellia_debug__GDAMinimumRule__

#include "TypeDefs.h"

#include <iostream>

#include "GlobalDofAssignment.h"

#include "LocalDofMapper.h"

#include "BasisReconciliation.h"

namespace Camellia
{

  struct AnnotatedEntity
  {
    GlobalIndexType cellID;
    unsigned sideOrdinal;    // -1 for volume-based constraint determination (i.e. for cases when the basis domain is the whole cell)
    unsigned subcellOrdinal; // subcell ordinal in the domain (cell for volume-based, side for side-based)
    unsigned dimension; // subcells can be constrained by subcells of higher dimension (i.e. this is not redundant!)
    
    bool operator < (const AnnotatedEntity & other) const
    {
      if (cellID < other.cellID) return true;
      if (cellID > other.cellID) return false;
      
      if (sideOrdinal < other.sideOrdinal) return true;
      if (sideOrdinal > other.sideOrdinal) return false;
      
      if (subcellOrdinal < other.subcellOrdinal) return true;
      if (subcellOrdinal > other.subcellOrdinal) return false;
      
      if (dimension < other.dimension) return true;
      if (dimension > other.dimension) return false;
      
      return false; // this is the case of equality.
    }
    
    bool operator == (const AnnotatedEntity & other) const
    {
      return !(*this < other) && !(other < *this);
    }
    
    bool operator != (const AnnotatedEntity & other) const
    {
      return !(*this == other);
    }
  };
  
  std::ostream& operator << (std::ostream& os, AnnotatedEntity& annotatedEntity);
  
  struct OwnershipInfo
{
  GlobalIndexType cellID;
  GlobalIndexType owningSubcellEntityIndex;
  unsigned dimension;
};

struct CellConstraints
{
  vector< vector< AnnotatedEntity > > subcellConstraints; // outer: subcell dim, inner: subcell ordinal in cell
  vector< vector< OwnershipInfo > > owningCellIDForSubcell; // outer vector indexed by subcell dimension; inner vector indexed by subcell ordinal in cell.  Pairs are (CellID, subcellIndex in MeshTopology)
//  vector< vector< vector<bool> > > sideSubcellConstraintEnforcedBySuper; // outermost vector indexed by side ordinal, then subcell dimension, then subcell ordinal.  When true, subcell does not need to be independently considered.
};

class GDAMinimumRule : public GlobalDofAssignment
{
  BasisReconciliation _br;
  map<GlobalIndexType, IndexType> _cellDofOffsets; // (cellID -> first partition-local dof index for that cell)  within the partition, offsets for the owned dofs in cell
  map<GlobalIndexType, GlobalIndexType> _globalCellDofOffsets; // (cellID -> first global dof index for that cell)
  GlobalIndexType _partitionDofOffset; // add to partition-local dof indices to get a global dof index
  GlobalIndexType _partitionDofCount; // how many dofs belong to the local partition
  Intrepid::FieldContainer<IndexType> _partitionDofCounts; // how many dofs belong to each MPI rank.
  GlobalIndexType _globalDofCount;

  set<IndexType> _partitionFluxIndexOffsets;
  set<IndexType> _partitionTraceIndexOffsets; // field indices are the complement of the other two

  map<int,set<IndexType> > _partitionIndexOffsetsForVarID; // TODO: factor out _partitionFluxIndexOffsets and _partitionTraceIndexOffsets using this container.

  typedef map<int, vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
  typedef map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
  typedef vector< SubCellOrdinalToMap > SubCellDofIndexInfo; // index to vector: subcell dimension

  map< GlobalIndexType, CellConstraints > _constraintsCache;
  map< GlobalIndexType, LocalDofMapperPtr > _dofMapperCache;
  map< GlobalIndexType, map<int, map<int, LocalDofMapperPtr> > > _dofMapperForVariableOnSideCache; // cellID --> side --> variable --> LocalDofMapper
  map< GlobalIndexType, SubCellDofIndexInfo> _ownedGlobalDofIndicesCache; // (cellID --> SubCellDofIndexInfo)
  map< GlobalIndexType, SubCellDofIndexInfo> _globalDofIndicesForCellCache; // (cellID --> SubCellDofIndexInfo) -- this has a lot of overlap in its data with the _ownedGlobalDofIndicesCache; could save some memory by only storing the difference
  map< pair<GlobalIndexType,pair<int,unsigned>>, set<GlobalIndexType>> _fittableGlobalIndicesCache; // keys: (cellID,(varID,sideOrdinal))
  
  vector<unsigned> allBasisDofOrdinalsVector(int basisCardinality);

  static string annotatedEntityToString(AnnotatedEntity &entity);

  typedef vector< SubBasisDofMapperPtr > BasisMap;
  BasisMap getBasisMap(GlobalIndexType cellID, SubCellDofIndexInfo& dofOwnershipInfo, VarPtr var);
  BasisMap getBasisMapVolumeRestrictedToSide(GlobalIndexType cellID, SubCellDofIndexInfo& dofOwnershipInfo, VarPtr var, int sideOrdinal);

  void getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal,
                           Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices);
  
  SubCellDofIndexInfo getOwnedGlobalDofIndices(GlobalIndexType cellID, CellConstraints &cellConstraints);

  set<GlobalIndexType> getFittableGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints, int sideOrdinal,
                                                   int varID = -1); // returns the global dof indices for basis functions which have support on the given side (i.e. their support intersected with the side has positive measure).  This is determined by taking the union of the global dof indices defined on all the constraining sides for the given side (the constraining sides are by definition unconstrained).  If varID of -1 is specified, returns dof indices corresponding to all variables; otherwise, returns dof indices only for the specified variable.

  vector<int> H1Order(GlobalIndexType cellID, unsigned sideOrdinal); // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...

  RefinementBranch volumeRefinementsForSideEntity(IndexType sideEntityIndex);

public:
  // these are public just for easier testing:
  BasisMap getBasisMap(GlobalIndexType cellID, SubCellDofIndexInfo& dofOwnershipInfo, VarPtr var, int sideOrdinal);
  
  CellConstraints getCellConstraints(GlobalIndexType cellID);
  LocalDofMapperPtr getDofMapper(GlobalIndexType cellID, CellConstraints &constraints, int varIDToMap = -1, int sideOrdinalToMap = -1);
  SubCellDofIndexInfo& getGlobalDofIndices(GlobalIndexType cellID, CellConstraints &cellConstraints);
  set<GlobalIndexType> getGlobalDofIndicesForIntegralContribution(GlobalIndexType cellID, int sideOrdinal); // assuming an integral is being done over the whole mesh skeleton, returns either an empty set or the global dof indices associated with the given side, depending on whether the cell "owns" the side for the purpose of such contributions.
  // ! returns the permutation that goes from the indicated cell's view of the subcell to the constraining cell's view.
  unsigned getConstraintPermutation(GlobalIndexType cellID, unsigned subcdim, unsigned subcord);
  // ! returns the permutation that goes from the indicated side's view of the subcell to the constraining side's view.
  unsigned getConstraintPermutation(GlobalIndexType cellID, unsigned sideOrdinal, unsigned subcdim, unsigned subcord);
public:
  GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                 unsigned initialH1OrderTrial, unsigned testOrderEnhancement);

  GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                 vector<int> initialH1OrderTrial, unsigned testOrderEnhancement);

  GlobalDofAssignmentPtr deepCopy();

  void didHRefine(const set<GlobalIndexType> &parentCellIDs);
  void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP);
  void didHUnrefine(const set<GlobalIndexType> &parentCellIDs);

  void didChangePartitionPolicy();
  

  ElementTypePtr elementType(GlobalIndexType cellID);

  std::set<GlobalIndexType> getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal);
  
  GlobalIndexType globalDofCount();
  
  //!! Returns the global dof indices for the indicated cell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID);
  
  //!! Returns the global dof indices for the indicated subcell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal);
  
  // ! Returns the global dof indices, in the same order as the basis ordinals, for a discontinuous variable.
  // ! For minimum-rule meshes, may throw an exception if invoked with a continuous variable's ID as argument.
  vector<GlobalIndexType> globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID);
  
  //!! Returns the global dof indices for the partition.
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber);

  set<GlobalIndexType> ownedGlobalDofIndicesForCell(GlobalIndexType cellID);

  set<GlobalIndexType> partitionOwnedGlobalFieldIndices();
  set<GlobalIndexType> partitionOwnedGlobalFluxIndices();
  set<GlobalIndexType> partitionOwnedGlobalTraceIndices();
  set<GlobalIndexType> partitionOwnedIndicesForVariables(set<int> varIDs);

  void interpretLocalData(GlobalIndexType cellID, const Intrepid::FieldContainer<double> &localData,
                          Intrepid::FieldContainer<double> &globalData,
                          Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal,
                                       const Intrepid::FieldContainer<double> &basisCoefficients,
                                       Intrepid::FieldContainer<double> &globalCoefficients,
                                       Intrepid::FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretGlobalCoefficients(GlobalIndexType cellID, Intrepid::FieldContainer<double> &localCoefficients,
                                   const Epetra_MultiVector &globalCoefficients);
  template <typename Scalar>
  void interpretGlobalCoefficients2(GlobalIndexType cellID, Intrepid::FieldContainer<Scalar> &localCoefficients,
                                    const TVectorPtr<Scalar> globalCoefficients);
  IndexType localDofCount(); // local to the MPI node

  PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex );
  void printConstraintInfo(GlobalIndexType cellID);
  void printGlobalDofInfo();
  void rebuildLookups();
};
}

#endif /* defined(__Camellia_debug__GDAMinimumRule__) */
