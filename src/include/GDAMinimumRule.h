// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  GDAMinimumRule.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#ifndef __Camellia_debug__GDAMinimumRule__
#define __Camellia_debug__GDAMinimumRule__

#include <iostream>

#include "BasisReconciliation.h"
#include "GlobalDofAssignment.h"
#include "LocalDofMapper.h"
#include "SubcellDofIndices.h"
#include "TypeDefs.h"

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
  unsigned owningSubcellOrdinal;
  unsigned dimension;
};

struct CellConstraints
{
  vector< vector< AnnotatedEntity > > subcellConstraints; // outer: subcell dim, inner: subcell ordinal in cell
  vector< vector< OwnershipInfo > > owningCellIDForSubcell; // outer vector indexed by subcell dimension; inner vector indexed by subcell ordinal in cell.  Pairs are (CellID, subcellIndex in MeshTopology)
//  vector< vector< vector<bool> > > sideSubcellConstraintEnforcedBySuper; // outermost vector indexed by side ordinal, then subcell dimension, then subcell ordinal.  When true, subcell does not need to be independently considered.
  
  /*
   spatialSliceConstraints: 
   When space-only trace/flux variables are defined in space-time meshes,
   then we need to treat these somewhat specially, because sometimes the geometrically
   constraining side must be a temporal interface.  When that happens, we need to know what
   same-dimensional "constraints" there are on the spatial slice.  Because here we will
   necessarily have a hanging node on the temporal interface (otherwise a spatial side
   would be available), then for a 1-irregular mesh we can guarantee that the entities constrained
   by that temporal side are not geometrically constrained in space.  So really what we are doing
   here is deciding ownership and orientation, as well as resolving any difference in polynomial
   degree.
   
   To keep the storage cost to a minimum, we only initialize this container when there is a temporal
   side constraint.  Even then, we only fill it in for those entities that we cannot resolve by the
   usual mechanism: those that are constrained by the temporal interface.  The idea is that the
   usual mechanism should be tried first, and if that fails, then this special space-time mechanism
   can be consulted.
   */
  
  Teuchos::RCP<CellConstraints> spatialSliceConstraints;
};

class GDAMinimumRule : public GlobalDofAssignment
{
  bool _checkConstraintConsistency = false;
  
  BasisReconciliation _br;
  map<GlobalIndexType, IndexType> _cellDofOffsets; // (cellID -> first partition-local dof index for that cell)  within the partition, offsets for the owned dofs in cell
  map<GlobalIndexType, GlobalIndexType> _globalCellDofOffsets; // (cellID -> first global dof index for that cell)
  GlobalIndexType _partitionDofOffset; // add to partition-local dof indices to get a global dof index
  GlobalIndexType _partitionDofCount; // how many dofs belong to the local partition
//  std::vector<GlobalIndexType> _partitionDofCounts; // how many dofs belong to each MPI rank.
  std::vector<GlobalIndexType> _partitionDofOffsets; // offsets for each partition, plus we store global dof count at the end
//  GlobalIndexType _globalDofCount;
  
  bool _hasSpaceOnlyTrialVariable;
  
  GlobalIndexType _partitionFieldDofCount;
  GlobalIndexType _partitionFluxDofCount;
  GlobalIndexType _partitionTraceDofCount;
  
  map< GlobalIndexType, CellConstraints > _constraintsCache;
  map< GlobalIndexType, LocalDofMapperPtr > _dofMapperCache;
  map< GlobalIndexType, map<int, map<int, LocalDofMapperPtr> > > _dofMapperForVariableOnSideCache; // cellID --> side --> variable --> LocalDofMapper
  map< GlobalIndexType, SubcellDofIndices> _ownedGlobalDofIndicesCache; // (cellID --> SubcellDofIndices)
  map< GlobalIndexType, SubcellDofIndices> _globalDofIndicesForCellCache; // (cellID --> SubcellDofIndices) -- this has a lot of overlap in its data with the _ownedGlobalDofIndicesCache; could save some memory by only storing the difference
  map< pair<GlobalIndexType,pair<int,unsigned>>, set<GlobalIndexType>> _fittableGlobalIndicesCache; // keys: (cellID,(varID,sideOrdinal))
  
  vector<unsigned> allBasisDofOrdinalsVector(int basisCardinality);

  static string annotatedEntityToString(AnnotatedEntity &entity);
  
  typedef vector< SubBasisDofMapperPtr > BasisMap;
  BasisMap getBasisMapDiscontinuousVolumeRestrictedToSide(GlobalIndexType cellID, SubcellDofIndices& dofOwnershipInfo, VarPtr var, int sideOrdinal);
  static BasisMap getRestrictedBasisMap(BasisMap &basisMap, const set<int> &basisDofOrdinalRestriction); // restricts to part of the basis

  typedef pair< IndexType, unsigned > CellPair;  // (cellIndex, sideOrdinal) -- sideOrdinal contains the specified entity
  CellPair cellContainingEntityWithLeastH1Order(int d, IndexType entityIndex, 
                                                std::vector< std::pair<std::string, int> > tags
                                                = std::vector< std::pair<std::string, int> >()); // if tags are specified, if the entity matches one or more of them, only matches cells that contain a side that also matches; throws an exception if none does.
  
  void distributeGlobalDofOwnership(const map<GlobalIndexType, SubcellDofIndices> &myOwnedGlobalDofs,
                                    const set<GlobalIndexType> &remoteCellIDs);
  
  void distributeCellGlobalDofs(const map<GlobalIndexType, SubcellDofIndices> &myCellGlobalDofs,
                                const set<GlobalIndexType> &remoteCellIDs);
  
  AnnotatedEntity* getConstrainingEntityInfo(GlobalIndexType cellID, CellConstraints &cellConstraints, VarPtr var, int d, int scord);
  void getConstrainingEntityInfo(GlobalIndexType cellID, CellConstraints &cellConstraints, VarPtr var, int d, int scord,
                                 AnnotatedEntity* &constrainingInfo, OwnershipInfo* &ownershipInfo, bool &spaceOnlyConstraint);
  
  vector<GlobalIndexType> getGlobalDofOrdinalsForSubcell(GlobalIndexType cellID, VarPtr var, int d, int scord);
  
  SubcellDofIndices & getOwnedGlobalDofIndices(GlobalIndexType cellID);

  set<GlobalIndexType> getFittableGlobalDofIndices(GlobalIndexType cellID, int sideOrdinal, int varID = -1); // returns the global dof indices for basis functions which have support on the given side (i.e. their support intersected with the side has positive measure).  This is determined by taking the union of the global dof indices defined on all the constraining sides for the given side (the constraining sides are by definition unconstrained).  If varID of -1 is specified, returns dof indices corresponding to all variables; otherwise, returns dof indices only for the specified variable.

  vector<int> H1Order(GlobalIndexType cellID, unsigned sideOrdinal); // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...
  
  RefinementBranch volumeRefinementsForSideEntity(IndexType sideEntityIndex);

protected:
  void clearCaches();
public:
  // these are public just for easier testing:
  BasisMap getBasisMap(GlobalIndexType cellID, SubcellDofIndices& dofOwnershipInfo, VarPtr var);
  BasisMap getBasisMap(GlobalIndexType cellID, SubcellDofIndices& dofOwnershipInfo, VarPtr var, int sideOrdinal);
  
  CellConstraints getCellConstraints(GlobalIndexType cellID);
  LocalDofMapperPtr getDofMapper(GlobalIndexType cellID, int varIDToMap = -1, int sideOrdinalToMap = -1);
  
  using GlobalDofAssignment::getGlobalDofIndices; // same name, different interface (a little ugly)
  SubcellDofIndices& getGlobalDofIndices(GlobalIndexType cellID);
//  set<GlobalIndexType> getGlobalDofIndicesForIntegralContribution(GlobalIndexType cellID, int sideOrdinal); // assuming an integral is being done over the whole mesh skeleton, returns either an empty set or the global dof indices associated with the given side, depending on whether the cell "owns" the side for the purpose of such contributions.
  // ! returns the permutation that goes from the indicated cell's view of the subcell to the constraining cell's view.
  unsigned getConstraintPermutation(GlobalIndexType cellID, unsigned subcdim, unsigned subcord);
  // ! returns the permutation that goes from the indicated side's view of the subcell to the constraining side's view.
  unsigned getConstraintPermutation(GlobalIndexType cellID, unsigned sideOrdinal, unsigned subcdim, unsigned subcord);
public:
  GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                 unsigned initialH1OrderTrial, unsigned testOrderEnhancement);

  GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                 vector<int> initialH1OrderTrial, unsigned testOrderEnhancement);
  
  // ! Default is false.  Checking constraint consistency is useful for debugging purposes, though.
  void setCheckConstraintConsistency(bool value);
  void setCellPRefinements(const map<GlobalIndexType,int>& pRefinements) override;
  
  GlobalDofAssignmentPtr deepCopy() override;
  
  void didHRefine(const set<GlobalIndexType> &parentCellIDs) override;
  void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) override;
  void willHUnrefine(const set<GlobalIndexType> &parentCellIDs) override;

  void didChangePartitionPolicy() override;
  
  int getH1OrderOnEdge(GlobalIndexType cellID, unsigned edgeOrdinal);
  
  GlobalIndexType globalDofCount() override;
  
  //!! Returns the global dof indices for the indicated cell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) override;
  
  //!! Returns the global dof indices for the indicated subcell.  Only guaranteed to provide correct values for cells that belong to the local partition.
  set<GlobalIndexType> globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal, bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime) override;
  
  // ! Returns the global dof indices, in the same order as the basis ordinals, for a discontinuous variable.
  // ! For minimum-rule meshes, may throw an exception if invoked with a continuous variable's ID as argument.
  vector<GlobalIndexType> globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID) override;
  
  //!! Returns the global dof indices for the partition.
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber) override;

  bool isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const override;

  GlobalIndexType numPartitionOwnedGlobalFieldIndices() override;
  GlobalIndexType numPartitionOwnedGlobalFluxIndices() override;
  GlobalIndexType numPartitionOwnedGlobalTraceIndices() override;

  using DofInterpreter::interpretLocalData;
  void interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localData,
                          Camellia::FieldContainer<double> &globalData,
                          Camellia::FieldContainer<GlobalIndexType> &globalDofIndices) override;
  void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal,
                                       const Camellia::FieldContainer<double> &basisCoefficients,
                                       Camellia::FieldContainer<double> &globalCoefficients,
                                       Camellia::FieldContainer<GlobalIndexType> &globalDofIndices) override;
  void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients,
                                   const Epetra_MultiVector &globalCoefficients) override;
  
  void interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients,
                                   const TVectorPtr<double> globalCoefficients) override;
  
  IndexType localDofCount() override; // local to the MPI node

  PartitionIndexType partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) override;
  void printConstraintInfo(GlobalIndexType cellID);
  void printGlobalDofInfo();
  void rebuildLookups() override;
  
  void setAllowMeshTopologyPruning(bool allowMeshTopologyPruning) override;
  bool allowMeshTopologyPruning() override;
  
  static void distributeSubcellDofIndices(Teuchos_CommPtr Comm,
                                          const map<GlobalIndexType, SubcellDofIndices> &rankLocalCellDofIndices,
                                          const map<GlobalIndexType, PartitionIndexType> &remoteCellOwners,
                                          map<GlobalIndexType, SubcellDofIndices> &remoteCellDofIndices);
};
}

#endif /* defined(__Camellia_debug__GDAMinimumRule__) */
