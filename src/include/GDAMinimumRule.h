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

#include "GlobalDofAssignment.h"

#include "LocalDofMapper.h"

#include "BasisReconciliation.h"

struct ConstrainingSubsideInfo {
  GlobalIndexType cellID;
  unsigned sideOrdinal;
  unsigned subsideOrdinalInSide;
  unsigned subsideVertexPermutation; // the composed inverse of the coarse side's permutation and the fine side's permutation
};

struct ConstrainingCellInfo {
  GlobalIndexType cellID;
  unsigned sideOrdinal;
  unsigned sideVertexPermutation; // the composed inverse of the coarse domain's permutation and the fine domain's permutation
  vector< vector< pair<GlobalIndexType,GlobalIndexType> > > owningCellIDForSideSubcell; // outer vector indexed by subcell dimension; inner vector indexed by subcell ordinal in side
};

struct CellConstraints {
  vector<ConstrainingCellInfo> sideConstraints; // one entry for each side
  vector< vector<ConstrainingSubsideInfo> > subsideConstraints; // outer vector: one entry for each side.  Inner vector: one entry for each subside.
  vector< vector< pair<GlobalIndexType,GlobalIndexType> > > owningCellIDForSubcell; // outer vector indexed by subcell dimension; inner vector indexed by subcell ordinal in cell.  Pairs are (CellID, subcellIndex in MeshTopology)
};

struct ConstrainedDofMap {
  set<int> localDofIndices;
  
};

class GDAMinimumRule : public GlobalDofAssignment {
  BasisReconciliation _br;
  map<GlobalIndexType, IndexType> _cellDofOffsets; // (cellID -> first partition-local dof index for that cell)  within the partition, offsets for the owned dofs in cell
  map<GlobalIndexType, GlobalIndexType> _globalCellDofOffsets; // (cellID -> first global dof index for that cell)
  GlobalIndexType _partitionDofOffset; // add to partition-local dof indices to get a global dof index
  GlobalIndexType _partitionDofCount; // how many dofs belong to the local partition
  GlobalIndexType _globalDofCount;
  
  map< GlobalIndexType, CellConstraints > _constraintsCache;

  typedef map<int, vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
  typedef map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
  typedef vector< SubCellOrdinalToMap > SubCellDofIndexInfo; // index to vector: subcell dimension
  
  CellConstraints getCellConstraints(GlobalIndexType cellID);
  LocalDofMapperPtr getDofMapper(GlobalIndexType cellID, CellConstraints &constraints);
  
  SubCellDofIndexInfo getOwnedGlobalDofIndices(GlobalIndexType cellID, CellConstraints &cellConstraints);
  
  int H1Order(GlobalIndexType cellID, unsigned sideOrdinal); // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...
public:
  GDAMinimumRule(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                 unsigned initialH1OrderTrial, unsigned testOrderEnhancement);
  
  void didHRefine(const set<GlobalIndexType> &parentCellIDs);
  void didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP);
  void didHUnrefine(const set<GlobalIndexType> &parentCellIDs);
  
  void didChangePartitionPolicy();
  
  ElementTypePtr elementType(GlobalIndexType cellID);
  GlobalIndexType globalDofCount();
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType partitionNumber);
  void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                          FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs);
  IndexType localDofCount(); // local to the MPI node
  
  void rebuildLookups();
};

#endif /* defined(__Camellia_debug__GDAMinimumRule__) */
