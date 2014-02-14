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

struct ConstrainingSubsideInfo {
  GlobalIndexType cellID;
  unsigned sideOrdinal;
  unsigned subsideOrdinalInSide;
};

struct ConstrainingCellInfo {
  GlobalIndexType cellID;
  unsigned sideOrdinal;
  vector< vector<GlobalIndexType> > owningCellIDForSideSubcell; // outer vector indexed by subcell dimension; inner vector indexed by subcell ordinal in side
};

struct CellConstraints {
  vector<ConstrainingCellInfo> sideConstraints; // one entry for each side
  vector< vector<ConstrainingSubsideInfo> > subsideConstraints; // outer vector: one entry for each side.  Inner vector: one entry for each subside.
  vector< vector<GlobalIndexType> > owningCellIDForSubcell; // outer vector indexed by subcell dimension; inner vector indexed by subcell ordinal in cell
  LocalDofMapperPtr dofMapper;
};

struct ConstrainedDofMap {
  set<int> localDofIndices;
  
};

class GDAMinimumRule : public GlobalDofAssignment {
  map<GlobalIndexType, IndexType> _cellDofOffsets; // (cellID -> first partition-local dof index for that cell)  within the partition, offsets for the owned dofs in cell
  GlobalIndexType _partitionDofOffset; // add to partition-local dof indices to get a global dof index
  GlobalIndexType _partitionDofCount; // how many dofs belong to the local partition
  GlobalIndexType _globalDofCount;
  
  CellConstraints getCellConstraints(GlobalIndexType cellID);
  
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
