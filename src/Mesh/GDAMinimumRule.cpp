//
//  GDAMinimumRule.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMinimumRule.h"

GDAMinimumRule::GDAMinimumRule(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
: GlobalDofAssignment(meshTopology,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement)
{
  
}


void GDAMinimumRule::didChangePartitionPolicy() {
  rebuildLookups();
}

void GDAMinimumRule::didHRefine(const set<GlobalIndexType> &parentCellIDs) {
  this->GlobalDofAssignment::didHRefine(parentCellIDs);
  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
    GlobalIndexType parentCellID = *cellIDIt;
    CellPtr parentCell = _meshTopology->getCell(parentCellID);
    vector<IndexType> childIDs = parentCell->getChildIndices();
    int parentH1Order = _cellH1Orders[parentCellID];
    for (vector<IndexType>::iterator childIDIt = childIDs.begin(); childIDIt != childIDs.end(); childIDIt++) {
      _cellH1Orders[*childIDIt] = parentH1Order;
      assignInitialElementType(*childIDIt);
    }
  }
  rebuildLookups();
}

void GDAMinimumRule::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) {
  this->GlobalDofAssignment::didPRefine(cellIDs, deltaP);
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    assignInitialElementType(*cellIDIt);
  }
  rebuildLookups();
}

void GDAMinimumRule::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) {
  this->GlobalDofAssignment::didHUnrefine(parentCellIDs);
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::didHUnrefine() unimplemented.\n";
  rebuildLookups();
}

ElementTypePtr GDAMinimumRule::elementType(GlobalIndexType cellID) {
  return Teuchos::rcp( (ElementType*) NULL);
//  return _elementTypeForCell[cellID];
}

GlobalIndexType GDAMinimumRule::globalDofCount() {
  // TODO: implement this
  cout << "WARNING: globalDofCount() unimplemented.\n";
  return 0;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForPartition(PartitionIndexType partitionNumber) {
  // TODO: implement this
  set<GlobalIndexType> globalDofIndices;
  cout << "WARNING: GDAMinimumRule::globalDofIndicesForPartition() unimplemented.\n";
  return globalDofIndices;
}

void GDAMinimumRule::interpretGlobalDofs(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs) {
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::interpretGlobalDofs() unimplemented.\n";
}

void GDAMinimumRule::interpretLocalDofs(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                                        FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) {
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::interpretLocalDofs() unimplemented.\n";
}

IndexType GDAMinimumRule::localDofCount() {
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

void GDAMinimumRule::rebuildLookups() {
  determineActiveElements(); // call to super: constructs cell partitionings
  
}