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
//  rebuildLookups();
}

void GDAMinimumRule::didHRefine(const set<GlobalIndexType> &parentCellIDs) {
//  rebuildLookups();
}

void GDAMinimumRule::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) {
  
//  rebuildLookups();
}

void GDAMinimumRule::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) {
//  rebuildLookups();
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
  
}