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

void GDAMinimumRule::didHRefine(const set<int> &parentCellIDs) {
//  rebuildLookups();
}

void GDAMinimumRule::didPRefine(const set<int> &cellIDs, int deltaP) {
  
//  rebuildLookups();
}

void GDAMinimumRule::didHUnrefine(const set<int> &parentCellIDs) {
//  rebuildLookups();
}

ElementTypePtr GDAMinimumRule::elementType(unsigned cellID) {
  return Teuchos::rcp( (ElementType*) NULL);
//  return _elementTypeForCell[cellID];
}

unsigned GDAMinimumRule::globalDofCount() {
  // TODO: implement this
  cout << "WARNING: globalDofCount() unimplemented.\n";
  return 0;
}

unsigned GDAMinimumRule::localDofCount() {
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

void GDAMinimumRule::rebuildLookups() {
  
}