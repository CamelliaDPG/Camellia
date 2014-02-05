//
//  GlobalDofAssignment.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/20/14.
//
//

#include "GlobalDofAssignment.h"

#include "Teuchos_GlobalMPISession.hpp"

// subclasses:
#include "GDAMinimumRule.h"
#include "GDAMaximumRule2D.h"

GlobalDofAssignment::GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                         DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                         unsigned initialH1OrderTrial, unsigned testOrderEnhancement) {
  _meshTopology = meshTopology;
  _varFactory = varFactory;
  _dofOrderingFactory = dofOrderingFactory;
  _partitionPolicy = partitionPolicy;
  _initialH1OrderTrial = initialH1OrderTrial;
  _testOrderEnhancement = testOrderEnhancement;
  
  set<unsigned> cellIDs = meshTopology->getActiveCellIndices();
  for (set<unsigned>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    unsigned cellID = *cellIDIt;
    _cellH1Orders[cellID] = _initialH1OrderTrial;
  }
  
  _numPartitions = Teuchos::GlobalMPISession::getNProc();  
  determineActiveElements();
}


GlobalIndexType GlobalDofAssignment::activeCellOffset() {
  return _activeCellOffset;
}

void GlobalDofAssignment::determineActiveElements() {
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();
  
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
  
  //  cout << "determineActiveElements(): there are "  << activeCellIDs.size() << " active elements.\n";
  _partitions.clear();
  _partitionForCellID.clear();
  FieldContainer<GlobalIndexType> partitionedMesh(_numPartitions,activeCellIDs.size());
  _partitionPolicy->partitionMesh(_meshTopology.get(),_numPartitions,partitionedMesh);
  //  cout << "partitionedMesh:\n" << partitionedMesh;
  
  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i<partitionedMesh.dimension(0); i++) {
    vector< GlobalIndexType > partition;
    for (int j=0; j<partitionedMesh.dimension(1); j++) {
      //      cout << "partitionedMesh(i,j) = " << partitionedMesh(i,j) << endl;
      if (partitionedMesh(i,j) == -1) break; // no more elements in this partition
      GlobalIndexType cellID = partitionedMesh(i,j);
      partition.push_back( cellID );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
    if (partitionNumber > i) {
      _activeCellOffset += partition.size();
    }
  }
}

void GlobalDofAssignment::setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy ) {
  _partitionPolicy = partitionPolicy;
  determineActiveElements();
  didChangePartitionPolicy();
}

// maximumRule2D provides support for legacy (MultiBasis) meshes
GlobalDofAssignmentPtr GlobalDofAssignment::maximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                                          DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                                          unsigned initialH1OrderTrial, unsigned testOrderEnhancement) {
  return Teuchos::rcp( new GDAMaximumRule2D(meshTopology,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}

GlobalDofAssignmentPtr GlobalDofAssignment::minumumRule(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                                        DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                                        unsigned initialH1OrderTrial, unsigned testOrderEnhancement) {
  return Teuchos::rcp( new GDAMinimumRule(meshTopology,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}