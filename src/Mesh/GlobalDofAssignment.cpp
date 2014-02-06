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
  
//  unsigned testOrder = initialH1OrderTrial + testOrderEnhancement;
  // assign some initial element types:
  set<IndexType> cellIndices = _meshTopology->getActiveCellIndices();
  set<GlobalIndexType> activeCellIDs;
  activeCellIDs.insert(cellIndices.begin(),cellIndices.end()); // for distributed mesh, we'd do some logic with cellID offsets for each MPI rank.  (cellID = cellIndex + cellIDOffsetForRank)
  
  unsigned spaceDim = _meshTopology->getSpaceDim();
  unsigned sideDim = spaceDim - 1;
  
  map<GlobalIndexType, unsigned> sideIndexParityAssignmentCount; // tracks the number of times each side in the mesh has been assigned a parity.
  for (set<GlobalIndexType>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    if (cell->isParent() || (cell->getParent().get() != NULL)) {
      // enforcing this allows us to assume that each face that isn't on the boundary will be treated exactly twice...
      cout << "GDAMaximumRule2D constructor only supports mesh topologies that are unrefined.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GDAMaximumRule2D constructor only supports mesh topologies that are unrefined.\n");
    }
//    DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(initialH1OrderTrial, *cell->topology());
//    DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testOrder, *cell->topology());
//    ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cell->topology());
//    _elementTypeForCell[cellID] = elemType;
//    _cellH1Orders[cellID] = _initialH1OrderTrial;
    
    assignInitialElementType(cellID);
    
    //    cout << "Assigned trialOrdering to cell " << cellID << ":\n" << *trialOrdering;
    
    unsigned sideCount = cell->topology()->getSideCount();
    vector<int> cellParities(sideCount);
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
      if (sideIndexParityAssignmentCount[sideEntityIndex] == 0) {
        cellParities[sideOrdinal] = 1;
      } else if (sideIndexParityAssignmentCount[sideEntityIndex] == 1) {
        cellParities[sideOrdinal] = -1;
      } else {
        cout << "Internal error during GDAMaxRule2D construction: encountered side more than twice.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: encountered side more than twice.");
      }
      sideIndexParityAssignmentCount[sideEntityIndex]++;
    }
    _cellSideParitiesForCellID[cellID] = cellParities;
  }
  
  _numPartitions = Teuchos::GlobalMPISession::getNProc();  
  determineActiveElements();
}


GlobalIndexType GlobalDofAssignment::activeCellOffset() {
  return _activeCellOffset;
}

void GlobalDofAssignment::assignInitialElementType( GlobalIndexType cellID ) {
  if (_cellH1Orders.find(cellID) == _cellH1Orders.end()) {
    _cellH1Orders[cellID] = _initialH1OrderTrial;
  }
  
  int testDegree = _cellH1Orders[cellID] + _testOrderEnhancement;
  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(_cellH1Orders[cellID], *cell->topology());
  DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testDegree, *cell->topology());
  ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cell->topology());
  _elementTypeForCell[cellID] = elemType;
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

void GlobalDofAssignment::didHRefine(const set<GlobalIndexType> &parentCellIDs) { // subclasses should call super
  
}
void GlobalDofAssignment::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) { // subclasses should call super
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    _cellH1Orders[*cellIDIt] += deltaP;
  }
  // the appropriate modifications to _elementTypeForCell are left to subclasses
}
void GlobalDofAssignment::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) { // subclasses should call super
  
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