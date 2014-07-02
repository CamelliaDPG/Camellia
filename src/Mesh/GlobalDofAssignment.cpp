//
//  GlobalDofAssignment.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/20/14.
//
//

#include "GlobalDofAssignment.h"

#include "Teuchos_GlobalMPISession.hpp"

#include "CamelliaDebugUtility.h"

// subclasses:
#include "GDAMinimumRule.h"
#include "GDAMaximumRule2D.h"

#include "CamelliaCellTools.h"

GlobalDofAssignment::GlobalDofAssignment(MeshTopologyPtr meshTopology, VarFactory varFactory,
                                         DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                         unsigned initialH1OrderTrial, unsigned testOrderEnhancement, bool enforceConformityLocally) {
  _meshTopology = meshTopology;
  _varFactory = varFactory;
  _dofOrderingFactory = dofOrderingFactory;
  _partitionPolicy = partitionPolicy;
  _initialH1OrderTrial = initialH1OrderTrial;
  _testOrderEnhancement = testOrderEnhancement;
  _enforceConformityLocally = enforceConformityLocally;
  
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
    
    assignInitialElementType(cellID);
    assignParities(cellID);
//    unsigned sideCount = CamelliaCellTools::getSideCount(*cell->topology());
//    
//    vector<int> cellParities(sideCount);
//    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
//      unsigned sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
//      if (sideIndexParityAssignmentCount[sideEntityIndex] == 0) {
//        cellParities[sideOrdinal] = 1;
//      } else if (sideIndexParityAssignmentCount[sideEntityIndex] == 1) {
//        cellParities[sideOrdinal] = -1;
//      } else {
//        cout << "Internal error during GDAMaxRule2D construction: encountered side more than twice.\n";
//        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: encountered side more than twice.");
//      }
//      sideIndexParityAssignmentCount[sideEntityIndex]++;
//    }
//    _cellSideParitiesForCellID[cellID] = cellParities;
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
  DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(_cellH1Orders[cellID], *cell->topology(), _enforceConformityLocally);
  DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testDegree, *cell->topology());
  ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cell->topology());
  _elementTypeForCell[cellID] = elemType;
}

void GlobalDofAssignment::assignParities( GlobalIndexType cellID ) {
  CellPtr cell = _meshTopology->getCell(cellID);

  unsigned sideCount = CamelliaCellTools::getSideCount(*cell->topology());
  
  vector<int> cellParities(sideCount);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighbor(sideOrdinal);
    GlobalIndexType neighborCellID = neighborInfo.first;
    if (neighborCellID == -1) { // boundary --> parity is 1
      cellParities[sideOrdinal] = 1;
    } else {
      CellPtr neighbor = _meshTopology->getCell(neighborCellID);
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighbor(neighborInfo.second);
      bool neighborIsPeer = neighborNeighborInfo.first == cellID;
      if (neighborIsPeer) { // then the lower cellID gets the positive parity
        cellParities[sideOrdinal] = (cellID < neighborCellID) ? 1 : -1;
      } else {
        CellPtr parent = cell->getParent();
        if (parent.get() == NULL) {
          cout << "ERROR: in assignParities(), encountered cell with non-peer neighbor but without parent.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "in assignParities(), encountered cell with non-peer neighbor but without parent");
        }
        // inherit parent's parity along the shared side:
        unsigned childOrdinal = parent->childOrdinal(cellID);
        unsigned parentSideOrdinal = parent->refinementPattern()->parentSideLookupForChild(childOrdinal)[sideOrdinal];
        cellParities[sideOrdinal] = _cellSideParitiesForCellID[parent->cellIndex()][parentSideOrdinal];
      }
    }
  }
  _cellSideParitiesForCellID[cellID] = cellParities;
  
  // if this cell is a parent, then we should treat its children as well (children without peer neighbors will inherit any parity flips)
  if (cell->isParent()) {
    vector<GlobalIndexType> childIndices = cell->getChildIndices();
    for (vector<GlobalIndexType>::iterator childIndexIt = childIndices.begin(); childIndexIt != childIndices.end(); childIndexIt++) {
      assignParities(*childIndexIt);
    }
  }
}

GlobalIndexType GlobalDofAssignment::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber) {
  if (partitionNumber == -1) {
    // determine the partition number for the cellIndex
    int partitionCellOffset = 0;
    for (PartitionIndexType i=0; i<partitionNumber; i++) {
      int numCellIDsForPartition = _cellIDsForElementType[i][elemTypePtr.get()].size();
      if (partitionCellOffset + numCellIDsForPartition > cellIndex) {
        partitionNumber = i;
        cellIndex -= partitionCellOffset; // rewrite as a local cellIndex
        break;
      }
      partitionCellOffset += numCellIDsForPartition;
    }
    if (partitionNumber == -1) {
      cout << "cellIndex is out of bounds.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex is out of bounds.");
    }
  }
  if ( ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) != _cellIDsForElementType[partitionNumber].end() )
      &&
      (_cellIDsForElementType[partitionNumber][elemTypePtr.get()].size() > cellIndex ) ) {
    return _cellIDsForElementType[partitionNumber][elemTypePtr.get()][cellIndex];
  } else return -1;
}

vector<GlobalIndexType> GlobalDofAssignment::cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr) {
  if (partitionNumber == -1) {
    cout << "cellIDsOfElementType called with partitionNumber==-1.  Returning empty vector.\n";
    return vector<GlobalIndexType>();
  }
  map<ElementType*, vector<GlobalIndexType> >::iterator cellIDsIt = _cellIDsForElementType[partitionNumber].find(elemTypePtr.get());
  if (cellIDsIt == _cellIDsForElementType[partitionNumber].end()) {
    return vector<GlobalIndexType>();
  }
  return cellIDsIt->second;
}

vector< GlobalIndexType > GlobalDofAssignment::cellsInPartition(PartitionIndexType partitionNumber) {
  int rank     = Teuchos::GlobalMPISession::getRank();
  if (partitionNumber == -1) {
    partitionNumber = rank;
  }
  return _partitions[partitionNumber];
}

FieldContainer<double> GlobalDofAssignment::cellSideParitiesForCell( GlobalIndexType cellID ) {
  vector<int> parities = _cellSideParitiesForCellID[cellID];
  int numSides = parities.size();
  FieldContainer<double> cellSideParities(1,numSides);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    cellSideParities(0,sideIndex) = parities[sideIndex];
  }
  return cellSideParities;
}

void GlobalDofAssignment::determineActiveElements() {
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();
  
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
  
  //  cout << "determineActiveElements(): there are "  << activeCellIDs.size() << " active elements.\n";
  _partitions.clear();
  _partitionForCellID.clear();
  FieldContainer<GlobalIndexType> partitionedMesh(_numPartitions,activeCellIDs.size());
  _partitionPolicy->partitionMesh(_meshTopology.get(),_numPartitions,partitionedMesh);
  
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
//    if (partitionNumber==0) cout << "partition " << i << ": ";
//    if (partitionNumber==0) print("",partition);
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

vector< ElementTypePtr > GlobalDofAssignment::elementTypes(PartitionIndexType partitionNumber) {
  if (partitionNumber != -1) {
    vector< ElementTypePtr > elemTypes;
    map< ElementType*, vector<GlobalIndexType> > cellIDsForElemType = _cellIDsForElementType[partitionNumber];
    for (map< ElementType*, vector<GlobalIndexType> >::iterator elemTypeIt = cellIDsForElemType.begin(); elemTypeIt != cellIDsForElemType.end(); elemTypeIt++) {
      elemTypes.push_back(Teuchos::rcp(elemTypeIt->first,false)); // false: doesn't own memory…
    }
    return elemTypes;
  } else {
    int numRanks = Teuchos::GlobalMPISession::getNProc();
    set< ElementType* > includedTypes;
    vector< ElementTypePtr > types;
    for (int rank=0; rank<numRanks; rank++) {
      vector< ElementTypePtr > elemTypesForRank = elementTypes(rank);
      for (vector< ElementTypePtr >::iterator typeForRankIt = elemTypesForRank.begin(); typeForRankIt != elemTypesForRank.end(); typeForRankIt++) {
        ElementTypePtr elemType = *typeForRankIt;
        if (includedTypes.find(elemType.get()) == includedTypes.end()) {
          types.push_back(elemType);
          includedTypes.insert(elemType.get());
        }
      }
    }
    return types;
  }
}

DofOrderingFactoryPtr GlobalDofAssignment::getDofOrderingFactory() {
  return _dofOrderingFactory;
}

ElementTypeFactory & GlobalDofAssignment::getElementTypeFactory() {
  return _elementTypeFactory;
}

GlobalIndexType GlobalDofAssignment::globalCellIndex(GlobalIndexType cellID) {
  int partitionNumber     = partitionForCellID(cellID);
  GlobalIndexType cellIndex = partitionLocalCellIndex(cellID, partitionNumber);
  ElementType* elemType = _elementTypeForCell[cellID].get();

  for (PartitionIndexType i=0; i<partitionNumber; i++) {
    cellIndex += _cellIDsForElementType[i][elemType].size();
  }
  return cellIndex;
}

int GlobalDofAssignment::getInitialH1Order() {
  return _initialH1OrderTrial;
}

PartitionIndexType GlobalDofAssignment::getPartitionCount() {
  return _numPartitions;
}

int GlobalDofAssignment::getTestOrderEnrichment() {
  return _testOrderEnhancement;
}

void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients) {
  DofOrderingPtr trialOrder = elementType(cellID)->trialOrderPtr;
  FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++) {
    int trialID = *trialIDIt;
    int sideCount = trialOrder->getNumSidesForVarID(trialID);
    for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++) {
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<double> fittedGlobalCoefficients;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++) {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, 0, fittedGlobalCoefficients[i]);
      }
    }
  }
}

void GlobalDofAssignment::setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy ) {
  _partitionPolicy = partitionPolicy;
  determineActiveElements();
  didChangePartitionPolicy();
}

PartitionIndexType GlobalDofAssignment::partitionForCellID( GlobalIndexType cellID ) {
  return _partitionForCellID[ cellID ];
}

IndexType GlobalDofAssignment::partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber) {
  if (partitionNumber == -1) {
    partitionNumber     = Teuchos::GlobalMPISession::getRank();
  }

  ElementType* elemType = _elementTypeForCell[cellID].get();
  vector<GlobalIndexType> cellIDsOfType = _cellIDsForElementType[partitionNumber][elemType];
  for (IndexType cellIndex = 0; cellIndex < cellIDsOfType.size(); cellIndex++) {
    if (cellIDsOfType[cellIndex] == cellID) {
      return cellIndex;
    }
  }
  return -1;
}

void GlobalDofAssignment::registerSolution(Solution* solution) {
  _registeredSolutions.push_back( solution );
}

void GlobalDofAssignment::unregisterSolution(Solution* solution) {
  for (vector< Solution* >::iterator solnIt = _registeredSolutions.begin();
       solnIt != _registeredSolutions.end(); solnIt++) {
    if ( *solnIt == solution ) {
      _registeredSolutions.erase(solnIt);
      return;
    }
  }
  cout << "GDAMaximumRule2D::unregisterSolution: Solution not found.\n";
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