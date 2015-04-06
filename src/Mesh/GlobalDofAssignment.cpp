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

#include "Solution.h"

#include "CamelliaCellTools.h"
#include "MPIWrapper.h"

#include "CondensedDofInterpreter.h"

using namespace Intrepid;
using namespace Camellia;
using namespace std;

GlobalDofAssignment::GlobalDofAssignment(MeshPtr mesh, VarFactory varFactory,
                                         DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                         unsigned initialH1OrderTrial, unsigned testOrderEnhancement, bool enforceConformityLocally) : DofInterpreter(mesh) {

  _mesh = mesh;
  _meshTopology = mesh->getTopology();
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

  for (set<GlobalIndexType>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
//    CellPtr cell = _meshTopology->getCell(cellID);
//    if (cell->isParent() || (cell->getParent().get() != NULL)) {
//      // enforcing this allows us to assume that each face that isn't on the boundary will be treated exactly twice...
//      cout << "GlobalDofAssignment constructor only supports mesh topologies that are unrefined.\n";
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GlobalDofAssignment constructor only supports mesh topologies that are unrefined.\n");
//    }

    assignInitialElementType(cellID);
    assignParities(cellID);
  }

  _numPartitions = Teuchos::GlobalMPISession::getNProc();

  _partitions = vector<set<GlobalIndexType> >(_numPartitions);

  // before repartitioning (which should happen immediately), put all active cells on rank 0
  _partitions[0] = _mesh->getActiveCellIDs();
  constructActiveCellMap();
}

GlobalDofAssignment::GlobalDofAssignment( GlobalDofAssignment &otherGDA ) : DofInterpreter(Teuchos::null) {  // subclass deepCopy() is responsible for filling this in post-construction
  _activeCellOffset = otherGDA._activeCellOffset;
  _cellSideParitiesForCellID = otherGDA._cellSideParitiesForCellID;

  _elementTypeFactory = otherGDA._elementTypeFactory;
  _enforceConformityLocally = otherGDA._enforceConformityLocally;

  _mesh = Teuchos::null;         // subclass deepCopy() is responsible for filling this in post-construction
  _meshTopology = Teuchos::null; // subclass deepCopy() is responsible for filling this in post-construction
  _varFactory = otherGDA._varFactory;
  _dofOrderingFactory = otherGDA._dofOrderingFactory;
  _partitionPolicy = otherGDA._partitionPolicy;;
  _initialH1OrderTrial = otherGDA._initialH1OrderTrial;
  _testOrderEnhancement = otherGDA._testOrderEnhancement;

  _cellH1Orders = otherGDA._cellH1Orders;
  _elementTypeForCell = otherGDA._elementTypeForCell;

  _cellIDsForElementType = otherGDA._cellIDsForElementType;

  _partitions = otherGDA._partitions;
  _partitionForCellID = otherGDA._partitionForCellID;

  _activeCellMap = Teuchos::rcp( new Epetra_Map(*otherGDA._activeCellMap) );

  _numPartitions = otherGDA._numPartitions;

  // we leave _registeredSolutions empty
  ///_registeredSolutions;
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
  DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(_cellH1Orders[cellID], cell->topology(), _enforceConformityLocally);
  DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testDegree, cell->topology());
  ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cell->topology());
  _elementTypeForCell[cellID] = elemType;

  if (cell->getParent() != Teuchos::null) {
    GlobalIndexType parentCellID = cell->getParent()->cellIndex();
    if (_elementTypeForCell.find(parentCellID) == _elementTypeForCell.end())
      assignInitialElementType(parentCellID);
  }
}

void GlobalDofAssignment::assignParities( GlobalIndexType cellID ) {
  CellPtr cell = _meshTopology->getCell(cellID);

  unsigned sideCount = cell->getSideCount();

  vector<int> cellParities(sideCount);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal);
    GlobalIndexType neighborCellID = neighborInfo.first;
    if (neighborCellID == -1) { // boundary --> parity is 1
      cellParities[sideOrdinal] = 1;
    } else {
      CellPtr neighbor = _meshTopology->getCell(neighborCellID);
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(neighborInfo.second);
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
        if (_cellSideParitiesForCellID.find(parent->cellIndex()) == _cellSideParitiesForCellID.end()) {
          assignParities(parent->cellIndex());
        }
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

const set< GlobalIndexType > & GlobalDofAssignment::cellsInPartition(PartitionIndexType partitionNumber) {
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

void GlobalDofAssignment::constructActiveCellMap() {
  const set<GlobalIndexType>* cellIDs = &cellsInPartition(-1);
  FieldContainer<GlobalIndexTypeToCast> myCellIDsFC(cellIDs->size());

  int localIndex = 0;
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs->begin(); cellIDIt != cellIDs->end(); cellIDIt++, localIndex++) {
    myCellIDsFC(localIndex) = *cellIDIt;
  }

  int indexBase = 0;
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  if (myCellIDsFC.size()==0)
    _activeCellMap = Teuchos::rcp( new Epetra_Map(-1, myCellIDsFC.size(), NULL, indexBase, Comm) );
  else
    _activeCellMap = Teuchos::rcp( new Epetra_Map(-1, myCellIDsFC.size(), &myCellIDsFC[0], indexBase, Comm) );
}

void GlobalDofAssignment::repartitionAndMigrate() {
  _partitionPolicy->partitionMesh(_mesh.get(),_numPartitions);
  for (vector< SolutionPtr >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++) {
    // if solution has a condensed dof interpreter, we should reinitialize the mapping from interpreted to global dofs
    Teuchos::RCP<DofInterpreter> dofInterpreter = (*solutionIt)->getDofInterpreter();
    CondensedDofInterpreter* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter*>(dofInterpreter.get());
    if (condensedDofInterpreter != NULL) {
      condensedDofInterpreter->reinitialize();
    }

    (*solutionIt)->initializeLHSVector(); // rebuild LHS vector; global dofs will have changed. (important for addSolution)
  }
}

void GlobalDofAssignment::didHRefine(const set<GlobalIndexType> &parentCellIDs) { // subclasses should call super
  int rank     = Teuchos::GlobalMPISession::getRank();
  // until we repartition, assign the new children to the parent's partition
  for (set<GlobalIndexType>::const_iterator cellIDIt=parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
    GlobalIndexType parentID = *cellIDIt;
    if (_partitions[rank].find(parentID) != _partitions[rank].end()) {
      _partitions[rank].erase(parentID);
      CellPtr parent = _meshTopology->getCell(parentID);
      vector<GlobalIndexType> childIDs = parent->getChildIndices();
      _partitions[rank].insert(childIDs.begin(),childIDs.end());
    }
  }
  constructActiveCellMap();
}

void GlobalDofAssignment::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) { // subclasses should call super
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    _cellH1Orders[*cellIDIt] += deltaP;
  }
  // the appropriate modifications to _elementTypeForCell are left to subclasses
}

void GlobalDofAssignment::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) { // subclasses should call super
  cout << "WARNING: GlobalDofAssignment::didHUnrefine unimplemented.  At minimum, should update partition to drop children, and add parent.\n";
  // TODO: address this -- of course, Mesh doesn't yet support h-unrefinements, so might want to do that first.
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

Teuchos::RCP<Epetra_Map> GlobalDofAssignment::getActiveCellMap() {
  return _activeCellMap;
}

int GlobalDofAssignment::getCubatureDegree(GlobalIndexType cellID) {
  ElementTypePtr elemType = this->elementType(cellID);
  return elemType->trialOrderPtr->maxBasisDegree() + elemType->testOrderPtr->maxBasisDegree();
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

bool GlobalDofAssignment::getPartitions(FieldContainer<GlobalIndexType> &partitions) {
  if (_partitions.size() == 0) return false; // false: no partitions set
  int numPartitions = _partitions.size();
  int maxSize = 0;
  for (int i=0; i<numPartitions; i++) {
    maxSize = std::max<int>((int)_partitions[i].size(),maxSize);
  }
  partitions.resize(numPartitions,maxSize);
  partitions.initialize(-1);
  for (int i=0; i<numPartitions; i++) {
    int j=0;
    for (set<GlobalIndexType>::iterator cellIDIt = _partitions[i].begin();
         cellIDIt != _partitions[i].end(); cellIDIt++) {
      partitions(i,j) = *cellIDIt;
      j++;
    }
  }
  return true; // true: partitions container filled
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
    const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++) {
      int sideOrdinal = *sideIt;
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
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, 0, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
//        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

void GlobalDofAssignment::projectParentCoefficientsOntoUnsetChildren() {
  set<GlobalIndexType> rankLocalCellIDs = cellsInPartition(-1);

  for (vector< SolutionPtr >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++) {
    SolutionPtr soln = *solutionIt;
    for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCellIDs.begin(); cellIDIt != rankLocalCellIDs.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      if (soln->cellHasCoefficientsAssigned(cellID)) continue;

      CellPtr cell = _meshTopology->getCell(cellID);
      CellPtr parent = cell->getParent();
      if (parent.get()==NULL) continue;
      GlobalIndexType parentCellID = parent->cellIndex();
      if (! soln->cellHasCoefficientsAssigned(parentCellID)) continue;

      int childOrdinal = -1;
      vector<IndexType> childIndices = parent->getChildIndices();
      for (int i=0; i<childIndices.size(); i++) {
        if (childIndices[i]==cellID) childOrdinal = i;
        else childIndices[i] = -1; // indication that Solution should not compute the projection for this child
      }
      if (childOrdinal == -1) {
        cout << "ERROR: child not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child not found!");
      }
//      cout << "determining cellID " << parent->cellIndex() << "'s child " << childOrdinal << "'s coefficients.\n";
      soln->projectOldCellOntoNewCells(parent->cellIndex(), _elementTypeForCell[parentCellID], childIndices);
    }
  }
}

void GlobalDofAssignment::setPartitions(FieldContainer<GlobalIndexType> &partitionedMesh) {
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();

  int partitionNumber     = Teuchos::GlobalMPISession::getRank();

  //  cout << "determineActiveElements(): there are "  << activeCellIDs.size() << " active elements.\n";
  _partitions.clear();
  _partitionForCellID.clear();

  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i<partitionedMesh.dimension(0); i++) {
    set< GlobalIndexType > partition;
    for (int j=0; j<partitionedMesh.dimension(1); j++) {
      //      cout << "partitionedMesh(i,j) = " << partitionedMesh(i,j) << endl;
      if (partitionedMesh(i,j) == -1) break; // no more elements in this partition
      GlobalIndexType cellID = partitionedMesh(i,j);
      partition.insert( cellID );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
    //    if (partitionNumber==0) cout << "partition " << i << ": ";
    //    if (partitionNumber==0) print("",partition);
    if (partitionNumber > i) {
      _activeCellOffset += partition.size();
    }
  }
  constructActiveCellMap();
  projectParentCoefficientsOntoUnsetChildren();
  rebuildLookups();
}

void GlobalDofAssignment::setPartitions(std::vector<std::set<GlobalIndexType> > &partitions) {
  int thisPartitionNumber     = Teuchos::GlobalMPISession::getRank();

  // not sure numProcs == partitions.size() is a great requirement to impose, but it is an assumption we make in some places,
  // so we require it here.
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  TEUCHOS_TEST_FOR_EXCEPTION(numProcs != partitions.size(), std::invalid_argument, "partitions.size() must be equal to numProcs!");

  _partitions = partitions;
  _partitionForCellID.clear();

  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i< _partitions.size(); i++) {
    set< GlobalIndexType > partition;
    for (set< GlobalIndexType >::iterator cellIDIt = partitions[i].begin(); cellIDIt != partitions[i].end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      _partitionForCellID[cellID] = i;
    }
    if (thisPartitionNumber > i) {
      _activeCellOffset += partition.size();
    }
  }
  constructActiveCellMap();
  projectParentCoefficientsOntoUnsetChildren();
  rebuildLookups();
}

void GlobalDofAssignment::setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy ) {
  _partitionPolicy = partitionPolicy;
  repartitionAndMigrate();
}

PartitionIndexType GlobalDofAssignment::partitionForCellID( GlobalIndexType cellID ) {
  if (_partitionForCellID.find(cellID) != _partitionForCellID.end()) {
    return _partitionForCellID[ cellID ];
  } else {
    return -1;
  }
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

vector<SolutionPtr> GlobalDofAssignment::getRegisteredSolutions() {
  return _registeredSolutions;
}

void GlobalDofAssignment::registerSolution(SolutionPtr solution) {
  _registeredSolutions.push_back( solution );
}

void GlobalDofAssignment::setMeshAndMeshTopology(MeshPtr mesh) {
  // make copies of the RCPs that don't own memory.
  _mesh = Teuchos::rcp(mesh.get(), false);
  _meshTopology = Teuchos::rcp(mesh->getTopology().get(), false);

  this->DofInterpreter::_mesh = _mesh;
}

void GlobalDofAssignment::unregisterSolution(SolutionPtr solution) {
  for (vector< SolutionPtr >::iterator solnIt = _registeredSolutions.begin();
       solnIt != _registeredSolutions.end(); solnIt++) {
    if ( *solnIt == solution ) {
      _registeredSolutions.erase(solnIt);
      return;
    }
  }
  cout << "GDAMaximumRule2D::unregisterSolution: Solution not found.\n";
}

// maximumRule2D provides support for legacy (MultiBasis) meshes
GlobalDofAssignmentPtr GlobalDofAssignment::maximumRule2D(MeshPtr mesh, VarFactory varFactory,
                                                          DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                                          unsigned initialH1OrderTrial, unsigned testOrderEnhancement) {
  return Teuchos::rcp( new GDAMaximumRule2D(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}

GlobalDofAssignmentPtr GlobalDofAssignment::minimumRule(MeshPtr mesh, VarFactory varFactory,
                                                        DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                                                        unsigned initialH1OrderTrial, unsigned testOrderEnhancement) {
  return Teuchos::rcp( new GDAMinimumRule(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}
