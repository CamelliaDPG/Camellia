//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  GDAMinimumRule.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMinimumRule.h"

#include "BasisFactory.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "MeshTestUtility.h"
#include "MPIWrapper.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"
#include "TimeLogger.h"

using namespace std;
using namespace Camellia;

namespace Camellia {
  // DEBUGGING: define a method to fill in the "identity" weights container with actual weights, to try to figure out
  //            where, if anywhere, the results are different.
  SubBasisReconciliationWeights weightsManualIdentity(SubBasisReconciliationWeights identityWeights)
  {
    SubBasisReconciliationWeights newWeights;
    newWeights.fineOrdinals = identityWeights.fineOrdinals;
    newWeights.coarseOrdinals = identityWeights.coarseOrdinals;
    newWeights.weights = Camellia::FieldContainer<double>(newWeights.fineOrdinals.size(),newWeights.coarseOrdinals.size());
    for (int i=0; i<newWeights.weights.dimension(0); i++)
    {
      newWeights.weights(i,i) = 1.0;
    }
    return newWeights;
  }
}

void printDofIndexInfo(GlobalIndexType cellID, SubcellDofIndices &subcellDofIndices)
{
  //typedef vector< SubCellOrdinalToMap > SubcellDofIndices; // index to vector: subcell dimension
  cout << "Dof Index info for cell ID " << cellID << ":\n";
  subcellDofIndices.print(cout);
}

GDAMinimumRule::GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
  : GlobalDofAssignment(mesh,varFactory,dofOrderingFactory,partitionPolicy, vector<int>(1,initialH1OrderTrial), testOrderEnhancement, false)
{
  _hasSpaceOnlyTrialVariable = varFactory->hasSpaceOnlyTrialVariable();
  TimeLogger::sharedInstance()->createTimeEntry("read SubcellDofIndices");
  TimeLogger::sharedInstance()->createTimeEntry("write SubcellDofIndices");
}

GDAMinimumRule::GDAMinimumRule(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               vector<int> initialH1OrderTrial, unsigned testOrderEnhancement)
  : GlobalDofAssignment(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement, false)
{
  _hasSpaceOnlyTrialVariable = varFactory->hasSpaceOnlyTrialVariable();
  TimeLogger::sharedInstance()->createTimeEntry("read SubcellDofIndices");
  TimeLogger::sharedInstance()->createTimeEntry("write SubcellDofIndices");
}

vector<unsigned> GDAMinimumRule::allBasisDofOrdinalsVector(int basisCardinality)
{
  vector<unsigned> ordinals(basisCardinality);
  for (int i=0; i<basisCardinality; i++)
  {
    ordinals[i] = i;
  }
  return ordinals;
}

bool GDAMinimumRule::allowMeshTopologyPruning()
{
  return _allowMeshTopologyPruning;
}

typedef pair< IndexType, unsigned > CellPair;
CellPair GDAMinimumRule::cellContainingEntityWithLeastH1Order(int d, IndexType entityIndex, std::vector< std::pair<std::string, int> > tags)
{
  set< CellPair > cellsForSubcell = _meshTopology->getCellsContainingEntity(d, entityIndex, tags);
  
  if (cellsForSubcell.size() == 0)
  {
    // repeat the call so we can see it in the debugger:
    cellsForSubcell = _meshTopology->getCellsContainingEntity(d, entityIndex, tags);
    TEUCHOS_TEST_FOR_EXCEPTION(cellsForSubcell.size() == 0, std::invalid_argument, "no cells found that match constraining entity");
  }
  
  // for now, we just use the first component of H1Order; this should be OK so long as refinements are isotropic,
  // but if / when we support anisotropic refinements, we'll want to revisit this (we need a notion of the
  // appropriate order along the *interface* in question)
  int leastH1Order = INT_MAX;
  set< CellPair > cellsWithLeastH1Order;
  for (set< CellPair >::iterator cellForSubcellIt = cellsForSubcell.begin(); cellForSubcellIt != cellsForSubcell.end(); cellForSubcellIt++)
  {
    IndexType subcellCellID = cellForSubcellIt->first;
    
    // For now, disallowing non-active cells.  This will be problematic, possibly, in the case of anisotropic h-refinements (unless we simply induce a regular refinement in cases where an active cell would not be the constraining cell for a hanging node).
    if (_meshTopology->isParent(subcellCellID)) continue;
    
    vector<int> subcellH1Order = getH1Order(subcellCellID);
    if ( subcellH1Order[0] == leastH1Order)
    {
      cellsWithLeastH1Order.insert(*cellForSubcellIt);
    }
    else if (subcellH1Order[0] < leastH1Order)
    {
      cellsWithLeastH1Order.clear();
      leastH1Order = subcellH1Order[0];
      cellsWithLeastH1Order.insert(*cellForSubcellIt);
    }
    if (cellsWithLeastH1Order.size() == 0)
    {
      cout << "ERROR: No cells found for constraining subside entity.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No cells found for constraining subside entity.");
    }
  }
  
  CellPair constrainingCellPair = *cellsWithLeastH1Order.begin(); // first one will have the earliest cell ID, given the sorting of set/pair.
  
  // if one of the cellsWithLeastH1Order is active, we prefer that...
  for (CellPair cellPair : cellsWithLeastH1Order)
  {
    GlobalIndexType cellID = cellPair.first;
    if (!_meshTopology->isParent(cellID))
    {
      // active cell: we prefer this one
      constrainingCellPair = cellPair;
      break;
    }
  }

  return constrainingCellPair;
}

void GDAMinimumRule::setCheckConstraintConsistency(bool value)
{
  _checkConstraintConsistency = value;
}

void GDAMinimumRule::clearCaches()
{
  _constraintsCache.clear(); // to free up memory, could clear this again after the lookups are rebuilt.  Having the cache is most important during the construction in rebuildLookups().
  _dofMapperCache.clear();
  _dofMapperForVariableOnSideCache.clear();
  _ownedGlobalDofIndicesCache.clear();
  _globalDofIndicesForCellCache.clear();
  _fittableGlobalIndicesCache.clear();
}

GlobalDofAssignmentPtr GDAMinimumRule::deepCopy()
{
  Teuchos::RCP<GDAMinimumRule> copy = Teuchos::rcp(new GDAMinimumRule(*this) );
  copy->clearCaches();
  return copy;
}

void GDAMinimumRule::didChangePartitionPolicy()
{
  rebuildLookups();
}

void GDAMinimumRule::didHRefine(const set<GlobalIndexType> &parentCellIDs)
{
  set<GlobalIndexType> neighborsOfNewElements;
  for (GlobalIndexType parentCellID : parentCellIDs)
  {
    if (_meshTopology->isValidCellIndex(parentCellID))
    {
  //    cout << "GDAMinimumRule: h-refining " << parentCellID << endl;
      CellPtr parentCell = _meshTopology->getCell(parentCellID);
      vector<IndexType> childIDs = parentCell->getChildIndices(_meshTopology);
      vector<int> parentH1Order = getH1Order(parentCellID);
      int parentDeltaP = getPRefinementDegree(parentCellID);
      for (IndexType childCellID : childIDs)
      {
        if (_meshTopology->isValidCellIndex(childCellID))
        {
          auto childDeltaP = getPRefinementDegree(childCellID);
          setPRefinementDegree(childCellID, max(parentDeltaP,childDeltaP));
          assignParities(childCellID);
          // determine neighbors, so their parities can be updated below:
          CellPtr childCell = _meshTopology->getCell(childCellID);

          unsigned childSideCount = childCell->getSideCount();
          for (int childSideOrdinal=0; childSideOrdinal<childSideCount; childSideOrdinal++)
          {
            GlobalIndexType neighborCellID = childCell->getNeighborInfo(childSideOrdinal, _meshTopology).first;
            if (neighborCellID != -1)
            {
              neighborsOfNewElements.insert(neighborCellID);
            }
          }
        }
      }
    }
  }
  // this set is not as lean as it might be -- we could restrict to peer neighbors, I think -- but it's a pretty cheap operation.
  for (set<GlobalIndexType>::iterator cellIDIt = neighborsOfNewElements.begin(); cellIDIt != neighborsOfNewElements.end(); cellIDIt++)
  {
    assignParities(*cellIDIt);
  }
//  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
//    GlobalIndexType parentCellID = *cellIDIt;
//    ElementTypePtr elemType = elementType(parentCellID);
//    for (vector< TSolution<double>* >::iterator solutionIt = _registeredSolutions.begin();
//         solutionIt != _registeredSolutions.end(); solutionIt++) {
//      // do projection
//      vector<IndexType> childIDsLocalIndexType = _meshTopology->getCell(parentCellID)->getChildIndices();
//      vector<GlobalIndexType> childIDs(childIDsLocalIndexType.begin(),childIDsLocalIndexType.end());
//      (*solutionIt)->projectOldCellOntoNewCells(parentCellID,elemType,childIDs);
//    }
//  }

  this->GlobalDofAssignment::didHRefine(parentCellIDs);
}

void GDAMinimumRule::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP)
{
  // before we let super set the new type, record the old type for any cells we see
  // (and potentially have local solution coefficients for)
  map<GlobalIndexType, ElementTypePtr> locallyKnownRefinedCellsOldTypes;
  const set<IndexType>* locallyKnownCellIndices = &_mesh->getTopology()->getLocallyKnownActiveCellIndices();
  for (GlobalIndexType cellID : cellIDs)
  {
    if (locallyKnownCellIndices->find(cellID) != locallyKnownCellIndices->end())
    {
      locallyKnownRefinedCellsOldTypes[cellID] = elementType(cellID);
    }
  }
  
  this->GlobalDofAssignment::didPRefine(cellIDs, deltaP);

  for (auto entry : locallyKnownRefinedCellsOldTypes)
  {
    GlobalIndexType cellID = entry.first;
    ElementTypePtr oldType = entry.second;
    for (SolutionPtr soln : _registeredSolutions)
    {
      // do projection
      vector<IndexType> childIDs(1,cellID); // "child" ID is just the cellID
      const int numLHSes = soln->numSolutions();
      for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
      {
        soln->projectOldCellOntoNewCells(cellID,oldType,childIDs,lhsOrdinal);
      }
    }
  }
  
  // the above assigns _cellPRefinements for active elements; now we take minimums for parents (inactive elements)
//  for (GlobalIndexType cellID : cellIDs)
//  {
//    CellPtr cell = _meshTopology->getCell(cellID);
//    CellPtr parent = cell->getParent();
//    while (parent.get() != NULL)
//    {
//      vector<IndexType> childIndices = parent->getChildIndices(_meshTopology);
//      int minDeltaP = getPRefinementDegree(cellID);
//      for (int childOrdinal=0; childOrdinal<childIndices.size(); childOrdinal++)
//      {
//        minDeltaP = min(getPRefinementDegree(childIndices[childOrdinal]), minDeltaP);
//      }
//      setPRefinementDegree(parent->cellIndex(), minDeltaP);
//      parent = parent->getParent();
//    }
//  }
}

void GDAMinimumRule::setCellPRefinements(const map<GlobalIndexType,int>& pRefinements)
{
  this->GlobalDofAssignment::setCellPRefinements(pRefinements);
  // for now, we *are* assuming that pRefinements is a global container. 
  
//  // the above assigns _cellPRefinements for active elements; now we take minimums for parents (inactive elements)
//  for (auto entry : pRefinements)
//  {
//    IndexType cellID = entry.first;
//    
//    CellPtr cell = _meshTopology->getCell(cellID);
//    CellPtr parent = cell->getParent();
//    while (parent.get() != NULL)
//    {
//      vector<IndexType> childIndices = parent->getChildIndices(_meshTopology);
//      int minDeltaP = getPRefinementDegree(cellID);
//      for (int childOrdinal=0; childOrdinal<childIndices.size(); childOrdinal++)
//      {
//        minDeltaP = min(getPRefinementDegree(childIndices[childOrdinal]), minDeltaP);
//      }
//      setPRefinementDegree(parent->cellIndex(), minDeltaP);
//      parent = parent->getParent();
//    }
//  }
}

void GDAMinimumRule::willHUnrefine(const set<GlobalIndexType> &parentCellIDs)
{
  this->GlobalDofAssignment::willHUnrefine(parentCellIDs);
  // will need to treat cell side parities here--probably suffices to redo those in parentCellIDs plus all their neighbors.
  // (but this should be handled at the GlobalDofAssignment level)
}

void GDAMinimumRule::distributeCellGlobalDofs(const map<GlobalIndexType, SubcellDofIndices> &myCellGlobalDofs,
                                              const set<GlobalIndexType> &remoteCellIDs)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::distributeCellGlobalDofs");
  int timerHandle = TimeLogger::sharedInstance()->startTimer("distributeCellGlobalDofs");
  
  int rank = _mesh->Comm()->getRank();
  map<GlobalIndexType,PartitionIndexType> remoteCellOwners;
  for (GlobalIndexType remoteCellID : remoteCellIDs)
  {
    int partitionForCell = _mesh->globalDofAssignment()->partitionForCellID(remoteCellID);
    if (partitionForCell == rank)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "remoteCellIDs includes a local cell");
    }
    else
    {
      remoteCellOwners[remoteCellID] = partitionForCell;
    }
  }
  map<GlobalIndexType, SubcellDofIndices> remoteCellGlobalDofs;
  auto epetraComm = MPIWrapper::EpetraComm(_mesh->Comm());
  distributeSubcellDofIndices(epetraComm, myCellGlobalDofs, remoteCellOwners, remoteCellGlobalDofs);
  for (GlobalIndexType cellID : remoteCellIDs)
  {
    if (remoteCellGlobalDofs.find(cellID) == remoteCellGlobalDofs.end())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "still missing SubcellDofIndices for remote cell after distributing cell global dof information");
    }
    _globalDofIndicesForCellCache[cellID] = remoteCellGlobalDofs[cellID];
  }
  
  TimeLogger::sharedInstance()->stopTimer(timerHandle);
}


void GDAMinimumRule::distributeGlobalDofOwnership(const map<GlobalIndexType, SubcellDofIndices> &myOwnedGlobalDofs,
                                                  const set<GlobalIndexType> &remoteCellIDs)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::distributeGlobalDofOwnership");
  int timerHandle = TimeLogger::sharedInstance()->startTimer("distributeGlobalDofOwnership");
  
  int rank = _mesh->Comm()->getRank();
  {
    // DEBUGGING: barrier for breakpoint.
    _mesh->Comm()->barrier();
  }
  map<GlobalIndexType,PartitionIndexType> remoteCellOwners;
  for (GlobalIndexType remoteCellID : remoteCellIDs)
  {
    int partitionForCell = _mesh->globalDofAssignment()->partitionForCellID(remoteCellID);
    if (partitionForCell == rank)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "remoteCellIDs includes a local cell");
    }
    else if (partitionForCell < 0)
    {
      cout << "Partition for cell " << remoteCellID << " not known on rank " << rank << endl;
      if (!_mesh->getTopology()->isValidCellIndex(remoteCellID))
      {
        cout << "Cell " << remoteCellID << " is not a valid cell index on rank " << rank << endl;
      }
      else
      {
        CellPtr remoteCell = _mesh->getTopology()->getCell(remoteCellID);
        if (remoteCell->isParent(_mesh->getTopology()))
        {
          cout << "Cell " << remoteCellID << " is a parent.\n";
        }
        else
        {
          cout << "Cell " << remoteCellID << " is active.\n";
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Partition not found for remoteCellID");
    }
    else
    {
      remoteCellOwners[remoteCellID] = partitionForCell;
    }
  }
  auto epetraComm = MPIWrapper::EpetraComm(_mesh->Comm());
  map<GlobalIndexType, SubcellDofIndices> remotelyOwnedGlobalDofs;
  distributeSubcellDofIndices(epetraComm, myOwnedGlobalDofs, remoteCellOwners, remotelyOwnedGlobalDofs);
  for (GlobalIndexType cellID : remoteCellIDs)
  {
    if (remotelyOwnedGlobalDofs.find(cellID) == remotelyOwnedGlobalDofs.end())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "still missing SubcellDofIndices for remote cell after distributing DofIndexInfo ownership information");
    }
    _ownedGlobalDofIndicesCache[cellID] = remotelyOwnedGlobalDofs[cellID];
  }
  TimeLogger::sharedInstance()->stopTimer(timerHandle);
}

void GDAMinimumRule::distributeSubcellDofIndices(Epetra_CommPtr Comm, const map<GlobalIndexType, SubcellDofIndices> &myOwnedGlobalDofs,
                                                 const map<GlobalIndexType, PartitionIndexType> &remoteCellOwners,
                                                 map<GlobalIndexType, SubcellDofIndices> &remotelyOwnedGlobalDofs)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::distributeSubcellDofIndices");
  int timerHandle = TimeLogger::sharedInstance()->startTimer("distributeSubcellDofIndices");
  
  // sort by partition # using map
  map<PartitionIndexType,set<GlobalIndexType>> requestMap;
  for (auto remoteCellEntry : remoteCellOwners)
  {
    GlobalIndexType remoteCellID = remoteCellEntry.first;
    PartitionIndexType partitionForCell = remoteCellEntry.second;
    requestMap[partitionForCell].insert(remoteCellID);
  }
  
  vector<int> myRequestOwners;
  vector<GlobalIndexTypeToCast> myRequest;
  for (auto requestEntry : requestMap)
  {
    PartitionIndexType partition = requestEntry.first;
    for (GlobalIndexType remoteCellID : requestEntry.second)
    {
      myRequest.push_back(remoteCellID);
      myRequestOwners.push_back(partition);
    }
  }
  
  int myRequestCount = myRequest.size();
  
  int globalRequestCount = MPIWrapper::sum(*Comm, myRequestCount);
  
  if (globalRequestCount != 0)
  {
    Teuchos::RCP<Epetra_Distributor> distributor = MPIWrapper::getDistributor(*Comm);
    
    GlobalIndexTypeToCast* myRequestPtr = NULL;
    int *myRequestOwnersPtr = NULL;
    if (myRequest.size() > 0)
    {
      myRequestPtr = &myRequest[0];
      myRequestOwnersPtr = &myRequestOwners[0];
    }
    int numCellsToExport = 0;
    GlobalIndexTypeToCast* cellIDsToExport = NULL;  // we are responsible for deleting the allocated arrays
    int* exportRecipients = NULL;
    
    distributor->CreateFromRecvs(myRequestCount, myRequestPtr, myRequestOwnersPtr, true, numCellsToExport, cellIDsToExport, exportRecipients);
    
    vector<int> sizes(numCellsToExport);
    vector<char> dataToExport;
    for (int cellOrdinal=0; cellOrdinal<numCellsToExport; cellOrdinal++)
    {
      GlobalIndexType cellID = cellIDsToExport[cellOrdinal];
      
      if (myOwnedGlobalDofs.find(cellID) == myOwnedGlobalDofs.end())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Entry for local cellID not found in mySubcellOwnership map");
      }
      const SubcellDofIndices* cellDofIndexInfo = &myOwnedGlobalDofs.find(cellID)->second;
      int cellIDSize = sizeof(cellID);
      int dataSize = cellDofIndexInfo->dataSize();
      sizes[cellOrdinal] = cellIDSize + dataSize;
      int startIndex = dataToExport.size();
      dataToExport.resize(dataToExport.size() + sizes[cellOrdinal]);
      
      char *writeLocation = &dataToExport[startIndex];
      memcpy(writeLocation, &cellID, cellIDSize);
      writeLocation = &dataToExport[startIndex+cellIDSize];
      cellDofIndexInfo->write(writeLocation, dataSize);
    }
    
    int importLength = 0;
    char* globalData = NULL;
    int* sizePtr = NULL;
    char* dataToExportPtr = NULL;
    if (numCellsToExport > 0)
    {
      sizePtr = &sizes[0];
      dataToExportPtr = &dataToExport[0];
    }
    int objSize = sizeof(char);
    distributor->Do(dataToExportPtr, objSize, sizePtr, importLength, globalData);
    const char* globalDataStart = globalData;
    const char* copyFromLocation = globalData;
    for (int cellOrdinal=0; cellOrdinal<myRequest.size(); cellOrdinal++)
    {
      GlobalIndexType cellID;
      memcpy(&cellID, copyFromLocation, sizeof(cellID));
      copyFromLocation += sizeof(cellID);
      SubcellDofIndices subcellDofIndices;
      int bytesRead = copyFromLocation - globalDataStart;
      int bytesLeft = importLength - bytesRead;
      TEUCHOS_TEST_FOR_EXCEPTION(bytesLeft <= 0, std::invalid_argument, "Invalid read requested");
      subcellDofIndices.read(copyFromLocation, bytesLeft);
      remotelyOwnedGlobalDofs[cellID] = subcellDofIndices;
    }
    
    if( cellIDsToExport != 0 ) delete [] cellIDsToExport;
    if( exportRecipients != 0 ) delete [] exportRecipients;
    if (globalData != 0 ) delete [] globalData;
  }
  
  TimeLogger::sharedInstance()->stopTimer(timerHandle);
}

GlobalIndexType GDAMinimumRule::globalDofCount()
{
  // assumes the lookups have been rebuilt since the last change that would affect the count

  // TODO: Consider working out a way to guard against a stale value here.  E.g. could have a "dirty" flag that gets set anytime there's a change to the refinements, and cleared when lookups are rebuilt.  If we're dirty when we get here, we rebuild before returning the global dof count.
  int numRanks = _partitionPolicy->Comm()->getSize();
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(_partitionDofOffsets.size()) <= numRanks, std::invalid_argument, "_partitionDofOffsets is not sized correctly");
  return _partitionDofOffsets[numRanks];
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForCell(GlobalIndexType cellID)
{
  set<GlobalIndexType> globalDofIndices;

  LocalDofMapperPtr dofMapper = getDofMapper(cellID);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();

  globalDofIndices.insert(globalIndexVector.begin(),globalIndexVector.end());
  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned int dim, unsigned int subcellOrdinal, bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime)
{
  LocalDofMapperPtr dofMapper = getDofMapper(cellID);
  bool includeSubSubcells = true;
  set<GlobalIndexType> globalDofIndices = dofMapper->globalIndicesForSubcell(varID, dim, subcellOrdinal, upgradeToContinuousBasis, tensorTopologyMeansSpaceTime, includeSubSubcells);
  
  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForPartition(PartitionIndexType partitionNumber)
{
  int rank = _partitionPolicy->Comm()->getRank();

  if (partitionNumber==-1) partitionNumber = rank;

  GlobalIndexType partitionDofOffset = _partitionDofOffsets[partitionNumber];
  GlobalIndexType nextPartitionDofOffset = _partitionDofOffsets[partitionNumber+1];
  vector<GlobalIndexType> globalDofIndicesVector(nextPartitionDofOffset - partitionDofOffset);
  int ordinal = 0;
  for (IndexType i=partitionDofOffset; i<nextPartitionDofOffset; i++)
  {
    globalDofIndicesVector[ordinal] = i;
    ordinal++;
  }

  return set<GlobalIndexType>(globalDofIndicesVector.begin(),globalDofIndicesVector.end());
}

vector<int> GDAMinimumRule::H1Order(GlobalIndexType cellID, unsigned sideOrdinal)
{
  // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...
  return getH1Order(cellID);
}

int GDAMinimumRule::getH1OrderOnEdge(GlobalIndexType cellID, unsigned edgeOrdinal)
{
  const int edgeDim = 1;
  AnnotatedEntity edgeConstraint = getCellConstraints(cellID).subcellConstraints[edgeDim][edgeOrdinal];
  GlobalIndexType constrainingCellID = edgeConstraint.cellID;
  // we're isotropic in p right now, with the exception of tensor-product cell topologies
  // for now, we don't support tensor-product cell topologies in this method
  if (_meshTopology->getCell(constrainingCellID)->topology()->getTensorialDegree() > 0)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "H1OrderOnEdge() does not yet support tensorial degree > 0");
  // otherwise, we can just return the H1Order for the cell:
  return getH1Order(constrainingCellID)[0];
}

void GDAMinimumRule::interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients,
                                                 const Epetra_MultiVector &globalCoefficients)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::interpretGlobalCoefficients");
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID);
  const vector<GlobalIndexType>* globalIndexVector = &dofMapper->globalIndices();

  if (globalCoefficients.NumVectors() == 1)
  {
    bool useFieldContainer = false;
    
    if (useFieldContainer)
    {
      Camellia::FieldContainer<double> globalCoefficientsFC(globalIndexVector->size());
      Epetra_BlockMap partMap = globalCoefficients.Map();
      for (int i=0; i<globalIndexVector->size(); i++)
      {
        GlobalIndexTypeToCast globalIndex = (*globalIndexVector)[i];
        int localIndex = partMap.LID(globalIndex);
        if (localIndex != -1)
        {
          globalCoefficientsFC[i] = globalCoefficients[0][localIndex];
        }
        else
        {
          // non-local coefficient: ignore
          globalCoefficientsFC[i] = 0;
        }
      }
      localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
    }
    else
    {
      Epetra_BlockMap partMap = globalCoefficients.Map();
      map<GlobalIndexType,double> globalCoefficientsToMap;
      for (int i=0; i<globalIndexVector->size(); i++)
      {
        GlobalIndexTypeToCast globalIndex = (*globalIndexVector)[i];
        int localIndex = partMap.LID(globalIndex);
        if (localIndex != -1)
        {
          if (globalCoefficients[0][localIndex] != 0.0)
          {
            globalCoefficientsToMap[globalIndex] = globalCoefficients[0][localIndex];
          }
        }
      }
//      if (globalCoefficientsToMap.size() == 0) cout << "***********************  Empty globalCoefficientsToMap **************************\n";
      dofMapper->mapGlobalCoefficients(globalCoefficientsToMap, localCoefficients);
    }
  }
  else
  {
    Camellia::FieldContainer<double> globalCoefficientsFC(globalIndexVector->size());
    Epetra_BlockMap partMap = globalCoefficients.Map();
    
    for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.NumVectors(); vectorOrdinal++)
    {
      for (int i=0; i<globalIndexVector->size(); i++)
      {
        GlobalIndexTypeToCast globalIndex = (*globalIndexVector)[i];
        int localIndex = partMap.LID(globalIndex);
        if (localIndex != -1)
        {
          globalCoefficientsFC(i) = globalCoefficients[vectorOrdinal][localIndex];
        }
        else
        {
          // non-local coefficient: ignore
          globalCoefficientsFC(i) = 0;
        }
      }
      auto vectorLocalCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
      auto numLocalCoefficients = vectorLocalCoefficients.size();
      if (vectorOrdinal == 0)
      {
        // resize localCoefficients according to the first returned guy:
        localCoefficients.resize(globalCoefficients.NumVectors(), numLocalCoefficients);
      }
      for (int localDofOrdinal=0; localDofOrdinal<numLocalCoefficients; localDofOrdinal++)
      {
        localCoefficients(vectorOrdinal,localDofOrdinal) = vectorLocalCoefficients(localDofOrdinal);
      }
    }
  }
//  cout << "For cellID " << cellID << ", mapping globalData:\n " << globalDataFC;
//  cout << " to localData:\n " << localDofs;
}

template <typename Scalar>
void GDAMinimumRule::interpretGlobalCoefficients2(GlobalIndexType cellID, Camellia::FieldContainer<Scalar> &localCoefficients, const TVectorPtr<Scalar> globalCoefficients)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::interpretGlobalCoefficients2");
  LocalDofMapperPtr dofMapper = getDofMapper(cellID);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();

  // DEBUGGING
//  if (cellID==4) {
//    cout << "interpretGlobalData, mapping report for cell " << cellID << ":\n";
//    dofMapper->printMappingReport();
//  }

  if (globalCoefficients->getNumVectors() == 1)
  {
    // TODO: Test whether this is actually correct
    Teuchos::ArrayRCP<Scalar> globalCoefficientsArray = globalCoefficients->getDataNonConst(globalIndexVector.size());
    Camellia::FieldContainer<Scalar> globalCoefficientsFC(globalIndexVector.size());
    Teuchos::ArrayView<Scalar> arrayView = globalCoefficientsArray.view(0,globalIndexVector.size());
    globalCoefficientsFC.setValues(arrayView);
    // ConstMapPtr partMap = globalCoefficients->getMap();
    // for (int i=0; i<globalIndexVector.size(); i++) {
    //   GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
    //   int localIndex = partMap->getLocalElement(globalIndex);
    //   if (localIndex != Teuchos::OrdinalTraits<IndexType>::invalid()) {
    //     globalCoefficientsFC[i] = globalCoefficients[0][localIndex];
    //   } else {
    //     // non-local coefficient: ignore
    //     globalCoefficientsFC[i] = 0;
    //   }
    // }
    localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "InterpretGlobalCoefficients not implemented for more than one vector");
    // Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar>> globalCoefficients2DArray = globalCoefficients->get2dViewNonConst(globalIndexVector.size());
    // Camellia::FieldContainer<Scalar> globalCoefficientsFC(globalCoefficients.numVectors(), globalIndexVector.size());
    // for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.numVectors(); vectorOrdinal++) {
    //   Teuchos::ArrayRCP<Scalar> globalCoefficientsArray = globalCoefficients2DArray[vectorOrdinal];
    //   Teuchos::ArrayView<Scalar> arrayView = globalCoefficientsArray.view(0,globalIndexVector.size());
    //   globalCoefficientsFC.setValues(arrayView);
    // }
    //   Camellia::FieldContainer<Scalar> globalCoefficientsFC(globalCoefficients.NumVectors(), globalIndexVector.size());
    //   Epetra_BlockMap partMap = globalCoefficients.Map();
    //   for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.NumVectors(); vectorOrdinal++) {
    //     for (int i=0; i<globalIndexVector.size(); i++) {
    //       GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
    //       int localIndex = partMap.LID(globalIndex);
    //       if (localIndex != -1) {
    //         globalCoefficientsFC(vectorOrdinal,i) = globalCoefficients[vectorOrdinal][localIndex];
    //       } else {
    //         // non-local coefficient: ignore
    //         globalCoefficientsFC(vectorOrdinal,i) = 0;
    //       }
    //     }
    // }
    // localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
  }
//  cout << "For cellID " << cellID << ", mapping globalData:\n " << globalDataFC;
//  cout << " to localData:\n " << localDofs;
}

template void GDAMinimumRule::interpretGlobalCoefficients2(GlobalIndexType cellID, Camellia::FieldContainer<double> &localCoefficients, const TVectorPtr<double> globalCoefficients);

void GDAMinimumRule::interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localData,
                                        Camellia::FieldContainer<double> &globalData, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::interpretLocalData");
  // we allow interpreting trial x trial, trial x other, or other x trial; the latter two involve only one mapping
  
  int numLocalTrialDofs = _mesh->getElementType(cellID)->trialOrderPtr->totalDofs();
  bool isMatrixAndNotSquare = true;
  int trialDim = -1;
  if (localData.rank() == 1)
  {
    // vector
    isMatrixAndNotSquare = false;
  }
  else
  {
    if (localData.dimension(0) == numLocalTrialDofs)
    {
      if (localData.dimension(1) == numLocalTrialDofs)
        isMatrixAndNotSquare = false;
      else
        trialDim = 0;
    }
    else if (localData.dimension(1) == numLocalTrialDofs)
    {
      trialDim = 1;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "One of the dimensions of localData must match the number of trial dofs on the cell");
    }
  }
  
//  { // DEBUGGING
//    if (isMatrixAndNotSquare)
//      cout << "isMatrixAndNotSquare is TRUE;";
//    else
//      cout << "isMatrixAndNotSquare is FALSE;";
//    if (localData.rank() == 1)
//      cout << " dimensions are " << localData.dimension(0) << " \n";
//    else
//      cout << " dimensions are " << localData.dimension(0) << " x " << localData.dimension(1) << " \n";
//  }
  
  LocalDofMapperPtr dofMapper = getDofMapper(cellID);

  // DEBUGGING
//  if (Teuchos::GlobalMPISession::getRank()==0) {
//    if (cellID==1) {
//      cout << "interpretLocalData, mapping report for cell " << cellID << ":\n";
//      dofMapper->printMappingReport();
//    }
//  }

  bool fittableDofsOnly = false;
  if (!isMatrixAndNotSquare)
  {
    globalData = dofMapper->mapLocalData(localData, fittableDofsOnly);
  }
  else
  {
    dofMapper->mapLocalDataMatrix(localData, fittableDofsOnly, globalData, trialDim);
  }
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();
  globalDofIndices.resize(globalIndexVector.size());
  for (int i=0; i<globalIndexVector.size(); i++)
  {
    globalDofIndices(i) = globalIndexVector[i];
  }

//  // mostly for debugging purposes, let's sort according to global dof index:
//  if (globalData.rank() == 1) {
//    map<GlobalIndexType, double> globalDataMap;
//    for (int i=0; i<globalIndexVector.size(); i++) {
//      GlobalIndexType globalIndex = globalIndexVector[i];
//      globalDataMap[globalIndex] = globalData(i);
//    }
//    int i=0;
//    for (map<GlobalIndexType, double>::iterator mapIt = globalDataMap.begin(); mapIt != globalDataMap.end(); mapIt++) {
//      globalDofIndices(i) = mapIt->first;
//      globalData(i) = mapIt->second;
//      i++;
//    }
//  } else if (globalData.rank() == 2) {
//    cout << "globalData.rank() == 2.\n";
//    Camellia::FieldContainer<double> globalDataCopy = globalData;
//    map<GlobalIndexType,int> globalIndexToOrdinalMap;
//    for (int i=0; i<globalIndexVector.size(); i++) {
//      GlobalIndexType globalIndex = globalIndexVector[i];
//      globalIndexToOrdinalMap[globalIndex] = i;
//    }
//    int i=0;
//    for (map<GlobalIndexType,int>::iterator i_mapIt = globalIndexToOrdinalMap.begin();
//         i_mapIt != globalIndexToOrdinalMap.end(); i_mapIt++) {
//      int j=0;
//      globalDofIndices(i) = i_mapIt->first;
//      for (map<GlobalIndexType,int>::iterator j_mapIt = globalIndexToOrdinalMap.begin();
//           j_mapIt != globalIndexToOrdinalMap.end(); j_mapIt++) {
//        globalData(i,j) = globalDataCopy(i_mapIt->second,j_mapIt->second);
//
//        j++;
//      }
//      i++;
//    }
//  }

//  cout << "localData:\n" << localData;
//  cout << "globalData:\n" << globalData;
//  if (cellID==1)
//    cout << "globalIndices:\n" << globalDofIndices;
}

void GDAMinimumRule::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Camellia::FieldContainer<double> &basisCoefficients,
                                                     Camellia::FieldContainer<double> &globalCoefficients, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::interpretLocalBasisCoefficients");
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, varID, sideOrdinal);
  
  if (dofMapper->isPermutation())
  {
    const map<int, GlobalIndexType>* permutationMap = &dofMapper->getPermutationMap();
    DofOrderingPtr dofOrdering = elementType(cellID)->trialOrderPtr;
    if (dofOrdering->hasBasisEntry(varID, sideOrdinal))
    {
      // we are mapping the whole basis in this case
      const vector<int>* localDofIndices = &dofOrdering->getDofIndices(varID, sideOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(localDofIndices->size() != basisCoefficients.size(), std::invalid_argument, "basisCoefficients should be sized to match either the basis cardinality (when whole bases are matched) or the cardinality of the restriction");
      int numEntries = basisCoefficients.size();
      globalCoefficients.resize(numEntries);
      globalDofIndices.resize(numEntries);
      const double* basisCoefficient = &basisCoefficients[0];
      double* globalCoefficient = &globalCoefficients[0];
      GlobalIndexType* globalDofIndex = &globalDofIndices[0];
      
      for (int localDofIndex : *localDofIndices)
      {
        auto permutationEntry = permutationMap->find(localDofIndex);
        // for hierarchical bases in particular, it can happen that we have a "permutation" map that simply drops
        // some of the degrees of freedom; this is due to the nesting of the polynomial bases.
        if (permutationEntry != permutationMap->end())
        {
          *globalCoefficient++ = *basisCoefficient++;
          *globalDofIndex++ = permutationEntry->second;
        }
      }
    }
    else
    {
      // we are restricting a volume basis to the side indicated
      TEUCHOS_TEST_FOR_EXCEPTION(!dofOrdering->hasBasisEntry(varID, VOLUME_INTERIOR_SIDE_INDEX), std::invalid_argument, "Basis not found");
      BasisPtr volumeBasis = dofOrdering->getBasis(varID);
      set<int> basisDofOrdinals = volumeBasis->dofOrdinalsForSide(sideOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(basisDofOrdinals.size() != basisCoefficients.size(), std::invalid_argument, "basisCoefficients should be sized to match either the basis cardinality (when whole bases are matched) or the cardinality of the restriction");
      const vector<int>* localDofIndices = &dofOrdering->getDofIndices(varID);
      int numEntries = basisCoefficients.size();
      globalCoefficients.resize(numEntries);
      globalDofIndices.resize(numEntries);
      const double* basisCoefficient = &basisCoefficients[0];
      double* globalCoefficient = &globalCoefficients[0];
      GlobalIndexType* globalDofIndex = &globalDofIndices[0];
      for (int basisDofOrdinal : basisDofOrdinals)
      {
        int localDofIndex = (*localDofIndices)[basisDofOrdinal];
        *globalCoefficient++ = *basisCoefficient++;
        auto permutationEntry = permutationMap->find(localDofIndex);
        *globalDofIndex++ = permutationEntry->second;
      }
    }
  }
  else
  {
    globalCoefficients = dofMapper->fitLocalCoefficients(basisCoefficients);
    const vector<GlobalIndexType>* globalIndexVector = &dofMapper->fittableGlobalIndices();
    globalDofIndices.resize(globalIndexVector->size());

    for (int i=0; i<globalIndexVector->size(); i++)
    {
      globalDofIndices(i) = (*globalIndexVector)[i];
    }
  }
}

bool GDAMinimumRule::isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const
{
  return (_partitionDofOffset <= globalDofIndex) && (globalDofIndex < _partitionDofOffset + _partitionDofCount);
}

IndexType GDAMinimumRule::localDofCount()
{
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

typedef vector< SubBasisDofMapperPtr > BasisMap;
// volume variable version
BasisMap GDAMinimumRule::getBasisMap(GlobalIndexType cellID, SubcellDofIndices& subcellDofIndices, VarPtr var)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getBasisMap");
  BasisMap varVolumeMap;
  
  vector<SubBasisMapInfo> subBasisMaps;
  SubBasisMapInfo subBasisMap;

  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  CellTopoPtr topo = cell->topology();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;

  // assumption is that the basis is defined on the whole cell
  BasisPtr basis = trialOrdering->getBasis(var->ID());
  
  // to begin, let's map the volume-interior dofs:
  vector<GlobalIndexType> globalDofOrdinals = subcellDofIndices.dofIndicesForVarOnSubcell(spaceDim,0,var->ID());
//  vector<GlobalIndexType> globalDofOrdinals = subcellDofIndices.subcellDofIndices[spaceDim][0][var->ID()];
  const vector<int>* basisDofOrdinalsVector = &basis->dofOrdinalsForInterior();
  set<int> basisDofOrdinals(basisDofOrdinalsVector->begin(), basisDofOrdinalsVector->end());

  if (basisDofOrdinals.size() > 0)
  {
    varVolumeMap.push_back(SubBasisDofMapper::subBasisDofMapper(basisDofOrdinals, globalDofOrdinals));
  }

  if (basisDofOrdinals.size() == basis->getCardinality())
  {
    // then we're done...
    return varVolumeMap;
  }
  
  static const int VOLUME_BASIS_SUBCORD = 0; // volume always has subcell ordinal 0 in volume
  
  int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
  
  for (int d = sideDim; d >= minimumConstraintDimension; d--)
  {
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int subcord=0; subcord < subcellCount; subcord++)
    {
      AnnotatedEntity subcellConstraint = getCellConstraints(cellID).subcellConstraints[d][subcord];
      DofOrderingPtr constrainingTrialOrdering = elementType(subcellConstraint.cellID)->trialOrderPtr;
      
      CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
      BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID());
      
      unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim,
                                                                                       subcellConstraint.sideOrdinal,
                                                                                       subcellConstraint.dimension,
                                                                                       subcellConstraint.subcellOrdinal);
      
      IndexType subcellEntityIndex = cell->entityIndex(d,subcord);
      IndexType constrainingEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
      
      bool subcellIsGeometricallyConstrained = (d != subcellConstraint.dimension) || (subcellEntityIndex != constrainingEntityIndex);
      
      CellTopoPtr constrainingTopo = constrainingCell->topology()->getSubcell(subcellConstraint.dimension,
                                                                              subcellOrdinalInConstrainingCell);
      
      SubBasisReconciliationWeights weightsForSubcell;
      
      CellPtr ancestralCell = cell->ancestralCellForSubcell(d, subcord, _meshTopology);
      
      RefinementBranch volumeRefinements = cell->refinementBranchForSubcell(d, subcord, _meshTopology);
      if (volumeRefinements.size()==0)
      {
        // could be, we'd do better to revise Cell::refinementBranchForSubcell() to ensure that we always have a refinement, but for now
        // we just create a RefinementBranch with a trivial refinement here:
        RefinementPatternPtr noRefinementPattern = RefinementPattern::noRefinementPattern(cell->topology());
        volumeRefinements = {{noRefinementPattern.get(),0}};
      }
      
      pair<unsigned, unsigned> ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(d, subcord, _meshTopology);
      
      unsigned ancestralSubcellOrdinal = ancestralSubcell.first;
      unsigned ancestralSubcellDimension = ancestralSubcell.second;
      
      if (ancestralSubcellOrdinal == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL)
      {
        cout << "Internal error: ancestral subcell ordinal was not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: ancestral subcell ordinal was not found.");
      }
    
      /***
       It is possible use the common computeConstrainedWeights() for both the case where the subcell dimension is the same as
       constraining and the one where it's different.  It appears, however, that this version of computeConstrainedWeights is a *LOT* 
       slower; overall runtime of runTests about doubled when I tried using this for both.
       ***/
      unsigned ancestralSubcellOrdinalInCell = ancestralCell->findSubcellOrdinal(subcellConstraint.dimension, constrainingEntityIndex);
      
      if (subcellConstraint.dimension != d)
      {
        // ancestralPermutation goes from canonical to cell's side's ancestor's ordering:
        unsigned ancestralCellPermutation = ancestralCell->subcellPermutation(ancestralSubcellDimension, ancestralSubcellOrdinal);
        // constrainingPermutation goes from canonical to the constraining side's ordering
        unsigned constrainingCellPermutation = constrainingCell->subcellPermutation(subcellConstraint.dimension, subcellOrdinalInConstrainingCell); // subcell permutation as seen from the perspective of the constraining cell's side
        
        // ancestralPermutationInverse goes from ancestral view to canonical
        unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralCellPermutation);
        unsigned ancestralToConstrainedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingCellPermutation);

        weightsForSubcell = BasisReconciliation::computeConstrainedWeights(d, basis, subcord,
                                                                           volumeRefinements,
                                                                           VOLUME_BASIS_SUBCORD,
                                                                           ancestralCell->topology(),
                                                                           subcellConstraint.dimension,
                                                                           constrainingBasis, subcellOrdinalInConstrainingCell,
                                                                           VOLUME_BASIS_SUBCORD, ancestralToConstrainedPermutation);
      }
      else
      {
        // from canonical to ancestral view:
        unsigned ancestralPermutation = ancestralCell->subcellPermutation(subcellConstraint.dimension, ancestralSubcellOrdinalInCell); // subcell permutation as seen from the perspective of the fine cell's ancestor
        // from canonical to constraining view:
        unsigned constrainingPermutation = constrainingCell->subcellPermutation(subcellConstraint.dimension,
                                                                                subcellOrdinalInConstrainingCell);
        
        // from ancestral to canonical:
        unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralPermutation);
        unsigned ancestralToConstrainingPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingPermutation);
        
        weightsForSubcell = _br.constrainedWeights(d, basis, subcord, volumeRefinements, constrainingBasis,
                                                   subcellOrdinalInConstrainingCell, ancestralToConstrainingPermutation);
      }
      
      // add sub-basis map for dofs interior to the constraining subcell
      // filter the weights whose coarse dofs are interior to this subcell, and create a SubBasisDofMapper for these (add it to varSideMap)
      SubBasisReconciliationWeights weightsForWholeSubcell = weightsForSubcell; // copy to save before we filter
      weightsForSubcell = BasisReconciliation::weightsForCoarseSubcell(weightsForSubcell, constrainingBasis,
                                                                       subcellConstraint.dimension, subcellOrdinalInConstrainingCell,
                                                                       false);
//      { // DEBUGGING
//        if (weightsForWholeSubcell.isIdentity)
//        {
//          SubBasisReconciliationWeights weightsForWholeSubcellManual = weightsManualIdentity(weightsForWholeSubcell);
//          SubBasisReconciliationWeights expectedWeights = BasisReconciliation::weightsForCoarseSubcell(weightsForWholeSubcellManual,
//                                                                                                       constrainingBasis,
//                                                                                                       subcellConstraint.dimension,
//                                                                                                       subcellOrdinalInConstrainingCell,
//                                                                                                       false);
//          bool equal = BasisReconciliation::equalWeights(expectedWeights, weightsForSubcell);
//          TEUCHOS_TEST_FOR_EXCEPTION(!equal, std::invalid_argument, "Something wrong with identity weights treatment");
//        }
//      }
      
      if ((weightsForSubcell.coarseOrdinals.size() > 0) && (weightsForSubcell.fineOrdinals.size() > 0))
      {
        CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint.cellID);
        OwnershipInfo ownershipInfo = constrainingCellConstraints.owningCellIDForSubcell[subcellConstraint.dimension][subcellOrdinalInConstrainingCell];
        SubcellDofIndices owningCellDofIndexInfo = getOwnedGlobalDofIndices(ownershipInfo.cellID);
        unsigned owningSubcellOrdinal = ownershipInfo.owningSubcellOrdinal;

//        unsigned owningSubcellOrdinal = _meshTopology->getCell(ownershipInfo.cellID)->findSubcellOrdinal(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
        vector<GlobalIndexType> globalDofOrdinalsForSubcell = owningCellDofIndexInfo.dofIndicesForVarOnSubcell(ownershipInfo.dimension,owningSubcellOrdinal,var->ID());
//        vector<GlobalIndexType> globalDofOrdinalsForSubcell = owningCellDofIndexInfo.subcellDofIndices[ownershipInfo.dimension][owningSubcellOrdinal][var->ID()];
        
        // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
        vector<int> basisOrdinalsVector = constrainingBasis->dofOrdinalsForSubcell(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
//        vector<int> basisOrdinalsVector(constrainingBasisOrdinalsForSubcell.begin(),constrainingBasisOrdinalsForSubcell.end());
        
        if (globalDofOrdinalsForSubcell.size() != basisOrdinalsVector.size())
        {
          cout << "globalDofOrdinalsForSubcell.size() != basisOrdinalsVector.size()\n";
          print("globalDofOrdinalsForSubcell", globalDofOrdinalsForSubcell);
          print("constrainingBasisOrdinalsForSubcell", basisOrdinalsVector);
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofOrdinalsForSubcell.size() != basisOrdinalsVector.size()");
        }
        
        vector<GlobalIndexType> globalDofOrdinals;
        for (int i=0; i<basisOrdinalsVector.size(); i++)
        {
          if (weightsForSubcell.coarseOrdinals.find(basisOrdinalsVector[i]) != weightsForSubcell.coarseOrdinals.end())
          {
            globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
          }
        }
        
        if (weightsForSubcell.coarseOrdinals.size() != globalDofOrdinals.size())
        {
          cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
          Camellia::print("coarseOrdinals", weightsForSubcell.coarseOrdinals);
          Camellia::print("globalDofOrdinals", globalDofOrdinals);
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
        }
        
        subBasisMap.weights = weightsForSubcell.weights;
        subBasisMap.globalDofOrdinals = globalDofOrdinals;
        subBasisMap.basisDofOrdinals = weightsForSubcell.fineOrdinals;
        subBasisMap.isIdentity = weightsForSubcell.isIdentity;
//        { // DEBUGGING
//          if (cellID == DEBUG_CELL_ID)
//          {
//            if (var->ID() == DEBUG_VAR_ID)
//            {
//              if (std::find(subBasisMap.globalDofOrdinals.begin(), subBasisMap.globalDofOrdinals.end(), DEBUG_GLOBAL_DOF) != subBasisMap.globalDofOrdinals.end())
//              {
//                if (subBasisMap.basisDofOrdinals.find(DEBUG_LOCAL_DOF) != subBasisMap.basisDofOrdinals.end())
//                {
//                  cout << CamelliaCellTools::entityTypeString(d) << " " << subcord << " on cell " << cellID << " constrained by ";
//                  cout << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << subcellOrdinalInConstrainingCell << " on cell ";
//                  cout << subcellConstraint.cellID << ".\n";
//                  
//                  Camellia::print("weights coarse ordinals", weightsForSubcell.coarseOrdinals);
//                  Camellia::print("subBasisMap.basisDofOrdinals", subBasisMap.basisDofOrdinals);
//                  Camellia::print("subBasisMap.globalDofOrdinals", subBasisMap.globalDofOrdinals);
//                  cout << "subBasisMap.weights:\n" << weightsForSubcell.weights;
//                }
//              }
//            }
//          }
//        }
        
        subBasisMaps.push_back(subBasisMap);
      }
      
      // volume version
      // process subcells of the coarse subcell (new code, new idea as of 2-8-16; passes tests thus far, but coverage isn't terribly thorough)
      for (int subsubcdim=minimumConstraintDimension; subsubcdim<subcellConstraint.dimension; subsubcdim++)
      {
        int subsubcellCount = constrainingTopo->getSubcellCount(subsubcdim);
        for (int subsubcellOrdinal = 0; subsubcellOrdinal < subsubcellCount; subsubcellOrdinal++)
        {
          // first question: is this subcell of the original constraining subcell further constrained?
          // (In a 1-irregular mesh, I believe this is only possible if the original constraint did not involve a hanging node--could be a permutation,
          //  or a trivial constraint.)
          // If it is further constrained, then I *think* this will naturally be handled at some other point.
          int sscOrdInOriginalConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(),
                                                                                      subcellConstraint.dimension,
                                                                                      subcellOrdinalInConstrainingCell,
                                                                                      subsubcdim, subsubcellOrdinal);
          
          CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint.cellID);
          AnnotatedEntity subsubcellConstraints = constrainingCellConstraints.subcellConstraints[subsubcdim][sscOrdInOriginalConstrainingCell];
          CellPtr subsubcellConstrainingCell = _meshTopology->getCell(subsubcellConstraints.cellID);
          int sscOrdInNewConstrainingCell = CamelliaCellTools::subcellOrdinalMap(subsubcellConstrainingCell->topology(), sideDim,
                                                                                 subsubcellConstraints.sideOrdinal, subsubcdim,
                                                                                 subsubcellConstraints.subcellOrdinal);
          
          bool furtherConstrained;
          int subsubcellPermutation = 0; // permutation between constrainingCell's view and that in the subsubcellConstrainingCell
          if (subsubcellConstraints.dimension != subsubcdim)
            furtherConstrained = true;
          else
          {
            
            IndexType constrainingEntityIndex = subsubcellConstrainingCell->entityIndex(subsubcdim, sscOrdInNewConstrainingCell);
            IndexType subsubcellEntityIndex = constrainingCell->entityIndex(subsubcdim, sscOrdInOriginalConstrainingCell);
            furtherConstrained = (constrainingEntityIndex != subsubcellEntityIndex);
            if (!furtherConstrained)
            {
              // from canonical to subcell's constraint view:
              unsigned sscOriginalConstrainingPermutation = constrainingCell->subcellPermutation(subsubcdim,
                                                                                                 sscOrdInOriginalConstrainingCell);
              // from canonical to subsubcell's constraint view:
              unsigned sscNewConstrainingPermutation = subsubcellConstrainingCell->subcellPermutation(subsubcellConstraints.dimension,
                                                                                                      sscOrdInNewConstrainingCell);
              
              CellTopoPtr subsubcellTopo = constrainingCell->topology()->getSubcell(subsubcdim, sscOrdInOriginalConstrainingCell);
              unsigned subcellPermutationInverse = CamelliaCellTools::permutationInverse(subsubcellTopo, sscOriginalConstrainingPermutation);
              subsubcellPermutation = CamelliaCellTools::permutationComposition(subsubcellTopo, subcellPermutationInverse, sscNewConstrainingPermutation);
            }
          }
          TEUCHOS_TEST_FOR_EXCEPTION(furtherConstrained && subcellIsGeometricallyConstrained, std::invalid_argument, "Mesh is not 1-irregular");
          if (furtherConstrained) continue;
          
          SubBasisReconciliationWeights weightsForSubSubcell = BasisReconciliation::weightsForCoarseSubcell(weightsForWholeSubcell, constrainingBasis,
                                                                                                            subsubcdim, sscOrdInOriginalConstrainingCell,
                                                                                                            false);
          
//          if (weightsForWholeSubcell.isIdentity)
//          {
//            SubBasisReconciliationWeights weightsForWholeSubcellManual = weightsManualIdentity(weightsForWholeSubcell);
//            SubBasisReconciliationWeights expectedWeights = BasisReconciliation::weightsForCoarseSubcell(weightsForWholeSubcellManual, constrainingBasis,
//                                                                                                         subsubcdim, sscOrdInOriginalConstrainingCell,
//                                                                                                         false);
//            bool equal = BasisReconciliation::equalWeights(expectedWeights, weightsForSubSubcell);
//            TEUCHOS_TEST_FOR_EXCEPTION(!equal, std::invalid_argument, "Something wrong with identity weights treatment");
//          }
          
          DofOrderingPtr sscConstrainingTrialOrdering = elementType(subsubcellConstraints.cellID)->trialOrderPtr;
          BasisPtr sscConstrainingBasis = sscConstrainingTrialOrdering->getBasis(var->ID());
          
          RefinementPatternPtr noRefinementPattern = RefinementPattern::noRefinementPattern(constrainingBasis->domainTopology());
          RefinementBranch noRefinements = {{noRefinementPattern.get(),0}};
          
          SubBasisReconciliationWeights coarseWeightPermutation = _br.constrainedWeights(subsubcdim, constrainingBasis,
                                                                                         sscOrdInOriginalConstrainingCell,
                                                                                         noRefinements, sscConstrainingBasis,
                                                                                         sscOrdInNewConstrainingCell,
                                                                                         subsubcellPermutation);
          
          //          SubBasisReconciliationWeights expectedComposition; // DEBUGGING
          //          { //DEBUGGING
          //            if (coarseWeightPermutation.isIdentity || weightsForSubSubcell.isIdentity)
          //            {
          //              SubBasisReconciliationWeights coarseWeightPermutationManual = coarseWeightPermutation.isIdentity ? weightsManualIdentity(coarseWeightPermutation) : coarseWeightPermutation;
          //              SubBasisReconciliationWeights weightsForSubSubcellManual = weightsForSubSubcell.isIdentity ? weightsManualIdentity(weightsForSubSubcell) : weightsForSubSubcell;
          //
          //              expectedComposition = BasisReconciliation::composedSubBasisReconciliationWeights(weightsForSubSubcellManual, coarseWeightPermutationManual);
          //            }
          //          }
          
          //              cout << "weightsForSubSubcell.weights, before applying permutation:\n" << weightsForSubSubcell.weights;
          //              cout << "coarseWeightPermutation.weights:\n" << coarseWeightPermutation.weights;
          
          weightsForSubSubcell = BasisReconciliation::composedSubBasisReconciliationWeights(weightsForSubSubcell, coarseWeightPermutation);
          
          
          // copied and pasted from above.  Could refactor:
          if ((weightsForSubSubcell.coarseOrdinals.size() > 0) && (weightsForSubSubcell.fineOrdinals.size() > 0))
          {
            vector<GlobalIndexType> globalDofOrdinalsForSubcell = getGlobalDofOrdinalsForSubcell(subsubcellConstraints.cellID,
                                                                                                 var, subsubcdim,
                                                                                                 sscOrdInNewConstrainingCell);
            
            // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
            
            const vector<int>* constrainingBasisOrdinalsForSubcell = &sscConstrainingBasis->dofOrdinalsForSubcell(subsubcdim,
                                                                                                                  sscOrdInNewConstrainingCell);
            vector<GlobalIndexType> globalDofOrdinals;
            for (int i=0; i<constrainingBasisOrdinalsForSubcell->size(); i++)
            {
              int constrainingBasisOrdinal = (*constrainingBasisOrdinalsForSubcell)[i];
              if (weightsForSubSubcell.coarseOrdinals.find(constrainingBasisOrdinal) != weightsForSubSubcell.coarseOrdinals.end())
              {
                globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
              }
            }
            
            if (weightsForSubSubcell.coarseOrdinals.size() != globalDofOrdinals.size())
            {
              cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
              Camellia::print("coarseOrdinals", weightsForSubSubcell.coarseOrdinals);
              Camellia::print("globalDofOrdinals", globalDofOrdinals);
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
            }
            
            subBasisMap.weights = weightsForSubSubcell.weights;
            subBasisMap.globalDofOrdinals = globalDofOrdinals;
            subBasisMap.basisDofOrdinals = weightsForSubSubcell.fineOrdinals;
            subBasisMap.isIdentity = weightsForSubSubcell.isIdentity;
            //            { // DEBUGGING
            //              if (cellID == DEBUG_CELL_ID)
            //              {
            //                if (var->ID() == DEBUG_VAR_ID)
            //                {
            //                  if (std::find(subBasisMap.globalDofOrdinals.begin(), subBasisMap.globalDofOrdinals.end(), DEBUG_GLOBAL_DOF) != subBasisMap.globalDofOrdinals.end())
            //                  {
            //                    if (subBasisMap.basisDofOrdinals.find(DEBUG_LOCAL_DOF) != subBasisMap.basisDofOrdinals.end())
            //                    {
            //                      cout << CamelliaCellTools::entityTypeString(d) << " " << subcord << " on cell " << cellID << " constrained by ";
            //                      cout << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << subcellOrdinalInConstrainingCell << " on cell ";
            //                      cout << subcellConstraint.cellID << "; " << CamelliaCellTools::entityTypeString(subsubcdim) << " " << subsubcellOrdinalInConstrainingCell;
            //                      cout << " on cell " << subcellConstraint.cellID << " constrained by ";
            //                      cout << CamelliaCellTools::entityTypeString(subsubcellConstraints.dimension) << " " << subcellOrdinalInConstrainingCell << " on cell ";
            //                      cout << subsubcellConstraints.cellID << ".\n";
            //
            //                      Camellia::print("weights coarse ordinals", weightsForSubSubcell.coarseOrdinals);
            //                      Camellia::print("subBasisMap.basisDofOrdinals", subBasisMap.basisDofOrdinals);
            //                      Camellia::print("subBasisMap.globalDofOrdinals", subBasisMap.globalDofOrdinals);
            //                      cout << "subBasisMap.weights:\n" << weightsForSubSubcell.weights;
            //                    }
            //                  }
            //                }
            //              }
            //            }
            
            subBasisMaps.push_back(subBasisMap);
            
            //              { // DEBUGGING
            //                cout << "weights for whole subcell:\n";
            //                Camellia::print("coarseOrdinals", weightsForWholeSubcell.coarseOrdinals);
            //                Camellia::print("fineOrdinals", weightsForWholeSubcell.fineOrdinals);
            //                cout << weightsForWholeSubcell.weights;
            //
            //                cout << "weights for sub-subcell:\n";
            //                Camellia::print("coarseOrdinals", weightsForSubSubcell.coarseOrdinals);
            //                Camellia::print("fineOrdinals", weightsForSubSubcell.fineOrdinals);
            //                cout << weightsForSubSubcell.weights;
            //              }
          }
        }
      }
    }
  }
  
  // now, we collect the local basis coefficients corresponding to each global ordinal
  // likely there is a more efficient way to do this, but for now this is our approach
  map< GlobalIndexType, map<int, double> > weightsForGlobalOrdinal;
  map< int, set<GlobalIndexType> > globalOrdinalsForFineOrdinal;
  
  for (vector<SubBasisMapInfo>::iterator subBasisIt = subBasisMaps.begin(); subBasisIt != subBasisMaps.end(); subBasisIt++)
  {
    subBasisMap = *subBasisIt;
    vector<GlobalIndexType> globalDofOrdinals = subBasisMap.globalDofOrdinals;
    set<int>* basisDofOrdinals = &subBasisMap.basisDofOrdinals;
    // weights are fine x coarse
    for (int j=0; j<subBasisMap.globalDofOrdinals.size(); j++)
    {
      GlobalIndexType globalDofOrdinal = globalDofOrdinals[j];
      map<int, double> fineOrdinalCoefficientsThusFar = weightsForGlobalOrdinal[globalDofOrdinal];
      auto fineOrdinalPtr = basisDofOrdinals->begin();
      for (int i=0; i<subBasisMap.basisDofOrdinals.size(); i++)
      {
        int fineOrdinal = *fineOrdinalPtr++;
        double coefficient;
        if (!subBasisMap.isIdentity)
          coefficient = subBasisMap.weights(i,j);
        else
          coefficient = (i == j) ? 1.0 : 0.0;
        
        if (coefficient != 0)
        {
          if (fineOrdinalCoefficientsThusFar.find(fineOrdinal) != fineOrdinalCoefficientsThusFar.end())
          {
            double tol = 1e-13;
            double previousCoefficient = fineOrdinalCoefficientsThusFar[fineOrdinal];
            if (abs(previousCoefficient-coefficient) > tol)
            {
              cout  << "ERROR: incompatible entries for fine ordinal " << fineOrdinal << " in representation of global ordinal " << globalDofOrdinal << endl;
              cout << "previousCoefficient = " << previousCoefficient << endl;
              cout << "coefficient = " << coefficient << endl;
              cout << "diff = " << abs(previousCoefficient - coefficient) << endl;
              cout << "Encountered the incompatible entry while processing variable " << var->name() << " on cell " << cellID << ", on volume" << endl;
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal Error: incompatible entries for fine ordinal " );
            }
          }
          fineOrdinalCoefficientsThusFar[fineOrdinal] = coefficient;
          globalOrdinalsForFineOrdinal[fineOrdinal].insert(globalDofOrdinal);
//          {
//            // DEBUGGING
//            if (cellID == DEBUG_CELL_ID)
//            {
//              if (var->ID() == DEBUG_VAR_ID)
//              {
//                if ((fineOrdinal == DEBUG_LOCAL_DOF) && (globalDofOrdinal == DEBUG_GLOBAL_DOF))
//                {
//                  cout << "subBasisMap.weights:\n" << subBasisMap.weights;
//                  cout << "Added weight " << coefficient << " for global dof " << globalDofOrdinal << " representation by local basis ordinal " << fineOrdinal << endl;
//                }
//              }
//            }
//          }
        }
      }
      weightsForGlobalOrdinal[globalDofOrdinal] = fineOrdinalCoefficientsThusFar;
    }
  }
  
  // partition global ordinals according to which fine ordinals they interact with -- this is definitely not super-efficient
  set<GlobalIndexType> partitionedGlobalDofOrdinals;
  vector< set<GlobalIndexType> > globalDofOrdinalPartitions;
  vector< set<int> > fineOrdinalsForPartition;
  
  for (map< GlobalIndexType, map<int, double> >::iterator globalWeightsIt = weightsForGlobalOrdinal.begin();
       globalWeightsIt != weightsForGlobalOrdinal.end(); globalWeightsIt++)
  {
    GlobalIndexType globalOrdinal = globalWeightsIt->first;
    if (partitionedGlobalDofOrdinals.find(globalOrdinal) != partitionedGlobalDofOrdinals.end()) continue;
    
    set<GlobalIndexType> partition;
    partition.insert(globalOrdinal);
    
    set<int> fineOrdinals;
    
    set<GlobalIndexType> globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed;
    
    map<int, double> fineCoefficients = globalWeightsIt->second;
    for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
    {
      int fineOrdinal = coefficientIt->first;
      fineOrdinals.insert(fineOrdinal);
      set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
      partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
    }
    
    globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
    
    while (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.size() != partition.size())
    {
      for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
      {
        GlobalIndexType globalOrdinal = *globalOrdIt;
        if (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.find(globalOrdinal) !=
            globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.end()) continue;
        map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
        for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
        {
          int fineOrdinal = coefficientIt->first;
          fineOrdinals.insert(fineOrdinal);
          set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
          partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
        }
        
        globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
      }
    }
    
    for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
    {
      GlobalIndexType globalOrdinal = *globalOrdIt;
      partitionedGlobalDofOrdinals.insert(globalOrdinal);
    }
    globalDofOrdinalPartitions.push_back(partition);
    fineOrdinalsForPartition.push_back(fineOrdinals);
  }
  
  for (int i=0; i<globalDofOrdinalPartitions.size(); i++)
  {
    set<GlobalIndexType> partition = globalDofOrdinalPartitions[i];
    set<int> fineOrdinals = fineOrdinalsForPartition[i];
    vector<int> fineOrdinalsVector(fineOrdinals.begin(), fineOrdinals.end());
    map<int,int> fineOrdinalRowLookup;
    for (int i=0; i<fineOrdinalsVector.size(); i++)
    {
      fineOrdinalRowLookup[fineOrdinalsVector[i]] = i;
    }
    Camellia::FieldContainer<double> weights(fineOrdinals.size(),partition.size());
    int col=0;
    for (set<GlobalIndexType>::iterator globalDofIt=partition.begin(); globalDofIt != partition.end(); globalDofIt++)
    {
      GlobalIndexType globalOrdinal = *globalDofIt;
      map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
      for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
      {
        int fineOrdinal = coefficientIt->first;
        double coefficient = coefficientIt->second;
        int row = fineOrdinalRowLookup[fineOrdinal];
        weights(row,col) = coefficient;
      }
      col++;
    }
    vector<GlobalIndexType> globalOrdinals(partition.begin(),partition.end());
    SubBasisDofMapperPtr subBasisMap = SubBasisDofMapper::subBasisDofMapper(fineOrdinals, globalOrdinals, weights);
    varVolumeMap.push_back(subBasisMap);
  }
  
  return varVolumeMap;
}

BasisMap GDAMinimumRule::getBasisMapDiscontinuousVolumeRestrictedToSide(GlobalIndexType cellID, SubcellDofIndices& dofOwnershipInfo, VarPtr var, int sideOrdinal)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getBasisMapDiscontinuousVolumeRestrictedToSide");
   BasisMap volumeMap = getBasisMap(cellID, dofOwnershipInfo, var);

  // We're interested in the restriction of the map to the side
  
  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  BasisPtr basis = trialOrdering->getBasis(var->ID());
  set<int> basisDofOrdinalsForSide = basis->dofOrdinalsForSide(sideOrdinal);
  
  // for now, the only case we will support is the one where each volume dof is identified with a single global dof
  // (this is the case when using discontinuous volume dofs, which is the case for ultraweak DPG as well as DG)
  // we assert that condition here:
  {
    for (SubBasisDofMapperPtr subBasisDofMapper : volumeMap)
    {
      // does this mapper involve any of the dof ordinals we're interested in?
      // if so, we require that it is a permutation -- i.e. no constraints imposed
      bool containsSideDofOrdinal = false;
      for (int basisDofOrdinal : basisDofOrdinalsForSide)
      {
        if (subBasisDofMapper->basisDofOrdinalFilter().find(basisDofOrdinal) != subBasisDofMapper->basisDofOrdinalFilter().end())
        {
          containsSideDofOrdinal = true;
          break;
        }
      }
      if (! containsSideDofOrdinal) break;
      
      TEUCHOS_TEST_FOR_EXCEPTION(!subBasisDofMapper->isPermutation(), std::invalid_argument, "getDofMapper() only supports side restrictions of volume variables in cases where the volume variables do not have any constraints imposed");
    }
  }

  vector<GlobalIndexType> globalDofOrdinals;
  for (int basisDofOrdinalForSide : basisDofOrdinalsForSide)
  {
    for (SubBasisDofMapperPtr subBasisDofMapper : volumeMap)
    {
      if (subBasisDofMapper->basisDofOrdinalFilter().find(basisDofOrdinalForSide) != subBasisDofMapper->basisDofOrdinalFilter().end())
      {
        // this dof mapper contains the guy we're looking for
        set<int> basisDofOrdinalSet = {basisDofOrdinalForSide};
        set<GlobalIndexType> mappedGlobalDofOrdinals = subBasisDofMapper->mappedGlobalDofOrdinalsForBasisOrdinals(basisDofOrdinalSet);
        if (mappedGlobalDofOrdinals.size() != 1)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getDofMapper() only supports side restrictions of volume variables if volume mapping is a permutation");
        }
        GlobalIndexType mappedGlobalDofOrdinal = *mappedGlobalDofOrdinals.begin();
        
        globalDofOrdinals.push_back(mappedGlobalDofOrdinal);
        break; // break out of subBasisDofMapper iteration loop -- we found our guy
      }
    }
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(globalDofOrdinals.size() != basisDofOrdinalsForSide.size(), std::invalid_argument, "Error: did not find globalDofOrdinals for all basisDofOrdinals on the side");
  
  BasisMap varVolumeMap;
  
  if (basisDofOrdinalsForSide.size() > 0)
  {
    varVolumeMap.push_back(SubBasisDofMapper::subBasisDofMapper(basisDofOrdinalsForSide, globalDofOrdinals));
  }

  return varVolumeMap;
}

// trace variable version
BasisMap GDAMinimumRule::getBasisMap(GlobalIndexType cellID, SubcellDofIndices& subcellDofIndices, VarPtr var, int sideOrdinal)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getBasisMap - trace variable variant");
  // TODO: looks like we don't use the subcellDofIndices argument.  Consider excising.
  
  vector<SubBasisMapInfo> subBasisMaps;

  SubBasisMapInfo subBasisMap;

//  const static int DEBUG_VAR_ID = 2;
//  const static GlobalIndexType DEBUG_CELL_ID = 44;
//  const static unsigned DEBUG_SIDE_ORDINAL = 2;
//  const static GlobalIndexType DEBUG_GLOBAL_DOF = 1316;
  
//  {
//    // DEBUGGING
//    if ((cellID == DEBUG_CELL_ID) && (sideOrdinal == DEBUG_SIDE_ORDINAL) && (var->ID() == DEBUG_VAR_ID))
//    {
//      cout << "Debug getBasisMap: set breakpoint here.\n";
//    }
//  }
  
   const static int DEBUG_VAR_ID = -1;
   const static GlobalIndexType DEBUG_CELL_ID = 0;
   const static unsigned DEBUG_SIDE_ORDINAL = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
   const static GlobalIndexType DEBUG_GLOBAL_DOF = -3;

//  if ((cellID==2) && (sideOrdinal==1) && (var->ID() == 0)) {
//    cout << "DEBUGGING: (cellID==2) && (sideOrdinal==1) && (var->ID() == 0).\n";
//  }

  // TODO: move the permutation computation outside of this method -- might include in CellConstraints, e.g. -- this obviously will not change from one var to the next, but we compute it redundantly each time...

  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  CellTopoPtr topo = cell->topology();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;
  CellTopoPtr sideTopo = topo->getSubcell(sideDim, sideOrdinal);

  // assumption is that the basis is defined on the side
  BasisPtr basis = trialOrdering->getBasis(var->ID(), sideOrdinal);
  int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
  
  CellConstraints cellConstraints = getCellConstraints(cellID);
  
  /*
   In a 1-irregular mesh, subcells count as processed when either:
   a) they belong to a constrained subcell
   b) they belong to an unconstrained subcell and are not themselves geometrically constrained
   */
  vector<vector<bool>> processedSubcells(sideDim + 1);
  for (int d=minimumConstraintDimension; d<=sideDim; d++)
  {
    int subcellCount = sideTopo->getSubcellCount(d);
    processedSubcells[d] = vector<bool>(subcellCount,false);
  }
  
  SubBasisReconciliationWeights wholeSubcellWeights;
  for (int d=sideDim; d >= minimumConstraintDimension; d--)
  {
    int subcellCount = sideTopo->getSubcellCount(d);
    for (int subcord=0; subcord < subcellCount; subcord++)
    {
      if (processedSubcells[d][subcord]) continue;
      unsigned subcordInCell = CamelliaCellTools::subcellOrdinalMap(topo, sideDim, sideOrdinal, d, subcord);
      AnnotatedEntity* subcellConstraint = getConstrainingEntityInfo(cellID, cellConstraints, var, d, subcordInCell);
      
      if (_checkConstraintConsistency)
      {
        IndexType subcellEntityIndex = cell->entityIndex(d, subcordInCell);
        bool cellIsActive = !cell->isParent(_meshTopology);
        // if cell is active, then also check that the constrainning entity belongs to an active cell
        // (this can be violated for anisotropic refinements, but that's not presently supported in 3D,
        //  and not what we're debugging right now.)
        if (!MeshTestUtility::constraintIsConsistent(_meshTopology, *subcellConstraint, d, subcellEntityIndex, cellIsActive))
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Constraint is not consistent");
        }
      }
      
      CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint->cellID);
      bool followGeometricConstraints = true;
      if (subcellConstraint != &cellConstraints.subcellConstraints[d][subcordInCell])
      {
        // then we are using a space-only trace, on a subcell geometrically constrained by a temporal interface
        // we use the fact that the mesh is 1-irregular, which means that this subcell will not be otherwise
        // constrained, so that we don't have to follow the geometric constraints at all.
        followGeometricConstraints = false;
      }
      
      DofOrderingPtr constrainingTrialOrdering = elementType(subcellConstraint->cellID)->trialOrderPtr;
      
      BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID(), subcellConstraint->sideOrdinal);
      unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim,
                                                                                       subcellConstraint->sideOrdinal,
                                                                                       subcellConstraint->dimension,
                                                                                       subcellConstraint->subcellOrdinal);
      
      IndexType subcellEntityIndex = cell->entityIndex(d,subcordInCell);
      IndexType constrainingEntityIndex = constrainingCell->entityIndex(subcellConstraint->dimension, subcellOrdinalInConstrainingCell);
      
      bool subcellIsGeometricallyConstrained = (d != subcellConstraint->dimension) || (subcellEntityIndex != constrainingEntityIndex);
      
      CellTopoPtr constrainingTopo = constrainingCell->topology()->getSubcell(subcellConstraint->dimension, subcellOrdinalInConstrainingCell);
      
      CellPtr ancestralCell;
      if (followGeometricConstraints)
        ancestralCell = cell->ancestralCellForSubcell(d, subcordInCell, _meshTopology);
      else
        ancestralCell = cell;
      
      RefinementBranch volumeRefinements;
      pair<unsigned, unsigned> ancestralSubcell;
      unsigned ancestralSubcellOrdinal;
      unsigned ancestralSubcellDimension;
      if (followGeometricConstraints)
      {
        volumeRefinements = cell->refinementBranchForSubcell(d, subcordInCell, _meshTopology);
        ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(d, subcordInCell, _meshTopology);
        ancestralSubcellOrdinal = ancestralSubcell.first;
        ancestralSubcellDimension = ancestralSubcell.second;
      }
      else
      {
        ancestralSubcellOrdinal = subcordInCell;
        ancestralSubcellDimension = d;
      }
      if (volumeRefinements.size()==0)
      {
        // could be, we'd do better to revise Cell::refinementBranchForSubcell() to ensure that we always have a refinement, but for now
        // we just create a RefinementBranch with a trivial refinement here:
        RefinementPatternPtr noRefinementPattern = RefinementPattern::noRefinementPattern(cell->topology());
        volumeRefinements = {{noRefinementPattern.get(),0}};
      }
      
      if (ancestralSubcellOrdinal == INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL)
      {
        cout << "Internal error: ancestral subcell ordinal was not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: ancestral subcell ordinal was not found.");
      }
      
      unsigned ancestralSideOrdinal;
      /*
       How we know that there is always an ancestralSideOrdinal to speak of:
       In the extreme case, consider what happens when you have a triangular refinement and an interior triangle selected,
       and you're concerned with one of its vertices: even then there is a side ordinal even then, and a unique one.  Whatever constraints
       there are eventually come through some constraining side (or the subcell of some constraining side).
       */
      if (ancestralSubcellDimension == sideDim)
      {
        ancestralSideOrdinal = ancestralSubcellOrdinal;
      }
      else if (!followGeometricConstraints)
      {
        ancestralSideOrdinal = sideOrdinal;
      }
      else
      {
        IndexType ancestralSubcellEntityIndex = ancestralCell->entityIndex(ancestralSubcellDimension, ancestralSubcellOrdinal);
        
        // for subcells constrained by subcells of unlike dimension, we can handle any side that contains the ancestral subcell,
        // but for like-dimensional constraints, we do need the ancestralSideOrdinal to be the ancestor of the side in subcellInfo...
        
        if (subcellConstraint->dimension == d)
        {
          IndexType descendantSideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
          
          ancestralSideOrdinal = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
          int sideCount = ancestralCell->getSideCount();
          for (int side=0; side<sideCount; side++)
          {
            IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
            if (ancestralSideEntityIndex == descendantSideEntityIndex)
            {
              ancestralSideOrdinal = side;
              break;
            }
            
            if (_meshTopology->entityIsAncestor(sideDim, ancestralSideEntityIndex, descendantSideEntityIndex))
            {
              ancestralSideOrdinal = side;
              break;
            }
          }
          
          if (ancestralSideOrdinal == INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
          {
            cout << "Error: no ancestor of side found.\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: no ancestor of side contains the ancestral subcell.");
          }
          
          {
            // a sanity check:
            vector<IndexType> sidesForSubcell = _meshTopology->getSidesContainingEntity(ancestralSubcellDimension, ancestralSubcellEntityIndex);
            IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, ancestralSideOrdinal);
            if (std::find(sidesForSubcell.begin(), sidesForSubcell.end(), ancestralSideEntityIndex) == sidesForSubcell.end())
            {
              cout << "Error: the ancestral side does not contain the ancestral subcell.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "the ancestral side does not contain the ancestral subcell.");
            }
          }
        }
        else
        {
          // find some side in the ancestral cell that contains the ancestral subcell, then.
          // (there should be at least two; which one shouldn't matter, except in the case of space-only trace variables
          vector<IndexType> sidesForSubcell = _meshTopology->getSidesContainingEntity(ancestralSubcellDimension, ancestralSubcellEntityIndex);
          
          ancestralSideOrdinal = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
          int sideCount = ancestralCell->getSideCount();
          // search among the spatial sides first:
          for (int side=0; side<sideCount; side++)
          {
            if (! ancestralCell->topology()->sideIsSpatial(side)) continue; // skip non-spatial sides
            IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
            if (std::find(sidesForSubcell.begin(), sidesForSubcell.end(), ancestralSideEntityIndex) != sidesForSubcell.end())
            {
              ancestralSideOrdinal = side;
              break;
            }
          }
          // if we haven't found among the spatial sides, then search in the non-spatial ones
          if (ancestralSideOrdinal == INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
          {
            for (int side=0; side<sideCount; side++)
            {
              if (ancestralCell->topology()->sideIsSpatial(side)) continue; // skip spatial sides
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
              if (std::find(sidesForSubcell.begin(), sidesForSubcell.end(), ancestralSideEntityIndex) != sidesForSubcell.end())
              {
                ancestralSideOrdinal = side;
                break;
              }
            }
          }
        }
      }
      
      if (ancestralSideOrdinal == INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
      {
        cout << "Error: ancestralSideOrdinal not found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: ancestralSideOrdinal not found.");
      }
      
      {
        // ancestralPermutation goes from canonical to cell's side's ancestor's ordering:
        unsigned ancestralCellPermutation = ancestralCell->subcellPermutation(ancestralSubcellDimension, ancestralSubcellOrdinal);
        // constrainingPermutation goes from canonical to the constraining side's ordering
        unsigned constrainingCellPermutation = constrainingCell->subcellPermutation(subcellConstraint->dimension, subcellOrdinalInConstrainingCell); // subcell permutation as seen from the perspective of the constraining cell's side
        
        // ancestralPermutationInverse goes from ancestral view to canonical
        unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralCellPermutation);
        unsigned ancestralToConstrainedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingCellPermutation);
        
        wholeSubcellWeights = BasisReconciliation::computeConstrainedWeights(d, basis, subcord, volumeRefinements, sideOrdinal,
                                                                             ancestralCell->topology(), subcellConstraint->dimension,
                                                                             constrainingBasis, subcellConstraint->subcellOrdinal,
                                                                             subcellConstraint->sideOrdinal,
                                                                             ancestralToConstrainedPermutation);
      }
      
      // add sub-basis map for dofs interior to the constraining subcell
      // filter the weights whose coarse dofs are interior to this subcell, and create a SubBasisDofMapper for these (add it to varSideMap)
      SubBasisReconciliationWeights subcellInteriorWeights = BasisReconciliation::weightsForCoarseSubcell(wholeSubcellWeights, constrainingBasis, subcellConstraint->dimension,
                                                                                                          subcellConstraint->subcellOrdinal, false);
      
      //        { //DEBUGGING
      //          if (wholeSubcellWeights.isIdentity)
      //          {
      //            SubBasisReconciliationWeights wholeSubcellWeightsManual = weightsManualIdentity(wholeSubcellWeights);
      //            SubBasisReconciliationWeights subcellInteriorWeightsManual = BasisReconciliation::weightsForCoarseSubcell(wholeSubcellWeightsManual, constrainingBasis,
      //                                                                                                                      subcellConstraint.dimension,
      //                                                                                                                      subcellConstraint.subcellOrdinal, false);
      //            bool equal = BasisReconciliation::equalWeights(subcellInteriorWeightsManual, subcellInteriorWeights);
      //            TEUCHOS_TEST_FOR_EXCEPTION(!equal, std::invalid_argument, "Something wrong with the identity map");
      //          }
      //        }
      
      if ((subcellInteriorWeights.coarseOrdinals.size() > 0) && (subcellInteriorWeights.fineOrdinals.size() > 0))
      {
        vector<GlobalIndexType> globalDofOrdinalsForSubcell = getGlobalDofOrdinalsForSubcell(subcellConstraint->cellID, var,
                                                                                             subcellConstraint->dimension,
                                                                                             subcellOrdinalInConstrainingCell);
        
        // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
        const vector<int>* constrainingBasisOrdinalsForSubcell = &constrainingBasis->dofOrdinalsForSubcell(subcellConstraint->dimension, subcellConstraint->subcellOrdinal);
        
        if (globalDofOrdinalsForSubcell.size() != constrainingBasisOrdinalsForSubcell->size())
        {
          // repeat the call for debugging convenience
          vector<GlobalIndexType> globalDofOrdinalsForSubcell = getGlobalDofOrdinalsForSubcell(subcellConstraint->cellID, var,
                                                                                               subcellConstraint->dimension,
                                                                                               subcellOrdinalInConstrainingCell);
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getGlobalDofOrdinalsForSubcell() does not match constrainingBasisOrdinalsForSubcell->size()");
        }
        
        vector<GlobalIndexType> globalDofOrdinals;
        for (int i=0; i<constrainingBasisOrdinalsForSubcell->size(); i++)
        {
          int constrainingBasisOrdinal = (*constrainingBasisOrdinalsForSubcell)[i];
          if (subcellInteriorWeights.coarseOrdinals.find(constrainingBasisOrdinal) != subcellInteriorWeights.coarseOrdinals.end())
          {
            globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
//            // DEBUGGING:
//            if ((globalDofOrdinalsForSubcell[i]==DEBUG_GLOBAL_DOF) && (cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL))
//            {
//              cout << "globalDofOrdinalsForSubcell includes global dof ordinal " << DEBUG_GLOBAL_DOF << ".\n";
//              auto fineOrdinalIt = subcellInteriorWeights.fineOrdinals.begin();
//              for (int fineWeightOrdinal=0; fineWeightOrdinal<subcellInteriorWeights.fineOrdinals.size(); fineWeightOrdinal++,
//                   fineOrdinalIt++)
//              {
//                cout << "fine ordinal " << *fineOrdinalIt << " weight for global dof " << DEBUG_GLOBAL_DOF << ": " << subcellInteriorWeights.weights(fineWeightOrdinal,i) << endl;
//              }
//            }
          }
        }
        
        if (subcellInteriorWeights.coarseOrdinals.size() != globalDofOrdinals.size())
        {
          cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
          Camellia::print("coarseOrdinals", subcellInteriorWeights.coarseOrdinals);
          Camellia::print("globalDofOrdinals", globalDofOrdinals);
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
        }
        
        // DEBUGGING:
        if ((cellID==DEBUG_CELL_ID) && (sideOrdinal==DEBUG_SIDE_ORDINAL) && (var->ID()==DEBUG_VAR_ID))
        {
          set<unsigned> globalDofOrdinalsSet(globalDofOrdinals.begin(),globalDofOrdinals.end());
          IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(subcellConstraint->dimension, subcellOrdinalInConstrainingCell);
          
          cout << "Determined constraints imposed by ";
          cout << CamelliaCellTools::entityTypeString(subcellConstraint->dimension) << " " << constrainingSubcellEntityIndex;
          //            cout << " (owned by " << CamelliaCellTools::entityTypeString(ownershipInfo->dimension) << " " << ownershipInfo->owningSubcellEntityIndex << ")\n";
          
          ostringstream basisOrdinalsString, globalOrdinalsString;
          basisOrdinalsString << "basisDofOrdinals mapped on cell " << DEBUG_CELL_ID;
          basisOrdinalsString << ", sideOrdinal " << DEBUG_SIDE_ORDINAL;
          
          globalOrdinalsString << "globalDofOrdinals mapped on cell " << DEBUG_CELL_ID;
          globalOrdinalsString << ", sideOrdinal " << DEBUG_SIDE_ORDINAL;
          
          Camellia::print(basisOrdinalsString.str(),subcellInteriorWeights.fineOrdinals);
          Camellia::print(globalOrdinalsString.str(), globalDofOrdinalsSet);
          if (subcellInteriorWeights.isIdentity) cout << "weights: identity\n";
          else cout << "weights:\n" << subcellInteriorWeights.weights;
        }
        
        subBasisMap.weights = subcellInteriorWeights.weights;
        subBasisMap.globalDofOrdinals = globalDofOrdinals;
        subBasisMap.basisDofOrdinals = subcellInteriorWeights.fineOrdinals;
        subBasisMap.isIdentity = subcellInteriorWeights.isIdentity;
        
        subBasisMaps.push_back(subBasisMap);
        
        {
          // DEBUGGING
          if ((DEBUG_CELL_ID == cellID) && (DEBUG_SIDE_ORDINAL == sideOrdinal) && (DEBUG_VAR_ID == var->ID()))
          {
            bool containsDebugDofIndex = false;
            for (GlobalIndexType globalDofIndex : globalDofOrdinals) {
              if (DEBUG_GLOBAL_DOF == globalDofIndex)
              {
                containsDebugDofIndex = true;
                break;
              }
            }
            if (containsDebugDofIndex)
            {
              cout << "mapped global dof ordinal " << DEBUG_GLOBAL_DOF << " on ";
              cout << CamelliaCellTools::entityTypeString(subcellConstraint->dimension);
              cout << " " << subcellOrdinalInConstrainingCell << " of cell " << subcellConstraint->cellID << endl;
            }
          }
        }
      }
      
      CellTopoPtr constrainingSideTopo = constrainingCell->topology()->getSide(subcellConstraint->sideOrdinal);
      
      // process subcells of the coarse subcell (new code, new idea as of 2-8-16; passes tests thus far, but coverage isn't terribly thorough)
      for (int subsubcdim=minimumConstraintDimension; subsubcdim<subcellConstraint->dimension; subsubcdim++)
      {
        int subsubcellCount = constrainingTopo->getSubcellCount(subsubcdim);
        for (int subsubcellOrdinal = 0; subsubcellOrdinal < subsubcellCount; subsubcellOrdinal++)
        {
          // first question: is this subcell of the original constraining subcell further constrained?
          // (In a 1-irregular mesh, I believe this is only possible if the original constraint did not involve a hanging node--could be a permutation,
          //  or a trivial constraint.)
          // If it is further constrained, then I *think* this will naturally be handled at some other point.
          int sscOrdInOriginalConstrainingSide = CamelliaCellTools::subcellOrdinalMap(constrainingSideTopo,
                                                                                      subcellConstraint->dimension,
                                                                                      subcellConstraint->subcellOrdinal,
                                                                                      subsubcdim, subsubcellOrdinal);
          
          int sscOrdInOriginalConstrainingCell = CamelliaCellTools::subcellOrdinalMap(
                                                                                      constrainingCell->topology(),
                                                                                      sideDim, subcellConstraint->sideOrdinal,
                                                                                      subsubcdim, sscOrdInOriginalConstrainingSide);
          
          CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint->cellID);
          
          AnnotatedEntity* subsubcellConstraints = getConstrainingEntityInfo(subcellConstraint->cellID, constrainingCellConstraints,
                                                                             var, subsubcdim, sscOrdInOriginalConstrainingCell);
          
          if (_checkConstraintConsistency)
          {
            IndexType subsubcellEntityIndex = constrainingCell->entityIndex(subsubcdim, sscOrdInOriginalConstrainingCell);
            bool cellIsActive = !cell->isParent(_meshTopology);
            // if cell is active, then also check that the constraining entity belongs to an active cell
            // (this can be violated for anisotropic refinements, but that's not presently supported in 3D,
            //  and not what we're debugging right now.)
            if (!MeshTestUtility::constraintIsConsistent(_meshTopology, *subsubcellConstraints, subsubcdim, subsubcellEntityIndex, cellIsActive))
            {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Constraint is not consistent");
            }
          }
          
          int sscOrdInNewConstrainingSide = subsubcellConstraints->subcellOrdinal;
          int sscOrdInNewConstrainingCell = -1;
          
          bool furtherConstrained;
          int subsubcellPermutation = 0; // permutation between constrainingCell's view and that in the subsubcellConstrainingCell
          if (subsubcellConstraints->dimension != subsubcdim)
            furtherConstrained = true;
          else
          {
            CellPtr subsubcellConstrainingCell = _meshTopology->getCell(subsubcellConstraints->cellID);
            sscOrdInNewConstrainingCell = CamelliaCellTools::subcellOrdinalMap(subsubcellConstrainingCell->topology(), sideDim,
                                                                               subsubcellConstraints->sideOrdinal, subsubcellConstraints->dimension,
                                                                               subsubcellConstraints->subcellOrdinal);
            IndexType constrainingEntityIndex = subsubcellConstrainingCell->entityIndex(subsubcdim, sscOrdInNewConstrainingCell);
            
            IndexType subsubcellEntityIndex = constrainingCell->entityIndex(subsubcdim, sscOrdInOriginalConstrainingCell);
            furtherConstrained = (constrainingEntityIndex != subsubcellEntityIndex);
            
            if (!furtherConstrained)
            {
              // from canonical to subcell's constraint view:
              unsigned sscOriginalConstrainingPermutation = constrainingCell->sideSubcellPermutation(subcellConstraint->sideOrdinal,
                                                                                                     subsubcdim,
                                                                                                     sscOrdInOriginalConstrainingSide);
              // from canonical to subsubcell's constraint view:
              unsigned sscNewConstrainingPermutation = subsubcellConstrainingCell->sideSubcellPermutation(subsubcellConstraints->sideOrdinal, subsubcellConstraints->dimension, subsubcellConstraints->subcellOrdinal);
              
              CellTopoPtr subsubcellTopo = constrainingSideTopo->getSubcell(subsubcdim, sscOrdInOriginalConstrainingSide);
              unsigned subcellPermutationInverse = CamelliaCellTools::permutationInverse(subsubcellTopo, sscOriginalConstrainingPermutation);
              subsubcellPermutation = CamelliaCellTools::permutationComposition(subsubcellTopo, subcellPermutationInverse, sscNewConstrainingPermutation);
            }
          }
          if (furtherConstrained && subcellIsGeometricallyConstrained)
          {
            CellPtr subsubcellConstrainingCell = _meshTopology->getCell(subsubcellConstraints->cellID);
            sscOrdInNewConstrainingCell = CamelliaCellTools::subcellOrdinalMap(subsubcellConstrainingCell->topology(), sideDim,
                                                                               subsubcellConstraints->sideOrdinal, subsubcellConstraints->dimension,
                                                                               subsubcellConstraints->subcellOrdinal);
            
            cout << "Mesh has a cascading constraint; on cell " << cellID;
            cout << ", " << CamelliaCellTools::entityTypeString(d) << " " << subcordInCell;
            cout << " is constrained by cell " << subcellConstraint->cellID;
            cout << ", " << CamelliaCellTools::entityTypeString(subcellConstraint->dimension) << " " << subcellOrdinalInConstrainingCell << endl;
            cout << "This has a subcell, " << CamelliaCellTools::entityTypeString(subsubcdim) << " " << sscOrdInOriginalConstrainingCell;
            cout << ", which is constrained by cell " << subsubcellConstraints->cellID;
            cout << ", " << CamelliaCellTools::entityTypeString(subsubcellConstraints->dimension) << " " << sscOrdInNewConstrainingCell;
            cout << endl;
            
            cout << "cell ancestors:\n";
            _meshTopology->printCellAncestors(cellID);
            _meshTopology->printCellAncestors(subcellConstraint->cellID);
            _meshTopology->printCellAncestors(subsubcellConstraints->cellID);
            
            cout << "All active cell ancestors:\n";
            _meshTopology->printActiveCellAncestors();
            
            _meshTopology->printAllEntitiesInBaseMeshTopology();
            
            bool meshIsConsistent = MeshTestUtility::checkConstraintConsistency(_mesh);
            
            if (meshIsConsistent)
              cout << "passes consistency check on rank " << _mesh->Comm()->getRank() << endl;
            else
              cout << "FAILS consistency check on rank " << _mesh->Comm()->getRank() << endl;
            
            
            TEUCHOS_TEST_FOR_EXCEPTION(furtherConstrained && subcellIsGeometricallyConstrained, std::invalid_argument, "Mesh has a cascading constraint (may not be 1-irregular)");
          }
          if (furtherConstrained) continue;
          
          SubBasisReconciliationWeights weightsForSubSubcell;
          weightsForSubSubcell = BasisReconciliation::weightsForCoarseSubcell(wholeSubcellWeights, constrainingBasis,
                                                                              subsubcdim,
                                                                              sscOrdInOriginalConstrainingSide,
                                                                              false);
          //            { //DEBUGGING
          //              if (wholeSubcellWeights.isIdentity)
          //              {
          //                SubBasisReconciliationWeights wholeSubcellWeightsManual = weightsManualIdentity(wholeSubcellWeights);
          //                SubBasisReconciliationWeights weightsForSubSubcellManual = BasisReconciliation::weightsForCoarseSubcell(wholeSubcellWeightsManual, constrainingBasis,
          //                                                                                                                        subsubcdim,
          //                                                                                                                        sscOrdInOriginalConstrainingSide,
          //                                                                                                                        false);
          //                bool equal = BasisReconciliation::equalWeights(weightsForSubSubcellManual, weightsForSubSubcell);
          //                TEUCHOS_TEST_FOR_EXCEPTION(!equal, std::invalid_argument, "Something wrong with the identity map");
          //              }
          //            }
          DofOrderingPtr sscConstrainingTrialOrdering = elementType(subsubcellConstraints->cellID)->trialOrderPtr;
          BasisPtr sscConstrainingBasis = sscConstrainingTrialOrdering->getBasis(var->ID(), subsubcellConstraints->sideOrdinal);
          
          RefinementPatternPtr noRefinementPattern = RefinementPattern::noRefinementPattern(constrainingBasis->domainTopology());
          RefinementBranch noRefinements = {{noRefinementPattern.get(),0}};
          
          SubBasisReconciliationWeights coarseWeightPermutation = _br.constrainedWeights(subsubcdim, constrainingBasis,
                                                                                         sscOrdInOriginalConstrainingSide,
                                                                                         noRefinements, sscConstrainingBasis,
                                                                                         sscOrdInNewConstrainingSide,
                                                                                         subsubcellPermutation);
          
          //            SubBasisReconciliationWeights expectedComposition; // DEBUGGING
          //            { //DEBUGGING
          //              if (coarseWeightPermutation.isIdentity || weightsForSubSubcell.isIdentity)
          //              {
          //                SubBasisReconciliationWeights coarseWeightPermutationManual = coarseWeightPermutation.isIdentity ? weightsManualIdentity(coarseWeightPermutation) : coarseWeightPermutation;
          //                SubBasisReconciliationWeights weightsForSubSubcellManual = weightsForSubSubcell.isIdentity ? weightsManualIdentity(weightsForSubSubcell) : weightsForSubSubcell;
          //
          //                expectedComposition = BasisReconciliation::composedSubBasisReconciliationWeights(weightsForSubSubcellManual, coarseWeightPermutationManual);
          //              }
          //            }
          
          //              cout << "weightsForSubSubcell.weights, before applying permutation:\n" << weightsForSubSubcell.weights;
          //              cout << "coarseWeightPermutation.weights:\n" << coarseWeightPermutation.weights;
          
          weightsForSubSubcell = BasisReconciliation::composedSubBasisReconciliationWeights(weightsForSubSubcell, coarseWeightPermutation);
          
          //            { // DEBUGGING
          //              if (coarseWeightPermutation.isIdentity || weightsForSubSubcell.isIdentity)
          //              {
          //                bool equal = BasisReconciliation::equalWeights(expectedComposition, weightsForSubSubcell);
          //                TEUCHOS_TEST_FOR_EXCEPTION(!equal, std::invalid_argument, "Something wrong with the identity map");
          //              }
          //            }
          
          //              cout << "weightsForSubSubcell.weights, after applying permutation:\n" << weightsForSubSubcell.weights;
          
          // copied and pasted from above.  Could refactor:
          if ((weightsForSubSubcell.coarseOrdinals.size() > 0) && (weightsForSubSubcell.fineOrdinals.size() > 0))
          {
            vector<GlobalIndexType> globalDofOrdinalsForSubcell = getGlobalDofOrdinalsForSubcell(subsubcellConstraints->cellID,
                                                                                                 var, subsubcdim,
                                                                                                 sscOrdInNewConstrainingCell);
            
            // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
            
            const vector<int>* constrainingBasisOrdinalsForSubcell = &sscConstrainingBasis->dofOrdinalsForSubcell(subsubcdim, sscOrdInNewConstrainingSide);
            if (globalDofOrdinalsForSubcell.size() != constrainingBasisOrdinalsForSubcell->size())
            {
              cout << "globalDofOrdinalsForSubcell.size() != constrainingBasisOrdinalsForSubcell->size()\n";
              print("globalDofOrdinalsForSubcell", globalDofOrdinalsForSubcell);
              print("constrainingBasisOrdinalsForSubcell", *constrainingBasisOrdinalsForSubcell);
              
              // repeat the call for debugging convenience
              globalDofOrdinalsForSubcell = getGlobalDofOrdinalsForSubcell(subcellConstraint->cellID, var, subsubcdim, sscOrdInOriginalConstrainingCell);
              
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofOrdinalsForSubcell.size() != constrainingBasisOrdinalsForSubcell->size()");
            }
            
            vector<GlobalIndexType> globalDofOrdinals;
            for (int i=0; i<constrainingBasisOrdinalsForSubcell->size(); i++)
            {
              int constrainingBasisOrdinal = (*constrainingBasisOrdinalsForSubcell)[i];
              if (weightsForSubSubcell.coarseOrdinals.find(constrainingBasisOrdinal) != weightsForSubSubcell.coarseOrdinals.end())
              {
                globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
              }
            }
            
            if (weightsForSubSubcell.coarseOrdinals.size() != globalDofOrdinals.size())
            {
              cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
              Camellia::print("coarseOrdinals", weightsForSubSubcell.coarseOrdinals);
              Camellia::print("globalDofOrdinals", globalDofOrdinals);
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
            }
            
            subBasisMap.weights = weightsForSubSubcell.weights;
            subBasisMap.globalDofOrdinals = globalDofOrdinals;
            subBasisMap.basisDofOrdinals = weightsForSubSubcell.fineOrdinals;
            subBasisMap.isIdentity = weightsForSubSubcell.isIdentity;
            subBasisMaps.push_back(subBasisMap);
            
            {
              // DEBUGGING
              if ((DEBUG_CELL_ID == cellID) && (DEBUG_SIDE_ORDINAL == sideOrdinal) && (DEBUG_VAR_ID == var->ID()))
              {
                bool containsDebugDofIndex = false;
                for (GlobalIndexType globalDofIndex : globalDofOrdinals) {
                  if (DEBUG_GLOBAL_DOF == globalDofIndex)
                  {
                    containsDebugDofIndex = true;
                    break;
                  }
                }
                if (containsDebugDofIndex)
                {
                  cout << "While processing cellID " << cellID << ", side " << sideOrdinal << ", ";
                  cout << "mapped global dof ordinal " << DEBUG_GLOBAL_DOF << " on ";
                  cout << CamelliaCellTools::entityTypeString(subsubcdim);
                  cout << " " << sscOrdInNewConstrainingCell << " of cell " << subsubcellConstraints->cellID << endl;
                }
              }
            }
          }
        }
        if (subcellIsGeometricallyConstrained)
        {
          // then by virtue of the 1-irregular refinement rule, we know that all sub-subcells have been appropriately treated
          for (int subsubcdim=minimumConstraintDimension; subsubcdim<d; subsubcdim++)
          {
            CellTopoPtr subcellTopo = sideTopo->getSubcell(d, subcord);
            int subsubcellCount = subcellTopo->getSubcellCount(subsubcdim);
            for (int subsubcellOrdinal = 0; subsubcellOrdinal < subsubcellCount; subsubcellOrdinal++)
            {
              int subsubcellOrdinalInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo, d, subcord, subsubcdim, subsubcellOrdinal);
              processedSubcells[subsubcdim][subsubcellOrdinalInSide] = true;
            }
          }
        }
        else
        {
          // then we will have appropriately treated only those subsubcells which are not geometrically constrained
          
          for (int subsubcdim=minimumConstraintDimension; subsubcdim<d; subsubcdim++)
          {
            CellTopoPtr subcellTopo = sideTopo->getSubcell(d, subcord);
            int subsubcellCount = subcellTopo->getSubcellCount(subsubcdim);
            for (int subsubcellOrdinal = 0; subsubcellOrdinal < subsubcellCount; subsubcellOrdinal++)
            {
              int subsubcellOrdinalInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo, d, subcord, subsubcdim, subsubcellOrdinal);
              int subsubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(cell->topology(), sideDim, sideOrdinal, subsubcdim, subsubcellOrdinalInSide);
              IndexType sscEntityIndex = cell->entityIndex(subsubcdim, subsubcellOrdinalInCell);
              
              AnnotatedEntity subsubcellConstraint = cellConstraints.subcellConstraints[subsubcdim][subsubcellOrdinalInCell];
              
              if (subsubcdim != subsubcellConstraint.dimension) // then there is definitely a geometric constraint
                continue;
              
              CellPtr constrainingCell = _meshTopology->getCell(subsubcellConstraint.cellID);
              unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim,
                                                                                               subsubcellConstraint.sideOrdinal,
                                                                                               subsubcellConstraint.dimension,
                                                                                               subsubcellConstraint.subcellOrdinal);
              IndexType sscConstrainingEntityIndex = constrainingCell->entityIndex(subsubcellConstraint.dimension, subcellOrdinalInConstrainingCell);
              
              if ((subsubcdim == subsubcellConstraint.dimension) && (sscEntityIndex == sscConstrainingEntityIndex))
              {
                // no geometric constraints
                processedSubcells[subsubcdim][subsubcellOrdinalInSide] = true;
              }
            }
          } 
        }
      }
      
    }
  }
  
  // now, we collect the local basis coefficients corresponding to each global ordinal
  // likely there is a more efficient way to do this, but for now this is our approach
  map< GlobalIndexType, map<int, double> > weightsForGlobalOrdinal;
  
  map< int, set<GlobalIndexType> > globalOrdinalsForFineOrdinal;
  
  for (vector<SubBasisMapInfo>::iterator subBasisIt = subBasisMaps.begin(); subBasisIt != subBasisMaps.end(); subBasisIt++)
  {
    subBasisMap = *subBasisIt;
    vector<GlobalIndexType> globalDofOrdinals = subBasisMap.globalDofOrdinals;
    set<int> basisDofOrdinals = subBasisMap.basisDofOrdinals;
    vector<int> basisDofOrdinalsVector(basisDofOrdinals.begin(),basisDofOrdinals.end());
    // weights are fine x coarse
    for (int j=0; j<subBasisMap.globalDofOrdinals.size(); j++)
    {
      GlobalIndexType globalDofOrdinal = globalDofOrdinals[j];
      map<int, double> fineOrdinalCoefficientsThusFar = weightsForGlobalOrdinal[globalDofOrdinal];
      for (int i=0; i<subBasisMap.basisDofOrdinals.size(); i++)
      {
        int fineOrdinal = basisDofOrdinalsVector[i];
        double coefficient;
        if (!subBasisMap.isIdentity)
          coefficient = subBasisMap.weights(i,j);
        else
          coefficient = (i == j) ? 1.0 : 0.0;
        if (coefficient != 0)
        {
          if (fineOrdinalCoefficientsThusFar.find(fineOrdinal) != fineOrdinalCoefficientsThusFar.end())
          {
            double tol = 1e-14;
            double previousCoefficient = fineOrdinalCoefficientsThusFar[fineOrdinal];
            if (abs(previousCoefficient-coefficient) > tol)
            {
              cout  << "ERROR: incompatible entries for fine ordinal " << fineOrdinal << " in representation of global ordinal " << globalDofOrdinal << endl;
              cout << "previousCoefficient = " << previousCoefficient << endl;
              cout << "coefficient = " << coefficient << endl;
              cout << "diff = " << abs(previousCoefficient - coefficient) << endl;
              cout << "Encountered the incompatible entry while processing variable " << var->name() << " on cell " << cellID << ", side " << sideOrdinal << endl;
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal Error: incompatible entries for fine ordinal " );
            }
          }
          fineOrdinalCoefficientsThusFar[fineOrdinal] = coefficient;
          globalOrdinalsForFineOrdinal[fineOrdinal].insert(globalDofOrdinal);
        }
      }
      weightsForGlobalOrdinal[globalDofOrdinal] = fineOrdinalCoefficientsThusFar;
      if (globalDofOrdinal == DEBUG_GLOBAL_DOF)
      {
        ostringstream os;
        os << "On cell " << cellID << ", side " << sideOrdinal << ", ";
        os << "var " << var->ID() << ", ";
        os << "weightsForGlobalOrdinal[" << DEBUG_GLOBAL_DOF << "]";
        
        Camellia::FieldContainer<double> dofCoords;
        trialOrdering->getDofCoords(_mesh->physicalCellNodesForCell(cellID),
                                    dofCoords, var->ID(), sideOrdinal);
        
        os << "(basis ordinal dof coords: ";
        for (auto fineOrdinalEntry : fineOrdinalCoefficientsThusFar)
        {
          int basisOrdinal = fineOrdinalEntry.first;
          os << basisOrdinal << ": (";
          
          for (int d=0; d<spaceDim-1; d++)
          {
            os << dofCoords(0,basisOrdinal,d) << ",";
          }
          os << dofCoords(0,basisOrdinal,spaceDim-1) << "), ";
        }
        os << ")";
        
        Camellia::print(os.str(), fineOrdinalCoefficientsThusFar);
      }
    }
  }
  
  // partition global ordinals according to which fine ordinals they interact with -- this is definitely not super-efficient
  set<GlobalIndexType> partitionedGlobalDofOrdinals;
  vector< set<GlobalIndexType> > globalDofOrdinalPartitions;
  vector< set<int> > fineOrdinalsForPartition;
  
  for (map< GlobalIndexType, map<int, double> >::iterator globalWeightsIt = weightsForGlobalOrdinal.begin();
       globalWeightsIt != weightsForGlobalOrdinal.end(); globalWeightsIt++)
  {
    GlobalIndexType globalOrdinal = globalWeightsIt->first;
    if (partitionedGlobalDofOrdinals.find(globalOrdinal) != partitionedGlobalDofOrdinals.end()) continue;
    
    set<GlobalIndexType> partition;
    partition.insert(globalOrdinal);
    
    set<int> fineOrdinals;
    
    set<GlobalIndexType> globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed;
    
    map<int, double> fineCoefficients = globalWeightsIt->second;
    for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
    {
      int fineOrdinal = coefficientIt->first;
      fineOrdinals.insert(fineOrdinal);
      set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
      partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
    }
    
    globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
    
    while (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.size() != partition.size())
    {
      for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
      {
        GlobalIndexType globalOrdinal = *globalOrdIt;
        if (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.find(globalOrdinal) !=
            globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.end()) continue;
        map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
        for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
        {
          int fineOrdinal = coefficientIt->first;
          fineOrdinals.insert(fineOrdinal);
          set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
          partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
        }
        
        globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
      }
    }
    
    for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++)
    {
      GlobalIndexType globalOrdinal = *globalOrdIt;
      partitionedGlobalDofOrdinals.insert(globalOrdinal);
    }
    globalDofOrdinalPartitions.push_back(partition);
    fineOrdinalsForPartition.push_back(fineOrdinals);
  }
  
  BasisMap varSideMap;
  for (int i=0; i<globalDofOrdinalPartitions.size(); i++)
  {
    set<GlobalIndexType> partition = globalDofOrdinalPartitions[i];
    set<int> fineOrdinals = fineOrdinalsForPartition[i];
    vector<int> fineOrdinalsVector(fineOrdinals.begin(), fineOrdinals.end());
    map<int,int> fineOrdinalRowLookup;
    for (int i=0; i<fineOrdinalsVector.size(); i++)
    {
      fineOrdinalRowLookup[fineOrdinalsVector[i]] = i;
    }
    Camellia::FieldContainer<double> weights(fineOrdinals.size(),partition.size());
    int col=0;
    for (set<GlobalIndexType>::iterator globalDofIt=partition.begin(); globalDofIt != partition.end(); globalDofIt++)
    {
      GlobalIndexType globalOrdinal = *globalDofIt;
      map<int, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
      for (map<int, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++)
      {
        int fineOrdinal = coefficientIt->first;
        double coefficient = coefficientIt->second;
        int row = fineOrdinalRowLookup[fineOrdinal];
        weights(row,col) = coefficient;
      }
      col++;
    }
    vector<GlobalIndexType> globalOrdinals(partition.begin(),partition.end());
    SubBasisDofMapperPtr subBasisMap = SubBasisDofMapper::subBasisDofMapper(fineOrdinals, globalOrdinals, weights);
    varSideMap.push_back(subBasisMap);
    
    // DEBUGGING
    //    if ((cellID==4) && (sideOrdinal==3) && (var->ID() ==0)) {
    //      cout << "Adding subBasisDofMapper.  Details:\n";
    //      Camellia::print("fineOrdinals", fineOrdinals);
    //      Camellia::print("globalOrdinals", globalOrdinals);
    //      cout << "weights:\n" << weights;
    //    }
  }
  
  return varSideMap;
}

CellConstraints GDAMinimumRule::getCellConstraints(GlobalIndexType cellID)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getCellConstraints");
  if (_constraintsCache.find(cellID) == _constraintsCache.end())
  {
//    cout << "Getting cell constraints for cellID " << cellID << endl;
    typedef pair< IndexType, unsigned > CellPair;

    CellPtr cell = _meshTopology->getCell(cellID);
    TEUCHOS_TEST_FOR_EXCEPTION(cell->isParent(_meshTopology), std::invalid_argument, "Error: cell is not active (it is a parent) -- constraint tables are not valid for inactive cells…");
    std::vector< std::pair<std::string, int> > tags;
    {
      // check for tag filters
      DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
      auto trialIDs = trialOrdering->getVarIDs();
      for (const auto & trialID : trialIDs)
      {
        VarPtr var = this->_varFactory->trial(trialID);
        if (var->hasTag())
        {
          auto tag = var->getTag();
          tags.push_back(tag);
        }
      }
    }
    CellTopoPtr topo = cell->topology();
    unsigned spaceDim = topo->getDimension();
    unsigned sideDim = spaceDim - 1;

    vector< vector< bool > > processedSubcells(spaceDim+1); // we process dimensions from high to low -- since we deal here with the volume basis case, we allocate spaceDim + 1 entries
    vector< vector< AnnotatedEntity > > constrainingSubcellInfo(spaceDim + 1);
    
    Teuchos::RCP<CellConstraints> spatialSliceConstraints;

    AnnotatedEntity emptyConstraintInfo;
    emptyConstraintInfo.cellID = -1;

    const int d_min = minimumSubcellDimensionForContinuityEnforcement();
    for (int d=d_min; d<=spaceDim; d++)
    {
      int scCount = topo->getSubcellCount(d);
      processedSubcells[d] = vector<bool>(scCount, false);
      constrainingSubcellInfo[d] = vector< AnnotatedEntity >(scCount, emptyConstraintInfo);
    }

    for (int d=sideDim; d >= d_min; d--)
    {
      int scCount = topo->getSubcellCount(d);
      for (int subcord=0; subcord < scCount; subcord++)   // subcell ordinals in cell
      {
        if (! processedSubcells[d][subcord])   // i.e. we don't yet know the constraining entity for this subcell
        {
          IndexType entityIndex = cell->entityIndex(d, subcord);

//          cout << "entity of dimension " << d << " with entity index " << entityIndex << ": " << endl;
//          _meshTopology->printEntityVertices(d, entityIndex);
          pair<IndexType,unsigned> constrainingEntity = _meshTopology->getConstrainingEntity(d, entityIndex); // the constraining entity of the same dimension as this entity

          unsigned constrainingEntityDimension = constrainingEntity.second;
          IndexType constrainingEntityIndex = constrainingEntity.first;

//          cout << "is constrained by entity of dimension " << constrainingEntity.second << "  with entity index " << constrainingEntity.first << ": " << endl;
//          _meshTopology->printEntityVertices(constrainingEntity.second, constrainingEntity.first);

          // does the constraining entity match one (or more) of the tags?  Then we will require that the constraining cell/side match, too.
          CellPair constrainingCellPair = cellContainingEntityWithLeastH1Order(constrainingEntityDimension, constrainingEntityIndex, tags);
          constrainingSubcellInfo[d][subcord].cellID = constrainingCellPair.first;

          unsigned constrainingCellID = constrainingCellPair.first;
          unsigned constrainingSideOrdinal = constrainingCellPair.second;

          CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);
          CellTopoPtr constrainingCellTopo = constrainingCell->topology();
          if (constrainingCellTopo->getTensorialDegree() > 0)
          {
            // presumption is space-time.  In this case, we prefer spatial sides when we have a choice
            // (because some trace/flux variables are only defined on spatial sides).  But if tags are specified, we
            // do *not* allow this because the new side might not match the tag; the current one is guaranteed to.
            if (constrainingEntityDimension < sideDim) // if constrainingEntityDimension == sideDim, then the constraining entity is a side, so there isn't another side on constrainingCell that contains the constraining entity
            {
              if (! constrainingCellTopo->sideIsSpatial(constrainingSideOrdinal))
              {
                // if constraining side matches some tags, then we require that any replacement side matches those, too.
                std::vector< std::pair<std::string, int> > tagsToMatch;
                for (const auto & tag : tags)
                {
                  auto oldSideEntityIndex = constrainingCell->entityIndex(sideDim, constrainingSideOrdinal);
                  const bool matchesTag = _meshTopology->entityMatchesTag(sideDim, oldSideEntityIndex, tag);
                  if (matchesTag)
                  {
                    tagsToMatch.push_back(tag);
                  }
                }
                
                for (unsigned newSideOrdinal=0; newSideOrdinal<constrainingCellTopo->getSideCount(); newSideOrdinal++)
                {
                  if (constrainingCellTopo->sideIsSpatial(newSideOrdinal))
                  {
                    // Does this spatial side contain the constraining entity?
                    unsigned subcellOrdinalInSide = constrainingCell->findSubcellOrdinalInSide(constrainingEntityDimension, constrainingEntityIndex, newSideOrdinal);
                    
                    if (subcellOrdinalInSide == INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
                    {
                      // this side does not contain the constraining entity
                      continue;
                    }
                    
                    bool newSideOrdinalFailsTagCheck = false;
                    for (const auto & tag : tagsToMatch)
                    {
                      auto newSideEntityIndex = constrainingCell->entityIndex(sideDim, newSideOrdinal);
                      const bool matchesTag = _meshTopology->entityMatchesTag(sideDim, newSideEntityIndex, tag);
                      if (!matchesTag)
                      {
                        newSideOrdinalFailsTagCheck = true;
                        break;
                      }
                    }
                    if (newSideOrdinalFailsTagCheck)
                    {
                      continue; // try next newSideOrdinal.
                    }
                    // if we get here, the side "newSideOrdinal" contains the constraining entity, and passes the tag check
                    constrainingSideOrdinal = newSideOrdinal;
                    break;
                  }
                }
              }
            }
            
            if (!constrainingCellTopo->sideIsSpatial(constrainingSideOrdinal) && _hasSpaceOnlyTrialVariable && (d < sideDim))
            {
              // then we need to record this subcell in the spatialSliceConstraints container
              if (spatialSliceConstraints == Teuchos::null)
              {
                spatialSliceConstraints = Teuchos::rcp( new CellConstraints );
                
                spatialSliceConstraints->subcellConstraints = vector< vector< AnnotatedEntity > >(spaceDim + 1);
                
                for (int d=0; d<=spaceDim; d++)
                {
                  int scCount = topo->getSubcellCount(d);
                  spatialSliceConstraints->subcellConstraints[d] = vector< AnnotatedEntity >(scCount, emptyConstraintInfo);
                }
                spatialSliceConstraints->owningCellIDForSubcell = vector< vector< OwnershipInfo > >(spaceDim+1);
              }
              
              if (spatialSliceConstraints->owningCellIDForSubcell[d].size() == 0)
              {
                spatialSliceConstraints->owningCellIDForSubcell[d].resize(scCount);
              }
              
              // here, the "constraining entity" will be the entity itself
              // (requires 1-irregular mesh)
              
              CellPair owningCellAndSide = cellContainingEntityWithLeastH1Order(d, entityIndex);
              auto owningCellID = owningCellAndSide.first;
              if (_meshTopology->isValidCellIndex(owningCellID))
              {
                CellPtr owningCell = _meshTopology->getCell(owningCellID);
                if (owningCell->isParent(_meshTopology))
                {
                  cout << "Internal GDAMinimumRule Error: selected owning cell is a parent (not active!) -- owners must be active cells.\n";
                  cout << "cell " << owningCellID << " is a parent.\n";
                  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "inactive owning cell!");
                }
                spatialSliceConstraints->owningCellIDForSubcell[d][subcord].cellID = owningCellID;
                unsigned owningSubcellOrdinal = owningCell->findSubcellOrdinal(d, entityIndex);
                spatialSliceConstraints->owningCellIDForSubcell[d][subcord].owningSubcellOrdinal = owningSubcellOrdinal;
              }
              else
              {
                cout << "GDAMinimumRule::getCellConstraints(): owning cellID " << owningCellID << " is not valid.\n";
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "owning cellID is not valid!");
              }
              
              CellPtr owningCell = _meshTopology->getCell(owningCellID);
              CellTopoPtr owningCellTopo = owningCell->topology();
              
              bool found = false;
              for (unsigned owningSideOrdinal=0; owningSideOrdinal<owningCellTopo->getSideCount(); owningSideOrdinal++)
              {
                if (owningCellTopo->sideIsSpatial(owningSideOrdinal))
                {
                  // spatial side.  Does it contain the entity?
                  unsigned subcellOrdinalInSide = owningCell->findSubcellOrdinalInSide(d, entityIndex, owningSideOrdinal);
                  if (subcellOrdinalInSide != INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL)
                  {
                    found = true;
                    
                    spatialSliceConstraints->subcellConstraints[d][subcord].cellID = owningCellID;
                    spatialSliceConstraints->subcellConstraints[d][subcord].subcellOrdinal = subcellOrdinalInSide;
                    spatialSliceConstraints->subcellConstraints[d][subcord].sideOrdinal = owningSideOrdinal;
                    spatialSliceConstraints->subcellConstraints[d][subcord].dimension = d;
                    break;
                  }
                }
              }
              
              TEUCHOS_TEST_FOR_EXCEPTION(!found, std::invalid_argument, "during space-time spatial slice handling, subcell not found in owning cell");
            }
          }

          constrainingSubcellInfo[d][subcord].sideOrdinal = constrainingSideOrdinal;
          unsigned subcellOrdinalInConstrainingSide = constrainingCell->findSubcellOrdinalInSide(constrainingEntityDimension, constrainingEntityIndex, constrainingSideOrdinal);

          constrainingSubcellInfo[d][subcord].subcellOrdinal = subcellOrdinalInConstrainingSide;
          constrainingSubcellInfo[d][subcord].dimension = constrainingEntityDimension;

          if (constrainingEntityDimension < d)
          {
            cout << "Internal error: constrainingEntityDimension < entity dimension.  This should not happen!\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constrainingEntityDimension < entity dimension.  This should not happen!");
          }

          if ((d==0) && (constrainingEntityDimension==0))
          {
            if (constrainingEntityIndex != entityIndex)
            {
              cout << "Internal error: one vertex constrained by another.  This should not happen!\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "one vertex constrained by another.  Shouldn't happen!");
            }
          }

//          cout << "constraining subcell of dimension " << d << ", ordinal " << subcord << " with:\n";
//          cout << "  cellID " << constrainingSubcellInfo[d][subcord].cellID;
//          cout << ", sideOrdinal " << constrainingSubcellInfo[d][subcord].sideOrdinal;
//          cout << ", subcellOrdinalInSide " << constrainingSubcellInfo[d][subcord].subcellOrdinalInSide;
//          cout << ", dimension " << constrainingSubcellInfo[d][subcord].dimension << endl;
        }
      }
    }

    // fill in constraining subcell info for the volume (namely, it constrains itself):
    constrainingSubcellInfo[spaceDim][0].cellID = cellID;
    constrainingSubcellInfo[spaceDim][0].subcellOrdinal = INVALID_CELLTOPOLOGY_SUBCELL_ORDINAL;
    constrainingSubcellInfo[spaceDim][0].sideOrdinal = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
    constrainingSubcellInfo[spaceDim][0].dimension = spaceDim;

    CellConstraints cellConstraints;
    cellConstraints.subcellConstraints = constrainingSubcellInfo;

    // determine subcell ownership from the perspective of the cell
    cellConstraints.owningCellIDForSubcell = vector< vector< OwnershipInfo > >(spaceDim+1);
    for (int d=d_min; d<spaceDim; d++)
    {
      vector<IndexType> scIndices = cell->getEntityIndices(d);
      unsigned scCount = scIndices.size();
      cellConstraints.owningCellIDForSubcell[d] = vector< OwnershipInfo >(scCount);
      for (int scOrdinal = 0; scOrdinal < scCount; scOrdinal++)
      {
        IndexType entityIndex = scIndices[scOrdinal];
        unsigned constrainingDimension = constrainingSubcellInfo[d][scOrdinal].dimension;
        IndexType constrainingEntityIndex;
        if (d==constrainingDimension)
        {
          constrainingEntityIndex = _meshTopology->getConstrainingEntity(d, entityIndex).first;
        }
        else
        {
          CellPtr constrainingCell = _meshTopology->getCell(constrainingSubcellInfo[d][scOrdinal].cellID);
          unsigned constrainingSubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(constrainingCell->topology(), sideDim, constrainingSubcellInfo[d][scOrdinal].sideOrdinal,
              constrainingDimension, constrainingSubcellInfo[d][scOrdinal].subcellOrdinal);
          constrainingEntityIndex = constrainingCell->entityIndex(constrainingDimension, constrainingSubcellOrdinalInCell);
        }
        CellPair owningCellAndSide = cellContainingEntityWithLeastH1Order(constrainingDimension, constrainingEntityIndex, tags);
        auto owningCellID = owningCellAndSide.first;
        if (_meshTopology->isValidCellIndex(owningCellID))
        {
          CellPtr owningCell = _meshTopology->getCell(owningCellID);
          if (owningCell->isParent(_meshTopology))
          {
            cout << "Internal GDAMinimumRule Error: selected owning cell is a parent (not active!) -- owners must be active cells.\n";
            cout << "cell " << owningCellID << " is a parent.\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "inactive owning cell!");
          }
          cellConstraints.owningCellIDForSubcell[d][scOrdinal].cellID = owningCellID;
          unsigned owningSubcellOrdinal = owningCell->findSubcellOrdinal(constrainingDimension, constrainingEntityIndex);
          cellConstraints.owningCellIDForSubcell[d][scOrdinal].owningSubcellOrdinal = owningSubcellOrdinal;
        }
        else
        {
          cout << "GDAMinimumRule::getCellConstraints(): owning cellID " << owningCellID << " is not valid.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "owning cellID is not valid!");
        }
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].dimension = constrainingDimension;
      }
    }
    // cell owns (and constrains) its interior:
    cellConstraints.owningCellIDForSubcell[spaceDim] = vector< OwnershipInfo >(1);
    cellConstraints.owningCellIDForSubcell[spaceDim][0].cellID = cellID;
    cellConstraints.owningCellIDForSubcell[spaceDim][0].owningSubcellOrdinal = 0;
    cellConstraints.owningCellIDForSubcell[spaceDim][0].dimension = spaceDim;

    cellConstraints.spatialSliceConstraints = spatialSliceConstraints;
    
    /* 5-28-15:
     
     Something like the idea (or at least the goal) of sideSubcellConstraintEnforcedBySuper is good, but
     I need to get mathematically and geometrically precise on what the domain of enforcement should be,
     and the commented-out code below does not suffice to get us to uniqueness of that domain.
     
     It *might* be something like: take the AnnotatedEntity that constrains a given subcell; local degrees of freedom
     belonging to that subcell should be eliminated from any "enforcement path" that does not begin with that AnnotatedEntity.
     
     */
    /* Finally, build sideSubcellConstraintEnforcedBySuper.
     The point of this is to guarantee that during getBasisMap, as we split off lower-dimensioned subcells,
     we never enforce constraints on anything less than the intersection of the fine and coarse domains.  The
     rule is, if the side that constrains the subcell is the same as the sub-subcell, then the sub-subcell
     constraint is taken care of, and we can mark "true" for that sub-subcell.
     
     */
//    int sideCount = topo->getSideCount();
//    cellConstraints.sideSubcellConstraintEnforcedBySuper = vector< vector< vector<bool> > >(sideCount);
//    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
//    {
//      cellConstraints.sideSubcellConstraintEnforcedBySuper[sideOrdinal] = vector< vector<bool> >(sideDim+1);
//      CellTopoPtr sideTopo = topo->getSide(sideOrdinal);
//      
//      // initialize all entries to false:
//      for (int scdim=0; scdim<=sideDim; scdim++)
//      {
//        int subcellCount = sideTopo->getSubcellCount(scdim);
//        cellConstraints.sideSubcellConstraintEnforcedBySuper[sideOrdinal][scdim] = vector<bool>(subcellCount,false);
//      }
//      
//      for (int scdim=sideDim; scdim>0; scdim--)
//      {
//        int subcellCount = sideTopo->getSubcellCount(scdim);
//        for (int scord=0; scord<subcellCount; scord++) // ordinal in side
//        {
////          { // DEBUGGING:
////            if ((sideOrdinal==5) && (cellID==18) && (scdim==1) && (scord==0))
////            {
////              cout << "(sideOrdinal==5) && (cellID==18) && (scdim==1) && (scord==0).\n"; //breakpoint here
////            }
////          }
//          
//          int scordInCell = CamelliaCellTools::subcellOrdinalMap(topo, sideDim, sideOrdinal, scdim, scord);
//          GlobalIndexType constrainingCellID = cellConstraints.subcellConstraints[scdim][scordInCell].cellID;
//          unsigned constrainingSideOrdinal = cellConstraints.subcellConstraints[scdim][scordInCell].sideOrdinal;
//          CellTopoPtr subcell = sideTopo->getSubcell(scdim, scord);
//          for (int sscdim=scdim-1; sscdim >= 0; sscdim--)
//          {
//            int subsubcellCount = subcell->getSubcellCount(sscdim);
//            for (int sscord=0; sscord<subsubcellCount; sscord++) // ordinal in subcell
//            {
//              int sscordInSide = CamelliaCellTools::subcellOrdinalMap(sideTopo, scdim, scord, sscdim, sscord);
//              int sscordInCell = CamelliaCellTools::subcellOrdinalMap(topo, sideDim, sideOrdinal, sscdim, sscordInSide);
//              GlobalIndexType sscConstrainingCellID = cellConstraints.subcellConstraints[sscdim][sscordInCell].cellID;
//              GlobalIndexType sscConstrainingSideOrdinal = cellConstraints.subcellConstraints[sscdim][sscordInCell].sideOrdinal;
//              if ((sscConstrainingCellID == constrainingCellID) && (constrainingSideOrdinal == sscConstrainingSideOrdinal))
//              {
//                cellConstraints.sideSubcellConstraintEnforcedBySuper[sideOrdinal][sscdim][sscordInSide] = true;
//              }
//            }
//          }
//        }
//      }
//    }
    
    _constraintsCache[cellID] = cellConstraints;

//    if (cellID==4) { // DEBUGGING
//      printConstraintInfo(cellID);
//    }
  }

  return _constraintsCache[cellID];
}

AnnotatedEntity* GDAMinimumRule::getConstrainingEntityInfo(GlobalIndexType cellID, CellConstraints &cellConstraints,
                                                           VarPtr var, int d, int scord)
{
  AnnotatedEntity* constrainingInfo;
  OwnershipInfo* ownershipInfo;
  bool spaceOnlyConstraint;
  getConstrainingEntityInfo(cellID, cellConstraints, var, d, scord, constrainingInfo, ownershipInfo, spaceOnlyConstraint);
  return constrainingInfo;
}

void GDAMinimumRule::getConstrainingEntityInfo(GlobalIndexType cellID, CellConstraints &cellConstraints,
                                               VarPtr var, int d, int scord,
                                               AnnotatedEntity* &constrainingInfo, OwnershipInfo* &ownershipInfo,
                                               bool &spaceOnlyConstraint)
{
  if (var->hasTag())
  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Need to do something to get cell constraints for a tag-filtered variable");
//    // the problem we are running into right now is that the side selected as a constraining side for a tag-filtered trace variable is one that
    // is not actually contained in the tag (though it contains some subcells that do).
    // probably what we want to do is actually to do something like getCellConstraints() but where it asks for the cells with least H1Order that contain the entity (which might be a vertex which is contained in higher-dimensional sides), we want somehow to make sure that a side selected for the constraint actually has a basis defined for the variable.  I will note that at the point where we call getCellConstraints(), we have a VarPtr, just as we do here.  I kind of think *this* point is too late, and that we should have done something at the point we called getCellConstraints(): check if hasTag() is true, and if so, we circumscribe the search to only include cells that have at least a side that matches, and then whatever we do that specifies (in cellConstraints.subcellConstraints[d][scord].sideOrdinal) which *side* owns the entity, that should also require that said side has the tag applied to it.
    
//    constrainingInfo = &cellConstraints.subcellConstraints[d][scord];
//    ownershipInfo = &cellConstraints.owningCellIDForSubcell[d][scord];
//    
//    DofOrderingPtr sscConstrainingTrialOrdering = elementType(constrainingInfo->cellID)->trialOrderPtr;
//    if (sscConstrainingTrialOrdering)
//    
//    BasisPtr sscConstrainingBasis = sscConstrainingTrialOrdering->getBasis(var->ID(), subsubcellConstraints->sideOrdinal);
//    
//    
  }
  if (var->isDefinedOnTemporalInterface())
  {
    constrainingInfo = &cellConstraints.subcellConstraints[d][scord];
    ownershipInfo = &cellConstraints.owningCellIDForSubcell[d][scord];
    spaceOnlyConstraint = false;
  }
  else
  {
    unsigned constrainingSideOrdinal = cellConstraints.subcellConstraints[d][scord].sideOrdinal;
    CellPtr cell = _meshTopology->getCell(cellID);
    if (cell->topology()->sideIsSpatial(constrainingSideOrdinal))
    {
      constrainingInfo = &cellConstraints.subcellConstraints[d][scord];
      ownershipInfo = &cellConstraints.owningCellIDForSubcell[d][scord];
      spaceOnlyConstraint = false;
    }
    else
    {
      int sideDim = _meshTopology->getDimension() - 1;
      if (cellConstraints.spatialSliceConstraints != Teuchos::null)
      {
        constrainingInfo = &cellConstraints.spatialSliceConstraints->subcellConstraints[d][scord];
        // spatialSliceConstraints only sets up ownership info for d < sideDim -- maybe it should be otherwise.  For now, we just set to NULL if d >= sideDim
        if (d < sideDim)
          ownershipInfo = &cellConstraints.spatialSliceConstraints->owningCellIDForSubcell[d][scord];
        else
          ownershipInfo = NULL;
        spaceOnlyConstraint = true;
      }
      else
      {
        constrainingInfo = NULL;
        ownershipInfo = NULL;
        spaceOnlyConstraint = true;
      }
    }
  }
}

// ! returns the permutation that goes from the ancestor of the indicated cell's view of the subcell to the constraining cell's view.
unsigned GDAMinimumRule::getConstraintPermutation(GlobalIndexType cellID, unsigned subcdim, unsigned subcord)
{
  CellPtr cell = _meshTopology->getCell(cellID);
  CellPtr ancestralCell = cell->ancestralCellForSubcell(subcdim, subcord, _meshTopology);

  pair<unsigned, unsigned> ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(subcdim, subcord, _meshTopology);

  unsigned ancestralSubcellOrdinal = ancestralSubcell.first;
  unsigned ancestralSubcellDimension = ancestralSubcell.second;
  unsigned ancestralPermutation = ancestralCell->subcellPermutation(ancestralSubcellDimension, ancestralSubcellOrdinal);

  CellConstraints cellConstraints = getCellConstraints(cellID);
  GlobalIndexType constrainingCellID = cellConstraints.subcellConstraints[subcdim][subcord].cellID;
  GlobalIndexType constrainingSubcellOrdinal = cellConstraints.subcellConstraints[subcdim][subcord].subcellOrdinal;
  GlobalIndexType constrainingSubcellDimension = cellConstraints.subcellConstraints[subcdim][subcord].dimension;

  CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);
  unsigned constrainingCellPermutation = constrainingCell->subcellPermutation(constrainingSubcellOrdinal, constrainingSubcellDimension);

  // ancestral and constrainingCell permutations are from canonical.  We want to go from ancestor to constraining:
  CellTopoPtr ancestralSubcellTopo = ancestralCell->topology()->getSubcell(ancestralSubcellDimension, ancestralSubcellOrdinal);
  unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(ancestralSubcellTopo, ancestralPermutation);
  unsigned ancestralToConstraint = CamelliaCellTools::permutationComposition(ancestralSubcellTopo, ancestralPermutationInverse, constrainingCellPermutation);

  return ancestralToConstraint;
}

// ! returns the permutation that goes from the ancestor of the indicated side's view of the subcell to the constraining cell's view.
unsigned GDAMinimumRule::getConstraintPermutation(GlobalIndexType cellID, unsigned sideOrdinal, unsigned subcdim, unsigned subcord)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Method not implemented");
}

set<GlobalIndexType> GDAMinimumRule::getFittableGlobalDofIndices(GlobalIndexType cellID, int sideOrdinal, int varID)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getFittableGlobalDofIndices");
  
  // returns the global dof indices for basis functions which have support on the given side.  This is determined by taking the union of the global dof indices defined on all the constraining sides for the given side (the constraining sides are by definition unconstrained).
  
  int sideDim = _meshTopology->getDimension() - 1;
  const int d_min = minimumSubcellDimensionForContinuityEnforcement();
  if (d_min > sideDim)
  {
    // no dofs on sides; return empty set
    return set<GlobalIndexType>();
  }
  
  pair<GlobalIndexType,pair<int,unsigned>> key = {cellID,{varID,sideOrdinal}};
  if (_fittableGlobalIndicesCache.find(key) != _fittableGlobalIndicesCache.end())
  {
    return _fittableGlobalIndicesCache[key];
  }
  
  SubcellDofIndices subcellDofIndices = getGlobalDofIndices(cellID);
  CellPtr cell = _meshTopology->getCell(cellID);
  

  CellConstraints constraints = getCellConstraints(cellID);
  GlobalIndexType constrainingCellID = constraints.subcellConstraints[sideDim][sideOrdinal].cellID;
  unsigned constrainingCellSideOrdinal = constraints.subcellConstraints[sideDim][sideOrdinal].sideOrdinal;

  CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);

  SubcellDofIndices constrainingCellDofIndexInfo = getGlobalDofIndices(constrainingCellID);

  set<GlobalIndexType> fittableDofIndices;

  /*
   Exactly *which* of the constraining side's dofs to include is a little tricky.  We want the ones interior to the side for sure (since these are
   not further constrained), and definitely want to exclude e.g. edge dofs that are further constrained by other edges.  But then it turns out we
   also want to exclude the vertices that are included in those constrained edges.  The general rule seems to be: if a subcell is excluded, then
   all its constituent subcells should also be excluded.
   */

  // iterate through all the subcells of the constraining side, collecting dof indices
  CellTopoPtr constrainingCellTopo = constrainingCell->topology();
  CellTopoPtr constrainingSideTopo = constrainingCellTopo->getSubcell(sideDim, constrainingCellSideOrdinal);
  set<unsigned> constrainingCellConstrainedNodes; // vertices in the constraining cell that belong to subcell entities that are constrained by other cells
  for (int d=sideDim; d>=0; d--)
  {
    int scCount = constrainingSideTopo->getSubcellCount(d);
    for (int scOrdinal=0; scOrdinal<scCount; scOrdinal++)
    {
      int scordInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(constrainingCellTopo, sideDim, constrainingCellSideOrdinal, d, scOrdinal);
      set<unsigned> scNodesInConstrainingCell;
      bool nodeNotKnownToBeConstrainedFound = false;
      if (d==0)
      {
        scNodesInConstrainingCell.insert(scordInConstrainingCell);
        if (constrainingCellConstrainedNodes.find(scordInConstrainingCell) == constrainingCellConstrainedNodes.end())
        {
          nodeNotKnownToBeConstrainedFound = true;
        }
      }
      else
      {
        int nodeCount = constrainingCellTopo->getNodeCount(d, scordInConstrainingCell);
        for (int nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++)
        {
          unsigned nodeInConstrainingCell = constrainingCellTopo->getNodeMap(d, scordInConstrainingCell, nodeOrdinal);
          scNodesInConstrainingCell.insert(nodeInConstrainingCell);
          if (constrainingCellConstrainedNodes.find(nodeInConstrainingCell) == constrainingCellConstrainedNodes.end())
          {
            nodeNotKnownToBeConstrainedFound = true;
          }
        }
      }

      if (!nodeNotKnownToBeConstrainedFound) continue; // all nodes constrained, so we can skip further processing

      IndexType subcellIndex = constrainingCell->entityIndex(d, scordInConstrainingCell);
      pair<IndexType,unsigned> constrainingSubcell = _meshTopology->getConstrainingEntity(d, subcellIndex);
      if ((constrainingSubcell.second == d) && (constrainingSubcell.first == subcellIndex))   // i.e. the subcell is not further constrained.
      {
        vector<GlobalIndexType> dofIndicesInConstrainingSide = constrainingCellDofIndexInfo.dofIndicesForVarOnSubcell(d,scordInConstrainingCell,varID);
        fittableDofIndices.insert(dofIndicesInConstrainingSide.begin(), dofIndicesInConstrainingSide.end());
      }
      else
      {
        constrainingCellConstrainedNodes.insert(scNodesInConstrainingCell.begin(),scNodesInConstrainingCell.end());
      }
    }
  }
  _fittableGlobalIndicesCache[key] = fittableDofIndices;
  return fittableDofIndices;
}

SubcellDofIndices & GDAMinimumRule::getOwnedGlobalDofIndices(GlobalIndexType cellID)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getOwnedGlobalDofIndices");
  if (_ownedGlobalDofIndicesCache.find(cellID) != _ownedGlobalDofIndicesCache.end())
  {
    return _ownedGlobalDofIndicesCache[cellID];
  }
  CellConstraints constraints = getCellConstraints(cellID);
  
  TEUCHOS_TEST_FOR_EXCEPTION(!_meshTopology->isValidCellIndex(cellID), std::invalid_argument, "Invalid cellID");
  
  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = cell->topology();
  
  bool cellIsActive = !cell->isParent(_meshTopology);
  if (!cellIsActive)
  {
    // then the cell will not own any global dof indices
    SubcellDofIndices scDofIndices(topo);
    _ownedGlobalDofIndicesCache[cellID] = scDofIndices;
    return _ownedGlobalDofIndicesCache[cellID];
  }
  
  int rank = _partitionPolicy->Comm()->getRank();
  //  cout << "GDAMinimumRule: Rebuilding lookups on rank " << rank << endl;
  set<GlobalIndexType>* myCellIDs = &_partitions[rank];
  
  if (myCellIDs->find(cellID) == myCellIDs->end())
  {
    cout << "getOwnedGlobalDofIndices() called for non-locally-owned active cellID " << cellID << " on rank " << rank << endl;
    print("myCellIDs", *myCellIDs);
    _meshTopology->printAllEntitiesInBaseMeshTopology();
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getOwnedGlobalDofIndices() called for non-local cellID");
  }
  
  int spaceDim = _meshTopology->getDimension();
  int sideDim = spaceDim - 1;

  SubcellDofIndices scInfo(topo);

  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  const map<int, VarPtr>* trialVars = &_varFactory->trialVars();

//  cout << "Owned global dof indices for cell " << cellID << endl;

  GlobalIndexType globalDofIndex = _globalCellDofOffsets[cellID]; // this cell's first globalDofIndex

  map< pair<unsigned,IndexType>, pair<unsigned, unsigned> > entitiesClaimed; // maps from the constraining entity claimed to the (d, scord) entry that claimed it.
  map< pair<unsigned,IndexType>, pair<unsigned, unsigned> > entitiesClaimedSpaceOnlyVariables; // for variables with var->isDefinedOnTemporalInterface()==false
  
  const int d_min = minimumSubcellDimensionForContinuityEnforcement();
  for (int d=d_min; d<=spaceDim; d++)
  {
    int scCount = topo->getSubcellCount(d);
    for (int scord=0; scord<scCount; scord++)
    {
      pair<unsigned, IndexType> owningSubcellEntity = {-1,-1};
      pair<unsigned, IndexType> owningSubcellEntitySpaceOnly = {-1,-1};

      for (auto varEntry : *trialVars)
      {
        VarPtr var = varEntry.second;
        unsigned scordForBasis;
        bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
        
        AnnotatedEntity* constrainingEntityInfo;
        OwnershipInfo* ownershipInfo;
        bool spaceOnlyConstraint;
        
        getConstrainingEntityInfo(cellID, constraints, var, d, scord, constrainingEntityInfo, ownershipInfo, spaceOnlyConstraint);
        
        map< pair<unsigned,IndexType>, pair<unsigned, unsigned> >* entitiesClaimedForVariable;
        if (spaceOnlyConstraint)
        {
          if (d >= sideDim) continue; // the subcell is itself a temporal side (or a volume); no basis defined on this...
          entitiesClaimedForVariable = &entitiesClaimedSpaceOnlyVariables;
        }
        else
        {
          entitiesClaimedForVariable = &entitiesClaimed;
        }
        
        if (ownershipInfo->cellID == cellID)   // owned by this cell: count all the constraining dofs as entries for this cell
        {
          GlobalIndexType constrainingCellID = constrainingEntityInfo->cellID;
          unsigned constrainingDimension = constrainingEntityInfo->dimension;
          DofOrderingPtr trialOrdering = elementType(constrainingCellID)->trialOrderPtr;
          BasisPtr basis; // the constraining basis for the subcell
          if (varHasSupportOnVolume)
          {
            if (d==spaceDim)
            {
              // then there is only one subcell ordinal (and there will be -1's in sideOrdinal and subcellOrdinalInSide....
              scordForBasis = 0;
            }
            else
            {
              scordForBasis = CamelliaCellTools::subcellOrdinalMap(_meshTopology->getCell(constrainingCellID)->topology(), sideDim,
                              constrainingEntityInfo->sideOrdinal,
                              constrainingDimension, constrainingEntityInfo->subcellOrdinal);
            }
            basis = trialOrdering->getBasis(var->ID());
          }
          else
          {
            if (d==spaceDim) continue; // side bases don't have any support on the interior of the cell...
            if (! trialOrdering->hasBasisEntry(var->ID(), constrainingEntityInfo->sideOrdinal) ) continue;
            scordForBasis = constrainingEntityInfo->subcellOrdinal; // the basis sees the side, so that's the view to use for subcell ordinal
            basis = trialOrdering->getBasis(var->ID(), constrainingEntityInfo->sideOrdinal);
          }
          int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
          if (minimumConstraintDimension > d) continue; // then we don't enforce (or own) anything for this subcell/basis combination

          CellPtr owningCell = _meshTopology->getCell(ownershipInfo->cellID);
          IndexType owningSubcellEntityIndex = owningCell->entityIndex(ownershipInfo->dimension, ownershipInfo->owningSubcellOrdinal);
          pair<unsigned, IndexType> owningSubcellEntityForVariable = make_pair(ownershipInfo->dimension, owningSubcellEntityIndex);
          if (entitiesClaimedForVariable == &entitiesClaimed)
          {
            // sanity check: if this has previously been set, then make sure it's the same
            if (owningSubcellEntity != pair<unsigned,IndexType>{-1,-1})
            {
              TEUCHOS_TEST_FOR_EXCEPTION(owningSubcellEntity != owningSubcellEntityForVariable, std::invalid_argument, "owningSubcellEntry changed");
            }
            owningSubcellEntity = owningSubcellEntityForVariable;
          }
          else
          {
            // sanity check: if this has previously been set, then make sure it's the same
            if (owningSubcellEntitySpaceOnly != pair<unsigned,IndexType>{-1,-1})
            {
              TEUCHOS_TEST_FOR_EXCEPTION(owningSubcellEntitySpaceOnly != owningSubcellEntityForVariable, std::invalid_argument, "owningSubcellEntitySpaceOnly changed");
            }
            owningSubcellEntitySpaceOnly = owningSubcellEntityForVariable;
          }
          
          if (entitiesClaimedForVariable->find(owningSubcellEntity) != entitiesClaimedForVariable->end())
          {
            // already processed this guy on this cell: just copy
            pair<unsigned,unsigned> previousConstrainedSubcell = (*entitiesClaimedForVariable)[owningSubcellEntity];
            auto & varRange = scInfo.dofIndexRangeForVarOnSubcell(previousConstrainedSubcell.first, previousConstrainedSubcell.second, var->ID());
            scInfo.setDofIndicesForVarOnSubcell(d, scord, var->ID(), varRange);
            continue;
          }

          int dofOrdinalCount = basis->dofOrdinalsForSubcell(constrainingDimension, scordForBasis).size();
          scInfo.setDofIndicesForVarOnSubcell(d, scord, var->ID(), {globalDofIndex,dofOrdinalCount});
          
          globalDofIndex += dofOrdinalCount;
          
//          vector<GlobalIndexType> globalDofIndices;
//
//          for (int i=0; i<dofOrdinalCount; i++)
//          {
//            globalDofIndices.push_back(globalDofIndex++);
//          }
//          
//          if (scInfo.subcellDofIndices.size() < d+1)
//          {
//            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: scInfo vector not big enough");
//          }
//          if (dofOrdinalCount > 0)
//          {
//            scInfo.subcellDofIndices[d][scord][var->ID()] = globalDofIndices;
//          }
        }
      }
      
      if (owningSubcellEntity != pair<unsigned,IndexType>{-1,-1})
      {
        entitiesClaimed[owningSubcellEntity] = make_pair(d, scord);
      }
      if (owningSubcellEntitySpaceOnly != pair<unsigned,IndexType>{-1,-1})
      {
        entitiesClaimedSpaceOnlyVariables[owningSubcellEntitySpaceOnly] = make_pair(d, scord);
      }
    }
  }
  _ownedGlobalDofIndicesCache[cellID] = scInfo;
  return _ownedGlobalDofIndicesCache[cellID];
}

SubcellDofIndices& GDAMinimumRule::getGlobalDofIndices(GlobalIndexType cellID)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getGlobalDofIndices");
  if (_globalDofIndicesForCellCache.find(cellID) == _globalDofIndicesForCellCache.end())
  {
    const set<GlobalIndexType>* myCellIDs = &this->cellsInPartition(-1);
    if (myCellIDs->find(cellID) == myCellIDs->end())
    {
      cout << "getGlobalDofIndices() called for non-owned cellID " << cellID << endl;
      print("owned cellIDs: ", *myCellIDs);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getGlobalDofIndices() called for non-owned cellID");
    }
    
    /**************** ESTABLISH OWNERSHIP ****************/
    SubcellDofIndices subcellDofIndices = getOwnedGlobalDofIndices(cellID);
    CellConstraints constraints = getCellConstraints(cellID);
    
    CellPtr cell = _meshTopology->getCell(cellID);
    CellTopoPtr topo = cell->topology();
    
    int spaceDim = topo->getDimension();
    int d_min = minimumSubcellDimensionForContinuityEnforcement();
    
    // fill in the other global dof indices (the ones not owned by this cell):
    for (int d=d_min; d<=spaceDim; d++)
    {
      int scCount = topo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++)
      {
        GlobalIndexType owningCellID = constraints.owningCellIDForSubcell[d][scord].cellID;
        
        if (owningCellID != cellID)   // this one not yet filled in
        {
          OwnershipInfo owningCellInfo = constraints.owningCellIDForSubcell[d][scord];
          unsigned owningCellScord = owningCellInfo.owningSubcellOrdinal;
          SubcellDofIndices* owningCellSubcellDofIndices = &getOwnedGlobalDofIndices(owningCellID);
          subcellDofIndices.subcellDofIndices[d][scord] = owningCellSubcellDofIndices->subcellDofIndices[owningCellInfo.dimension][owningCellScord];
        }
      }
    }
    _globalDofIndicesForCellCache[cellID] = subcellDofIndices;
  }
  
  return _globalDofIndicesForCellCache[cellID];
}

//std::set<GlobalIndexType> GDAMinimumRule::getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal)
//{
//  return getFittableGlobalDofIndices(cellID, sideOrdinal, varID);
//}

vector<GlobalIndexType> GDAMinimumRule::globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID)
{
  map<int, VarPtr> trialVars = _varFactory->trialVars();
  VarPtr trialVar = trialVars[varID];
  
  EFunctionSpace fs = efsForSpace(trialVar->space());
  TEUCHOS_TEST_FOR_EXCEPTION(!functionSpaceIsDiscontinuous(fs), std::invalid_argument, "globalDofIndicesForFieldVariable() only supports discontinuous field variables right now");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVar->varType() != FIELD, std::invalid_argument, "globalDofIndicesForFieldVariable() requires a discontinuous field variable");
  
  SubcellDofIndices subcellDofIndices = getOwnedGlobalDofIndices(cellID);
  
  int spaceDim = _mesh->getTopology()->getDimension();
  vector<GlobalIndexType> globalIndices = subcellDofIndices.dofIndicesForVarOnSubcell(spaceDim,0,trialVar->ID());
  
  return globalIndices;
}

vector<GlobalIndexType> GDAMinimumRule::getGlobalDofOrdinalsForSubcell(GlobalIndexType cellID, VarPtr var, int d, int scord)
{
  SubcellDofIndices* globalDofIndicesForCell = &getGlobalDofIndices(cellID);
  
  {
    // If the passed-in cell is not the owner, then the ordering of the vector could be incorrect.
    // Therefore, we require that the passed-in cell *is* the owner.
    
    CellConstraints constraints = getCellConstraints(cellID);
    GlobalIndexType owningCellID = constraints.owningCellIDForSubcell[d][scord].cellID;
    TEUCHOS_TEST_FOR_EXCEPTION(owningCellID != cellID, std::invalid_argument, "owningCellID is not the same as cellID");
  }
  
//  vector<GlobalIndexType> globalDofOrdinalsForSubcell = globalDofIndicesForCell->subcellDofIndices[d][scord][var->ID()];
  return globalDofIndicesForCell->dofIndicesForVarOnSubcell(d,scord,var->ID());
}

LocalDofMapperPtr GDAMinimumRule::getDofMapper(GlobalIndexType cellID, int varIDToMap, int sideOrdinalToMap)
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::getDofMapper");
  if ((varIDToMap == -1) && (sideOrdinalToMap == -1))
  {
    // a mapper for the whole dof ordering: we cache these separately...
    if (_dofMapperCache.find(cellID) != _dofMapperCache.end())
    {
      return _dofMapperCache[cellID];
    }
  }
  else
  {
    map< GlobalIndexType, map<int, map<int, LocalDofMapperPtr> > >::iterator cellMapEntry = _dofMapperForVariableOnSideCache.find(cellID);
    if (cellMapEntry != _dofMapperForVariableOnSideCache.end())
    {
      map<int, map<int, LocalDofMapperPtr> >::iterator sideMapEntry = cellMapEntry->second.find(sideOrdinalToMap);
      if (sideMapEntry != cellMapEntry->second.end())
      {
        map<int, LocalDofMapperPtr>::iterator varMapEntry = sideMapEntry->second.find(varIDToMap);
        if (varMapEntry != sideMapEntry->second.end())
        {
          return varMapEntry->second;
        }
      }
    }
  }

  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = cell->topology();
  int sideCount = topo->getSideCount();
  int spaceDim = topo->getDimension();

  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  map<int, VarPtr> trialVars = _varFactory->trialVars();

  typedef vector< SubBasisDofMapperPtr > BasisMap;
  map< int, BasisMap > volumeMap; // keys are variable IDs
  vector< map< int, BasisMap > > sideMaps(sideCount);

  bool determineFittableGlobalDofOrdinalsOnSides = false;
  
  if (varIDToMap != -1)
  {
    // replace trialVars appropriately
    map<int, VarPtr> oneEntryTrialVars;
    oneEntryTrialVars[varIDToMap] = trialVars[varIDToMap];
    trialVars = oneEntryTrialVars;
  }
  
  for (auto varEntry : trialVars)
  {
    VarPtr var = varEntry.second;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
    if (varHasSupportOnVolume)
    {
      // we will only care about this variable on the sides if it's not discontinuous
      Camellia::EFunctionSpace fs = efsForSpace(var->space());
      if (!functionSpaceIsDiscontinuous(fs))
      {
        determineFittableGlobalDofOrdinalsOnSides = true;
        break;
      }
    }
    else
    {
      // there is a trace variable of interest; we should determine fittable global dof ordinals on sides
      determineFittableGlobalDofOrdinalsOnSides = true;
      break;
    }
  }
  
  vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(sideCount);
  
  if (determineFittableGlobalDofOrdinalsOnSides)
  {
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      if ((sideOrdinalToMap != -1) && (sideOrdinal != sideOrdinalToMap)) continue; // skip this side...
      fittableGlobalDofOrdinalsOnSides[sideOrdinal] = getFittableGlobalDofIndices(cellID, sideOrdinal);
  //    cout << "sideOrdinal " << sideOrdinal;
  //    Camellia::print(", fittableGlobalDofIndices", fittableGlobalDofOrdinalsOnSides[sideOrdinal]);
    }
  }

  set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;

  /**************** ESTABLISH OWNERSHIP ****************/
  SubcellDofIndices subcellDofIndices = getGlobalDofIndices(cellID);

  /**************** CREATE SUB-BASIS MAPPERS ****************/

  for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++)
  {
    VarPtr var = varIt->second;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);

    if (varHasSupportOnVolume)
    {
      if (sideOrdinalToMap == -1)
      {
        volumeMap[var->ID()] = getBasisMap(cellID, subcellDofIndices, var);
        // first, get interior dofs
        vector<GlobalIndexType> interiorDofIndices = subcellDofIndices.dofIndicesForVarOnSubcell(spaceDim,0,var->ID());
        fittableGlobalDofOrdinalsInVolume.insert(interiorDofIndices.begin(), interiorDofIndices.end());
        
        // now, sides
        for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
        {
          set<GlobalIndexType> fittableGlobalDofOrdinalsOnSide = getFittableGlobalDofIndices(cellID, sideOrdinal, var->ID());
          fittableGlobalDofOrdinalsInVolume.insert(fittableGlobalDofOrdinalsOnSide.begin(),fittableGlobalDofOrdinalsOnSide.end());
        }
      }
      else
      {
        if (functionSpaceIsDiscontinuous(efsForSpace(var->space())))
        {
          // then there is no chance of any minimum-rule constraints to impose
          // and we need to specially "extract" the side degrees of freedom
          // (this comes up when imposing BCs in a DG context)
          volumeMap[var->ID()] = getBasisMapDiscontinuousVolumeRestrictedToSide(cellID, subcellDofIndices, var, sideOrdinalToMap);
          for (auto subMap : volumeMap[var->ID()])
          {
            fittableGlobalDofOrdinalsInVolume.insert(subMap->mappedGlobalDofOrdinals().begin(),subMap->mappedGlobalDofOrdinals().end());
          }
        }
        else
        {
          // if the function space is not discontinuous, then we restrict the usual basis map to the global dof ordinals on the side
          DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
          BasisPtr basis = trialOrdering->getBasis(var->ID());
          set<int> basisDofOrdinalsForSide = basis->dofOrdinalsForSide(sideOrdinalToMap);
          
          BasisMap unrestrictedMap = getBasisMap(cellID, subcellDofIndices, var);
          volumeMap[var->ID()] = getRestrictedBasisMap(unrestrictedMap, basisDofOrdinalsForSide);
          set<GlobalIndexType> fittableGlobalDofOrdinalsOnSide = getFittableGlobalDofIndices(cellID, sideOrdinalToMap, var->ID());
          fittableGlobalDofOrdinalsInVolume.insert(fittableGlobalDofOrdinalsOnSide.begin(),fittableGlobalDofOrdinalsOnSide.end());
        }
      }
    }
    else
    {
      for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
      {
        if ((sideOrdinalToMap != -1) && (sideOrdinal != sideOrdinalToMap)) continue; // skip this side...
        if (! trialOrdering->hasBasisEntry(var->ID(), sideOrdinal)) continue; // skip this side/var combo...
        sideMaps[sideOrdinal][var->ID()] = getBasisMap(cellID, subcellDofIndices, var, sideOrdinal);
      }
    }
  }

  set<GlobalIndexType> emptyGlobalIDSet; // the "extra" guys to map
  LocalDofMapperPtr dofMapper = Teuchos::rcp( new LocalDofMapper(trialOrdering,volumeMap,fittableGlobalDofOrdinalsInVolume,
                                                                 sideMaps,fittableGlobalDofOrdinalsOnSides,emptyGlobalIDSet,
                                                                 varIDToMap,sideOrdinalToMap) );
  if ((varIDToMap == -1) && (sideOrdinalToMap == -1))
  {
    // a mapper for the whole dof ordering: we cache these...
    _dofMapperCache[cellID] = dofMapper;
    return _dofMapperCache[cellID];
  }
  else
  {
    _dofMapperForVariableOnSideCache[cellID][sideOrdinalToMap][varIDToMap] = dofMapper;
    return dofMapper;
  }
}

BasisMap GDAMinimumRule::getRestrictedBasisMap(BasisMap &basisMap, const set<int> &basisDofOrdinalRestriction) // restricts to part of the basis
{
  BasisMap newBasisMap;
  for (SubBasisDofMapperPtr subBasisMap : basisMap)
  {
    subBasisMap = subBasisMap->restrictDofOrdinalFilter(basisDofOrdinalRestriction);
    if (subBasisMap->basisDofOrdinalFilter().size() > 0)
    {
      newBasisMap.push_back(subBasisMap);
    }
  }
  return newBasisMap;
}

GlobalIndexType GDAMinimumRule::numPartitionOwnedGlobalFieldIndices()
{
  return _partitionFieldDofCount;
}

GlobalIndexType GDAMinimumRule::numPartitionOwnedGlobalFluxIndices()
{
  return _partitionFluxDofCount;
}

GlobalIndexType GDAMinimumRule::numPartitionOwnedGlobalTraceIndices()
{
  return _partitionTraceDofCount;
}

PartitionIndexType GDAMinimumRule::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex )
{
  // find the first rank whose offset is above globalDofIndex; the one prior to that will be the owner
  auto upperBoundIt = std::upper_bound(_partitionDofOffsets.begin(), _partitionDofOffsets.end(), globalDofIndex);
  int firstRankPastGlobalDofIndex = upperBoundIt - _partitionDofOffsets.begin();
  int partition = firstRankPastGlobalDofIndex - 1;

  // since the final entry in _partitionDofOffsets is for the global dof count (and does not correspond to any rank),
  // we should be at most one away from that
  TEUCHOS_TEST_FOR_EXCEPTION((partition < 0) || (partition >= _partitionDofOffsets.size() - 1), std::invalid_argument, "partition for globalDofIndex not found");
  // sanity check:
  {
    TEUCHOS_TEST_FOR_EXCEPTION(globalDofIndex < _partitionDofOffsets[partition], std::invalid_argument, "internal error: partition found for globalDofIndex does not contain it.");
    TEUCHOS_TEST_FOR_EXCEPTION(globalDofIndex >= _partitionDofOffsets[partition+1], std::invalid_argument, "internal error: partition found for globalDofIndex does not contain it.");
  }
  
  return partition;
//  PartitionIndexType numRanks = _partitionDofCounts.size();
//  GlobalIndexType totalDofCount = 0;
//  for (PartitionIndexType i=0; i<numRanks; i++)
//  {
//    totalDofCount += _partitionDofCounts(i);
//    if (totalDofCount > globalDofIndex)
//    {
//      return i;
//    }
//  }
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid globalDofIndex");
}

void GDAMinimumRule::printConstraintInfo(GlobalIndexType cellID)
{
  CellConstraints cellConstraints = getCellConstraints(cellID);
  cout << "***** Constraints for cell " << cellID << " ****** \n";
  CellPtr cell = _meshTopology->getCell(cellID);
  int spaceDim = cell->topology()->getDimension();
  const int d_min = minimumSubcellDimensionForContinuityEnforcement();
  for (int d=d_min; d<spaceDim; d++)
  {
    cout << "  dimension " << d << " subcells: " << endl;
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<subcellCount; scord++)
    {
      AnnotatedEntity constraintInfo = cellConstraints.subcellConstraints[d][scord];
      cout << "    ordinal " << scord  << " constrained by cell " << constraintInfo.cellID;
      cout << ", side " << constraintInfo.sideOrdinal << "'s dimension " << constraintInfo.dimension << " subcell ordinal " << constraintInfo.subcellOrdinal << endl;
    }
  }

  cout << "Ownership info:\n";
  for (int d=d_min; d<spaceDim; d++)
  {
    cout << "  dimension " << d << " subcells: " << endl;
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<subcellCount; scord++)
    {
      cout << "    ordinal " << scord  << " owned by cell " << cellConstraints.owningCellIDForSubcell[d][scord].cellID << endl;
    }
  }
}

void GDAMinimumRule::printGlobalDofInfo()
{
//  bool ownedGlobalDofsOnly = false;
//  set<GlobalIndexType> cellIDs = _meshTopology->getMyActiveCellIndices();
//  for (GlobalIndexType cellID : cellIDs)
//  {
//    SubcellDofIndices subcellDofIndices = ownedGlobalDofsOnly ? getOwnedGlobalDofIndices(cellID) : getGlobalDofIndices(cellID);
//    printDofIndexInfo(cellID, subcellDofIndices);
//  }
  
  for (auto entry : _globalDofIndicesForCellCache)
  {
    printDofIndexInfo(entry.first, entry.second);
  }
  
}

void GDAMinimumRule::rebuildLookups()
{
  TEUCHOS_FUNC_TIME_MONITOR("GDAMinimumRule::rebuildLookups()");
  int timerHandle = TimeLogger::sharedInstance()->startTimer("rebuildLookups");
  
  clearCaches();
  determineMinimumSubcellDimensionForContinuityEnforcement();
  
  int spaceDim = _meshTopology->getDimension();
  int d_min = minimumSubcellDimensionForContinuityEnforcement();
  
  _partitionFieldDofCount = 0;
  _partitionFluxDofCount = 0;
  _partitionTraceDofCount = 0;

  int rank = _partitionPolicy->Comm()->getRank();
//  cout << "GDAMinimumRule: Rebuilding lookups on rank " << rank << endl;
  set<GlobalIndexType>* myCellIDs = &_partitions[rank];

  map<int, VarPtr> trialVars = _varFactory->trialVars();

  _cellDofOffsets.clear(); // within the partition, offsets for the owned dofs in cell

  // Here, we check and warn if mesh must be 1-irregular but isn't.
  // (Need to examine variables to see if there are any that require reconciliation for d < sideDim; otherwise we can do without 1-irregularity.)
  int irregularity = _mesh->irregularity();
  if (irregularity > 1)
  {
    // this is only supported if only face continuity is required -- i.e. d_min == spaceDim - 1
    if (d_min != spaceDim - 1)
    {
      cout << "ERROR: " << spaceDim << "D mesh is " << irregularity << "-irregular, but min continuity dimension is " << d_min << ".\n";
      TEUCHOS_TEST_FOR_EXCEPTION(d_min != spaceDim - 1, std::invalid_argument, "Mesh irregularity > 1 is only supported when only face continuity is required");
    }
  }
  else
  {
//    cout << "Mesh is " << irregularity << "-irregular.\n";
  }
  
  map<GlobalIndexType, SubcellDofIndices> myCellDofIndices;
  set<GlobalIndexType> nonLocalCellNeighbors; // cells whose dofs I need to label, but aren't in myCells
  
  bool isPureView = (NULL == dynamic_cast<MeshTopology*>(_meshTopology.get()));
  bool hasBaseMeshTopology = (NULL != _meshTopology->baseMeshTopology());  // SubExtrudedMeshTopology does not have a base MeshTopology, for example
  if (isPureView && hasBaseMeshTopology)
  {
    // then we should get dof info for ancestors of the fine cells owned by _meshTopology->baseMeshTopology()
    // add these to nonLocalCellNeighbors
    set<IndexType> ancestorsOfInterest = _meshTopology->getActiveCellIndicesForAncestorsOfMyCellsInBaseMeshTopology();
    for (IndexType cellID : ancestorsOfInterest)
    {
      if (myCellIDs->find(cellID) == myCellIDs->end())
      {
        nonLocalCellNeighbors.insert(cellID);
      }
    }
  }
  
  // new approach to nonLocalCellNeighbors (using halo):
  // the nonLocalCellNeighbors of interest are, at most, the active ones in the cell halo of my cell IDs
  // together with the cell halo of the ancestors of interest, now sitting in nonLocalCellNeighbors
  set<GlobalIndexType> cellHalo;
  _meshTopology->cellHalo(cellHalo, *myCellIDs, _minContinuityDimension);
  _meshTopology->cellHalo(cellHalo, nonLocalCellNeighbors, _minContinuityDimension);
  // get just the active ones that aren't among myCells
  for (GlobalIndexType cellID : cellHalo)
  {
    // we use _partitionForCellID to determine whether the cell is active or not.
    // (If it's not there, then we won't know who to ask for the global dofs, so any active cell whose global dofs we
    //  need should be listed in _partitionForCellID.)
    if ((_partitionForCellID.find(cellID) != _partitionForCellID.end()) && (myCellIDs->find(cellID) == myCellIDs->end()))
    {
      nonLocalCellNeighbors.insert(cellID);
    }
  }
  
  _partitionDofCount = 0; // how many dofs we own locally
  for (GlobalIndexType cellID : *myCellIDs)
  {
    _cellDofOffsets[cellID] = _partitionDofCount;
    CellPtr cell = _meshTopology->getCell(cellID);
    CellTopoPtr topo = cell->topology();
    CellConstraints constraints = getCellConstraints(cellID);
    
    // getOwnedGlobalDofIndices will use the cell's global dof offset, which we still have to compute
    // We use zero for now, and adjust below
    _globalCellDofOffsets[cellID] = 0;
    SubcellDofIndices* ownedGlobalDofIndices = &getOwnedGlobalDofIndices(cellID);
    myCellDofIndices[cellID] = *ownedGlobalDofIndices;
    
    for (auto varEntry : trialVars)
    {
      VarPtr var = varEntry.second;
      int dofsForVariable = ownedGlobalDofIndices->dofIndexCountForVariable(var);
      switch (var->varType()) {
        case FLUX:
          _partitionFluxDofCount += dofsForVariable;
          break;
        case TRACE:
          _partitionTraceDofCount += dofsForVariable;
          break;
        default:
          _partitionFieldDofCount += dofsForVariable;
          break;
      }
      _partitionDofCount += dofsForVariable;
    }
  }

  int numRanks = _partitionPolicy->Comm()->getSize();
  vector<GlobalIndexType> partitionDofCounts(numRanks);
  MPIWrapper::allGather(_partitionPolicy->Comm(), partitionDofCounts, _partitionDofCount);
  
  _partitionDofOffsets.resize(numRanks+1); // global dof count will be at the end
  _partitionDofOffsets[0] = 0;
  for (int i=1; i<numRanks+1; i++)
  {
    _partitionDofOffsets[i] = _partitionDofOffsets[i-1] + partitionDofCounts[i-1];
  }
  
  _partitionDofOffset = _partitionDofOffsets[rank];
  
  for (GlobalIndexType cellID : *myCellIDs)
  {
    _globalCellDofOffsets[cellID] = _cellDofOffsets[cellID] + _partitionDofOffset;
  }
  
  // Now that we have the global dof offsets for our cells, we adjust the ownedGlobalDofIndices container accordingly
  for (GlobalIndexType cellID : *myCellIDs)
  {
    SubcellDofIndices* ownedGlobalDofIndices = &getOwnedGlobalDofIndices(cellID);

    CellTopoPtr topo = _meshTopology->getCell(cellID)->topology();

    GlobalIndexType globalCellDofOffset = _globalCellDofOffsets[cellID];
    
    ownedGlobalDofIndices->addOffsetToDofIndices(globalCellDofOffset);
    
//    for (auto varEntry : trialVars)
//    {
//      VarPtr var = varEntry.second;
//      for (int d=d_min; d<=spaceDim; d++)
//      {
//        int scCount = topo->getSubcellCount(d);
//        for (int scord=0; scord<scCount; scord++)
//        {
//          if (ownedGlobalDofIndices->subcellDofIndices[d].find(scord) != ownedGlobalDofIndices->subcellDofIndices[d].end())
//          {
//            if (ownedGlobalDofIndices->subcellDofIndices[d][scord].find(var->ID()) != ownedGlobalDofIndices->subcellDofIndices[d][scord].end())
//            {
//              vector<GlobalIndexType>* varDofs = &ownedGlobalDofIndices->subcellDofIndices[d][scord][var->ID()];
//              for (vector<GlobalIndexType>::iterator varDofIt = varDofs->begin(); varDofIt != varDofs->end(); varDofIt++)
//              {
//                *varDofIt += globalCellDofOffset;
//              }
//            }
//          }
//        }
//      }
//    }
    
    // copy the adjusted guy back into myCellDofIndices:
    myCellDofIndices[cellID] = *ownedGlobalDofIndices;
  }
  
  distributeGlobalDofOwnership(myCellDofIndices, nonLocalCellNeighbors);
  
  for (GlobalIndexType cellID : *myCellIDs)
  {
    getGlobalDofIndices(cellID); // make sure the _globalDofIndicesForCellCache is filled for all my cellIDs
  }
  distributeCellGlobalDofs(_globalDofIndicesForCellCache, nonLocalCellNeighbors);
  
  TimeLogger::sharedInstance()->stopTimer(timerHandle);
}

void GDAMinimumRule::setAllowMeshTopologyPruning(bool allowMeshTopologyPruning)
{
  const bool oldSetting = _allowMeshTopologyPruning;
  if (oldSetting != allowMeshTopologyPruning)
  {
    // We disallow mesh topology pruning (aka properly distributed MeshTopology) when periodicBCs are present
    if (_mesh->getTopology()->getPeriodicBCs().size() == 0)
    {
      this->_allowMeshTopologyPruning = allowMeshTopologyPruning;
      if (!_allowMeshTopologyPruning)
      {
        MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_meshTopology.get());
        if (meshTopo != NULL)
        {
          // if _meshTopology is not a MeshTopology, we assume pruning is not possible/has not taken place
          if (meshTopo->pruningOrdinal() >= 0) // pruningOrdinal starts at -1; will be 0 after first pruning
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Must set allowMeshTopologyPruning to false *before* any pruning has taken place…");
          }
        }
      }
    }
  }
}

namespace Camellia
{
  std::ostream& operator << (std::ostream& os, AnnotatedEntity& annotatedEntity)
  {
    os << "cell " << annotatedEntity.cellID;
    if (annotatedEntity.sideOrdinal != INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
    {
      os << "'s side " << annotatedEntity.sideOrdinal;
    }
    os << "'s " << CamelliaCellTools::entityTypeString(annotatedEntity.dimension);
    os << " " << annotatedEntity.subcellOrdinal;
    
    return os;
  }
}
