//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
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

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CondensedDofInterpreter.h"
#include "ElementType.h"
#include "GnuPlotUtil.h" // for writing out mesh skeleton when an error is encountered...
#include "Solution.h"

using namespace Camellia;
using namespace std;

GlobalDofAssignment::GlobalDofAssignment(MeshPtr mesh, VarFactoryPtr varFactory,
    DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
    vector<int> initialH1OrderTrial, int testOrderEnhancement, bool enforceConformityLocally) : DofInterpreter(mesh)
{

  _mesh = mesh;
  _meshTopology = mesh->getTopology();
  _varFactory = varFactory;
  _dofOrderingFactory = dofOrderingFactory;
  _partitionPolicy = partitionPolicy;
  _initialH1OrderTrial = initialH1OrderTrial;
  _testOrderEnhancement = testOrderEnhancement;
  _enforceConformityLocally = enforceConformityLocally;
  
  // For now, disallow mesh topology pruning (aka properly distributed MeshTopology) when periodicBCs are present
  if (mesh->getTopology()->getPeriodicBCs().size() > 0)
  {
    _allowMeshTopologyPruning = false;
  }
  
  Epetra_CommPtr Comm = _mesh->getTopology()->Comm();
  
  if (_mesh->getTopology()->Comm() != Teuchos::null)
  {
    set<GlobalIndexType> localActiveCellIDs = _mesh->getTopology()->getLocallyKnownActiveCellIndices();
    
    for (IndexType cellID : localActiveCellIDs)
    {
      assignParities(cellID);
    }
    
    _numPartitions = Comm->NumProc();
    
    int globalActiveCellCount = _mesh->getTopology()->activeCellCount();
    set<GlobalIndexType> myCellIDs = _mesh->getTopology()->getMyActiveCellIndices();
    int numMyCellIDs = myCellIDs.size();

    int myPartitionOffset;
    Comm->ScanSum(&numMyCellIDs, &myPartitionOffset, 1);
    myPartitionOffset -= numMyCellIDs;
    
    vector<int> gatheredPartitionOffsets(_numPartitions,0);
    Comm->GatherAll(&myPartitionOffset, &gatheredPartitionOffsets[0], 1);
    
    vector<GlobalIndexTypeToCast> partitionOrderedCells(globalActiveCellCount,0);
    vector<GlobalIndexTypeToCast> gatheredPartitionOrderedCells(globalActiveCellCount,0);
    
    int i=0;
    for (GlobalIndexType cellID : myCellIDs)
    {
      partitionOrderedCells[i + myPartitionOffset] = cellID;
      i++;
    }
    // since this does not contain the same number of entries on each rank, can't use GatherAll
    Comm->SumAll(&partitionOrderedCells[0], &gatheredPartitionOrderedCells[0], globalActiveCellCount);
    
    _partitions = vector<set<GlobalIndexType> >(_numPartitions);
    for (int partitionOrdinal=0; partitionOrdinal<_numPartitions; partitionOrdinal++)
    {
      set<GlobalIndexType>* partitionCellIDs = &_partitions[partitionOrdinal];
      int startOffset = gatheredPartitionOffsets[partitionOrdinal];
      int endOffset = (partitionOrdinal < _numPartitions-1) ? gatheredPartitionOffsets[partitionOrdinal+1] : globalActiveCellCount;
      for (int i=startOffset; i<endOffset; i++)
      {
        partitionCellIDs->insert(gatheredPartitionOrderedCells[i]);
      }
    }
  }
  else
  {
    // MeshTopology not already distributed
    set<GlobalIndexType> activeCellIDs = _mesh->getActiveCellIDsGlobal();
    
    for (IndexType cellID : activeCellIDs)
    {
      assignParities(cellID);
    }
    
    _numPartitions = _partitionPolicy->Comm()->NumProc();
    _partitions = vector<set<GlobalIndexType> >(_numPartitions);
    
    // before repartitioning (which should happen immediately), put all active cells on rank 0
    _partitions[0] = activeCellIDs;
  }

  constructActiveCellMap();
  determineMinimumSubcellDimensionForContinuityEnforcement();
}

GlobalDofAssignment::GlobalDofAssignment( GlobalDofAssignment &otherGDA ) : DofInterpreter(Teuchos::null)    // subclass deepCopy() is responsible for filling this in post-construction
{
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

  _cellPRefinements = otherGDA._cellPRefinements;
  _elementTypesForCellTopology = otherGDA._elementTypesForCellTopology;

  _partitions = otherGDA._partitions;
  _partitionForCellID = otherGDA._partitionForCellID;

  _activeCellMap = Teuchos::rcp( new Epetra_Map(*otherGDA._activeCellMap) );
//  _activeCellMap2 = Teuchos::rcp( new Tpetra::Map<>(*otherGDA._activeCellMap2) );

  _numPartitions = otherGDA._numPartitions;

  _allowMeshTopologyPruning = otherGDA._allowMeshTopologyPruning;
  
  // we leave _registeredSolutions empty
  ///_registeredSolutions;
}

GlobalIndexType GlobalDofAssignment::activeCellOffset()
{
  return _activeCellOffset;
}

void GlobalDofAssignment::assignParities( GlobalIndexType cellID )
{
  CellPtr cell = _mesh->getTopology()->getCell(cellID);

  unsigned sideCount = cell->getSideCount();

  vector<int> cellParities(sideCount);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, _meshTopology);
    GlobalIndexType neighborCellID = neighborInfo.first;
    if (neighborCellID == -1)   // boundary --> parity is 1
    {
      cellParities[sideOrdinal] = 1;
    }
    else
    {
      CellPtr neighbor = _meshTopology->getCell(neighborCellID);
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(neighborInfo.second, _meshTopology);
      bool cellParitySet = false;
      
      // if the neighbor sees us as its neighbor, then we're certainly peers
      if (neighborNeighborInfo.first == cellID)
      {
        // then the lower cellID gets the positive parity
        cellParities[sideOrdinal] = (cellID < neighborCellID) ? 1 : -1;
        cellParitySet = true;
      }
      else
      {
        // If not, then it's still possible that we have a descendant that is peers with neighbor (or neighbor's descendant),
        // in the case of anisotropic refinements.
        
        unsigned sideDim = _meshTopology->getDimension() - 1;
        IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
        vector<IndexType> cellIndicesForSide = _meshTopology->getCellsForSide(sideEntityIndex); // max 2 entries
        
        GlobalIndexType neighborDescendantID = -1, cellDescendantID = -1;
        for (IndexType cellIndexForSide : cellIndicesForSide)
        {
          CellPtr cellForSide = _meshTopology->getCell(cellIndexForSide);
          if (cellForSide->isDescendant(neighborCellID))
          {
            neighborDescendantID = cellIndexForSide;
          }
          else if (cellForSide->isDescendant(cellID))
          {
            cellDescendantID = cellIndexForSide;
          }
        }
        
        if ((neighborDescendantID != -1) && (cellDescendantID != -1))
        {
          // a descendant that is a peer: we "inherit" parity of the descendant
          // then the lower cellID gets the positive parity
          cellParities[sideOrdinal] = (cellDescendantID < neighborDescendantID) ? 1 : -1;
          cellParitySet = true;
        }
      }
      if (!cellParitySet)
      {
        CellPtr parent = cell->getParent();
        if (parent.get() == NULL)
        {
          cout << "ERROR: in assignParities(), encountered cell with non-peer neighbor but without parent.\n";
          cout << "cell ID: " << cellID << endl;
          cout << "neighbor cell ID: " << neighborCellID << endl;
          cout << "sideOrdinal: " << sideOrdinal << endl;
          if (_mesh->Comm()->NumProc() == 1)
          {
            GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/bad_mesh", _mesh, true);
            cout << "Wrote labeled mesh to /tmp/bad_mesh.\n";
          }
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "in assignParities(), encountered cell with non-peer neighbor but without parent");
        }
        // inherit parent's parity along the shared side:
        unsigned childOrdinal = parent->childOrdinal(cellID);
        unsigned parentSideOrdinal = parent->refinementPattern()->parentSideLookupForChild(childOrdinal)[sideOrdinal];
        if (_cellSideParitiesForCellID.find(parent->cellIndex()) == _cellSideParitiesForCellID.end())
        {
          assignParities(parent->cellIndex());
        }
        cellParities[sideOrdinal] = _cellSideParitiesForCellID[parent->cellIndex()][parentSideOrdinal];
      }
    }
  }
  // track which sides flipped parities, so that solution's fluxes can be corrected accordingly
  vector<int> sidesWithChangedParities;
  if (_cellSideParitiesForCellID.find(cellID) != _cellSideParitiesForCellID.end())
  {
    vector<int> oldSideParities = _cellSideParitiesForCellID[cellID];
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      if (oldSideParities[sideOrdinal] != cellParities[sideOrdinal])
      {
        sidesWithChangedParities.push_back(sideOrdinal);
      }
    }
  }
  _cellSideParitiesForCellID[cellID] = cellParities;

  for (TSolutionPtr<double> soln : _registeredSolutions)
  {
    int numSolutions = soln->numSolutions();
    for (int solutionOrdinal=0; solutionOrdinal<numSolutions; solutionOrdinal++)
    {
      if (soln->cellHasCoefficientsAssigned(cellID,solutionOrdinal))
      {
        soln->reverseParitiesForLocalCoefficients(cellID, sidesWithChangedParities, solutionOrdinal);
      }
    }
  }
  
  // if this cell is a parent, then we should treat its children as well (children without peer neighbors will inherit any parity flips)
  if (cell->isParent(_meshTopology))
  {
    vector<GlobalIndexType> childIndices = cell->getChildIndices(_meshTopology);
    for (GlobalIndexType childCellIndex : childIndices)
    {
      if (_meshTopology->isValidCellIndex(childCellIndex))
      {
        // if childCellIndex isn't a valid cell index, the cell for cellID must be a ghost cell, and we don't actually need parities
        // to be correct on that side...
        assignParities(childCellIndex);
      }
    }
  }
}

//GlobalIndexType GlobalDofAssignment::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber)
//{
//  if (partitionNumber == -1)
//  {
//    // determine the partition number for the cellIndex
//    int partitionCellOffset = 0;
//    for (PartitionIndexType i=0; i<partitionNumber; i++)
//    {
//      int numCellIDsForPartition = _cellIDsForElementType[i][elemTypePtr.get()].size();
//      if (partitionCellOffset + numCellIDsForPartition > cellIndex)
//      {
//        partitionNumber = i;
//        cellIndex -= partitionCellOffset; // rewrite as a local cellIndex
//        break;
//      }
//      partitionCellOffset += numCellIDsForPartition;
//    }
//    if (partitionNumber == -1)
//    {
//      cout << "cellIndex is out of bounds.\n";
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIndex is out of bounds.");
//    }
//  }
//  if ( ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) != _cellIDsForElementType[partitionNumber].end() )
//       &&
//       (_cellIDsForElementType[partitionNumber][elemTypePtr.get()].size() > cellIndex ) )
//  {
//    return _cellIDsForElementType[partitionNumber][elemTypePtr.get()][cellIndex];
//  }
//  else return -1;
//}

vector<GlobalIndexType> GlobalDofAssignment::cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemType)
{
  if (partitionNumber == -1)
  {
    cout << "cellIDsOfElementType called with partitionNumber==-1.  Returning empty vector.\n";
    return vector<GlobalIndexType>();
  }
  if (_meshTopology->isDistributed() && (partitionNumber != _meshTopology->Comm()->MyPID()))
  {
    cout << "For distributed MeshTopology, cellIDsOfElementType() only supports the current partition.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "For distributed MeshTopology, cellIDsOfElementType() only supports the current partition");
  }
  
  // first, let's find the key that corresponds to elemType
  bool keyFound = false;
  pair<CellTopologyKey,int> key;
  for (auto entry : _elementTypesForCellTopology)
  {
    if (*(entry.second) == *elemType)
    {
      key = entry.first;
      keyFound = true;
      break;
    }
  }
  if (keyFound)
  {
    vector<GlobalIndexType> cellsForElementType;
    const set<GlobalIndexType>* cellsInPartition = &_partitions[partitionNumber];
    for (GlobalIndexType cellID : *cellsInPartition)
    {
      if (getElementTypeLookupKey(cellID) == key)
      {
        auto cellElemType = elementType(cellID);
        if (*cellElemType == *elemType)
        {
          cellsForElementType.push_back(cellID);
        }
      }
    }
    return cellsForElementType;
  }
  else
  {
    // check for tag-indexed element type
    // let's look for the tuple that corresponds to elemType
    bool tupleFound = false;
    std::tuple<CellTopologyKey,int,std::set<tag>, std::map<int, std::set< tag > > > tuple;
    for (auto entry : _elementTypesForKeyAndTags)
    {
      if (*(entry.second) == *elemType)
      {
        tuple = entry.first;
        tupleFound = true;
        break;
      }
    }
    if (tupleFound)
    {
      vector<GlobalIndexType> cellsForElementType;
      const set<GlobalIndexType>* cellsInPartition = &_partitions[partitionNumber];
      key = {std::get<0>(tuple), std::get<1>(tuple)};
      for (GlobalIndexType cellID : *cellsInPartition)
      {
        if (getElementTypeLookupKey(cellID) == key)
        {
          // then get the full ElementType and compare
          auto cellElemType = elementType(cellID);
          if (*cellElemType == *elemType)
          {
            cellsForElementType.push_back(cellID);
          }
        }
      }
      return cellsForElementType;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching cells found");
  }
}

const set< GlobalIndexType > & GlobalDofAssignment::cellsInPartition(PartitionIndexType partitionNumber) const
{
  int rank = _partitionPolicy->Comm()->MyPID();
  if (partitionNumber == -1)
  {
    partitionNumber = rank;
  }
  return _partitions[partitionNumber];
}

Camellia::FieldContainer<double> GlobalDofAssignment::cellSideParitiesForCell( GlobalIndexType cellID )
{
  if (_cellSideParitiesForCellID.find(cellID) == _cellSideParitiesForCellID.end())
  {
    assignParities(cellID);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(_cellSideParitiesForCellID.find(cellID) == _cellSideParitiesForCellID.end(),
                             std::invalid_argument, "_cellSideParities is not set for the provided cell!");
  vector<int> parities = _cellSideParitiesForCellID[cellID];
  int numSides = parities.size();
  Camellia::FieldContainer<double> cellSideParities(1,numSides);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++)
  {
    cellSideParities(0,sideIndex) = parities[sideIndex];
  }
  return cellSideParities;
}

void GlobalDofAssignment::constructActiveCellMap()
{
  const set<GlobalIndexType>* myCellIDs = &cellsInPartition(-1);
  vector<GlobalIndexTypeToCast> myCellIDsVector(myCellIDs->begin(), myCellIDs->end());

  int indexBase = 0;
  if (myCellIDsVector.size()==0)
    _activeCellMap = Teuchos::rcp( new Epetra_Map(-1, myCellIDsVector.size(), NULL, indexBase, *_partitionPolicy->Comm()) );
  else
    _activeCellMap = Teuchos::rcp( new Epetra_Map(-1, myCellIDsVector.size(), &myCellIDsVector[0], indexBase, *_partitionPolicy->Comm()) );
}

void GlobalDofAssignment::constructActiveCellMap2()
{
  using LO = ::Tpetra::Details::DefaultTypes::local_ordinal_type;
  using GO = ::Tpetra::Details::DefaultTypes::global_ordinal_type;
  
  const set<GlobalIndexType> & cellIDs = cellsInPartition(-1);
  vector<GO> myCellIDsVector(cellIDs.size());
  
  int i=0;
  for (const GlobalIndexType & cellID : cellIDs)
  {
    myCellIDsVector[i++] = static_cast<GO>(cellID);
  }
  
  Teuchos::ArrayView< const GO > myCellIDsAV(myCellIDsVector);

  int indexBase = 0;
  if (myCellIDsVector.size()==0)
    _activeCellMap2 = Teuchos::rcp( new Tpetra::Map<>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                    myCellIDsVector.size(), indexBase, _partitionPolicy->TeuchosComm()) );
  else
    _activeCellMap2 = Teuchos::rcp( new Tpetra::Map<>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                    myCellIDsAV, indexBase, _partitionPolicy->TeuchosComm()) );
}

void GlobalDofAssignment::determineMinimumSubcellDimensionForContinuityEnforcement()
{
  int myMinimumSubcellDimension = _meshTopology->getDimension();
  set<pair<CellTopologyKey,int>> myCellKeys;
  const set<GlobalIndexType>* myCellIDs = &cellsInPartition(-1);
  
  std::set<ElementType> elemTypes; // not elemTypePtr; use ElementType's == implementation for de-duplication
  for (GlobalIndexType myCellID : *myCellIDs)
  {
    auto elemTypePtr = this->elementType(myCellID);
    elemTypes.insert(*elemTypePtr);
  }
  for (const auto &elemType : elemTypes)
  {
    int elemMin = elemType.trialOrderPtr->minimumSubcellDimensionForContinuity();
    myMinimumSubcellDimension = min(myMinimumSubcellDimension,elemMin);
    if (myMinimumSubcellDimension == 0) break; // can't go lower than 0
  }
  int globalMinimumSubcellDimension;
  _partitionPolicy->Comm()->MinAll(&myMinimumSubcellDimension, &globalMinimumSubcellDimension, 1);
  _minContinuityDimension = globalMinimumSubcellDimension;
}

void GlobalDofAssignment::didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
{
  // look for any cells whose parities might need to be reassigned
  // these include the unrefined cell, its neighbors, and its neighbors' descendants along the shared sides.
  for (GlobalIndexType parentID : cellIDs)
  {
    if (_meshTopology->isValidCellIndex(parentID))
    {
      assignParities(parentID);
      // neighbors and their descendants on the side should be invalidated
      CellPtr parentCell = _meshTopology->getCell(parentID);
      unsigned int sideCount = parentCell->getSideCount();
      for (unsigned int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        const auto neighborInfo = parentCell->getNeighborInfo(sideOrdinal, _meshTopology);
        const auto & neighborCellID = neighborInfo.first;
        const auto & mySideIndexInNeighbor = neighborInfo.second;
        if (neighborCellID != -1)
        {
          assignParities(neighborCellID);
        }
        if (_meshTopology->isValidCellIndex(neighborCellID))
        {
          CellPtr neighborCell = _meshTopology->getCell(neighborCellID);
          const bool leafNodesOnly = false;
          vector< pair< GlobalIndexType, unsigned> > neighborDescendants = neighborCell->getDescendantsForSide(mySideIndexInNeighbor, _meshTopology, leafNodesOnly);
          for (const auto &descendantEntry : neighborDescendants)
          {
            const GlobalIndexType & descendantID = descendantEntry.first;
            if (_meshTopology->isValidCellIndex(descendantID))
            {
              assignParities(descendantID);
            }
          }
        }
      }
    }
  }
  
}

ElementTypePtr GlobalDofAssignment::elementType(GlobalIndexType cellID)
{
  pair<CellTopologyKey, int> key = getElementTypeLookupKey(cellID);
  
  auto varTagFilters = _varFactory->tagFilters();
  using tag = std::pair<std::string,int>;
  std::map<int, std::set<tag> > tagsForSide;
  std::set<tag> tagsForVolume;
  
  int volumeDim = _meshTopology->getDimension();
  int sideDim   = volumeDim - 1;
  
  for (auto varTag : varTagFilters)
  {
    vector<EntitySetPtr> entitySets = _meshTopology->baseMeshTopology()->getEntitySetsForTagID(varTag.first, varTag.second);
    for (EntitySetPtr entitySet : entitySets)
    {
      auto volumeOrdinals = entitySet->subcellOrdinals(_meshTopology, cellID, volumeDim);
      if (volumeOrdinals.size() > 0) tagsForVolume.insert(varTag);
      
      auto sideOrdinals = entitySet->subcellOrdinals(_meshTopology, cellID, sideDim);
      for (const auto & sideOrdinal : sideOrdinals)
      {
        tagsForSide[sideOrdinal].insert(varTag);
      }
    }
  }
  
  if ((tagsForVolume.size() == 0) && (tagsForSide.size() == 0))
  {
    return getElementTypeForKey(key);
  }
  else
  {
    return getElementTypeForKeyAndTags(key,tagsForVolume,tagsForSide);
  }
}

const map<GlobalIndexType,int>& GlobalDofAssignment::getCellPRefinements() const
{
  return _cellPRefinements;
}

void GlobalDofAssignment::setCellPRefinements(const map<GlobalIndexType,int>& pRefinements)
{
  _cellPRefinements = pRefinements;
}

void GlobalDofAssignment::repartitionAndMigrate()
{
  _partitionPolicy->partitionMesh(_mesh.get(),_numPartitions);
  // if our MeshTopology is the base, then we can prune
  
  if (_allowMeshTopologyPruning)
  {
    // then clear cell parities, as these may now be outdated
    // (there may be a more granular approach, but parity determination is relatively cheap)
    _cellSideParitiesForCellID.clear();
  }
  
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_meshTopology.get());
  if (meshTopo != NULL)
  {
    if (_allowMeshTopologyPruning)
    {
      int dimForNeighborRelation = minimumSubcellDimensionForContinuityEnforcement(); // collective operation
      set<GlobalIndexType> cellsToInclude = cellsInPartition(-1); // my cells
      auto & knownCells = meshTopo->getLocallyKnownActiveCellIndices();
      for (auto cellID : knownCells)
      {
        if (meshTopo->cellHasCurvedEdges(cellID))
        {
          cellsToInclude.insert(cellID);
        }
      }
      meshTopo->pruneToInclude(_partitionPolicy->Comm(), cellsToInclude, dimForNeighborRelation);
    }
  }
  for (vector< TSolutionPtr<double> >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++)
  {
    // if solution has a condensed dof interpreter, we should reinitialize the mapping from interpreted to global dofs
    Teuchos::RCP<DofInterpreter> dofInterpreter = (*solutionIt)->getDofInterpreter();
    CondensedDofInterpreter<double>* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter<double>*>(dofInterpreter.get());
    if (condensedDofInterpreter != NULL)
    {
      condensedDofInterpreter->reinitialize();
    }

    (*solutionIt)->initializeLHSVector(); // rebuild LHS vector; global dofs will have changed. (important for addSolution)
  }
}

void GlobalDofAssignment::didHRefine(const set<GlobalIndexType> &parentCellIDs)   // subclasses should call super
{
  // until we repartition, assign the new children to the parent's partition
  for (set<GlobalIndexType>::const_iterator cellIDIt=parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++)
  {
    GlobalIndexType parentID = *cellIDIt;
    PartitionIndexType partitionForParent = partitionForCellID(parentID);
    if (partitionForParent != -1) // this check allows us to be robust against getting the notification twice.
    {
      // If we don't know the parent cell, then for the moment we won't see even the child indices
      // I'm hoping that we will soon be able to relax the requirement that every partition knows about
      // every other partition's cell assignments; _partitionForCellID is one of the few containers
      // that grows with the number of global cells.
      
      if (_meshTopology->isValidCellIndex(parentID))
      {
        CellPtr parent = _meshTopology->getCell(parentID);
        vector<GlobalIndexType> childIDs = parent->getChildIndices(_meshTopology);
        _partitions[partitionForParent].insert(childIDs.begin(),childIDs.end());
        for (GlobalIndexType childID : childIDs)
        {
          _partitionForCellID[childID] = partitionForParent;
        }
      }
      _partitions[partitionForParent].erase(parentID);
      _partitionForCellID.erase(parentID);
    }
  }
  constructActiveCellMap();
}

void GlobalDofAssignment::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP)   // subclasses should call super
{
  for (GlobalIndexType cellID : cellIDs)
  {
    if (_cellPRefinements.find(cellID) == _cellPRefinements.end())
    {
      _cellPRefinements[cellID] = deltaP;
    }
    else
    {
      _cellPRefinements[cellID] += deltaP;
    }
  }
}

void GlobalDofAssignment::willHUnrefine(const set<GlobalIndexType> &parentCellIDs)   // subclasses should call super
{
  
  projectChildCoefficientsOntoParents(parentCellIDs);
  
  for (GlobalIndexType parentID : parentCellIDs)
  {
    PartitionIndexType partitionForParent = partitionForCellID(parentID);
    if (partitionForParent == -1) // this check allows us to be robust against getting the notification twice.
    {
      if (_meshTopology->isValidCellIndex(parentID))
      {
        CellPtr parent = _meshTopology->getCell(parentID);
        vector<GlobalIndexType> childIDs = parent->getChildIndices(_meshTopology);
        
        GlobalIndexType firstChildID = childIDs[0];
        
        partitionForParent = partitionForCellID(firstChildID);
        
        _partitions[partitionForParent].insert(parentID);
        _partitionForCellID[parentID] = partitionForParent;
        
        for (GlobalIndexType childID : childIDs)
        {
          auto partition = _partitionForCellID[childID];
          _partitions[partition].erase(childID);
          _partitionForCellID.erase(childID);
        }
      }
    }
  }
  
  // TODO: do something to ensure that the active cell map is reconstructed after the unrefinement actually takes place
  constructActiveCellMap();
}

vector< ElementTypePtr > GlobalDofAssignment::elementTypes(PartitionIndexType partitionNumber)
{
  // Ultimately, it probably will be an error to call this with argument that is not equal to the current rank.
  if (partitionNumber != -1)
  {
    const set<GlobalIndexType> cellIDsForPartition = this->cellsInPartition(partitionNumber);
    
    vector< ElementTypePtr > elemTypes;
    set< ElementType > elemTypesSet;
    
    for (const GlobalIndexType & cellID : cellIDsForPartition)
    {
      auto elemTypePtr = elementType(cellID);
      if (elemTypesSet.find(*elemTypePtr) == elemTypesSet.end() )
      {
        elemTypesSet.insert(*elemTypePtr);
        elemTypes.push_back(elemTypePtr);
      }
    }
    return elemTypes;
  }
  else
  {
    int numRanks = _mesh->Comm()->NumProc();
    
    vector< ElementTypePtr > elemTypes;
    set< ElementType > elemTypesSet;
    
    for (int rank=0; rank<numRanks; rank++)
    {
      const set<GlobalIndexType> & cellIDsForPartition = this->cellsInPartition(rank);
      for (const GlobalIndexType & cellID : cellIDsForPartition)
      {
        auto elemTypePtr = elementType(cellID);
        if (elemTypesSet.find(*elemTypePtr) == elemTypesSet.end() )
        {
          elemTypesSet.insert(*elemTypePtr);
          elemTypes.push_back(elemTypePtr);
        }
      }
    }
    
    return elemTypes;
  }
}

Teuchos::RCP<Epetra_Map> GlobalDofAssignment::getActiveCellMap()
{
  return _activeCellMap;
}

MapPtr GlobalDofAssignment::getActiveCellMap2()
{
  return _activeCellMap2;
}

int GlobalDofAssignment::getCubatureDegree(GlobalIndexType cellID)
{
  ElementTypePtr elemType = this->elementType(cellID);
  return elemType->trialOrderPtr->maxBasisDegree() + elemType->testOrderPtr->maxBasisDegree();
}

DofOrderingFactoryPtr GlobalDofAssignment::getDofOrderingFactory()
{
  return _dofOrderingFactory;
}

ElementTypeFactory & GlobalDofAssignment::getElementTypeFactory()
{
  return _elementTypeFactory;
}

ElementTypePtr GlobalDofAssignment::getElementTypeForKey(pair<CellTopologyKey,int> key)
{
  if (_elementTypesForCellTopology.find(key) == _elementTypesForCellTopology.end())
  {
    int deltaP = key.second;
    vector<int> H1Order = getH1OrderForPRefinement(deltaP);
    const int numPComponents = int(H1Order.size());
    vector<int> testDegree(numPComponents);
    for (int pComponent=0; pComponent<numPComponents; pComponent++)
    {
      testDegree[pComponent] = H1Order[pComponent] + _testOrderEnhancement;
    }
    
    CellTopoPtr cellTopo = CellTopology::cellTopology(key.first);
    DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(H1Order, cellTopo, _enforceConformityLocally);
    DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testDegree, cellTopo);
    ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cellTopo);
    _elementTypesForCellTopology[key] = elemType;
  }
  return _elementTypesForCellTopology[key];
}

using tag = std::pair<std::string,int>;
ElementTypePtr GlobalDofAssignment::getElementTypeForKeyAndTags(pair<CellTopologyKey,int> key,
                                                                const std::set< tag > &tagsForVolume,
                                                                const std::map<int, std::set< tag > > &tagsForSide)
{
//  CellTopologyKey first = key.first;
//  int second = key.second;
//  std::set<tag> third = tagsForVolume;
//  std::map<int,tag> fourth = tagsForSide;
  std::tuple<CellTopologyKey,int,std::set<tag>, std::map<int, std::set< tag > > > keyTagsTuple = make_tuple(key.first,key.second,tagsForVolume,tagsForSide);
  if (_elementTypesForKeyAndTags.find(keyTagsTuple) == _elementTypesForKeyAndTags.end())
  {
    int deltaP = key.second;
    vector<int> H1Order = getH1OrderForPRefinement(deltaP);
    const int numPComponents = int(H1Order.size());
    vector<int> testDegree(numPComponents);
    for (int pComponent=0; pComponent<numPComponents; pComponent++)
    {
      testDegree[pComponent] = H1Order[pComponent] + _testOrderEnhancement;
    }
    
    CellTopoPtr cellTopo = CellTopology::cellTopology(key.first);
    DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(H1Order, cellTopo, _enforceConformityLocally,tagsForVolume,tagsForSide);
    DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testDegree, cellTopo,tagsForVolume,tagsForSide);
    ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cellTopo);
    _elementTypesForKeyAndTags[keyTagsTuple] = elemType;
  }
  return _elementTypesForKeyAndTags[keyTagsTuple];
}

pair<CellTopologyKey,int> GlobalDofAssignment::getElementTypeLookupKey(GlobalIndexType cellID)
{
  int deltaP = getPRefinementDegree(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(!_meshTopology->isValidCellIndex(cellID), std::invalid_argument, "cellID not found in MeshTopology!");
  CellTopologyKey cellTopoKey = _meshTopology->getCell(cellID)->topology()->getKey();
  
  pair<CellTopologyKey, int> key = {cellTopoKey,deltaP};
  return key;
}

map< pair<CellTopologyKey,int>, set<GlobalIndexType> > GlobalDofAssignment::getElementTypePartitionForCells(const  set<GlobalIndexType> &cellIDs)
{
  map< pair<CellTopologyKey,int>, set<GlobalIndexType> > elemTypePartitions;
  
  for (const auto & cellID : cellIDs)
  {
    auto elemTypeKey = getElementTypeLookupKey(cellID);
    elemTypePartitions[elemTypeKey].insert(cellID);
  }
  return elemTypePartitions;
}

MeshPartitionPolicyPtr GlobalDofAssignment::getPartitionPolicy()
{
  return _partitionPolicy;
}

int GlobalDofAssignment::getPRefinementDegree(GlobalIndexType cellID)
{
  if (_cellPRefinements.find(cellID) == _cellPRefinements.end())
  {
    return 0;
  }
  else
  {
    return _cellPRefinements[cellID];
  }
}

ElementTypePtr GlobalDofAssignment::getSampleElementType()
{
  if (_elementTypesForCellTopology.size() == 0)
  {
    // force determination of some cell, if any is known
    const auto & activeCellIDs = _mesh->getTopology()->getLocallyKnownActiveCellIndices();
    if (activeCellIDs.size() == 0) return Teuchos::null;
    const auto & cellID = *activeCellIDs.begin();
    return elementType(cellID);
  }
  return _elementTypesForCellTopology.begin()->second;
}

GlobalIndexType GlobalDofAssignment::globalCellIndex(GlobalIndexType cellID)
{
  int partitionNumber     = partitionForCellID(cellID);
  if (partitionNumber == -1)
  {
    // no partition number found -- cell is presumably inactive.  Return -1
    return -1;
  }
  GlobalIndexType cellIndex = partitionLocalCellIndex(cellID, partitionNumber);
  ElementTypePtr elemType = elementType(cellID);
  
  for (PartitionIndexType i=0; i<partitionNumber; i++)
  {
    cellIndex += cellIDsOfElementType(i, elemType).size();
  }
  return cellIndex;
}

vector<int> GlobalDofAssignment::getH1Order(GlobalIndexType cellID)
{
  int deltaP = getPRefinementDegree(cellID);
  return getH1OrderForPRefinement(deltaP);
}

vector<int> GlobalDofAssignment::getH1OrderForPRefinement(int deltaP)
{
  if (deltaP == 0) return _initialH1OrderTrial;
  
  vector<int> H1Order = _initialH1OrderTrial;
  for (int d=0; d<H1Order.size(); d++)
  {
    H1Order[d] += deltaP;
  }
  
  return H1Order;
}

vector<int> GlobalDofAssignment::getInitialH1Order()
{
  return _initialH1OrderTrial;
}

MeshPtr GlobalDofAssignment::getMesh()
{
  return _mesh;
}

MeshTopologyViewPtr GlobalDofAssignment::getMeshTopology()
{
  return _meshTopology;
}

bool GlobalDofAssignment::getPartitions(Camellia::FieldContainer<GlobalIndexType> &partitions)
{
  if (_partitions.size() == 0) return false; // false: no partitions set
  int numPartitions = _partitions.size();
  int maxSize = 0;
  for (int i=0; i<numPartitions; i++)
  {
    maxSize = std::max<int>((int)_partitions[i].size(),maxSize);
  }
  partitions.resize(numPartitions,maxSize);
  partitions.initialize(-1);
  for (int i=0; i<numPartitions; i++)
  {
    int j=0;
    for (set<GlobalIndexType>::iterator cellIDIt = _partitions[i].begin();
         cellIDIt != _partitions[i].end(); cellIDIt++)
    {
      partitions(i,j) = *cellIDIt;
      j++;
    }
  }
  return true; // true: partitions container filled
}

PartitionIndexType GlobalDofAssignment::getPartitionCount()
{
  return _numPartitions;
}

int GlobalDofAssignment::getTestOrderEnrichment()
{
  return _testOrderEnhancement;
}

void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localCoefficients,
                                                     Epetra_MultiVector &globalCoefficients, int columnOrdinal)
{
  DofOrderingPtr trialOrder = elementType(cellID)->trialOrderPtr;
  Camellia::FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      Camellia::FieldContainer<double> fittedGlobalCoefficients;
      Camellia::FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, columnOrdinal, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
//        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

template <typename Scalar>
void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localCoefficients,
                                                     TVectorPtr<Scalar> globalCoefficients, int columnOrdinal)
{
  DofOrderingPtr trialOrder = elementType(cellID)->trialOrderPtr;
  Camellia::FieldContainer<Scalar> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      Camellia::FieldContainer<Scalar> fittedGlobalCoefficients;
      Camellia::FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients->replaceGlobalValue(globalDofIndex, columnOrdinal, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
      }
    }
  }
}
template void GlobalDofAssignment::interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localCoefficients,
                                                              TVectorPtr<double> globalCoefficients, int columnOrdinal);

// ! Returns the smallest dimension along which continuity will be enforced.  GlobalDofAssignment's implementation
// ! assumes that the function spaces for the bases defined on cells determine this (e.g. H^1-conforming basis --> 0).
int GlobalDofAssignment::minimumSubcellDimensionForContinuityEnforcement() const
{
  return _minContinuityDimension;
}

void GlobalDofAssignment::projectChildCoefficientsOntoParents(const std::set<GlobalIndexType> &parentCellIDs)
{
  // rule is that the parent coefficients end up on the first child's partition
  // therefore, we need to send all child coefficients that do not belong to the same partition as the first child to that partition
  int myRank = _partitionPolicy->Comm()->MyPID();
  
  // we will use Solution::importSolutionForOffRankCells(cellIDs)
  // we build up a list of off-rank cells whose data we need
  set<GlobalIndexType> offRankCellsNeeded;
  
  set<GlobalIndexType> localParentCellIDs; // the ones that end up on this rank
  
  for (auto parentCellID : parentCellIDs)
  {
    if (_meshTopology->isValidCellIndex(parentCellID))
    {
      auto parentCell = _meshTopology->getCell(parentCellID);
      auto childIndices = parentCell->getChildIndices(_meshTopology);
      auto firstChildID = childIndices[0];
      auto destinationRank = partitionForCellID(firstChildID);
      if (destinationRank == myRank) // if first child is locally owned, then we may have some data to request
      {
        localParentCellIDs.insert(parentCellID);
        for (int childOrdinal=1; childOrdinal<childIndices.size(); childOrdinal++)
        {
          auto childID = childIndices[childOrdinal];
          auto childPartition = partitionForCellID(childID);
          TEUCHOS_TEST_FOR_EXCEPTION(childPartition == -1, std::invalid_argument, "child cell ID " + std::to_string(childID) + ": partition not found.  Perhaps it's not an active cell?");
          if (childPartition != myRank)
          {
            offRankCellsNeeded.insert(childID);
          }
        }
      }
    }
  }
  
  for (TSolutionPtr<double> soln : _registeredSolutions)
  {
    soln->importSolutionForOffRankCells(offRankCellsNeeded);
  }
  
  for (TSolutionPtr<double> soln : _registeredSolutions)
  {
    const int numLHSes = soln->numSolutions();
    for (GlobalIndexType parentCellID : localParentCellIDs)
    {
      for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
      {
        soln->projectChildrenOntoParent(parentCellID, lhsOrdinal);
      }
    }
  }
}

void GlobalDofAssignment::projectParentCoefficientsOntoUnsetChildren()
{
  const set<GlobalIndexType> & rankLocalCellIDs = cellsInPartition(-1);

  for (TSolutionPtr<double> soln : _registeredSolutions)
  {
    const int numLHSes = soln->numSolutions();
    for (GlobalIndexType cellID : rankLocalCellIDs)
    {
      for (int lhsOrdinal=0; lhsOrdinal<numLHSes; lhsOrdinal++)
      {
        if (soln->cellHasCoefficientsAssigned(cellID,lhsOrdinal)) continue;

        CellPtr cell = _meshTopology->getCell(cellID);
        CellPtr parent = cell->getParent();
        if (parent == Teuchos::null) continue;
        
        // proceed up the refinement hierarchy looking for an ancestor that has solution data
        std::vector<GlobalIndexType> projectionsNeeded {cellID}; // from coarsest to finest, these are the ancestral cells that need solution coefficients
        CellPtr ancestor = parent;
        while (true)
        {
          GlobalIndexType ancestorCellID = ancestor->cellIndex();
          if (soln->cellHasCoefficientsAssigned(ancestorCellID,lhsOrdinal))
          {
            break;
          }
          else
          {
            projectionsNeeded.insert(projectionsNeeded.begin(), ancestorCellID);
          }
          if (ancestor->getParent() == Teuchos::null)
          {
            break;
          }
          else
          {
            ancestor = ancestor->getParent();
          }
        }
        GlobalIndexType ancestorCellID = ancestor->cellIndex();
        if (! soln->cellHasCoefficientsAssigned(ancestorCellID,lhsOrdinal)) continue;

        for (auto childCellID : projectionsNeeded)
        {
          CellPtr childCell = _meshTopology->getCell(childCellID);
          parent = childCell->getParent();
          int childOrdinal = -1;
          vector<IndexType> childIndices = parent->getChildIndices(_meshTopology);
          for (int i=0; i<childIndices.size(); i++)
          {
            if (childIndices[i]==childCellID) childOrdinal = i;
            else childIndices[i] = -1; // indication that Solution should not compute the projection for this child
          }
          if (childOrdinal == -1)
          {
            cout << "ERROR: child not found.\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "child not found!");
          }
    //      cout << "determining cellID " << parent->cellIndex() << "'s child " << childOrdinal << "'s coefficients.\n";
          soln->projectOldCellOntoNewCells(parent->cellIndex(), elementType(parent->cellIndex()), childIndices, lhsOrdinal);
        }
        
        
      }
    }
  }
}

void GlobalDofAssignment::setPartitions(Camellia::FieldContainer<GlobalIndexType> &partitionedMesh, bool rebuildDofLookups)
{
//  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndicesGlobal();

  int partitionNumber     = _partitionPolicy->Comm()->MyPID();
  int partitionCount      = _partitionPolicy->Comm()->NumProc();

  TEUCHOS_TEST_FOR_EXCEPTION(partitionedMesh.dimension(0) > partitionCount, std::invalid_argument,
                             "Number of partitions exceeds the maximum MPI rank; this is unsupported");

  //  cout << "determineActiveElements(): there are "  << activeCellIDs.size() << " active elements.\n";
  _partitions.clear();
  _partitionForCellID.clear();

  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i<partitionedMesh.dimension(0); i++)
  {
    set< GlobalIndexType > partition;
    for (int j=0; j<partitionedMesh.dimension(1); j++)
    {
      //      cout << "partitionedMesh(i,j) = " << partitionedMesh(i,j) << endl;
      if (partitionedMesh(i,j) == -1) break; // no more elements in this partition
      GlobalIndexType cellID = partitionedMesh(i,j);
      partition.insert( cellID );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
    //    if (partitionNumber==0) cout << "partition " << i << ": ";
    //    if (partitionNumber==0) print("",partition);
    if (partitionNumber > i)
    {
      _activeCellOffset += partition.size();
    }
  }
  
  constructActiveCellMap();
  if (rebuildDofLookups)
  {
    determineMinimumSubcellDimensionForContinuityEnforcement();
    projectParentCoefficientsOntoUnsetChildren();
    rebuildLookups();
  }
}

void GlobalDofAssignment::setPartitions(std::vector<std::set<GlobalIndexType> > &partitions, bool rebuildDofLookups)
{
  int thisPartitionNumber = _partitionPolicy->Comm()->MyPID();

  // not sure numProcs == partitions.size() is a great requirement to impose, but it is an assumption we make in some places,
  // so we require it here.
  int numProcs = _partitionPolicy->Comm()->NumProc();
  TEUCHOS_TEST_FOR_EXCEPTION(numProcs != partitions.size(), std::invalid_argument, "partitions.size() must be equal to numProcs!");

  _partitions = partitions;
  _partitionForCellID.clear();

  _activeCellOffset = 0;
  for (PartitionIndexType i=0; i< _partitions.size(); i++)
  {
    for (set< GlobalIndexType >::iterator cellIDIt = partitions[i].begin(); cellIDIt != partitions[i].end(); cellIDIt++)
    {
      GlobalIndexType cellID = *cellIDIt;
      _partitionForCellID[cellID] = i;
    }
    if (thisPartitionNumber > i)
    {
      _activeCellOffset += partitions[i].size();
    }
  }
  constructActiveCellMap();
  if (rebuildDofLookups)
  {
    determineMinimumSubcellDimensionForContinuityEnforcement();
    projectParentCoefficientsOntoUnsetChildren();
    rebuildLookups();
  }
}

void GlobalDofAssignment::setPartitionPolicy( MeshPartitionPolicyPtr partitionPolicy, bool shouldRepartitionAndMigrate )
{
  _partitionPolicy = partitionPolicy;
  if (shouldRepartitionAndMigrate)
  {
    repartitionAndMigrate();
  }
}

int GlobalDofAssignment::sideParity(GlobalIndexType cellID, unsigned int sideOrdinal)
{
  auto parityEntry = _cellSideParitiesForCellID.find(cellID);
  if (parityEntry == _cellSideParitiesForCellID.end())
  {
    assignParities(cellID);
    parityEntry = _cellSideParitiesForCellID.find(cellID);
    TEUCHOS_TEST_FOR_EXCEPTION(parityEntry == _cellSideParitiesForCellID.end(), std::invalid_argument, "No entry for parity found even after call to assignParities()");
  }
  
  const auto & parities = parityEntry->second;
  TEUCHOS_TEST_FOR_EXCEPTION(sideOrdinal >= parities.size(), std::invalid_argument, "sideOrdinal " + std::to_string(sideOrdinal) + " is out of bounds; there are " + std::to_string(parities.size()) + " sides…" );
  return parities[sideOrdinal];
}

PartitionIndexType GlobalDofAssignment::partitionForCellID( GlobalIndexType cellID )
{
  if (_partitionForCellID.find(cellID) != _partitionForCellID.end())
  {
    return _partitionForCellID[ cellID ];
  }
  else
  {
    return -1;
  }
}

IndexType GlobalDofAssignment::partitionLocalCellIndex(GlobalIndexType cellID, int partitionNumber)
{
  if (partitionNumber == -1)
  {
    partitionNumber     = _partitionPolicy->Comm()->MyPID();
  }

  ElementTypePtr elemType = elementType(cellID);
  vector<GlobalIndexType> cellIDsOfType = cellIDsOfElementType(partitionNumber, elemType);
  for (IndexType cellIndex = 0; cellIndex < cellIDsOfType.size(); cellIndex++)
  {
    if (cellIDsOfType[cellIndex] == cellID)
    {
      return cellIndex;
    }
  }
  return -1;
}

vector<TSolutionPtr<double>> GlobalDofAssignment::getRegisteredSolutions()
{
  return _registeredSolutions;
}

void GlobalDofAssignment::registerSolution(TSolutionPtr<double> solution)
{
  // Make a new, weak RCP, since Solution already has a pointer to Mesh, and Mesh has a pointer to us.
  TSolutionPtr<double> weakSolnPtr = Teuchos::rcp(solution.get(),false);
  _registeredSolutions.push_back( weakSolnPtr );
}

void GlobalDofAssignment::setMeshAndMeshTopology(MeshPtr mesh)
{
  // make copies of the RCPs that don't own memory.
  _mesh = Teuchos::rcp(mesh.get(), false);
  _meshTopology = Teuchos::rcp(mesh->getTopology().get(), false);

  this->DofInterpreter::_mesh = _mesh;
}

void GlobalDofAssignment::setPRefinementDegree(GlobalIndexType cellID, int deltaP)
{
  if (deltaP == 0)
  {
    _cellPRefinements.erase(cellID);
  }
  else
  {
    _cellPRefinements[cellID] = deltaP;
  }
}

void GlobalDofAssignment::unregisterSolution(TSolutionPtr<double> solution)
{
  for (vector< TSolutionPtr<double> >::iterator solnIt = _registeredSolutions.begin();
       solnIt != _registeredSolutions.end(); solnIt++)
  {
    if ( *solnIt == solution )
    {
      _registeredSolutions.erase(solnIt);
      return;
    }
  }
  cout << "GlobalDofAssignment::unregisterSolution: Solution not found.\n";
}

// maximumRule2D provides support for legacy (MultiBasis) meshes
GlobalDofAssignmentPtr GlobalDofAssignment::maximumRule2D(MeshPtr mesh, VarFactoryPtr varFactory,
    DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
    unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
{
  return Teuchos::rcp( new GDAMaximumRule2D(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}

GlobalDofAssignmentPtr GlobalDofAssignment::minimumRule(MeshPtr mesh, VarFactoryPtr varFactory,
    DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
    unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
{
  return Teuchos::rcp( new GDAMinimumRule(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement) );
}
