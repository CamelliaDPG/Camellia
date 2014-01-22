//
//  GDAMaximumRule2D.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMaximumRule2D.h"
#include "MultiBasis.h"
#include "BasisFactory.h" // TODO: make BasisFactory a member of DofOrderingFactory.

#include <Teuchos_GlobalMPISession.hpp>

GDAMaximumRule2D::GDAMaximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                   MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement, bool enforceMBFluxContinuity)
: GlobalDofAssignment(meshTopology,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement)
{
  _enforceMBFluxContinuity = enforceMBFluxContinuity;
  
  unsigned testOrder = initialH1OrderTrial + testOrderEnhancement;
  // assign some initial element types:
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();
  
  unsigned spaceDim = _meshTopology->getSpaceDim();
  unsigned sideDim = spaceDim - 1;
  
  map<unsigned, unsigned> sideIndexParityAssignmentCount; // tracks the number of times each side in the mesh has been assigned a parity.
  for (set<unsigned>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    unsigned cellID = *cellIDIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    if (cell->isParent() || (cell->getParent().get() != NULL)) {
      // enforcing this allows us to assume that each face that isn't on the boundary will be treated exactly twice...
      cout << "GDAMaximumRule2D constructor only supports mesh topologies that are unrefined.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GDAMaximumRule2D constructor only supports mesh topologies that are unrefined.\n");
    }
    DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(initialH1OrderTrial, *cell->topology());
    DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testOrder, *cell->topology());
    ElementTypePtr elemType = _elementTypeFactory.getElementType(trialOrdering,testOrdering,cell->topology());
    _elementTypeForCell[cellID] = elemType;
    
//    cout << "Assigned trialOrdering to cell " << cellID << ":\n" << *trialOrdering;
    
    unsigned sideCount = cell->topology()->getSideCount();
    vector<int> cellParities(sideCount);
    for (int sideIndex=0; sideIndex<sideCount; sideIndex++) {
      unsigned sideEntityIndex = cell->entityIndex(sideDim, sideIndex);
      if (sideIndexParityAssignmentCount[sideEntityIndex] == 0) {
        cellParities[sideIndex] = 1;
      } else if (sideIndexParityAssignmentCount[sideEntityIndex] == 1) {
        cellParities[sideIndex] = -1;
      } else {
        cout << "Internal error during GDAMaxRule2D construction: encountered side more than twice.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: encountered side more than twice.");
      }
    }
    _cellSideParitiesForCellID[cellID] = cellParities;
  }
  
  rebuildLookups();
}

void GDAMaximumRule2D::addDofPairing(int cellID1, int dofIndex1, int cellID2, int dofIndex2) {
  int firstCellID, firstDofIndex;
  int secondCellID, secondDofIndex;
  if (cellID1 < cellID2) {
    firstCellID = cellID1;
    firstDofIndex = dofIndex1;
    secondCellID = cellID2;
    secondDofIndex = dofIndex2;
  } else if (cellID1 > cellID2) {
    firstCellID = cellID2;
    firstDofIndex = dofIndex2;
    secondCellID = cellID1;
    secondDofIndex = dofIndex1;
  } else { // cellID1 == cellID2
    firstCellID = cellID1;
    secondCellID = cellID1;
    if (dofIndex1 < dofIndex2) {
      firstDofIndex = dofIndex1;
      secondDofIndex = dofIndex2;
    } else if (dofIndex1 > dofIndex2) {
      firstDofIndex = dofIndex2;
      secondDofIndex = dofIndex1;
    } else {
      // if we get here, the following test will throw an exception:
      TEUCHOS_TEST_FOR_EXCEPTION( ( dofIndex1 == dofIndex2 ) && ( cellID1 == cellID2 ),
                                 std::invalid_argument,
                                 "attempt to identify (cellID1, dofIndex1) with itself.");
      return; // to appease the compiler, which otherwise warns that variables will be used unset...
    }
  }
  pair<int,int> key = make_pair(secondCellID,secondDofIndex);
  pair<int,int> value = make_pair(firstCellID,firstDofIndex);
  if ( _dofPairingIndex.find(key) != _dofPairingIndex.end() ) {
    // we already have an entry for this key: need to fix the linked list so it goes from greatest to least
    pair<int,int> existing = _dofPairingIndex[key];
    pair<int,int> entry1, entry2, entry3;
    entry3 = key; // know this is the greatest of the three
    int existingCellID = existing.first;
    int newCellID = value.first;
    int existingDofIndex = existing.second;
    int newDofIndex = value.second;
    
    if (existingCellID < newCellID) {
      entry1 = existing;
      entry2 = value;
    } else if (existingCellID > newCellID) {
      entry1 = value;
      entry2 = existing;
    } else { // cellID1 == cellID2
      if (existingDofIndex < newDofIndex) {
        entry1 = existing;
        entry2 = value;
      } else if (existingDofIndex > newDofIndex) {
        entry1 = value;
        entry2 = existing;
      } else {
        // existing == value --> we're trying to create an entry that already exists
        return;
      }
    }
    // go entry3 -> entry2 -> entry1 -> whatever entry1 pointed at before
    _dofPairingIndex[entry3] = entry2;
    //    cout << "Added DofPairing (cellID,dofIndex) --> (earlierCellID,dofIndex) : (" << entry3.first;
    //    cout << "," << entry3.second << ") --> (";
    //    cout << entry2.first << "," << entry2.second << ")." << endl;
    // now, there may already be an entry for entry2, so best to use recursion:
    addDofPairing(entry2.first,entry2.second,entry1.first,entry1.second);
  } else {
    _dofPairingIndex[key] = value;
    //    cout << "Added DofPairing (cellID,dofIndex) --> (earlierCellID,dofIndex) : (";
    //    cout << secondCellID << "," << secondDofIndex << ") --> (";
    //    cout << firstCellID << "," << firstDofIndex << ")." << endl;
  }
}

void GDAMaximumRule2D::buildLocalToGlobalMap() {
  _localToGlobalMap.clear();
  set<unsigned>::iterator cellIDIt;
  
  determineDofPairings();
  
  int globalIndex = 0;
  vector< int > trialIDs = _varFactory.trialIDs();
  
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();
  
  for (cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    unsigned cellID = *cellIDIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);

      VarPtr trialVar = _varFactory.trial(trialID);
      bool isFluxOrTrace = (trialVar->varType() == FLUX) || (trialVar->varType() == TRACE);

      if (! isFluxOrTrace ) {
        // then all these dofs are interior, so there's no overlap with other elements...
        int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,0);
        for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
          int localDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,0);
          _localToGlobalMap[make_pair(cellID,localDofIndex)] = globalIndex;
          //          cout << "added localToGlobalMap(cellID=" << cellID << ", localDofIndex=" << localDofIndex;
          //          cout << ") = " << globalIndex << "\n";
          globalIndex++;
        }
      } else {
        int numSides = cell->topology()->getSideCount();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
            int myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
            pair<int, int> myKey = make_pair(cellID,myLocalDofIndex);
            pair<int, int> myValue;
            if ( _dofPairingIndex.find(myKey) != _dofPairingIndex.end() ) {
              int earlierCellID = _dofPairingIndex[myKey].first;
              int earlierLocalDofIndex = _dofPairingIndex[myKey].second;
              if (_localToGlobalMap.find(_dofPairingIndex[myKey]) == _localToGlobalMap.end() ) {
                // error: we haven't processed the earlier key yet...
                TEUCHOS_TEST_FOR_EXCEPTION( true,
                                           std::invalid_argument,
                                           "global indices are being processed out of order (should be by cellID, then localDofIndex).");
              } else {
                _localToGlobalMap[myKey] = _localToGlobalMap[_dofPairingIndex[myKey]];
                //                cout << "added flux/trace localToGlobalMap(cellID,localDofIndex)=(" << cellID << ", " << myLocalDofIndex;
                //                cout << ") = " << _localToGlobalMap[myKey] << "\n";
              }
            } else {
              // this test is necessary because for conforming traces, getDofIndex() may not be 1-1
              // and therefore we might have already assigned a globalDofIndex to myKey....
              if ( _localToGlobalMap.find(myKey) == _localToGlobalMap.end() ) {
                _localToGlobalMap[myKey] = globalIndex;
                //                cout << "added flux/trace localToGlobalMap(cellID,localDofIndex)=(" << cellID << ", " << myLocalDofIndex;
                //                cout << ") = " << globalIndex << "\n";
                globalIndex++;
              }
            }
          }
        }
      }
    }
  }
  _numGlobalDofs = globalIndex;
}

void GDAMaximumRule2D::buildTypeLookups() {
  _elementTypes.clear();
  _elementTypesForPartition.clear();
  _cellIDsForElementType.clear();
  _globalCellIndexToCellID.clear();
  _partitionedCellSideParitiesForElementType.clear();
  _partitionedPhysicalCellNodesForElementType.clear();
  set< ElementType* > elementTypeSet; // keep track of which ones we've seen globally (avoid duplicates in _elementTypes)
  map< ElementType*, int > globalCellIndices;
  
  for (int partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    _cellIDsForElementType.push_back( map< ElementType*, vector<int> >() );
    _elementTypesForPartition.push_back( vector< ElementTypePtr >() );
    _partitionedPhysicalCellNodesForElementType.push_back( map< ElementType*, FieldContainer<double> >() );
    _partitionedCellSideParitiesForElementType.push_back( map< ElementType*, FieldContainer<double> >() );
    vector< unsigned >::iterator elemIterator;
    
    // this should loop over the elements in the partition instead
    for (elemIterator=_partitions[partitionNumber].begin();
         elemIterator != _partitions[partitionNumber].end(); elemIterator++) {
      unsigned cellID = *elemIterator;
      ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
      if ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) == _cellIDsForElementType[partitionNumber].end() ) {
        _elementTypesForPartition[partitionNumber].push_back(elemTypePtr);
      }
      if (elementTypeSet.find( elemTypePtr.get() ) == elementTypeSet.end() ) {
        elementTypeSet.insert( elemTypePtr.get() );
        _elementTypes.push_back( elemTypePtr );
      }
      _cellIDsForElementType[partitionNumber][elemTypePtr.get()].push_back(cellID);
    }
    
    // now, build cellSideParities and physicalCellNodes lookups
    vector< ElementTypePtr >::iterator elemTypeIt;
    for (elemTypeIt=_elementTypesForPartition[partitionNumber].begin(); elemTypeIt != _elementTypesForPartition[partitionNumber].end(); elemTypeIt++) {
      //ElementTypePtr elemType = _elementTypeFactory.getElementType((*elemTypeIt)->trialOrderPtr,
      //                                                                 (*elemTypeIt)->testOrderPtr,
      //                                                                 (*elemTypeIt)->cellTopoPtr);
      ElementTypePtr elemType = *elemTypeIt; // don't enforce uniquing here (if we wanted to, we
      //   would also need to call elem.setElementType for
      //   affected elements...)
      int spaceDim = elemType->cellTopoPtr->getDimension();
      int numSides = elemType->cellTopoPtr->getSideCount();
      int numNodes = elemType->cellTopoPtr->getNodeCount();
      vector<int> cellIDs = _cellIDsForElementType[partitionNumber][elemType.get()];
      int numCells = cellIDs.size();
      FieldContainer<double> physicalCellNodes( numCells, numNodes, spaceDim ) ;
      FieldContainer<double> cellSideParities( numCells, numSides );
      vector<int>::iterator cellIt;
      int cellIndex = 0;
      
      // store physicalCellNodes:
      verticesForCells(physicalCellNodes, cellIDs);
      for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
        int cellID = *cellIt;
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          cellSideParities(cellIndex,sideIndex) = _cellSideParitiesForCellID[cellID][sideIndex];
        }
        _partitionLocalCellIndices[cellID] = cellIndex;
        unsigned globalCellIndex = globalCellIndices[elemType.get()]++;
        _globalCellIndices[cellID] = globalCellIndex;
        _globalCellIndexToCellID[elemType.get()][globalCellIndex] = cellID;
      }
      _partitionedPhysicalCellNodesForElementType[partitionNumber][elemType.get()] = physicalCellNodes;
      _partitionedCellSideParitiesForElementType[partitionNumber][elemType.get()] = cellSideParities;
    }
  }
  // finally, build _physicalCellNodesForElementType and _cellSideParitiesForElementType:
  _physicalCellNodesForElementType.clear();
  for (vector< ElementTypePtr >::iterator elemTypeIt = _elementTypes.begin();
       elemTypeIt != _elementTypes.end(); elemTypeIt++) {
    ElementType* elemType = elemTypeIt->get();
    int numCells = globalCellIndices[elemType];
    int spaceDim = elemType->cellTopoPtr->getDimension();
    int numSides = elemType->cellTopoPtr->getSideCount();
    _physicalCellNodesForElementType[elemType] = FieldContainer<double>(numCells,numSides,spaceDim);
  }
  // copy from the local (per-partition) FieldContainers to the global ones
  for (int partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    vector< ElementTypePtr >::iterator elemTypeIt;
    for (elemTypeIt  = _elementTypesForPartition[partitionNumber].begin();
         elemTypeIt != _elementTypesForPartition[partitionNumber].end(); elemTypeIt++) {
      ElementType* elemType = elemTypeIt->get();
      FieldContainer<double> partitionedPhysicalCellNodes = _partitionedPhysicalCellNodesForElementType[partitionNumber][elemType];
      FieldContainer<double> partitionedCellSideParities = _partitionedCellSideParitiesForElementType[partitionNumber][elemType];
      
      int numCells = partitionedPhysicalCellNodes.dimension(0);
      int numSides = partitionedPhysicalCellNodes.dimension(1);
      int spaceDim = partitionedPhysicalCellNodes.dimension(2);
      
      // this copying can be made more efficient by copying a whole FieldContainer at a time
      // (but it's probably not worth it, for now)
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = _cellIDsForElementType[partitionNumber][elemType][cellIndex];
        int globalCellIndex = _globalCellIndices[cellID];
        TEUCHOS_TEST_FOR_EXCEPTION( cellID != _globalCellIndexToCellID[elemType][globalCellIndex],
                                   std::invalid_argument, "globalCellIndex -> cellID inconsistency detected" );
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          for (int dim=0; dim<spaceDim; dim++) {
            _physicalCellNodesForElementType[elemType](globalCellIndex,sideIndex,dim)
            = partitionedPhysicalCellNodes(cellIndex,sideIndex,dim);
          }
        }
      }
    }
  }
}

void GDAMaximumRule2D::determineActiveElements() {
#ifdef HAVE_MPI
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
#else
  int partitionNumber     = 0;
#endif
  
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();
  _partitions.clear();
  _partitionForCellID.clear();
  FieldContainer<int> partitionedMesh(_numPartitions,activeCellIDs.size());
  _partitionPolicy->partitionMesh(_meshTopology.get(),_numPartitions,partitionedMesh);
  for (int i=0; i<partitionedMesh.dimension(0); i++) {
    vector< unsigned > partition;
    for (int j=0; j<partitionedMesh.dimension(1); j++) {
      if (partitionedMesh(i,j) < 0) break; // no more elements in this partition
      unsigned cellID = partitionedMesh(i,j);
      partition.push_back( cellID );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
  }
}

void GDAMaximumRule2D::determineDofPairings() {
  _dofPairingIndex.clear();
  
  vector< int > trialIDs = _varFactory.trialIDs();
  
  set<unsigned> activeCellIDs = _meshTopology->getActiveCellIndices();
  
  for (set<unsigned>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    unsigned cellID = *cellIDIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
    
    if ( cell->isParent() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"elemPtr is in _activeElements, but is a parent...");
    }
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      VarPtr trialVar = _varFactory.trial(trialID);
      bool isFluxOrTrace = (trialVar->varType() == FLUX) || (trialVar->varType() == TRACE);
      if ( isFluxOrTrace ) {
        int numSides = cell->topology()->getSideCount();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int myNumDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          pair<CellPtr, unsigned> neighborInfo = _meshTopology->getCellAncestralNeighbor(cellID, sideIndex);
          CellPtr neighbor = neighborInfo.first;
          int mySideIndexInNeighbor = neighborInfo.second;
          
          if (neighbor.get() != NULL) {
            // check that the bases agree in #dofs:
            bool hasMultiBasis = neighbor->isParent();
            
            unsigned neighborCellID = neighbor->cellIndex();
            if ( ! neighbor->isParent() ) {
              int neighborNumDofs = _elementTypeForCell[neighborCellID]->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              if ( !hasMultiBasis && (myNumDofs != neighborNumDofs) ) { // neither a multi-basis, and we differ: a problem
                TEUCHOS_TEST_FOR_EXCEPTION(myNumDofs != neighborNumDofs,
                                           std::invalid_argument,
                                           "Element and neighbor don't agree on basis along shared side.");
              }
            }
            
            // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
            //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendants along that side...
            vector< pair<unsigned,unsigned> > descendantsForSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor);
            vector< pair<unsigned,unsigned> >:: iterator entryIt;
            int descendantIndex = -1;
            for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
              descendantIndex++;
              int descendantSubSideIndexInMe = neighborChildPermutation(descendantIndex, descendantsForSide.size());
              neighborCellID = (*entryIt).first;
              mySideIndexInNeighbor = (*entryIt).second;
              neighbor = _meshTopology->getCell(neighborCellID);
              int neighborNumDofs = _elementTypeForCell[neighborCellID]->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              
              for (int dofOrdinal=0; dofOrdinal<neighborNumDofs; dofOrdinal++) {
                int myLocalDofIndex;
                if ( descendantsForSide.size() > 1 ) {
                  // multi-basis
                  myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex,descendantSubSideIndexInMe);
                } else {
                  myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
                }
                
                // neighbor's dofs are in reverse order from mine along each side
                // TODO: generalize this to some sort of permutation for 3D meshes...
                int permutedDofOrdinal = neighborDofPermutation(dofOrdinal,neighborNumDofs);
                
                int neighborLocalDofIndex = _elementTypeForCell[neighborCellID]->trialOrderPtr->getDofIndex(trialID,permutedDofOrdinal,mySideIndexInNeighbor);
                addDofPairing(cellID, myLocalDofIndex, neighborCellID, neighborLocalDofIndex);
              }
            }
            
            if ( hasMultiBasis && _enforceMBFluxContinuity ) {
              // marry the last node from one MB leaf to first node from the next
              // note that we're doing this for both traces and fluxes, but with traces this is redundant
              BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID,sideIndex);
              MultiBasis<>* multiBasis = (MultiBasis<> *) basis.get(); // Dynamic cast would be better
              vector< pair<int,int> > adjacentDofOrdinals = multiBasis->adjacentVertexOrdinals();
              for (vector< pair<int,int> >::iterator dofPairIt = adjacentDofOrdinals.begin();
                   dofPairIt != adjacentDofOrdinals.end(); dofPairIt++) {
                int firstOrdinal  = dofPairIt->first;
                int secondOrdinal = dofPairIt->second;
                int firstDofIndex  = elemTypePtr->trialOrderPtr->getDofIndex(trialID,firstOrdinal, sideIndex);
                int secondDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,secondOrdinal,sideIndex);
                addDofPairing(cellID,firstDofIndex, cellID, secondDofIndex);
              }
            }
          }
        }
      }
    }
  }
  // now that we have all the dofPairings, reduce the map so that all the pairings point at the earliest paired index
  for (set<unsigned>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    int cellID = *cellIDIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      
      VarPtr trialVar = _varFactory.trial(trialID);
      bool isFluxOrTrace = (trialVar->varType() == FLUX) || (trialVar->varType() == TRACE);
      
      if ( isFluxOrTrace ) {
        int numSides = cell->topology()->getSideCount();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
            int myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
            pair<int, int> myKey = make_pair(cellID,myLocalDofIndex);
            if (_dofPairingIndex.find(myKey) != _dofPairingIndex.end()) {
              pair<int, int> myValue = _dofPairingIndex[myKey];
              while (_dofPairingIndex.find(myValue) != _dofPairingIndex.end()) {
                myValue = _dofPairingIndex[myValue];
              }
              _dofPairingIndex[myKey] = myValue;
            }
          }
        }
      }
    }
  }
}

void GDAMaximumRule2D::determinePartitionDofIndices() {
  _partitionedGlobalDofIndices.clear();
  _partitionForGlobalDofIndex.clear();
  _partitionLocalIndexForGlobalDofIndex.clear();
  set<int> dofIndices;
  set<int> previouslyClaimedDofIndices;
  for (int i=0; i<_numPartitions; i++) {
    dofIndices.clear();
    vector< unsigned >::iterator cellIDIt;
    for (cellIDIt =  _partitions[i].begin(); cellIDIt != _partitions[i].end(); cellIDIt++) {
      unsigned cellID = *cellIDIt;
      ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
      int numLocalDofs = elemTypePtr->trialOrderPtr->totalDofs();
      for (int localDofIndex=0; localDofIndex < numLocalDofs; localDofIndex++) {
        pair<unsigned,unsigned> key = make_pair(cellID, localDofIndex);
        map< pair<unsigned,unsigned>, unsigned >::iterator mapEntryIt = _localToGlobalMap.find(key);
        if ( mapEntryIt == _localToGlobalMap.end() ) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "entry not found.");
        }
        int dofIndex = (*mapEntryIt).second;
        if ( previouslyClaimedDofIndices.find( dofIndex ) == previouslyClaimedDofIndices.end() ) {
          dofIndices.insert( dofIndex );
          _partitionForGlobalDofIndex[ dofIndex ] = i;
          previouslyClaimedDofIndices.insert(dofIndex);
        }
      }
    }
    _partitionedGlobalDofIndices.push_back( dofIndices );
    int partitionDofIndex = 0;
    for (set<int>::iterator dofIndexIt = dofIndices.begin();
         dofIndexIt != dofIndices.end(); dofIndexIt++) {
      int globalDofIndex = *dofIndexIt;
      _partitionLocalIndexForGlobalDofIndex[globalDofIndex] = partitionDofIndex++;
    }
  }
}

void GDAMaximumRule2D::didChangePartitionPolicy() {
  rebuildLookups();
}

void GDAMaximumRule2D::didHRefine(set<int> &parentCellIDs) {
  rebuildLookups();
}

void GDAMaximumRule2D::didPRefine(set<int> &cellIDs, int deltaP) {
  
  rebuildLookups();
}

void GDAMaximumRule2D::didHUnrefine(set<int> &parentCellIDs) {
  rebuildLookups();
}

ElementTypePtr GDAMaximumRule2D::elementType(unsigned cellID) {
  return _elementTypeForCell[cellID];
}

void GDAMaximumRule2D::getMultiBasisOrdering(DofOrderingPtr &originalNonParentOrdering,
                                 CellPtr parent, unsigned sideIndex, unsigned parentSideIndexInNeighbor,
                                 CellPtr nonParent) {
  unsigned nonParentPolyOrder = _dofOrderingFactory->trialPolyOrder( _elementTypeForCell[nonParent->cellIndex()]->trialOrderPtr );

  map< int, BasisPtr > varIDsToUpgrade = multiBasisUpgradeMap(parent,sideIndex,nonParentPolyOrder);
  originalNonParentOrdering = _dofOrderingFactory->upgradeSide(originalNonParentOrdering,
                                                              *nonParent->topology(),
                                                              varIDsToUpgrade,parentSideIndexInNeighbor);
}

unsigned GDAMaximumRule2D::globalDofCount() {
  return _numGlobalDofs;
}

unsigned GDAMaximumRule2D::localDofCount() {
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

void GDAMaximumRule2D::matchNeighbor(unsigned cellID, int sideIndex) {
  // sets new ElementType to match elem to neighbor on side sideIndex
  
  CellPtr cell = _meshTopology->getCell(cellID);
  const shards::CellTopology cellTopo = *cell->topology();
  
  pair< CellPtr, unsigned > neighborRecord = _meshTopology->getCellAncestralNeighbor(cellID, sideIndex);
  CellPtr neighbor = neighborRecord.first;
  unsigned sideIndexInNeighbor = neighborRecord.second;
  
  if (neighbor.get() == NULL) {
    // no neighbors (boundary): return
    return;
  }
  
  unsigned neighborCellID = neighbor->cellIndex();
  
  // h-refinement handling:
  bool neighborIsBroken = (neighbor->isParent() && (neighbor->childrenForSide(sideIndexInNeighbor).size() > 1));
  bool elementIsBroken  = (cell->isParent() && (neighbor->childrenForSide(sideIndex).size() > 1));
  if ( neighborIsBroken || elementIsBroken ) {
    bool bothBroken = ( neighborIsBroken && elementIsBroken );
    CellPtr nonParent, parent; // for the case that one is a parent and the other isn't
    int parentSideIndexInNeighbor, neighborSideIndexInParent;
    if ( !bothBroken ) {
      if (! elementIsBroken ) {
        nonParent = cell;
        parent = neighbor;
        parentSideIndexInNeighbor = sideIndex;
        neighborSideIndexInParent = sideIndexInNeighbor;
      } else {
        nonParent = neighbor;
        parent = cell;
        parentSideIndexInNeighbor = sideIndexInNeighbor;
        neighborSideIndexInParent = sideIndex;
      }
    }
    
    if (bothBroken) {
      // match all the children -- we assume RefinementPatterns are compatible (e.g. divisions always by 1/2s)
      vector< pair<unsigned,unsigned> > childrenForSide = cell->childrenForSide(sideIndex);
      for (int childIndexInSide=0; childIndexInSide < childrenForSide.size(); childIndexInSide++) {
        int childIndex = childrenForSide[childIndexInSide].first;
        int childSideIndex = childrenForSide[childIndexInSide].second;
        matchNeighbor(cell->children()[childIndex]->cellIndex(),childSideIndex);
      }
      // all our children matched => we're done:
      return;
    } else { // one broken
      vector< pair< unsigned, unsigned> > childrenForSide = parent->childrenForSide(neighborSideIndexInParent);
      
      if ( childrenForSide.size() > 1 ) { // then parent is broken along side, and neighbor isn't...
        { // MultiBasis
          Teuchos::RCP<DofOrdering> nonParentTrialOrdering = _elementTypeForCell[nonParent->cellIndex()]->trialOrderPtr;
          
          getMultiBasisOrdering( nonParentTrialOrdering, parent, neighborSideIndexInParent,
                                parentSideIndexInNeighbor, nonParent );
          ElementTypePtr nonParentType = _elementTypeFactory.getElementType(nonParentTrialOrdering,
                                                                            _elementTypeForCell[nonParent->cellIndex()]->testOrderPtr,
                                                                            nonParent->topology() );
          setElementType(nonParent->cellIndex(), nonParentType, true); // true: only a side upgrade
          //nonParent->setElementType(nonParentType);
          // debug code:
          if ( nonParentTrialOrdering->hasSideVarIDs() ) { // then we can check whether there is a multi-basis
            if ( ! _dofOrderingFactory->sideHasMultiBasis(nonParentTrialOrdering, parentSideIndexInNeighbor) ) {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "failed to add multi-basis to neighbor");
            }
          }
        }
        
        // by virtue of having assigned the multi-basis, we've already matched p-order ==> we're done
        return;
      }
    }
  }
  // p-refinement handling:
  const shards::CellTopology neighborTopo = *neighbor->topology();
  Teuchos::RCP<DofOrdering> elemTrialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  Teuchos::RCP<DofOrdering> elemTestOrdering = _elementTypeForCell[cellID]->testOrderPtr;
  
  Teuchos::RCP<DofOrdering> neighborTrialOrdering = _elementTypeForCell[neighborCellID]->trialOrderPtr;
  Teuchos::RCP<DofOrdering> neighborTestOrdering = _elementTypeForCell[neighborCellID]->testOrderPtr;
  
  int changed = _dofOrderingFactory->matchSides(elemTrialOrdering, sideIndex, cellTopo,
                                                neighborTrialOrdering, sideIndexInNeighbor, neighborTopo);
  // changed == 1 for me, 2 for neighbor, 0 for neither, -1 for PatchBasis
  if (changed==1) {
    vector< VarPtr > traces = _varFactory.traceVars();
    vector< VarPtr > fluxes = _varFactory.fluxVars();
    vector< VarPtr > tracesAndFluxes(traces.begin(),traces.end());
    tracesAndFluxes.insert(tracesAndFluxes.end(), fluxes.begin(), fluxes.end());
    int fluxTraceCount = tracesAndFluxes.size();
    TEUCHOS_TEST_FOR_EXCEPTION(fluxTraceCount == 0,
                               std::invalid_argument,
                               "BilinearForm has no traces or fluxes, but somehow element was upgraded...");
    
    int boundaryVarID = tracesAndFluxes[0]->ID();
    int neighborSidePolyOrder = BasisFactory::basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,sideIndexInNeighbor));
    int mySidePolyOrder = BasisFactory::basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,sideIndex));
    TEUCHOS_TEST_FOR_EXCEPTION(mySidePolyOrder != neighborSidePolyOrder,
                               std::invalid_argument,
                               "After matchSides(), the appropriate sides don't have the same order.");
    int testPolyOrder = _dofOrderingFactory->testPolyOrder(elemTestOrdering);
    if (testPolyOrder < mySidePolyOrder + _testOrderEnhancement) {
      elemTestOrdering = _dofOrderingFactory->testOrdering( mySidePolyOrder + _testOrderEnhancement, cellTopo);
    }
    ElementTypePtr newType = _elementTypeFactory.getElementType(elemTrialOrdering, elemTestOrdering,
                                                                cell->topology() );
    setElementType(cellID, newType, true); // true:
    //    elem->setElementType( _elementTypeFactory.getElementType(elemTrialOrdering, elemTestOrdering,
    //                                                             elem->elementType()->cellTopoPtr ) );
    //return ELEMENT_NEEDED_NEW;
  } else if (changed==2) {
    // if need be, upgrade neighborTestOrdering as well.
    vector< VarPtr > traces = _varFactory.traceVars();
    vector< VarPtr > fluxes = _varFactory.fluxVars();
    vector< VarPtr > tracesAndFluxes(traces.begin(),traces.end());
    tracesAndFluxes.insert(tracesAndFluxes.end(), fluxes.begin(), fluxes.end());
    int fluxTraceCount = tracesAndFluxes.size();
    TEUCHOS_TEST_FOR_EXCEPTION(fluxTraceCount == 0,
                               std::invalid_argument,
                               "BilinearForm has no traces or fluxes, but somehow neighbor was upgraded...");
    TEUCHOS_TEST_FOR_EXCEPTION(neighborTrialOrdering.get() == _elementTypeForCell[neighborCellID]->trialOrderPtr.get(),
                               std::invalid_argument,
                               "neighborTrialOrdering was supposed to be upgraded, but remains unchanged...");
    int boundaryVarID = tracesAndFluxes[0]->ID();
    int sidePolyOrder = BasisFactory::basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,sideIndexInNeighbor));
    int mySidePolyOrder = BasisFactory::basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,sideIndex));
    TEUCHOS_TEST_FOR_EXCEPTION(mySidePolyOrder != sidePolyOrder,
                               std::invalid_argument,
                               "After matchSides(), the appropriate sides don't have the same order.");
    int testPolyOrder = _dofOrderingFactory->testPolyOrder(neighborTestOrdering);
    if (testPolyOrder < sidePolyOrder + _testOrderEnhancement) {
      neighborTestOrdering = _dofOrderingFactory->testOrdering( sidePolyOrder + _testOrderEnhancement, neighborTopo);
    }
    ElementTypePtr newType = _elementTypeFactory.getElementType(neighborTrialOrdering, neighborTestOrdering,
                                                                neighbor->topology() );
    setElementType( neighborCellID, newType, true); // true: sideUpgradeOnly
    //return NEIGHBOR_NEEDED_NEW;
  } else if (changed == -1) { // PatchBasis
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "PatchBasis not supported by GDAMaximumRule2D.  Use GDAMinimumRule.");
  } else {
    //return NEITHER_NEEDED_NEW;
  }
}

map< int, BasisPtr > GDAMaximumRule2D::multiBasisUpgradeMap(CellPtr parent, unsigned sideIndex, unsigned bigNeighborPolyOrder) {
  vector< pair< unsigned, unsigned> > childrenForSide = parent->childrenForSide(sideIndex);
  map< int, BasisPtr > varIDsToUpgrade;
  vector< map< int, BasisPtr > > childVarIDsToUpgrade;
  vector< pair< unsigned, unsigned> >::iterator entryIt;
  for ( entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
    int childCellIndex = (*entryIt).first;
    int childSideIndex = (*entryIt).second;
    CellPtr childCell = parent->children()[childCellIndex];
    
    while (childCell->isParent() && (childCell->childrenForSide(childSideIndex).size() == 1) ) {
      pair<unsigned, unsigned> childEntry = childCell->childrenForSide(childSideIndex)[0];
      childSideIndex = childEntry.second;
      childCell = _meshTopology->getCell(childEntry.first);
    }
    
    if ( childCell->isParent() && (childCell->childrenForSide(childSideIndex).size() > 1)) {
      childVarIDsToUpgrade.push_back( multiBasisUpgradeMap(childCell,childSideIndex,bigNeighborPolyOrder) );
    } else {
      DofOrderingPtr childTrialOrder = _elementTypeForCell[childCell->cellIndex()]->trialOrderPtr;
      
      int childPolyOrder = _dofOrderingFactory->trialPolyOrder( childTrialOrder );
      
      if (bigNeighborPolyOrder > childPolyOrder) {
        // upgrade child p along side
        // NOTE: THIS is ugly--a side effect
        childTrialOrder = _dofOrderingFactory->setSidePolyOrder(childTrialOrder, childSideIndex, bigNeighborPolyOrder, false);
        ElementTypePtr newChildType = _elementTypeFactory.getElementType(childTrialOrder,
                                                                         _elementTypeForCell[childCell->cellIndex()]->testOrderPtr,
                                                                         _elementTypeForCell[childCell->cellIndex()]->cellTopoPtr );
        setElementType(childCell->cellIndex(), newChildType, true); // true: only a side upgrade
      }
      
      pair< DofOrderingPtr,int > entry = make_pair(childTrialOrder,childSideIndex);
      vector< pair< DofOrderingPtr,int > > childTrialOrdersForSide;
      childTrialOrdersForSide.push_back(entry);
      childVarIDsToUpgrade.push_back( _dofOrderingFactory->getMultiBasisUpgradeMap(childTrialOrdersForSide) );
    }
  }
  map< int, BasisPtr >::iterator varMapIt;
  for (varMapIt=childVarIDsToUpgrade[0].begin(); varMapIt != childVarIDsToUpgrade[0].end(); varMapIt++) {
    int varID = (*varMapIt).first;
    vector< BasisPtr > bases;
    int numChildrenInSide = childVarIDsToUpgrade.size();
    for (int childIndex = 0; childIndex<numChildrenInSide; childIndex++) {
      //permute child index (this is from neighbor's point of view)
      int permutedChildIndex = neighborChildPermutation(childIndex,numChildrenInSide);
      if (! childVarIDsToUpgrade[permutedChildIndex][varID].get()) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "null basis");
      }
      bases.push_back(childVarIDsToUpgrade[permutedChildIndex][varID]);
    }
    BasisPtr multiBasis = BasisFactory::getMultiBasis(bases);
    // debugging:
    //    ((MultiBasis*)multiBasis.get())->printInfo();
    varIDsToUpgrade[varID] = multiBasis;
  }
  return varIDsToUpgrade;
}

int GDAMaximumRule2D::neighborChildPermutation(int childIndex, int numChildrenInSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numChildrenInSide - childIndex - 1;
}

int GDAMaximumRule2D::neighborDofPermutation(int dofIndex, int numDofsForSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numDofsForSide - dofIndex - 1;
}

void GDAMaximumRule2D::rebuildLookups() {
  _cellSideUpgrades.clear();
  determineActiveElements();
  buildTypeLookups(); // build data structures for efficient lookup by element type
  buildLocalToGlobalMap();
  determinePartitionDofIndices();
}

void GDAMaximumRule2D::setElementType(unsigned cellID, ElementTypePtr newType, bool sideUpgradeOnly) {
  CellPtr cell = _meshTopology->getCell(cellID);
  if (sideUpgradeOnly) { // need to track in _cellSideUpgrades
    ElementTypePtr oldType;
    map<int, pair<ElementTypePtr, ElementTypePtr> >::iterator existingEntryIt = _cellSideUpgrades.find(cellID);
    if (existingEntryIt != _cellSideUpgrades.end() ) {
      oldType = (existingEntryIt->second).first;
    } else {
      oldType = _elementTypeForCell[cellID];
      if (oldType.get() == newType.get()) {
        // no change is actually happening
        return;
      }
    }
    //    cout << "setting element type for cellID " << cellID << " (sideUpgradeOnly=" << sideUpgradeOnly << ")\n";
    //    cout << "trialOrder old size: " << oldType->trialOrderPtr->totalDofs() << endl;
    //    cout << "trialOrder new size: " << newType->trialOrderPtr->totalDofs() << endl;
    _cellSideUpgrades[cellID] = make_pair(oldType,newType);
  }
  _elementTypeForCell[cellID] = newType;
}


void GDAMaximumRule2D::verticesForCell(FieldContainer<double>& vertices, int cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);
  vector<unsigned> vertexIndices = cell->vertices();
  int numVertices = vertexIndices.size();
  int spaceDim = _meshTopology->getSpaceDim();
  
  //vertices.resize(numVertices,dimension);
  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    vector<double> vertex = _meshTopology->getVertex(vertexIndices[vertexIndex]);
    for (int d=0; d<spaceDim; d++) {
      vertices(vertexIndex,d) = vertex[d];
    }
  }
}

void GDAMaximumRule2D::verticesForCells(FieldContainer<double>& vertices, vector<int> &cellIDs) {
  // all cells represented in cellIDs must have the same topology
  int spaceDim = _meshTopology->getSpaceDim();
  int numCells = cellIDs.size();
  
  if (numCells == 0) {
    vertices.resize(0,0,0);
    return;
  }
  unsigned firstCellID = cellIDs[0];
  int numVertices = _meshTopology->getCell(firstCellID)->vertices().size();
  
  vertices.resize(numCells,numVertices,spaceDim);
  
  Teuchos::Array<int> dim; // for an individual cell
  dim.push_back(numVertices);
  dim.push_back(spaceDim);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = cellIDs[cellIndex];
    FieldContainer<double> cellVertices(dim,&vertices(cellIndex,0,0));
    this->verticesForCell(cellVertices, cellID);
    //    cout << "vertices for cellID " << cellID << ":\n" << cellVertices;
  }
  
  //  cout << "all vertices:\n" << vertices;
}