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
#include "Solution.h"

#include "CamelliaCellTools.h"

#include <Teuchos_GlobalMPISession.hpp>

GDAMaximumRule2D::GDAMaximumRule2D(MeshPtr mesh, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                   MeshPartitionPolicyPtr partitionPolicy, unsigned initialH1OrderTrial, unsigned testOrderEnhancement, bool enforceMBFluxContinuity)
: GlobalDofAssignment(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement, true)
{
//  cout << "Entered constructor of GDAMaximumRule2D.\n";
  _enforceMBFluxContinuity = enforceMBFluxContinuity;
}

void GDAMaximumRule2D::addDofPairing(GlobalIndexType cellID1, IndexType dofIndex1, GlobalIndexType cellID2, IndexType dofIndex2) {
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
  pair<GlobalIndexType,IndexType> key = make_pair(secondCellID,secondDofIndex);
  pair<GlobalIndexType,IndexType> value = make_pair(firstCellID,firstDofIndex);
  if ( _dofPairingIndex.find(key) != _dofPairingIndex.end() ) {
    // we already have an entry for this key: need to fix the linked list so it goes from greatest to least
    pair<GlobalIndexType,IndexType> existing = _dofPairingIndex[key];
    pair<GlobalIndexType,IndexType> entry1, entry2, entry3;
    entry3 = key; // know this is the greatest of the three
    GlobalIndexType existingCellID = existing.first;
    GlobalIndexType newCellID = value.first;
    IndexType existingDofIndex = existing.second;
    IndexType newDofIndex = value.second;
    
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
  set<GlobalIndexType>::iterator cellIDIt;
  
  determineDofPairings();
  
  GlobalIndexType globalIndex = 0;
  vector< int > trialIDs = _varFactory.trialIDs();
  
  set<IndexType> activeCellIndices = _meshTopology->getActiveCellIndices();
  set<GlobalIndexType> activeCellIDs(activeCellIndices.begin(),activeCellIndices.end());
  
  for (cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
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
        int numSides = CamelliaCellTools::getSideCount(*cell->topology());
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
            int myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
            pair<int, int> myKey = make_pair(cellID,myLocalDofIndex);
            pair<int, int> myValue;
            if ( _dofPairingIndex.find(myKey) != _dofPairingIndex.end() ) {
//              GlobalIndexType earlierCellID = _dofPairingIndex[myKey].first;
//              GlobalIndexType earlierLocalDofIndex = _dofPairingIndex[myKey].second;
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
//  cout << "GDAMaximumRule2D::buildTypeLookups().\n";
  _elementTypesForPartition.clear();
  _cellIDsForElementType.clear();
  _globalCellIndexToCellID.clear();
  _partitionedCellSideParitiesForElementType.clear();
  _partitionedPhysicalCellNodesForElementType.clear();
  _elementTypeList.clear();  // it appears to be important not to clear this before the guys that have ElementType* entries.  (Otherwise, we sometimes get a seg fault, presumably because the RCP has freed the memory pointed to, and the attempt to clear a map< ElementType*, blah > or whatever appears to be an attempt to access the freed memory??  I'm not sure.  What I know is that we had a crash that would occur in MeshTestSuite::testJesseAnisotropicRefinement(), and when I moved this from being the first cleared container to being the last, that crash went away...)
  set< ElementType* > elementTypeSet; // keep track of which ones we've seen globally (avoid duplicates in _elementTypeList)
  map< ElementType*, GlobalIndexType > globalCellIndices;
  
//  cout << "_numPartitions = " << _numPartitions << endl;
  
//  int rank = Teuchos::GlobalMPISession::getRank();
  
  GlobalIndexType totalCellCount = _meshTopology->cellCount();
  
  for (PartitionIndexType partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    _cellIDsForElementType.push_back( map< ElementType*, vector<GlobalIndexType> >() );
    _elementTypesForPartition.push_back( vector< ElementTypePtr >() );
    _partitionedPhysicalCellNodesForElementType.push_back( map< ElementType*, FieldContainer<double> >() );
    _partitionedCellSideParitiesForElementType.push_back( map< ElementType*, FieldContainer<double> >() );
    set< GlobalIndexType >::const_iterator elemIterator;
    
//    cout << "_partitions[" << partitionNumber << "] has " << _partitions[partitionNumber].size() << " cells.\n";
    
    // this should loop over the elements in the partition instead
    for (elemIterator=_partitions[partitionNumber].begin();
         elemIterator != _partitions[partitionNumber].end(); elemIterator++) {
      GlobalIndexType cellID = *elemIterator;
      if (cellID > totalCellCount) {
        cout << "cellID " << cellID << " is out of range (0," << totalCellCount << ").\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID is out of range.\n");
      }
      ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
      if ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) == _cellIDsForElementType[partitionNumber].end() ) {
        _elementTypesForPartition[partitionNumber].push_back(elemTypePtr);
      }
      if (elementTypeSet.find( elemTypePtr.get() ) == elementTypeSet.end() ) {
        elementTypeSet.insert( elemTypePtr.get() );
        _elementTypeList.push_back( elemTypePtr );
      }
//      cout << "adding cellID " << cellID << " to partition " << partitionNumber << " and element type " << elemTypePtr.get() << endl;
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
      int numSides = CamelliaCellTools::getSideCount(*elemType->cellTopoPtr);
      int numNodes = elemType->cellTopoPtr->getNodeCount();
      vector<GlobalIndexType> cellIDs = _cellIDsForElementType[partitionNumber][elemType.get()];
      GlobalIndexType numCells = cellIDs.size();
      FieldContainer<double> physicalCellNodes( numCells, numNodes, spaceDim ) ;
      FieldContainer<double> cellSideParities( numCells, numSides );
      vector<GlobalIndexType>::iterator cellIt;
      GlobalIndexType cellIndex = 0;
      
      // store physicalCellNodes:
      verticesForCells(physicalCellNodes, cellIDs);
      for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
        int cellID = *cellIt;
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          cellSideParities(cellIndex,sideIndex) = _cellSideParitiesForCellID[cellID][sideIndex];
        }
        _partitionLocalCellIndices[cellID] = cellIndex++;
        unsigned globalCellIndex = globalCellIndices[elemType.get()]++;
        _globalCellIndices[cellID] = globalCellIndex;
        _globalCellIndexToCellID[elemType.get()][globalCellIndex] = cellID;
//        cout << "On rank " << rank << ", setting _globalCellIndexToCellID[" << elemType.get() << "][" << globalCellIndex << "] = " << cellID << endl;
      }
      _partitionedPhysicalCellNodesForElementType[partitionNumber][elemType.get()] = physicalCellNodes;
      _partitionedCellSideParitiesForElementType[partitionNumber][elemType.get()] = cellSideParities;
    }
  }
  // finally, build _physicalCellNodesForElementType and _cellSideParitiesForElementType:
  _physicalCellNodesForElementType.clear();
  for (vector< ElementTypePtr >::iterator elemTypeIt = _elementTypeList.begin();
       elemTypeIt != _elementTypeList.end(); elemTypeIt++) {
    ElementType* elemType = elemTypeIt->get();
    GlobalIndexType numCells = globalCellIndices[elemType];
    int spaceDim = elemType->cellTopoPtr->getDimension();
    int numSides = CamelliaCellTools::getSideCount(*elemType->cellTopoPtr);
    _physicalCellNodesForElementType[elemType] = FieldContainer<double>(numCells,numSides,spaceDim);
  }
  // copy from the local (per-partition) FieldContainers to the global ones
  for (PartitionIndexType partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    vector< ElementTypePtr >::iterator elemTypeIt;
    for (elemTypeIt  = _elementTypesForPartition[partitionNumber].begin();
         elemTypeIt != _elementTypesForPartition[partitionNumber].end(); elemTypeIt++) {
      ElementType* elemType = elemTypeIt->get();
      FieldContainer<double> partitionedPhysicalCellNodes = _partitionedPhysicalCellNodesForElementType[partitionNumber][elemType];
      FieldContainer<double> partitionedCellSideParities = _partitionedCellSideParitiesForElementType[partitionNumber][elemType];
      
      IndexType numCells = partitionedPhysicalCellNodes.dimension(0);
      int numSides = partitionedPhysicalCellNodes.dimension(1);
      int spaceDim = partitionedPhysicalCellNodes.dimension(2);
      
      // this copying can be made more efficient by copying a whole FieldContainer at a time
      // (but it's probably not worth it, for now)
      for (IndexType cellIndex=0; cellIndex<numCells; cellIndex++) {
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

//GlobalIndexType GDAMaximumRule2D::cellID(Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, PartitionIndexType partitionNumber) {
//  if (partitionNumber == -1) {
//    if (( _globalCellIndexToCellID.find( elemTypePtr.get() ) != _globalCellIndexToCellID.end() ) &&
//        ( _globalCellIndexToCellID[elemTypePtr.get()].find( cellIndex ) != _globalCellIndexToCellID[elemTypePtr.get()].end() ) )
//      
//      return _globalCellIndexToCellID[elemTypePtr.get()][ cellIndex ];
//    else {
//      cout << "did not find global cellIndex " << cellIndex << " for element type " << elemTypePtr.get() << endl;
//      return -1;
//    }
//  } else {
//    if ( ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) != _cellIDsForElementType[partitionNumber].end() )
//        &&
//        (_cellIDsForElementType[partitionNumber][elemTypePtr.get()].size() > cellIndex ) ) {
//      return _cellIDsForElementType[partitionNumber][elemTypePtr.get()][cellIndex];
//    } else return -1;
//  }
//}

int GDAMaximumRule2D::cellPolyOrder(GlobalIndexType cellID) {
  return _dofOrderingFactory->trialPolyOrder(_elementTypeForCell[cellID]->trialOrderPtr);
}

FieldContainer<double> & GDAMaximumRule2D::cellSideParities( ElementTypePtr elemTypePtr ) {
#ifdef HAVE_MPI
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
#else
  int partitionNumber     = 0;
#endif
  return _partitionedCellSideParitiesForElementType[ partitionNumber ][ elemTypePtr.get() ];
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
        int numSides = CamelliaCellTools::getSideCount(*cell->topology());
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int myNumDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          pair<unsigned, unsigned> neighborInfo = cell->getNeighbor(sideIndex);
          unsigned neighborCellID = neighborInfo.first;
          unsigned mySideIndexInNeighbor = neighborInfo.second;
          
          bool neighborIsPeer = (neighborCellID != -1) && (_meshTopology->getCell(neighborCellID)->getNeighbor(mySideIndexInNeighbor).first==cellID);
          
          if (neighborIsPeer) {
            CellPtr neighbor = _meshTopology->getCell(neighborCellID);
            // check that the bases agree in #dofs:
            bool hasMultiBasis = neighbor->isParent();
            
            unsigned neighborCellID = neighbor->cellIndex();
            if ( ! neighbor->isParent() ) {
              int neighborNumDofs = _elementTypeForCell[neighborCellID]->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              if ( !hasMultiBasis && (myNumDofs != neighborNumDofs) ) { // neither a multi-basis, and we differ: a problem
                cout << "Element and neighbor don't agree on basis along shared side.\n";
                TEUCHOS_TEST_FOR_EXCEPTION(myNumDofs != neighborNumDofs,
                                           std::invalid_argument,
                                           "Element and neighbor don't agree on basis along shared side.");
              }
            }
            
            // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
            //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendants along that side...
            vector< pair<GlobalIndexType,unsigned> > descendantsForSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor);
            vector< pair<GlobalIndexType,unsigned> >:: iterator entryIt;
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
        int numSides = CamelliaCellTools::getSideCount(*cell->topology());
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
  set<GlobalIndexType> dofIndices;
  set<GlobalIndexType> previouslyClaimedDofIndices;
  for (int i=0; i<_numPartitions; i++) {
    dofIndices.clear();
    set< GlobalIndexType >::const_iterator cellIDIt;
    for (cellIDIt =  _partitions[i].begin(); cellIDIt != _partitions[i].end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      ElementTypePtr elemTypePtr = _elementTypeForCell[cellID];
      int numLocalDofs = elemTypePtr->trialOrderPtr->totalDofs();
      for (int localDofIndex=0; localDofIndex < numLocalDofs; localDofIndex++) {
        pair<GlobalIndexType,IndexType> key = make_pair(cellID, localDofIndex);
        map< pair<GlobalIndexType,IndexType>, GlobalIndexType >::iterator mapEntryIt = _localToGlobalMap.find(key);
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
    IndexType partitionDofIndex = 0;
    for (set<GlobalIndexType>::iterator dofIndexIt = dofIndices.begin();
         dofIndexIt != dofIndices.end(); dofIndexIt++) {
      GlobalIndexType globalDofIndex = *dofIndexIt;
      _partitionLocalIndexForGlobalDofIndex[globalDofIndex] = partitionDofIndex++;
    }
  }
}

void GDAMaximumRule2D::didChangePartitionPolicy() {
  rebuildLookups();
}

void GDAMaximumRule2D::didHRefine(const set<GlobalIndexType> &parentCellIDs) {
  for (set<GlobalIndexType>::iterator parentCellIt = parentCellIDs.begin(); parentCellIt != parentCellIDs.end(); parentCellIt++) {
    GlobalIndexType parentCellID = *parentCellIt;
    CellPtr parent = _meshTopology->getCell(parentCellID);
    ElementTypePtr elemType;
    if (_cellSideUpgrades.find(parentCellID) != _cellSideUpgrades.end()) {
      // this cell has had its sides upgraded, so the elemType Solution knows about is stored in _cellSideUpgrades
      elemType = _cellSideUpgrades[parentCellID].first;
    } else {
      elemType = _elementTypeForCell[parentCellID];
    }
    
    vector< CellPtr > children = parent->children();
    for (vector< CellPtr >::iterator childIt = children.begin(); childIt != children.end(); childIt++) {
      CellPtr child = *childIt;
      _cellH1Orders[child->cellIndex()] = _cellH1Orders[parentCellID];
      assignInitialElementType(child->cellIndex());
    }
    for (vector< CellPtr >::iterator childIt = children.begin(); childIt != children.end(); childIt++) {
      CellPtr child = *childIt;
      int sideCount = CamelliaCellTools::getSideCount(*child->topology());
      _cellSideParitiesForCellID[child->cellIndex()] = vector<int>(sideCount); // 
      for (int sideIndex=0; sideIndex<sideCount; sideIndex++) {
        matchNeighbor(child->cellIndex(), sideIndex); // we'll do this more often than necessary.  Could be smarter about it.
      }
    }
    for (vector< Solution* >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++) {
      // do projection
      vector<IndexType> childIDsLocalIndexType = _meshTopology->getCell(parentCellID)->getChildIndices();
      vector<GlobalIndexType> childIDs(childIDsLocalIndexType.begin(),childIDsLocalIndexType.end());
      (*solutionIt)->processSideUpgrades(_cellSideUpgrades,parentCellIDs); // cellIDs argument: skip these...
      (*solutionIt)->projectOldCellOntoNewCells(parentCellID,elemType,childIDs);
    }
    // with the exception of the cellIDs upgrades, _cellSideUpgrades have been processed,
    // so we delete everything except those
    // (there's probably a more sophisticated way to delete these from the map, b
    map< GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > remainingCellSideUpgrades;
    for (set<GlobalIndexType>::iterator cellIt=parentCellIDs.begin(); cellIt != parentCellIDs.end(); cellIt++) {
      int cellID = *cellIt;
      if (_cellSideUpgrades.find(cellID) != _cellSideUpgrades.end()) {
        remainingCellSideUpgrades[cellID] = _cellSideUpgrades[cellID];
      }
    }
    _cellSideUpgrades = remainingCellSideUpgrades;
  }
  this->GlobalDofAssignment::didHRefine(parentCellIDs);
}

void GDAMaximumRule2D::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) {
  this->GlobalDofAssignment::didPRefine(cellIDs,deltaP);
  
  set<GlobalIndexType>::const_iterator cellIt;
  map<GlobalIndexType, ElementTypePtr > oldTypes;
  for (cellIt=cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    oldTypes[cellID] = _elementTypeForCell[cellID];
  }
  for (cellIt=cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    CellPtr cell = _meshTopology->getCell(cellID);
    ElementTypePtr oldElemType = oldTypes[cellID];
    const shards::CellTopology cellTopo = *(cell->topology());
    //   a. create new DofOrderings for trial and test
    Teuchos::RCP<DofOrdering> currentTrialOrdering, currentTestOrdering;
    currentTrialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
    currentTestOrdering  = _elementTypeForCell[cellID]->testOrderPtr;
    Teuchos::RCP<DofOrdering> newTrialOrdering = _dofOrderingFactory->pRefineTrial(currentTrialOrdering,
                                                                                   cellTopo,deltaP);
//    cout << "in p-refinement, old poly order is " << _dofOrderingFactory->trialPolyOrder(currentTrialOrdering) << endl;
//    cout << "in p-refinement, new poly order is " << _dofOrderingFactory->trialPolyOrder(newTrialOrdering) << endl;
    
    Teuchos::RCP<DofOrdering> newTestOrdering;
    // determine what newTestOrdering should be:
    int trialPolyOrder = _dofOrderingFactory->trialPolyOrder(newTrialOrdering);
    int testPolyOrder = _dofOrderingFactory->testPolyOrder(currentTestOrdering);
    if (testPolyOrder < trialPolyOrder + _testOrderEnhancement ) {
      newTestOrdering = _dofOrderingFactory->testOrdering( trialPolyOrder + _testOrderEnhancement, cellTopo);
    } else {
      newTestOrdering = currentTestOrdering;
    }
    
    ElementTypePtr newType = _elementTypeFactory.getElementType(newTrialOrdering, newTestOrdering,
                                                                cell->topology() );
    setElementType(cellID,newType,false); // false: *not* sideUpgradeOnly
    
    //    elem->setElementType( _elementTypeFactory.getElementType(newTrialOrdering, newTestOrdering,
    //                                                             elem->elementType()->cellTopoPtr ) );
    int numSides = CamelliaCellTools::getSideCount(*cell->topology());
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
      // get and match the big neighbor along the side, if we're a small element…
      pair<GlobalIndexType, int> neighborInfo = cell->getNeighbor(sideOrdinal);
      GlobalIndexType neighborToMatch = neighborInfo.first;
      int neighborSideIndex = neighborInfo.second;
      
      if (neighborToMatch != -1) { // then we have a neighbor to match along that side...
        matchNeighbor(neighborToMatch,neighborSideIndex);
      }
    }
    for (vector< Solution* >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++) {
      // do projection: for p-refinements, the "child" is the same cell
      vector<GlobalIndexType> childIDs(1,cellID);
      (*solutionIt)->processSideUpgrades(_cellSideUpgrades,cellIDs);
      (*solutionIt)->projectOldCellOntoNewCells(cellID,oldElemType,childIDs);
    }
    _cellSideUpgrades.clear(); // these have been processed by all solutions that will ever have a chance to process them.
  }
  
//  rebuildLookups();
}

void GDAMaximumRule2D::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) {
  this->GlobalDofAssignment::didHUnrefine(parentCellIDs);
  for (set<GlobalIndexType>::iterator parentCellIt = parentCellIDs.begin(); parentCellIt != parentCellIDs.end(); parentCellIt++) {
    GlobalIndexType parentCellID = *parentCellIt;
    CellPtr parent = _meshTopology->getCell(parentCellID);
    
    int sideCount = CamelliaCellTools::getSideCount(*parent->topology());
    for (int sideIndex=0; sideIndex<sideCount; sideIndex++) {
      matchNeighbor(parentCellID, sideIndex);
    }
  }
  
//  rebuildLookups();
}

ElementTypePtr GDAMaximumRule2D::elementType(GlobalIndexType cellID) {
  return _elementTypeForCell[cellID];
}

vector< Teuchos::RCP< ElementType > > GDAMaximumRule2D::elementTypes(PartitionIndexType partitionNumber) {
  if ((partitionNumber != -1) && (partitionNumber < _numPartitions)) {
    return _elementTypesForPartition[partitionNumber];
  } else if (partitionNumber == -1) {
    return _elementTypeList;
  } else {
    vector< Teuchos::RCP< ElementType > > noElementTypes;
    return noElementTypes;
  }
}

int GDAMaximumRule2D::getH1Order(GlobalIndexType cellID) {
  return cellPolyOrder(cellID);
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

GlobalIndexType GDAMaximumRule2D::globalCellIndex(GlobalIndexType cellID) {
//  int rank = Teuchos::GlobalMPISession::getRank();
//  cout << "on rank " << rank << ", cellID " << cellID << " has globalCellIndex " << _globalCellIndices[cellID] << endl;
  return _globalCellIndices[cellID];
}

GlobalIndexType GDAMaximumRule2D::globalDofIndex(GlobalIndexType cellID, IndexType localDofIndex) {
  pair<GlobalIndexType,IndexType> key = make_pair(cellID, localDofIndex);
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType >::iterator mapEntryIt = _localToGlobalMap.find(key);
  if ( mapEntryIt == _localToGlobalMap.end() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "entry not found.");
  }
  return (*mapEntryIt).second;
}

GlobalIndexType GDAMaximumRule2D::globalDofCount() {
  return _numGlobalDofs;
}

set<GlobalIndexType> GDAMaximumRule2D::globalDofIndicesForCell(GlobalIndexType cellID) {
  set<GlobalIndexType> dofIndices;
  int localDofCount = elementType(cellID)->trialOrderPtr->totalDofs();
  for (int localDofOrdinal=0; localDofOrdinal<localDofCount; localDofOrdinal++) {
    dofIndices.insert(globalDofIndex(cellID, localDofOrdinal));
  }
  return dofIndices;
}

set<GlobalIndexType> GDAMaximumRule2D::globalDofIndicesForPartition(PartitionIndexType partitionNumber) {
  return _partitionedGlobalDofIndices[partitionNumber];
}

set<GlobalIndexType> GDAMaximumRule2D::partitionOwnedGlobalFieldIndices() {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  set<GlobalIndexType> fieldIndices;
  set<GlobalIndexType> cellIDs = cellsInPartition(-1);
  vector< VarPtr > fieldVars = _varFactory.fieldVars();
  for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementTypePtr elemTypePtr = elementType(cellID);
    vector< VarPtr >::iterator fieldIt;
    for (fieldIt = fieldVars.begin(); fieldIt != fieldVars.end() ; fieldIt++){
      int fieldID = (*fieldIt)->ID();
      int sideOrdinal = 0;
      int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(fieldID,sideOrdinal);
      for (int basisOrdinal = 0; basisOrdinal<basisCardinality; basisOrdinal++) {
        int localDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(fieldID, basisOrdinal, sideOrdinal);
        GlobalIndexType globalIndex = globalDofIndex(cellID, localDofIndex);
        
        if (partitionForGlobalDofIndex(globalIndex) == rank)
          fieldIndices.insert(globalIndex);
      }
    }
  }
  return fieldIndices;
}

set<GlobalIndexType> GDAMaximumRule2D::partitionOwnedIndicesForVariables(set<int> varIDs) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  set<GlobalIndexType> varIndices;
  set<GlobalIndexType> cellIDs = cellsInPartition(-1);
  for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementTypePtr elemTypePtr = elementType(cellID);
    set< int >::iterator varIt;
    for (varIt = varIDs.begin(); varIt != varIDs.end(); varIt++){
      int varID = *varIt;
      int numSides = elemTypePtr->trialOrderPtr->getNumSidesForVarID(varID);
      for (int sideOrdinal = 0; sideOrdinal<numSides; sideOrdinal++) {
        int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(varID,sideOrdinal);
        for (int basisOrdinal = 0; basisOrdinal<basisCardinality; basisOrdinal++) {
          int localDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(varID, basisOrdinal, sideOrdinal);
          GlobalIndexType globalIndex = globalDofIndex(cellID, localDofIndex);
          
          if (partitionForGlobalDofIndex(globalIndex) == rank)
            varIndices.insert(globalIndex);
        }
      }
    }
  }
  return varIndices;
}

set<GlobalIndexType> GDAMaximumRule2D::partitionOwnedGlobalFluxIndices() {
  int rank = Teuchos::GlobalMPISession::getRank();

  set<GlobalIndexType> fluxIndices;
  vector< VarPtr > fluxVars = _varFactory.fluxVars();
  set<GlobalIndexType> cellIDs = cellsInPartition(-1);
  for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementTypePtr elemTypePtr = elementType(cellID);
    int sideCount = CamelliaCellTools::getSideCount(*elemTypePtr->cellTopoPtr);
    vector< VarPtr >::iterator fluxIt;
    for (fluxIt = fluxVars.begin(); fluxIt != fluxVars.end(); fluxIt++){
      int fluxID = (*fluxIt)->ID();
      for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++) {
        int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(fluxID,sideOrdinal);
        for (int basisOrdinal = 0; basisOrdinal<basisCardinality; basisOrdinal++) {
          int localDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(fluxID, basisOrdinal, sideOrdinal);
          GlobalIndexType globalIndex = globalDofIndex(cellID, localDofIndex);
          
          if (partitionForGlobalDofIndex(globalIndex) == rank)
            fluxIndices.insert(globalIndex);

        }
      }
    }
  }
  return fluxIndices;
}

set<GlobalIndexType> GDAMaximumRule2D::partitionOwnedGlobalTraceIndices() {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  set<GlobalIndexType> traceIndices;
  vector< VarPtr > traceVars = _varFactory.traceVars();
  set<GlobalIndexType> cellIDs = cellsInPartition(-1);
  for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementTypePtr elemTypePtr = elementType(cellID);
    int sideCount = CamelliaCellTools::getSideCount(*elemTypePtr->cellTopoPtr);
    vector< VarPtr >::iterator traceIt;
    for (traceIt = traceVars.begin(); traceIt != traceVars.end(); traceIt++){
      int traceID = (*traceIt)->ID();
      for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++) {
        int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(traceID,sideOrdinal);
        for (int basisOrdinal = 0; basisOrdinal<basisCardinality; basisOrdinal++) {
          int localDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(traceID, basisOrdinal, sideOrdinal);
          GlobalIndexType globalIndex = globalDofIndex(cellID, localDofIndex);
          
          if (partitionForGlobalDofIndex(globalIndex) == rank)
            traceIndices.insert(globalIndex);
          
        }
      }
    }
  }
  return traceIndices;
}

void GDAMaximumRule2D::interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localCoefficients,
                                                   const Epetra_MultiVector &globalCoefficients) {
  int numDofs = elementType(cellID)->trialOrderPtr->totalDofs();
  if (localCoefficients.rank()==1) {
    if (globalCoefficients.NumVectors() != 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "if localCoefficients rank==1, globalCoefficients.NumVectors() should be 1");
    }
    for (int dofIndex=0; dofIndex<numDofs; dofIndex++) {
      GlobalIndexTypeToCast globalIndex = globalDofIndex(cellID, dofIndex);
      int localIndex = globalCoefficients.Map().LID(globalIndex);
      localCoefficients(dofIndex) = globalCoefficients[0][localIndex];
    }
  } else if (localCoefficients.rank()==2) {
    if (globalCoefficients.NumVectors() != localCoefficients.dimension(0)) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "if localCoefficients rank==2, globalCoefficients.NumVectors() should match localCoefficients.dimension 0");
    }
    for (int vectorOrdinal=0; vectorOrdinal < globalCoefficients.NumVectors(); vectorOrdinal++) {
      for (int dofIndex=0; dofIndex<numDofs; dofIndex++) {
        GlobalIndexTypeToCast globalIndex = globalDofIndex(cellID, dofIndex);
        int localIndex = globalCoefficients.Map().LID(globalIndex);
        localCoefficients(vectorOrdinal, dofIndex) = globalCoefficients[vectorOrdinal][localIndex];
      }
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported localCoefficients rank");
  }
}

void GDAMaximumRule2D::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs, FieldContainer<double> &globalDofs,
                                          FieldContainer<GlobalIndexType> &globalDofIndices) {
  // for maximum rule, our interpretation is just a one-to-one mapping
  // TODO: add some sanity checks for the arguments here.
  globalDofs = localDofs; // copy -- this may be a vector or a (square) matrix
  int numDofs = localDofs.dimension(0);
  globalDofIndices.resize(numDofs);
  for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
    globalDofIndices[dofOrdinal] = globalDofIndex(cellID, dofOrdinal);
  }
}

void GDAMaximumRule2D::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                                       FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices) {
  globalCoefficients = basisCoefficients; // copy
  if (basisCoefficients.rank() != 1) {
    cout << "basisCoefficients must be a rank 1 container.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basisCoefficients must be a rank 1 container.");
  }
  int numDofs = basisCoefficients.dimension(0);
  globalDofIndices.resize(numDofs);
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  for (int basisDofOrdinal=0; basisDofOrdinal<numDofs; basisDofOrdinal++) {
    int cellDofIndex = trialOrdering->getDofIndex(varID, basisDofOrdinal, sideOrdinal);
    globalDofIndices[basisDofOrdinal] = globalDofIndex(cellID, cellDofIndex);
  }
}

unsigned GDAMaximumRule2D::localDofCount() {
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

void GDAMaximumRule2D::matchNeighbor(GlobalIndexType cellID, int sideIndex) {
  // sets new ElementType to match elem to neighbor on side sideIndex
  
//  cout << "matching neighbor for cell " << cellID << " with side ordinal " << sideIndex << endl;
  
  CellPtr cell = _meshTopology->getCell(cellID);
  const shards::CellTopology cellTopo = *cell->topology();
  
  pair< unsigned, unsigned > neighborRecord = cell->getNeighbor(sideIndex);
  GlobalIndexType neighborCellID = neighborRecord.first;
  unsigned sideIndexInNeighbor = neighborRecord.second;
  
  if (neighborCellID == -1) {
    // no neighbors (boundary): set parity and return
    _cellSideParitiesForCellID[cellID][sideIndex] = 1;
    return;
  }
  
  CellPtr neighbor = _meshTopology->getCell(neighborCellID);
  
  if (_cellSideParitiesForCellID.find(cellID) == _cellSideParitiesForCellID.end()) {
    int sideCount = CamelliaCellTools::getSideCount(*cell->topology());
    _cellSideParitiesForCellID[cellID] = vector<int>(sideCount);
  }
  if (_cellSideParitiesForCellID.find(neighborCellID) == _cellSideParitiesForCellID.end()) {
    int neighborSideCount = CamelliaCellTools::getSideCount(*neighbor->topology());
    _cellSideParitiesForCellID[neighborCellID] = vector<int>(neighborSideCount);
  }
  if ((_cellSideParitiesForCellID[cellID][sideIndex] == 0) && (_cellSideParitiesForCellID[neighborCellID][sideIndexInNeighbor] == 0)) {
    // then lower cellID gets positive parity
    _cellSideParitiesForCellID[cellID][sideIndex] = (cellID < neighborCellID) ? 1 : -1;
    _cellSideParitiesForCellID[neighborCellID][sideIndex] = (cellID < neighborCellID) ? -1 : 1;
  } else if (_cellSideParitiesForCellID[cellID][sideIndex] == 0) {
    _cellSideParitiesForCellID[cellID][sideIndex] = -_cellSideParitiesForCellID[neighborCellID][sideIndexInNeighbor];
  } else {
    _cellSideParitiesForCellID[neighborCellID][sideIndexInNeighbor] = -_cellSideParitiesForCellID[cellID][sideIndex];
  }
  
  // check whether neighbor and cell are peers: this happens if the neighbor relationship commutes:
  bool neighborIsPeer = neighbor->getNeighbor(sideIndexInNeighbor).first == cellID;
  if ( !neighborIsPeer ) {
    // TODO: figure out if this is correct: (I believe this will always result in a peer relationship, but if not, there'd be danger of an infinite recursion.)
    matchNeighbor(neighborCellID, sideIndexInNeighbor);
    return;
  }
  
  // h-refinement handling:
  bool neighborIsBroken = (neighbor->isParent() && (neighbor->childrenForSide(sideIndexInNeighbor).size() > 1));
  bool elementIsBroken  = (cell->isParent() && (cell->childrenForSide(sideIndex).size() > 1));
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
      vector< pair<GlobalIndexType,unsigned> > childrenForSide = cell->childrenForSide(sideIndex);
      for (int childIndexInSide=0; childIndexInSide < childrenForSide.size(); childIndexInSide++) {
        int childCellIndex = childrenForSide[childIndexInSide].first;
        int childSideIndex = childrenForSide[childIndexInSide].second;
        matchNeighbor(childCellIndex,childSideIndex);
      }
      // all our children matched => we're done:
      return;
    } else { // one broken
      vector< pair< GlobalIndexType, unsigned> > childrenForSide = parent->childrenForSide(neighborSideIndexInParent);
      
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
    int neighborSidePolyOrder = BasisFactory::basisFactory()->basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,sideIndexInNeighbor));
    int mySidePolyOrder = BasisFactory::basisFactory()->basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,sideIndex));
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
    int sidePolyOrder = BasisFactory::basisFactory()->basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,sideIndexInNeighbor));
    int mySidePolyOrder = BasisFactory::basisFactory()->basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,sideIndex));
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
  vector< pair< GlobalIndexType, unsigned> > childrenForSide = parent->childrenForSide(sideIndex);
  map< int, BasisPtr > varIDsToUpgrade;
  vector< map< int, BasisPtr > > childVarIDsToUpgrade;
  vector< pair< GlobalIndexType, unsigned> >::iterator entryIt;
  for ( entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
    GlobalIndexType childCellIndex = (*entryIt).first;
    int childSideIndex = (*entryIt).second;
    CellPtr childCell = _meshTopology->getCell(childCellIndex);
    
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
    BasisPtr multiBasis = BasisFactory::basisFactory()->getMultiBasis(bases);
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

IndexType GDAMaximumRule2D::neighborDofPermutation(IndexType dofIndex, IndexType numDofsForSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numDofsForSide - dofIndex - 1;
}

PartitionIndexType GDAMaximumRule2D::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  if ( _partitionForGlobalDofIndex.find( globalDofIndex ) == _partitionForGlobalDofIndex.end() ) {
    return -1;
  }
  return _partitionForGlobalDofIndex[ globalDofIndex ];
}

IndexType GDAMaximumRule2D::partitionLocalCellIndex(GlobalIndexType cellID) {
  return _partitionLocalCellIndices[cellID];
}

GlobalIndexType GDAMaximumRule2D::partitionLocalIndexForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  return _partitionLocalIndexForGlobalDofIndex[ globalDofIndex ];
}

FieldContainer<double> & GDAMaximumRule2D::physicalCellNodes( ElementTypePtr elemTypePtr) {
#ifdef HAVE_MPI
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
#else
  int partitionNumber     = 0;
#endif
  return _partitionedPhysicalCellNodesForElementType[ partitionNumber ][ elemTypePtr.get() ];
}

FieldContainer<double> & GDAMaximumRule2D::physicalCellNodesGlobal( ElementTypePtr elemTypePtr ) {
  return _physicalCellNodesForElementType[ elemTypePtr.get() ];
}

void GDAMaximumRule2D::rebuildLookups() {
//  cout << "GDAMaximumRule2D::rebuildLookups().\n";
  _cellSideUpgrades.clear();
  buildTypeLookups(); // build data structures for efficient lookup by element type
  buildLocalToGlobalMap();
  determinePartitionDofIndices();
  
  // now discard any old coefficients
  for (vector< Solution* >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++) {
    (*solutionIt)->discardInactiveCellCoefficients();
  }
}

void GDAMaximumRule2D::setElementType(GlobalIndexType cellID, ElementTypePtr newType, bool sideUpgradeOnly) {
  CellPtr cell = _meshTopology->getCell(cellID);
  if (sideUpgradeOnly) { // need to track in _cellSideUpgrades
    ElementTypePtr oldType;
    map<GlobalIndexType, pair<ElementTypePtr, ElementTypePtr> >::iterator existingEntryIt = _cellSideUpgrades.find(cellID);
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

void GDAMaximumRule2D::verticesForCell(FieldContainer<double>& vertices, GlobalIndexType cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);
  vector<IndexType> vertexIndices = cell->vertices();
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

void GDAMaximumRule2D::verticesForCells(FieldContainer<double>& vertices, vector<GlobalIndexType> &cellIDs) {
  // all cells represented in cellIDs must have the same topology
  int spaceDim = _meshTopology->getSpaceDim();
  GlobalIndexType numCells = cellIDs.size();
  
  if (numCells == 0) {
    vertices.resize(0,0,0);
    return;
  }
  GlobalIndexType firstCellID = cellIDs[0];
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