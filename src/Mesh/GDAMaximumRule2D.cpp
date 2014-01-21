//
//  GDAMaximumRule2D.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMaximumRule2D.h"

#include "MultiBasis.h"

GDAMaximumRule2D::GDAMaximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                   MeshPartitionPolicyPtr partitionPolicy, bool enforceMBFluxContinuity)
: GlobalDofAssignment(meshTopology,varFactory,dofOrderingFactory,partitionPolicy)
{
  _enforceMBFluxContinuity = enforceMBFluxContinuity;
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
  vector<ElementPtr>::iterator elemIterator;
  
  // debug code:
  //  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
  //    ElementPtr elemPtr = *(elemIterator);
  //    cout << "cellID " << elemPtr->cellID() << "'s trialOrdering:\n";
  //    cout << *(elemPtr->elementType()->trialOrderPtr);
  //  }
  
  determineDofPairings();
  
  int globalIndex = 0;
  vector< int > trialIDs = _bilinearForm->trialIDs();
  
  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    int cellID = elemPtr->cellID();
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      
      if (! _bilinearForm->isFluxOrTrace(trialID) ) {
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
        int numSides = elemPtr->numSides();
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
    vector< ElementPtr >::iterator elemIterator;
    
    // this should loop over the elements in the partition instead
    for (elemIterator=_partitions[partitionNumber].begin();
         elemIterator != _partitions[partitionNumber].end(); elemIterator++) {
      ElementPtr elem = *elemIterator;
      ElementTypePtr elemTypePtr = elem->elementType();
      if ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) == _cellIDsForElementType[partitionNumber].end() ) {
        _elementTypesForPartition[partitionNumber].push_back(elemTypePtr);
      }
      if (elementTypeSet.find( elemTypePtr.get() ) == elementTypeSet.end() ) {
        elementTypeSet.insert( elemTypePtr.get() );
        _elementTypes.push_back( elemTypePtr );
      }
      _cellIDsForElementType[partitionNumber][elemTypePtr.get()].push_back(elem->cellID());
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
        ElementPtr elem = _elements[cellID];
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          cellSideParities(cellIndex,sideIndex) = _cellSideParitiesForCellID[cellID][sideIndex];
        }
        elem->setCellIndex(cellIndex++);
        elem->setGlobalCellIndex(globalCellIndices[elemType.get()]++);
        _globalCellIndexToCellID[elemType.get()][elem->globalCellIndex()] = cellID;
        TEUCHOS_TEST_FOR_EXCEPTION( elem->cellID() != _globalCellIndexToCellID[elemType.get()][elem->globalCellIndex()],
                                   std::invalid_argument, "globalCellIndex -> cellID inconsistency detected" );
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
        int globalCellIndex = _elements[cellID]->globalCellIndex();
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
  
  _activeElements.clear();
  vector<ElementPtr>::iterator elemIterator;
  
  for (elemIterator = _elements.begin(); elemIterator != _elements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    if ( elemPtr->isActive() ) {
      _activeElements.push_back(elemPtr);
    }
  }
  _partitions.clear();
  _partitionForCellID.clear();
  FieldContainer<int> partitionedMesh(_numPartitions,_activeElements.size());
  _partitionPolicy->partitionMesh(_meshTopology.get(),_numPartitions,partitionedMesh);
  for (int i=0; i<_numPartitions; i++) {
    vector<ElementPtr> partition;
    for (int j=0; j<_activeElements.size(); j++) {
      if (partitionedMesh(i,j) < 0) break; // no more elements in this partition
      int cellID = partitionedMesh(i,j);
      partition.push_back( _elements[cellID] );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
  }
}

void GDAMaximumRule2D::determineDofPairings() {
  _dofPairingIndex.clear();
  vector<ElementPtr>::iterator elemIterator;
  
  vector< int > trialIDs = _bilinearForm->trialIDs();
  
  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    int cellID = elemPtr->cellID();
    
    if ( elemPtr->isParent() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"elemPtr is in _activeElements, but is a parent...");
    }
    if ( ! elemPtr->isActive() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"elemPtr is in _activeElements, but is inactive...");
    }
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      
      if (_bilinearForm->isFluxOrTrace(trialID) ) {
        int numSides = elemPtr->numSides();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int myNumDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          Element* neighborPtr;
          int mySideIndexInNeighbor;
          elemPtr->getNeighbor(neighborPtr, mySideIndexInNeighbor, sideIndex);
          
          int neighborCellID = neighborPtr->cellID(); // may be -1 if it's the boundary
          if (neighborCellID != -1) {
            Teuchos::RCP<Element> neighbor = _elements[neighborCellID];
            // check that the bases agree in #dofs:
            
            bool hasMultiBasis = neighbor->isParent();
            
            if ( ! neighbor->isParent() ) {
              int neighborNumDofs = neighbor->elementType()->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              if ( !hasMultiBasis && (myNumDofs != neighborNumDofs) ) { // neither a multi-basis, and we differ: a problem
                TEUCHOS_TEST_FOR_EXCEPTION(myNumDofs != neighborNumDofs,
                                           std::invalid_argument,
                                           "Element and neighbor don't agree on basis along shared side.");
              }
            }
            
            // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
            //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendants along that side...
            vector< pair<int,int> > descendantsForSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor);
            vector< pair<int,int> >:: iterator entryIt;
            int descendantIndex = -1;
            for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
              descendantIndex++;
              int descendantSubSideIndexInMe = neighborChildPermutation(descendantIndex, descendantsForSide.size());
              neighborCellID = (*entryIt).first;
              mySideIndexInNeighbor = (*entryIt).second;
              neighbor = _elements[neighborCellID];
              int neighborNumDofs = neighbor->elementType()->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              
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
                
                int neighborLocalDofIndex = neighbor->elementType()->trialOrderPtr->getDofIndex(trialID,permutedDofOrdinal,mySideIndexInNeighbor);
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
  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    int cellID = elemPtr->cellID();
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      if (_bilinearForm->isFluxOrTrace(trialID) ) {
        int numSides = elemPtr->numSides();
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
    vector< ElementPtr >::iterator elemIterator;
    for (elemIterator =  _partitions[i].begin(); elemIterator != _partitions[i].end(); elemIterator++) {
      ElementPtr elem = *elemIterator;
      ElementTypePtr elemTypePtr = elem->elementType();
      int numLocalDofs = elemTypePtr->trialOrderPtr->totalDofs();
      int cellID = elem->cellID();
      for (int localDofIndex=0; localDofIndex < numLocalDofs; localDofIndex++) {
        pair<int,int> key = make_pair(cellID, localDofIndex);
        map< pair<int,int>, int >::iterator mapEntryIt = _localToGlobalMap.find(key);
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

int GDAMaximumRule2D::neighborChildPermutation(int childIndex, int numChildrenInSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numChildrenInSide - childIndex - 1;
}

int GDAMaximumRule2D::neighborDofPermutation(int dofIndex, int numDofsForSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numDofsForSide - dofIndex - 1;
}

void GDAMaximumRule2D::verticesForCell(FieldContainer<double>& vertices, int cellID) {
  CellPtr cell = _meshTopology->getCell(cellID);
  vector<unsigned> vertexIndices = cell->vertices();
  int numVertices = vertexIndices.size();
  int spaceDim = _meshTopology->getSpaceDim();
  
  //vertices.resize(numVertices,dimension);
  for (unsigned vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    for (int d=0; d<spaceDim; d++) {
      vertices(vertexIndex,d) = _meshTopology->getVertex(vertexIndices[vertexIndex])[d];
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