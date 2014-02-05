//
//  MeshTestUtility.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "MeshTestUtility.h"
#include "MultiBasis.h"

#include "GDAMaximumRule2D.cpp"

bool MeshTestUtility::checkMeshConsistency(Teuchos::RCP<Mesh> mesh) {
  bool success = true;
  success = checkMeshDofConnectivities(mesh);
  // now, check element types:
  vector< ElementPtr > activeElements = mesh->activeElements();
  GlobalIndexType numElements = activeElements.size();
  for (GlobalIndexType cellIndex=0; cellIndex<numElements; cellIndex++) {
    Teuchos::RCP<Element> elem = activeElements[cellIndex];
    GlobalIndexType cellID = mesh->getElement(elem->cellID())->cellID();
    if ( cellID != elem->cellID() ) {
      success = false;
      cout << "cellID for element doesn't match its index in mesh->elements() --";
      cout <<  elem->cellID() << " != " << cellID << endl;
    }
    if ( cellID != mesh->cellID(elem->elementType(), elem->globalCellIndex()) ) {
      success = false;
      cout << "cellID index in mesh->elements() doesn't match what's reported by mesh->cellID(elemType,cellIndex): ";
      cout <<  cellID << " != " << mesh->cellID(elem->elementType(), elem->globalCellIndex()) << endl;
    }
    // check that the vertices are lined up correctly
    int numSides = elem->numSides();
    for (int sideIndex = 0; sideIndex<numSides; sideIndex++) {
      //      Element* neighbor;
      int mySideIndexInNeighbor;
      Teuchos::RCP<Element> neighbor = mesh->ancestralNeighborForSide(elem, sideIndex, mySideIndexInNeighbor);
      int myParity = mesh->parityForSide(cellID,sideIndex);
      if ( mesh->boundary().boundaryElement(cellID,sideIndex) ) { // on boundary
        if ( myParity != 1 ) {
          success = false;
          cout << "Mesh consistency FAILURE: cellID " << cellID << " has parity != 1 on boundary; sideIndex = " << sideIndex << endl;
        }
      } else { //not on boundary
        int neighborCellID = neighbor->cellID();
        int neighborParity = mesh->parityForSide(neighborCellID,mySideIndexInNeighbor);
        if (neighborParity != -myParity) {
          success = false;
          cout << "Mesh consistency FAILURE: cellID " << cellID << " has parity != -neighborParity on boundary; sideIndex = " << sideIndex << endl;
          cout << "neighbor parity = " << neighborParity << " and myparity = " << myParity << endl;
          cout << "side index in neighbor is " << mySideIndexInNeighbor << endl;
          vector<double> centroid = mesh->getCellCentroid(cellID);
          cout << "element centroid for cellID " << cellID << " is " << centroid[0] << "," << centroid[1] << endl;
        }
        // this check needs to be modified for 3D
        // TODO: modify for 3D
        if ( neighborCellID == elem->getNeighborCellID(sideIndex) ) { // peers, then
          FieldContainer<double> myVertices;
          FieldContainer<double> neighborVertices;
          mesh->verticesForSide(myVertices,cellID,sideIndex);
          mesh->verticesForSide(neighborVertices,neighborCellID,mySideIndexInNeighbor);
          int numPoints = myVertices.dimension(0);
          for (int i=0; i<numPoints; i++) { // numPoints
            int neighborVertexIndex = numPoints - 1 - i; // should be in reverse order, based on our 2D layout strategy
            if ( ( myVertices(i,0) != neighborVertices(neighborVertexIndex,0) ) 
                || ( myVertices(i,0) != neighborVertices(neighborVertexIndex,0) ) ) {
              cout << "cellID " << cellID << " and " << neighborCellID << " do not agree on shared edge " << endl;
              cout << "cellID " << cellID << " vertices: " << endl;
              cout << myVertices;
              cout << "cellID " << neighborCellID << " vertices: " << endl;
              cout << neighborVertices;
              success = false;
            }
          }
        }
      }
    }
  }
  // check that bases agree along shared edges:
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.11,-0.2,0.313,-0.4,0.54901,-0.6,0.73134,-0.810,0.912,-1.0};
  
  FieldContainer<double> testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    testPoints1D(i, 0) = x[i];
  }
  
  // TODO: get this working for MultiBasis
  //neighborBasesAgreeOnSides(mesh,testPoints1D);
  
  return success;
}

bool MeshTestUtility::checkMeshDofConnectivities(Teuchos::RCP<Mesh> mesh) {
  int numCells = mesh->activeElements().size();
  bool success = true;
  int numGlobalDofs = mesh->numGlobalDofs();
  vector<int> globalDofIndexHitCount(numGlobalDofs,0);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    Teuchos::RCP<Element> elem = mesh->activeElements()[cellIndex];
    GlobalIndexType cellID = elem->cellID();
    DofOrdering trialOrder = *(elem->elementType()->trialOrderPtr.get());
    vector< int > trialIDs = mesh->bilinearForm()->trialIDs();
    for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      int numSides = trialOrder.getNumSidesForVarID(trialID);
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        int numBasisDofs = trialOrder.getBasisCardinality(trialID, sideIndex);
        for (int dofOrdinal=0; dofOrdinal<numBasisDofs; dofOrdinal++) {
          // a very basic check on the mesh dof ordering: the globalDofIndices for all localDofs should not be negative!
          int localDofIndex = trialOrder.getDofIndex(trialID, dofOrdinal, sideIndex);
          int globalDofIndex = mesh->globalDofIndex(cellID,localDofIndex);
          if (globalDofIndex < 0) {
            cout << "mesh->globalDofIndex(" << cellID << "," << localDofIndex << ") = " << globalDofIndex << " < 0.  Error!";
            success = false;
          } else if (globalDofIndex >= mesh->numGlobalDofs()) {
            cout << "mesh->globalDofIndex(" << cellID << "," << localDofIndex << ") = " << globalDofIndex << " >= mymesh->numGlobalDofs().  Error!";
            success = false;
          } else {
            globalDofIndexHitCount[globalDofIndex]++;
          }
        }
        
        // now a more subtle check: given the mesh layout (that all vertices are specified CCW),
        // the dofs for boundary variables (fluxes & traces) should be reversed between element and its neighbor
        if (mesh->bilinearForm()->isFluxOrTrace(trialID)) {
          if (! mesh->boundary().boundaryElement(cellID, sideIndex)) { // not boundary...
            //            if (neighbor->cellID() != -1) { // not boundary...
            int ancestralSideIndexInNeighbor;
            Teuchos::RCP<Element> neighbor = mesh->ancestralNeighborForSide(elem, sideIndex, ancestralSideIndexInNeighbor);
            
            Teuchos::RCP<DofOrdering> neighborTrialOrder = neighbor->elementType()->trialOrderPtr;
            int neighborNumBasisDofs = neighborTrialOrder->getBasisCardinality(trialID,ancestralSideIndexInNeighbor);
            if (neighborNumBasisDofs != numBasisDofs) {
              if ( mesh->usePatchBasis() ) {
                cout << "FAILURE: usePatchBasis==true, but neighborNumBasisDofs != numBasisDofs.\n";
                success = false;
                continue;
              }
              if ( neighbor->isParent() ) {
                // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
                //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendants along that side...
                vector< pair<int,int> > descendantsForSide = neighbor->getDescendantsForSide(ancestralSideIndexInNeighbor);
                vector< pair<int,int> >:: iterator entryIt;
                int descendantIndex = -1;
                for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
                  descendantIndex++;
                  int neighborSubSideIndexInMe = GDAMaximumRule2D::neighborChildPermutation(descendantIndex, descendantsForSide.size());
                  int neighborCellID = (*entryIt).first;
                  int mySideIndexInNeighbor = (*entryIt).second;
                  neighbor = mesh->getElement(neighborCellID);
                  int neighborNumDofs = neighbor->elementType()->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
                  
                  for (int dofOrdinal=0; dofOrdinal<neighborNumDofs; dofOrdinal++) {
                    int myLocalDofIndex;
                    if ((descendantsForSide.size() > 1) && !mesh->usePatchBasis()) {
                      myLocalDofIndex = elem->elementType()->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex,neighborSubSideIndexInMe);
                    } else {
                      myLocalDofIndex = elem->elementType()->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
                    }
                    int globalDofIndex = mesh->globalDofIndex(cellID,myLocalDofIndex);
                    
                    // neighbor's dofs are in reverse order from mine along each side
                    int permutedDofOrdinal = GDAMaximumRule2D::neighborDofPermutation(dofOrdinal,neighborNumDofs);
                    
                    int neighborLocalDofIndex = neighbor->elementType()->trialOrderPtr->getDofIndex(trialID,permutedDofOrdinal,mySideIndexInNeighbor);
                    int neighborsGlobalDofIndex = mesh->globalDofIndex(neighbor->cellID(),neighborLocalDofIndex);                
                    if (neighborsGlobalDofIndex != globalDofIndex) {
                      
                      cout << "FAILURE: checkDofConnectivities--(cellID, localDofIndex) : (" << cellID << ", " << myLocalDofIndex << ") != (";
                      cout << neighborCellID << ", " << neighborLocalDofIndex << ") -- ";
                      cout << globalDofIndex << " != " << neighborsGlobalDofIndex << "\n";
                      success = false;
                    }
                  }
                }
              } else if (neighbor->getNeighborCellID(ancestralSideIndexInNeighbor) != cellID) {
                // elem is small, neighbor big
                // first, find my leaf index in neighbor:
                int ancestorCellID = neighbor->getNeighborCellID(ancestralSideIndexInNeighbor);
                Teuchos::RCP<Element> ancestor = mesh->getElement(ancestorCellID);
                int ancestorSideIndex = neighbor->getSideIndexInNeighbor(ancestralSideIndexInNeighbor);
                vector< pair<int,int> > descendantsForSide = ancestor->getDescendantsForSide(ancestorSideIndex);
                int descendantIndex = 0;
                int leafIndexInNeighbor = -1;
                for (vector< pair<int,int> >::iterator entryIt = descendantsForSide.begin(); 
                     entryIt != descendantsForSide.end();  entryIt++, descendantIndex++) {
                  if (entryIt->first == cellID) {
                    leafIndexInNeighbor = GDAMaximumRule2D::neighborChildPermutation(descendantIndex, descendantsForSide.size());
                    break;
                  }
                }
                if (leafIndexInNeighbor == -1) {
                  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not determine leafIndexInNeigbor.");
                }
                // check whether the basis is the right size:
                MultiBasis<>* neighborMultiBasis = (MultiBasis<>*) neighbor->elementType()->trialOrderPtr->getBasis(trialID,ancestralSideIndexInNeighbor).get();
                BasisPtr neighborLeafBasis = neighborMultiBasis->getLeafBasis(leafIndexInNeighbor);
                if (numBasisDofs != neighborLeafBasis->getCardinality()) {
                  success = false;
                  cout << "FAILURE: cellID " << cellID << "'s basis for trialID " << trialID;
                  cout << " along sideIndex " << sideIndex << " has cardinality " << numBasisDofs;
                  cout << ", but neighbor leaf basis along that side (cellID " << neighbor->cellID();
                  cout << ", sideIndex " << ancestralSideIndexInNeighbor;
                  cout << ", leaf node " << leafIndexInNeighbor;
                  cout << ") has cardinality " << neighborLeafBasis->getCardinality() << endl;
                } else {
                  // cardinalities match, check that global dofs line up
                  for (int dofOrdinal = 0; dofOrdinal < numBasisDofs; dofOrdinal++) {
                    int permutedDofOrdinal = GDAMaximumRule2D::neighborDofPermutation(dofOrdinal, numBasisDofs);
                    int neighborDofOrdinal = neighborMultiBasis->relativeToAbsoluteDofOrdinal(permutedDofOrdinal, 
                                                                                              leafIndexInNeighbor);
                    int neighborLocalDofIndex = neighbor->elementType()->trialOrderPtr->getDofIndex(trialID, neighborDofOrdinal,ancestralSideIndexInNeighbor);
                    int neighborGlobalDofIndex = mesh->globalDofIndex(neighbor->cellID(), neighborLocalDofIndex);
                    int myLocalDofIndex = elem->elementType()->trialOrderPtr->getDofIndex(trialID, dofOrdinal, sideIndex);
                    int myGlobalDofIndex = mesh->globalDofIndex(cellID, myLocalDofIndex);
                    if (neighborGlobalDofIndex != myGlobalDofIndex) {
                      success = false;
                      cout << "FAILURE: checkDofConnectivities--(cellID, localDofIndex) : (" << cellID << ", ";
                      cout << myLocalDofIndex << ") != (";
                      cout << neighbor->cellID() << ", " << neighborLocalDofIndex << ") -- ";
                      cout << myGlobalDofIndex << " != " << neighborGlobalDofIndex << "\n";
                    }
                  }
                }
              } else {
                cout << "FAILURE: cellID " << cellID << "'s basis for trialID " << trialID;
                cout << " along sideIndex " << sideIndex << " has cardinality " << numBasisDofs;
                cout << ", but neighbor along that side (cellID " << neighbor->cellID();
                cout << ", sideIndex " << ancestralSideIndexInNeighbor << ") has cardinality " << neighborNumBasisDofs << endl;
                success = false;
              }
            } else { // (neighborNumBasisDofs == numBasisDofs)
              if (! neighbor->isParent() ) { 
                for (int dofOrdinal=0; dofOrdinal<numBasisDofs; dofOrdinal++) {
                  int permutedDofOrdinal = GDAMaximumRule2D::neighborDofPermutation(dofOrdinal,numBasisDofs);
                  int neighborsLocalDofIndex = neighborTrialOrder->getDofIndex(trialID, permutedDofOrdinal, ancestralSideIndexInNeighbor);
                  GlobalIndexType neighborsGlobalDofIndex = mesh->globalDofIndex(neighbor->cellID(),neighborsLocalDofIndex);
                  int localDofIndex = trialOrder.getDofIndex(trialID, dofOrdinal, sideIndex);
                  GlobalIndexType globalDofIndex = mesh->globalDofIndex(cellID,localDofIndex);
                  if (neighborsGlobalDofIndex != globalDofIndex) {
                    cout << "FAILURE: cellID " << cellID << "'s neighbor " << sideIndex << "'s globalDofIndex " << neighborsGlobalDofIndex << " doesn't match element globalDofIndex " << globalDofIndex << ". (trialID, element dofOrdinal)=(" << trialID << "," << dofOrdinal << ")" << endl;
                    success = false;
                  }
                }
              } else { // neighbor->isParent()
                // for PatchBasis:
                for (int dofOrdinal=0; dofOrdinal<numBasisDofs; dofOrdinal++) {
                  int localDofIndex = trialOrder.getDofIndex(trialID, dofOrdinal, sideIndex);
                  GlobalIndexType globalDofIndex = mesh->globalDofIndex(cellID,localDofIndex);
                  vector< pair<int,int> > descendantsForSide = neighbor->getDescendantsForSide(ancestralSideIndexInNeighbor);
                  vector< pair<int,int> >:: iterator entryIt;
                  for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
                    GlobalIndexType neighborCellID = (*entryIt).first;
                    int mySideIndexInNeighbor = (*entryIt).second;
                    neighbor = mesh->getElement(neighborCellID);
                    neighborTrialOrder = neighbor->elementType()->trialOrderPtr;
                    int permutedDofOrdinal = GDAMaximumRule2D::neighborDofPermutation(dofOrdinal,numBasisDofs);
                    int neighborsLocalDofIndex = neighborTrialOrder->getDofIndex(trialID, permutedDofOrdinal, mySideIndexInNeighbor);
                    GlobalIndexType neighborsGlobalDofIndex = mesh->globalDofIndex(neighbor->cellID(),neighborsLocalDofIndex);
                    if (neighborsGlobalDofIndex != globalDofIndex) {
                      cout << "FAILURE: cellID " << cellID << "'s neighbor on side " << sideIndex;
                      cout << " (cellID " << neighborCellID << ")'s globalDofIndex " << neighborsGlobalDofIndex;
                      cout << " doesn't match element globalDofIndex " << globalDofIndex;
                      cout << ". (trialID, element dofOrdinal): (" << trialID << "," << dofOrdinal << ")" << endl;
                      cout << "         (cellID,localDofIndex): (" << cellID << "," << localDofIndex << ") ≠ (";
                      cout << neighborCellID << "," << neighborsLocalDofIndex << ")\n";
                      success = false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (int i=0; i<numGlobalDofs; i++) {
    if ( globalDofIndexHitCount[i] == 0 ) {
      success = false;
      cout << "FAILURE: meshDofConnectivity: globalDofIndex " << i << " is unreachable.\n";
    }
  }
  return success;
}

