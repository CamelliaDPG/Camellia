//
//  MeshTestUtility.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "MeshTestUtility.h"

bool MeshTestUtility::checkMeshConsistency(Teuchos::RCP<Mesh> mesh) {
  bool success = true;
  success = checkMeshDofConnectivities(mesh);
  // now, check element types:
  int numElements = mesh->activeElements().size();
  for (int cellIndex=0; cellIndex<numElements; cellIndex++) {
    Teuchos::RCP<Element> elem = mesh->activeElements()[cellIndex];
    int cellID = mesh->elements()[elem->cellID()]->cellID();
    if ( cellID != elem->cellID() ) {
      success = false;
      cout << "cellID for element doesn't match its index in mesh->elements() --";
      cout <<  elem->cellID() << " != " << cellID << endl;
    }
    if ( cellID != mesh->cellID(elem->elementType(), elem->globalCellIndex()) ) {
      success = false;
      cout << "cellID index in mesh->elements() doesn't match what's reported by mesh->cellID(elemType,cellIndex) --";
      cout <<  cellID << " != " << mesh->cellID(elem->elementType(), elem->globalCellIndex()) << endl;
    }
    // check that the vertices are lined up correctly
    int numSides = elem->numSides();
    for (int sideIndex = 0; sideIndex<numSides; sideIndex++) {
      //      Element* neighbor;
      int mySideIndexInNeighbor;
      //       elem->getNeighbor(neighbor,mySideIndexInNeighbor,sideIndex);
      Teuchos::RCP<Element> neighbor = mesh->ancestralNeighborForSide(elem, sideIndex, mySideIndexInNeighbor);
      int neighborCellID = neighbor->cellID();
      int myParity = mesh->parityForSide(cellID,sideIndex);
      if ( mesh->boundary().boundaryElement(cellID,sideIndex) ) { // on boundary
        if ( myParity != 1 ) {
          success = false;
          cout << "Mesh consistency FAILURE: cellID " << cellID << " has parity != 1 on boundary; sideIndex = " << sideIndex << endl;
        }
      } else { //not on boundary
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
    int cellID = elem->cellID();
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
          
          // now a more subtle check: given the mesh layout (that all vertices are specified CCW),
          // the dofs for boundary variables (fluxes & traces) should be reversed between element and its neighbor
          if (mesh->bilinearForm()->isFluxOrTrace(trialID)) {
            Element* neighbor;
            int mySideIndexInNeighbor;
            elem->getNeighbor(neighbor,mySideIndexInNeighbor,sideIndex);
            if (neighbor->cellID() != -1) { // not boundary...
              Teuchos::RCP<DofOrdering> neighborTrialOrder = neighbor->elementType()->trialOrderPtr;
              int neighborNumBasisDofs = neighborTrialOrder->getBasisCardinality(trialID,mySideIndexInNeighbor);
              if (neighborNumBasisDofs != numBasisDofs) {
                if ( mesh->usePatchBasis() ) {
                  cout << "FAILURE: usePatchBasis==true, but neighborNumBasisDofs != numBasisDofs.\n";
                  success = false;
                  continue;
                }
                if ( neighbor->isParent() ) {
                  // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
                  //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendants along that side...
                  int numDofs = min(neighborNumBasisDofs,numBasisDofs); // if there IS a multi-basis, we match the smaller basis with it...
                  vector< pair<int,int> > descendantsForSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor);
                  vector< pair<int,int> >:: iterator entryIt;
                  int descendantIndex = -1;
                  for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
                    descendantIndex++;
                    int neighborSubSideIndexInMe = mesh->neighborChildPermutation(descendantIndex, descendantsForSide.size());
                    int neighborCellID = (*entryIt).first;
                    mySideIndexInNeighbor = (*entryIt).second;
                    neighbor = mesh->elements()[neighborCellID].get();
                    for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
                      int myLocalDofIndex;
                      if ((descendantsForSide.size() > 1) && !mesh->usePatchBasis()) {
                        myLocalDofIndex = elem->elementType()->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex,neighborSubSideIndexInMe);
                      } else {
                        myLocalDofIndex = elem->elementType()->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
                      }
                      globalDofIndex = mesh->globalDofIndex(cellID,myLocalDofIndex);
                      
                      // neighbor's dofs are in reverse order from mine along each side
                      int permutedDofOrdinal = mesh->neighborDofPermutation(dofOrdinal,numDofs);
                      
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
                } else {
                  cout << "FAILURE: cellID " << cellID << "'s basis for trialID " << trialID;
                  cout << " along sideIndex " << sideIndex << " has cardinality " << numBasisDofs;
                  cout << ", but neighbor along that side (cellID " << neighbor->cellID();
                  cout << ", sideIndex " << mySideIndexInNeighbor << ") has cardinality " << neighborNumBasisDofs << endl;
                  success = false;
                }
              } else { // (neighborNumBasisDofs == numBasisDofs)
                if (! neighbor->isParent() ) { 
                  int permutedDofOrdinal = mesh->neighborDofPermutation(dofOrdinal,numBasisDofs);
                  int neighborsLocalDofIndex = neighborTrialOrder->getDofIndex(trialID, permutedDofOrdinal, mySideIndexInNeighbor);
                  int neighborsGlobalDofIndex = mesh->globalDofIndex(neighbor->cellID(),neighborsLocalDofIndex);                
                  if (neighborsGlobalDofIndex != globalDofIndex) {
                    cout << "FAILURE: cellID " << cellID << "'s neighbor " << sideIndex << "'s globalDofIndex " << neighborsGlobalDofIndex << " doesn't match element globalDofIndex " << globalDofIndex << ". (trialID, element dofOrdinal)=(" << trialID << "," << dofOrdinal << ")" << endl;
                    success = false;
                  }
                } else { // neighbor->isParent()
                  // for PatchBasis:
                  vector< pair<int,int> > descendantsForSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor);
                  vector< pair<int,int> >:: iterator entryIt;
                  int descendantIndex = -1;
                  for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
                    int neighborCellID = (*entryIt).first;
                    mySideIndexInNeighbor = (*entryIt).second;
                    neighbor = mesh->elements()[neighborCellID].get();
                    neighborTrialOrder = neighbor->elementType()->trialOrderPtr;
                    int permutedDofOrdinal = mesh->neighborDofPermutation(dofOrdinal,numBasisDofs);
                    int neighborsLocalDofIndex = neighborTrialOrder->getDofIndex(trialID, permutedDofOrdinal, mySideIndexInNeighbor);
                    int neighborsGlobalDofIndex = mesh->globalDofIndex(neighbor->cellID(),neighborsLocalDofIndex);                
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

