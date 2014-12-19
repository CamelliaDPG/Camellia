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

#include "CamelliaCellTools.h"

#include "GDAMaximumRule2D.h"

#include "IndexType.h"

#include "Epetra_SerialComm.h"

#include "BasisCache.h"

#include "CubatureFactory.h"

using namespace Camellia;

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
      if ( mesh->getTopology()->getCell(cellID)->isBoundary(sideIndex) ) { // on boundary
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
          CellPtr cell = mesh->getTopology()->getCell(cellID);
          if (! mesh->getTopology()->getCell(cellID)->isBoundary(sideIndex)) { // not boundary...
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

bool MeshTestUtility::determineRefTestPointsForNeighbors(MeshTopologyPtr meshTopo, CellPtr fineCell, unsigned int sideOrdinal,
                                                         FieldContainer<double> &fineSideRefPoints, FieldContainer<double> &fineCellRefPoints,
                                                         FieldContainer<double> &coarseSideRefPoints, FieldContainer<double> &coarseCellRefPoints) {
  unsigned spaceDim = meshTopo->getSpaceDim();
  unsigned sideDim = spaceDim - 1;
  
  if (spaceDim == 1) {
    fineSideRefPoints.resize(0,0);
    coarseSideRefPoints.resize(0,0);
    fineCellRefPoints.resize(1,1);
    coarseCellRefPoints.resize(1,1);
    
    FieldContainer<double> lineRefNodes(2,1);
    CellTopoPtr line = CellTopology::line();

    CamelliaCellTools::refCellNodesForTopology(lineRefNodes, line);
    
    fineCellRefPoints[0] = lineRefNodes[sideOrdinal];
    unsigned neighborSideOrdinal = fineCell->getNeighborInfo(sideOrdinal).second;
    if (neighborSideOrdinal != -1) {
      coarseCellRefPoints[0] = lineRefNodes[neighborSideOrdinal];
      return true;
    } else {
      return false;
    }
  }
  pair<GlobalIndexType, unsigned> neighborInfo = fineCell->getNeighborInfo(sideOrdinal);
  if (neighborInfo.first == -1) {
    // boundary
    return false;
  }
  CellPtr neighborCell = meshTopo->getCell(neighborInfo.first);
  if (neighborCell->isParent()) {
    return false; // fineCell isn't the finer of the two...
  }

  CellTopoPtr fineSideTopo = fineCell->topology()->getSubcell(sideDim, sideOrdinal);
  
  CubatureFactory cubFactory;
  int cubDegree = 4; // fairly arbitrary choice: enough to get a decent number of test points...
  Teuchos::RCP<Cubature<double> > fineSideTopoCub = cubFactory.create(fineSideTopo, cubDegree);
  
  int numCubPoints = fineSideTopoCub->getNumPoints();
  
  FieldContainer<double> cubPoints(numCubPoints, sideDim);
  FieldContainer<double> cubWeights(numCubPoints); // we neglect these...
  
  fineSideTopoCub->getCubature(cubPoints, cubWeights);
  
  FieldContainer<double> sideRefCellNodes(fineSideTopo->getNodeCount(),sideDim);
  CamelliaCellTools::refCellNodesForTopology(sideRefCellNodes, fineSideTopo);

  int numTestPoints = numCubPoints + fineSideTopo->getNodeCount();
  
  FieldContainer<double> testPoints(numTestPoints, sideDim);
  for (int ptOrdinal=0; ptOrdinal<testPoints.dimension(0); ptOrdinal++) {
    if (ptOrdinal<fineSideTopo->getNodeCount()) {
      for (int d=0; d<sideDim; d++) {
        testPoints(ptOrdinal,d) = sideRefCellNodes(ptOrdinal,d);
      }
    } else {
      for (int d=0; d<sideDim; d++) {
        testPoints(ptOrdinal,d) = cubPoints(ptOrdinal-fineSideTopo->getNodeCount(),d);
      }
    }
  }
  
  fineSideRefPoints = testPoints;
  fineCellRefPoints.resize(numTestPoints, spaceDim);
  
  CamelliaCellTools::mapToReferenceSubcell(fineCellRefPoints, testPoints, sideDim, sideOrdinal, fineCell->topology());
  
  CellTopoPtr coarseSideTopo = neighborCell->topology()->getSubcell(sideDim, neighborInfo.second);
  
  unsigned fineSideAncestorPermutation = fineCell->ancestralPermutationForSubcell(sideDim, sideOrdinal);
  unsigned coarseSidePermutation = neighborCell->subcellPermutation(sideDim, neighborInfo.second);
  
  unsigned coarseSideAncestorPermutationInverse = CamelliaCellTools::permutationInverse(coarseSideTopo, coarseSidePermutation);
  
  unsigned composedPermutation = CamelliaCellTools::permutationComposition(coarseSideTopo, fineSideAncestorPermutation, coarseSideAncestorPermutationInverse); // goes from coarse ordering to fine.
  
  RefinementBranch fineRefBranch = fineCell->refinementBranchForSide(sideOrdinal);
  
  FieldContainer<double> fineSideNodes(fineSideTopo->getNodeCount(), sideDim);  // relative to the ancestral cell whose neighbor is compatible
  if (fineRefBranch.size() == 0) {
    CamelliaCellTools::refCellNodesForTopology(fineSideNodes, coarseSideTopo, composedPermutation);
  } else {
    FieldContainer<double> ancestralSideNodes(coarseSideTopo->getNodeCount(), sideDim);
    CamelliaCellTools::refCellNodesForTopology(ancestralSideNodes, coarseSideTopo, composedPermutation);
    
    RefinementBranch fineSideRefBranch = RefinementPattern::sideRefinementBranch(fineRefBranch, sideOrdinal);
    fineSideNodes = RefinementPattern::descendantNodes(fineSideRefBranch, ancestralSideNodes);
  }
  
  BasisCachePtr sideTopoCache = Teuchos::rcp( new BasisCache(fineSideTopo, 1, false) );
  sideTopoCache->setRefCellPoints(testPoints);
  
  // add cell dimension
  fineSideNodes.resize(1,fineSideNodes.dimension(0), fineSideNodes.dimension(1));
  sideTopoCache->setPhysicalCellNodes(fineSideNodes, vector<GlobalIndexType>(), false);
  coarseSideRefPoints = sideTopoCache->getPhysicalCubaturePoints();
  
  // strip off the cell dimension:
  coarseSideRefPoints.resize(coarseSideRefPoints.dimension(1),coarseSideRefPoints.dimension(2));
  
  coarseCellRefPoints.resize(numTestPoints,spaceDim);
  CamelliaCellTools::mapToReferenceSubcell(coarseCellRefPoints, coarseSideRefPoints, sideDim, neighborInfo.second, neighborCell->topology());

  return true; // containers filled....
}

bool MeshTestUtility::fcsAgree(const FieldContainer<double> &fc1, const FieldContainer<double> &fc2, double tol, double &maxDiff) {
  if (fc1.size() != fc2.size()) {
    maxDiff = -1.0; // a signal something's wrong…
    return false;
  }
  maxDiff = 0.0;
  for (int i=0; i<fc1.size(); i++) {
    maxDiff = max(maxDiff, abs(fc1[i] - fc2[i]));
  }
  return (maxDiff <= tol);
}

bool MeshTestUtility::neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh) {
  int numGlobalDofs = mesh->globalDofCount();
  Epetra_SerialComm Comm;
  Epetra_BlockMap map(numGlobalDofs, 1, 0, Comm);
  Epetra_Vector testSolutionCoefficients(map);
  for (int i=0; i<numGlobalDofs; i++) {
    testSolutionCoefficients[i] =  0.1 * i;
  }

  return neighborBasesAgreeOnSides(mesh, testSolutionCoefficients);
}

bool MeshTestUtility::neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, Epetra_MultiVector &globalSolutionCoefficients) {
  bool success = true;
  MeshTopologyPtr meshTopo = mesh->getTopology();
  int spaceDim = meshTopo->getSpaceDim();
  
  set<IndexType> activeCellIndices = meshTopo->getActiveCellIndices();
  for (set<IndexType>::iterator cellIt=activeCellIndices.begin(); cellIt != activeCellIndices.end(); cellIt++) {
    IndexType cellIndex = *cellIt;
    CellPtr cell = meshTopo->getCell(cellIndex);
    
    BasisCachePtr fineCellBasisCache = BasisCache::basisCacheForCell(mesh, cellIndex);
    ElementTypePtr fineElemType = mesh->getElementType(cellIndex);
    DofOrderingPtr fineElemTrialOrder = fineElemType->trialOrderPtr;
    
    FieldContainer<double> fineSolutionCoefficients(fineElemTrialOrder->totalDofs());
    mesh->globalDofAssignment()->interpretGlobalCoefficients(cellIndex, fineSolutionCoefficients, globalSolutionCoefficients);
//    if ((cellIndex==0) || (cellIndex==2)) {
//      cout << "MeshTestUtility: local coefficients for cell " << cellIndex << ":\n" << fineSolutionCoefficients;
//    }

    unsigned sideCount = cell->getSideCount();
    for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      FieldContainer<double> fineSideRefPoints, fineCellRefPoints, coarseSideRefPoints, coarseCellRefPoints;
      bool hasCoarserNeighbor = determineRefTestPointsForNeighbors(meshTopo, cell, sideOrdinal, fineSideRefPoints, fineCellRefPoints, coarseSideRefPoints, coarseCellRefPoints);
      if (!hasCoarserNeighbor) continue;
      
      pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal);
      
      CellPtr neighborCell = meshTopo->getCell(neighborInfo.first);
      
      unsigned numTestPoints = coarseCellRefPoints.dimension(0);
      
      //        cout << "testing neighbor agreement between cell " << cellIndex << " and " << neighborCell->cellIndex() << endl;
      
      // if we get here, the cell has a neighbor on this side, and is at least as fine as that neighbor.
      
      BasisCachePtr fineSideBasisCache = fineCellBasisCache->getSideBasisCache(sideOrdinal);
      
      fineCellBasisCache->setRefCellPoints(fineCellRefPoints);
      
      BasisCachePtr coarseCellBasisCache = BasisCache::basisCacheForCell(mesh, neighborInfo.first);
      BasisCachePtr coarseSideBasisCache = coarseCellBasisCache->getSideBasisCache(neighborInfo.second);
    
      coarseCellBasisCache->setRefCellPoints(coarseCellRefPoints);
      
      bool hasSidePoints = (fineSideRefPoints.size() > 0);
      if (hasSidePoints) {
        fineSideBasisCache->setRefCellPoints(fineSideRefPoints);
        coarseSideBasisCache->setRefCellPoints(coarseSideRefPoints);
      }
      
      // sanity check: do physical points match?

      FieldContainer<double> fineCellPhysicalCubaturePoints = fineCellBasisCache->getPhysicalCubaturePoints();
      FieldContainer<double> coarseCellPhysicalCubaturePoints = coarseCellBasisCache->getPhysicalCubaturePoints();

      double tol = 1e-14;
      double maxDiff = 0;
      if (! fcsAgree(coarseCellPhysicalCubaturePoints, fineCellPhysicalCubaturePoints, tol, maxDiff) ) {
        cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: fine and coarse cell cubature points do not agree.\n";
        success = false;
        continue;
      }
      
      if (hasSidePoints) {
        FieldContainer<double> fineSidePhysicalCubaturePoints = fineSideBasisCache->getPhysicalCubaturePoints();
        FieldContainer<double> coarseSidePhysicalCubaturePoints = coarseSideBasisCache->getPhysicalCubaturePoints();
        
        double tol = 1e-14;
        double maxDiff = 0;
        if (! fcsAgree(fineSidePhysicalCubaturePoints, fineCellPhysicalCubaturePoints, tol, maxDiff) ) {
          cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: fine side and cell cubature points do not agree.\n";
          success = false;
          continue;
        }
        if (! fcsAgree(coarseSidePhysicalCubaturePoints, coarseCellPhysicalCubaturePoints, tol, maxDiff) ) {
          cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: coarse side and cell cubature points do not agree.\n";
          success = false;
          continue;
        }
        if (! fcsAgree(coarseSidePhysicalCubaturePoints, fineSidePhysicalCubaturePoints, tol, maxDiff) ) {
          cout << "ERROR: MeshTestUtility::neighborBasesAgreeOnSides internal error: fine and coarse side cubature points do not agree.\n";
          success = false;
          continue;
        }
      }
      
      ElementTypePtr coarseElementType = mesh->getElementType(neighborInfo.first);
      DofOrderingPtr coarseElemTrialOrder = coarseElementType->trialOrderPtr;
      
      FieldContainer<double> coarseSolutionCoefficients(coarseElemTrialOrder->totalDofs());
      mesh->globalDofAssignment()->interpretGlobalCoefficients(neighborInfo.first, coarseSolutionCoefficients, globalSolutionCoefficients);
      
      set<int> varIDs = fineElemTrialOrder->getVarIDs();
      for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
        int varID = *varIt;
//        cout << "MeshTestUtility: varID " << varID << ":\n";
        bool isTraceVar = mesh->bilinearForm()->isFluxOrTrace(varID);
        BasisPtr fineBasis, coarseBasis;
        BasisCachePtr fineCache, coarseCache;
        if (isTraceVar) {
          if (! hasSidePoints) continue; // then nothing to do for traces
          fineBasis = fineElemTrialOrder->getBasis(varID, sideOrdinal);
          coarseBasis = coarseElemTrialOrder->getBasis(varID, neighborInfo.second);
          fineCache = fineSideBasisCache;
          coarseCache = coarseSideBasisCache;
        } else {
          fineBasis = fineElemTrialOrder->getBasis(varID);
          coarseBasis = coarseElemTrialOrder->getBasis(varID);
          fineCache = fineCellBasisCache;
          coarseCache = coarseCellBasisCache;
          
          IntrepidExtendedTypes::EFunctionSpaceExtended fs = fineBasis->functionSpace();
          if ((fs == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL)
              || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL)
              || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_TENSOR_HVOL)) {
            // volume L^2 basis: no continuities expected...
            continue;
          }
        }
        FieldContainer<double> localValuesFine = *fineCache->getTransformedValues(fineBasis, OP_VALUE);
        FieldContainer<double> localValuesCoarse = *coarseCache->getTransformedValues(coarseBasis, OP_VALUE);
        
        bool scalarValued = (localValuesFine.rank() == 3);
        
        // the following used if vector or tensor-valued:
        Teuchos::Array<int> valueDim;
        unsigned componentsPerValue = 1;
        FieldContainer<double> valueContainer; // just used for enumeration computation...
        if (!scalarValued) {
          localValuesFine.dimensions(valueDim);
          // clear first three:
          valueDim.erase(valueDim.begin());
          valueDim.erase(valueDim.begin());
          valueDim.erase(valueDim.begin());
          valueContainer.resize(valueDim);
          componentsPerValue = valueContainer.size();
        }
        
        //          if (localValuesFine.rank() != 3) {
        //            cout << "WARNING: MeshTestUtility::neighborBasesAgreeOnSides() only supports scalar-valued bases right now.  Skipping check for varID " << varID << endl;
        //            continue;
        //          }
        
        FieldContainer<double> localPointValuesFine(fineElemTrialOrder->totalDofs());
        FieldContainer<double> localPointValuesCoarse(coarseElemTrialOrder->totalDofs());
        
        for (int valueComponentOrdinal=0; valueComponentOrdinal<componentsPerValue; valueComponentOrdinal++) {
          Teuchos::Array<int> valueMultiIndex(valueContainer.rank());
          
          if (!scalarValued)
            valueContainer.getMultiIndex(valueMultiIndex, valueComponentOrdinal);
          
          Teuchos::Array<int> localValuesMultiIndex(localValuesFine.rank());
          
          for (int r=0; r<valueMultiIndex.size(); r++) {
            localValuesMultiIndex[r+3] = valueMultiIndex[r];
          }
          
          for (int ptOrdinal=0; ptOrdinal<numTestPoints; ptOrdinal++) {
            localPointValuesCoarse.initialize(0);
            localPointValuesFine.initialize(0);
            localValuesMultiIndex[2] = ptOrdinal;
            
            double fineSolutionValue = 0, coarseSolutionValue = 0;
            
            for (int basisOrdinal=0; basisOrdinal < fineBasis->getCardinality(); basisOrdinal++) {
              int fineDofIndex;
              if (isTraceVar)
                fineDofIndex = fineElemTrialOrder->getDofIndex(varID, basisOrdinal, sideOrdinal);
              else
                fineDofIndex = fineElemTrialOrder->getDofIndex(varID, basisOrdinal);
              if (scalarValued) {
                localPointValuesFine(fineDofIndex) = localValuesFine(0,basisOrdinal,ptOrdinal);
              } else {
                localValuesMultiIndex[1] = basisOrdinal;
                localPointValuesFine(fineDofIndex) = localValuesFine.getValue(localValuesMultiIndex);
              }
              
              fineSolutionValue += fineSolutionCoefficients(fineDofIndex) * localPointValuesFine(fineDofIndex);
            }
            for (int basisOrdinal=0; basisOrdinal < coarseBasis->getCardinality(); basisOrdinal++) {
              int coarseDofIndex;
              if (isTraceVar)
                coarseDofIndex = coarseElemTrialOrder->getDofIndex(varID, basisOrdinal, neighborInfo.second);
              else
                coarseDofIndex = coarseElemTrialOrder->getDofIndex(varID, basisOrdinal);
              if (scalarValued) {
                localPointValuesCoarse(coarseDofIndex) = localValuesCoarse(0,basisOrdinal,ptOrdinal);
              } else {
                localValuesMultiIndex[1] = basisOrdinal;
                localPointValuesCoarse(coarseDofIndex) = localValuesCoarse.getValue(localValuesMultiIndex);
              }
              coarseSolutionValue += coarseSolutionCoefficients(coarseDofIndex) * localPointValuesCoarse(coarseDofIndex);
            }
            
            if (abs(coarseSolutionValue - fineSolutionValue) > 1e-13) {
              success = false;
              cout << "coarseSolutionValue (" << coarseSolutionValue << ") and fineSolutionValue (" << fineSolutionValue << ") differ by " << abs(coarseSolutionValue - fineSolutionValue);
              cout << " at point " << ptOrdinal << " for varID " << varID << ".  ";
              cout << "This may be an indication that something is amiss with the global-to-local map.\n";
            } else {
//              // DEBUGGING:
//              cout << "solution value at point (";
//              for (int d=0; d<spaceDim-1; d++) {
//                cout << fineSidePhysicalCubaturePoints(0,ptOrdinal,d) << ", ";
//              }
//              cout << fineSidePhysicalCubaturePoints(0,ptOrdinal,spaceDim-1) << "): ";
//              cout << coarseSolutionValue << endl;
            }
            
            FieldContainer<double> globalValuesFromFine, globalValuesFromCoarse;
            FieldContainer<GlobalIndexType> globalDofIndicesFromFine, globalDofIndicesFromCoarse;
            
            mesh->globalDofAssignment()->interpretLocalData(cellIndex, localPointValuesFine, globalValuesFromFine, globalDofIndicesFromFine);
            mesh->globalDofAssignment()->interpretLocalData(neighborInfo.first, localPointValuesCoarse, globalValuesFromCoarse, globalDofIndicesFromCoarse);
            
            std::map<GlobalIndexType, double> fineValuesMap;
            std::map<GlobalIndexType, double> coarseValuesMap;
            
            for (int i=0; i<globalDofIndicesFromCoarse.size(); i++) {
              GlobalIndexType globalDofIndex = globalDofIndicesFromCoarse[i];
              coarseValuesMap[globalDofIndex] = globalValuesFromCoarse[i];
            }
            
            double maxDiff = 0;
            for (int i=0; i<globalDofIndicesFromFine.size(); i++) {
              GlobalIndexType globalDofIndex = globalDofIndicesFromFine[i];
              fineValuesMap[globalDofIndex] = globalValuesFromFine[i];
              
              double diff = abs( fineValuesMap[globalDofIndex] - coarseValuesMap[globalDofIndex]);
              maxDiff = max(diff, maxDiff);
              if (diff > tol) {
                success = false;
                cout << "interpreted fine and coarse disagree at point (";
                for (int d=0; d<spaceDim; d++) {
                  cout << fineCellPhysicalCubaturePoints(0,ptOrdinal,d);
                  if (d==spaceDim-1)
                    cout <<  ").\n";
                  else
                    cout << ", ";
                }
              }
            }
            if (maxDiff > tol) {
              cout << "maxDiff: " << maxDiff << endl;
              cout << "globalValuesFromFine:\n" << globalValuesFromFine;
              cout << "globalValuesFromCoarse:\n" << globalValuesFromCoarse;
              
              cout << "globalDofIndicesFromFine:\n" << globalDofIndicesFromFine;
              cout <<  "globalDofIndicesFromCoarse:\n" << globalDofIndicesFromCoarse;
              
              continue; // only worth testing further if we passed the above
            }
          }
        }
      }
    }
  }
  
//  cout << "Completed neighborBasesAgreeOnSides.\n";
  return success;
}

