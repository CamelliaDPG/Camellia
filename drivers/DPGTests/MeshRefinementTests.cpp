//
//  MeshRefinementTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "DofOrdering.h"
#include "MeshRefinementTests.h"
#include "BilinearFormUtility.h"
#include "MeshFactory.h"

#include "PoissonFormulation.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

bool MeshRefinementTests::checkMultiElementStiffness(Teuchos::RCP<Mesh> mesh) {
  bool success = true;
  
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    if ( ! checkMultiElementStiffness(mesh, cellID) ) {
      success = false;
    }
  }
  return success;  
}

bool MeshRefinementTests::checkMultiElementStiffness(Teuchos::RCP<Mesh> mesh, int cellID) {
  bool success = true;
  
  double tol = 1e-14;
  double maxDiff;
  
  FieldContainer<double> expectedValues;
  FieldContainer<double> physicalCellNodes;
  FieldContainer<double> actualValues;
  FieldContainer<double> sideParities;
  
  set<int> brokenSideSet;
  
  ElementPtr elem = mesh->getElement(cellID);
  
  // if the element is a child, then h = _h_small; otherwise, h = _h:
  double h = elem->isChild() ? _h_small : _h;
  
  // determine expected values:
  multiBrokenSides(brokenSideSet,elem);
  ElementTypePtr elemType = elem->elementType();
  sideParities = mesh->cellSideParitiesForCell(cellID);
  { // check that the test can proceed:
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    int testID  =  *(testOrder->getVarIDs().begin()); // just grab the first one
    BasisPtr testBasis = testOrder->getBasis(testID);
    if (! testBasis->isConforming()) {
      static bool haveWarned = false;
      if (!haveWarned) {
        cout << "checkMultiElementStiffness(): test relies on conforming test basis, but we don't have one.  Skipping test with a PASS.\n";
        haveWarned = true;
      }

      return success;
    }
  }
  preStiffnessExpectedMulti(expectedValues,h,brokenSideSet,elemType,sideParities);
  
  if (cellID == 1) {
//    cout << "MultiBasis expectedValues for cell " << cellID << ":\n";
//    cout << expectedValues;
  }
  
  // get actual values:
  physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
  BilinearFormUtility::computeStiffnessMatrixForCell(actualValues, mesh, cellID);
  
  if ( !fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Failure in element " << cellID <<  " stiffness computation (using Multi-Basis); maxDiff = " << maxDiff << endl;
    cout << "expectedValues:\n" << expectedValues;
    cout << "actualValues:\n" << actualValues;
    success = false;
  }
  
  return success;
}

bool MeshRefinementTests::checkPatchElementStiffness(Teuchos::RCP<Mesh> mesh) {
  bool success = true;
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    if ( ! checkPatchElementStiffness(mesh, cellID ) ) {
      success = false;
    }
  }
  return success;
}

bool MeshRefinementTests::checkPatchElementStiffness(Teuchos::RCP<Mesh> mesh, int cellID) {
  bool success = true;
  
  double tol = 1e-14;
  double maxDiff;
  
  FieldContainer<double> expectedValues;
  FieldContainer<double> physicalCellNodes;
  FieldContainer<double> actualValues;
  FieldContainer<double> sideParities;
  
  map<int,int> parentSideMap;
  
  ElementPtr elem = mesh->getElement(cellID);

  // if the element is a child, then h = _h_small; otherwise, h = _h:
  double h = elem->isChild() ? _h_small : _h;
  
  // determine expected values:
  patchParentSideIndices(parentSideMap,mesh,elem);
  ElementTypePtr elemType = elem->elementType();
  sideParities = mesh->cellSideParitiesForCell(cellID);
  
  { // check that the test can proceed:
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    int testID  =  *(testOrder->getVarIDs().begin()); // just grab the first one
    BasisPtr testBasis = testOrder->getBasis(testID);
    if (! testBasis->isConforming()) {
      static bool haveWarned = false;
      if (! haveWarned ) {
        cout << "checkPatchElementStiffness(): test relies on conforming test basis, but we don't have one.  Skipping test with a PASS.\n";
        haveWarned = true;
      }
      return success;
    }
  }
  
  preStiffnessExpectedPatch(expectedValues,h,parentSideMap,elemType,sideParities);
  
  // get actual values:
  physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
  BilinearFormUtility::computeStiffnessMatrixForCell(actualValues, mesh, cellID);
  
  if ( !fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Failure in element " << cellID <<  " stiffness computation; maxDiff = " << maxDiff << endl;
    cout << "expectedValues:\n" << expectedValues;
    cout << "actualValues:\n" << actualValues;
    success = false;
  }
  return success;
}

void MeshRefinementTests::multiBrokenSides(set<int> &brokenSideSet, ElementPtr elem) {
  brokenSideSet.clear();
  int numSides = elem->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {  
    ElementPtr neighbor;
    int sideIndexInNeighbor;
    neighbor = elem->getNeighbor(sideIndexInNeighbor, sideIndex);
    if (neighbor.get() == NULL) {
      // boundary or this is a small side; either way, not broken
      continue;
    }
    if (neighbor->isParent() && (neighbor->childIndicesForSide(sideIndexInNeighbor).size() > 1)) {
      // if neighbor is a parent *and* broken along the given side, then MultiBasis…
      brokenSideSet.insert(sideIndex);
    }
  }
}

void MeshRefinementTests::patchParentSideIndices(map<int,int> &parentSideIndices, Teuchos::RCP<Mesh> mesh, ElementPtr elem) {
  // returns map from child side index --> child index in the matching parent side (0 or 1)
  parentSideIndices.clear();
  int numSides = elem->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    if (elem->getNeighborCellID(sideIndex) == -1) {
      int ancestralNeighborSideIndex;
      ElementPtr ancestralNeighbor = mesh->ancestralNeighborForSide(elem, sideIndex, ancestralNeighborSideIndex);
      if (ancestralNeighbor.get() != NULL) { // NOT boundary side
        int parentSideIndex = elem->parentSideForSideIndex(sideIndex);
        parentSideIndices[sideIndex] = elem->indexInParentSide(parentSideIndex);
      }
    }
  }
}

void MeshRefinementTests::preStiffnessExpectedUniform(FieldContainer<double> &preStiff, 
                                                      double h, ElementTypePtr elemType,
                                                      FieldContainer<double> &sideParities) {
  double h_ratio = h / 2.0; // because the master line has length 2...
  
  DofOrderingPtr trialOrder = elemType->trialOrderPtr;
  DofOrderingPtr testOrder = elemType->testOrderPtr;
  preStiff.resize(1,testOrder->totalDofs(),trialOrder->totalDofs());
  int testID  =  *(testOrder->getVarIDs().begin()); // just grab the first one
  int trialID = *(trialOrder->getVarIDs().begin()); // just grab the first one (the flux)
  
  int numPoints = 4;
  FieldContainer<double> refPoints2D(numPoints,2); // quad nodes (for bilinear basis)
  refPoints2D(0,0) = -1.0;
  refPoints2D(0,1) = -1.0;
  refPoints2D(1,0) =  1.0;
  refPoints2D(1,1) = -1.0;
  refPoints2D(2,0) =  1.0;
  refPoints2D(2,1) =  1.0;
  refPoints2D(3,0) = -1.0;
  refPoints2D(3,1) =  1.0;
  
  int phi_ordinals[numPoints]; // phi_i = 1 at node x_i, and 0 at other nodes
  
  double tol = 1e-15;
  BasisPtr testBasis = testOrder->getBasis(testID);
  FieldContainer<double> testValues(testBasis->getCardinality(),refPoints2D.dimension(0));
  testBasis->getValues(testValues, refPoints2D, Intrepid::OPERATOR_VALUE);
  for (int testOrdinal=0; testOrdinal<testBasis->getCardinality(); testOrdinal++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      if (abs(testValues(testOrdinal,pointIndex)-1.0) < tol) {
        phi_ordinals[pointIndex] = testOrdinal;
      }
    }
  }
  
  BasisPtr trialBasis = trialOrder->getBasis(trialID,0); // uniform: all sides should be the same
  
  numPoints = 2;
  FieldContainer<double> refPoints1D(numPoints,1); // line nodes (for linear basis)
  refPoints2D(0,0) = -1.0;
  refPoints2D(1,0) = 1.0;
  int v_ordinals[numPoints];
  
  FieldContainer<double> trialValues(trialBasis->getCardinality(),refPoints1D.dimension(0));
  trialBasis->getValues(trialValues, refPoints1D, Intrepid::OPERATOR_VALUE);
  for (int trialOrdinal=0; trialOrdinal<trialBasis->getCardinality(); trialOrdinal++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      if (abs(trialValues(trialOrdinal,pointIndex)-1.0) < tol) {
        v_ordinals[pointIndex] = trialOrdinal;
      }
    }
  }
  
  map<int,pair<int,int> > phiNodesForSide; // sideIndex --> pair( phiNodeIndex for v0, phiNodeIndex for v1)
  phiNodesForSide[0] = make_pair(0,1);
  phiNodesForSide[1] = make_pair(1,2);
  phiNodesForSide[2] = make_pair(2,3);
  phiNodesForSide[3] = make_pair(3,0);
  
  preStiff.initialize(0.0);
  int numSides = 4;
  int phiNodes[2];
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    phiNodes[0] = phiNodesForSide[sideIndex].first;
    phiNodes[1] = phiNodesForSide[sideIndex].second;
    for (int nodeIndex=0; nodeIndex<2; nodeIndex++) { // loop over the line's two nodes
      int testOrdinal = phi_ordinals[ phiNodes[nodeIndex] ]; // ordinal of the test that "agrees"
      int testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
      int trialDofIndex = trialOrder->getDofIndex(trialID,nodeIndex,sideIndex);
      preStiff(0,testDofIndex,trialDofIndex) += 2.0/3.0 * h_ratio * sideParities(0,sideIndex); // "2/3" computed manually
      testOrdinal = phi_ordinals[ phiNodes[1-nodeIndex] ]; // ordinal of the test that "disagrees"
      testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
      preStiff(0,testDofIndex,trialDofIndex) += 1.0/3.0 * h_ratio * sideParities(0,sideIndex); // "1/3" computed manually
    }
  }
}

void MeshRefinementTests::preStiffnessExpectedPatch(FieldContainer<double> &preStiff, double h, 
                                                    const map<int,int> &sidesWithBiggerNeighborToIndexInParentSide,
                                                    ElementTypePtr elemType,
                                                    FieldContainer<double> &sideParities) {
  double h_ratio = h / 2.0; // because the master line has length 2...
  
  DofOrderingPtr trialOrder = elemType->trialOrderPtr;
  DofOrderingPtr testOrder = elemType->testOrderPtr;
  preStiff.resize(1,testOrder->totalDofs(),trialOrder->totalDofs());
  int testID  =  *(testOrder->getVarIDs().begin()); // just grab the first one
  int trialID = *(trialOrder->getVarIDs().begin()); // just grab the first one (the flux)
  
  int numPoints = 4;
  FieldContainer<double> refPoints2D(numPoints,2); // quad nodes (for bilinear basis)
  refPoints2D(0,0) = -1.0;
  refPoints2D(0,1) = -1.0;
  refPoints2D(1,0) =  1.0;
  refPoints2D(1,1) = -1.0;
  refPoints2D(2,0) =  1.0;
  refPoints2D(2,1) =  1.0;
  refPoints2D(3,0) = -1.0;
  refPoints2D(3,1) =  1.0;
  
  int phi_ordinals[numPoints]; // phi_i = 1 at node x_i, and 0 at other nodes
  
  double tol = 1e-15;
  BasisPtr testBasis = testOrder->getBasis(testID);
  FieldContainer<double> testValues(testBasis->getCardinality(),refPoints2D.dimension(0));
  testBasis->getValues(testValues, refPoints2D, Intrepid::OPERATOR_VALUE);
  for (int testOrdinal=0; testOrdinal<testBasis->getCardinality(); testOrdinal++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      if (abs(testValues(testOrdinal,pointIndex)-1.0) < tol) {
        phi_ordinals[pointIndex] = testOrdinal;
      }
    }
  }
  
  BasisPtr trialBasis = trialOrder->getBasis(trialID,0); // uniform: all sides should be the same
  
  numPoints = 2;
  FieldContainer<double> refPoints1D(numPoints,1); // line nodes (for linear basis)
  refPoints2D(0,0) = -1.0;
  refPoints2D(1,0) = 1.0;
  int v_ordinals[numPoints];
  
  FieldContainer<double> trialValues(trialBasis->getCardinality(),refPoints1D.dimension(0));
  trialBasis->getValues(trialValues, refPoints1D, Intrepid::OPERATOR_VALUE);
  for (int trialOrdinal=0; trialOrdinal<trialBasis->getCardinality(); trialOrdinal++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      if (abs(trialValues(trialOrdinal,pointIndex)-1.0) < tol) {
        v_ordinals[pointIndex] = trialOrdinal;
      }
    }
  }
  
  map<int,pair<int,int> > phiNodesForSide; // sideIndex --> pair( phiNodeIndex for v0, phiNodeIndex for v1)
  phiNodesForSide[0] = make_pair(0,1);
  phiNodesForSide[1] = make_pair(1,2);
  phiNodesForSide[2] = make_pair(2,3);
  phiNodesForSide[3] = make_pair(3,0);
  
  preStiff.initialize(0.0);
  int numSides = 4;
  int phiNodes[2];
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    phiNodes[0] = phiNodesForSide[sideIndex].first;
    phiNodes[1] = phiNodesForSide[sideIndex].second;
    
    double agreeValue[2], disagreeValue[2];
    
    agreeValue[0]    = 2.0 / 3.0;  // for non-patch bases, if nodes for phi and v0 line up
    disagreeValue[0] = 1.0 / 3.0;  // for non-patch bases, if nodes for phi and v0 don't line up
    agreeValue[1]    = 2.0 / 3.0;  // for non-patch bases, if nodes for phi and v1 line up
    disagreeValue[1] = 1.0 / 3.0;  // for non-patch bases, if nodes for phi and v1 don't line up
    
    map<int,int>::const_iterator patchFoundIt = sidesWithBiggerNeighborToIndexInParentSide.find(sideIndex);
    bool hasPatchBasis = (patchFoundIt != sidesWithBiggerNeighborToIndexInParentSide.end());
    if ( hasPatchBasis ) {
      int indexInParentSide = patchFoundIt->second;
      if (indexInParentSide == 0) {
        agreeValue[0]    = 5.0 / 6.0;
        disagreeValue[0] = 2.0 / 3.0;
        agreeValue[1]    = 1.0 / 3.0;
        disagreeValue[1] = 1.0 / 6.0;
      } else if (indexInParentSide == 1) {
        agreeValue[0]    = 1.0 / 3.0;
        disagreeValue[0] = 1.0 / 6.0;
        agreeValue[1]    = 5.0 / 6.0;
        disagreeValue[1] = 2.0 / 3.0;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported indexInParentSide.");
      }
    }
    
    for (int nodeIndex=0; nodeIndex<2; nodeIndex++) { // loop over the line's two nodes
      int testOrdinal = phi_ordinals[ phiNodes[nodeIndex] ]; // ordinal of the test that "agrees"
      int testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
      int trialDofIndex = trialOrder->getDofIndex(trialID,nodeIndex,sideIndex);
      preStiff(0,testDofIndex,trialDofIndex) += agreeValue[nodeIndex] * h_ratio * sideParities(0,sideIndex);
      testOrdinal = phi_ordinals[ phiNodes[1-nodeIndex] ]; // ordinal of the test that "disagrees"
      testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
      preStiff(0,testDofIndex,trialDofIndex) += disagreeValue[nodeIndex] * h_ratio * sideParities(0,sideIndex);
    }
  }
}

void MeshRefinementTests::preStiffnessExpectedMulti(FieldContainer<double> &preStiff, double h,
                                                    const set<int> &brokenSides, ElementTypePtr elemType,
                                                    FieldContainer<double> &sideParities) {
  double h_ratio = h / 2.0; // because the master line has length 2...
  
  DofOrderingPtr trialOrder = elemType->trialOrderPtr;
  DofOrderingPtr testOrder = elemType->testOrderPtr;
  preStiff.resize(1,testOrder->totalDofs(),trialOrder->totalDofs());
  int testID  =  *(testOrder->getVarIDs().begin()); // just grab the first one
  int trialID = *(trialOrder->getVarIDs().begin()); // just grab the first one (the flux)
  
  int numPoints = 4;
  FieldContainer<double> refPoints2D(numPoints,2); // quad nodes (for bilinear basis)
  refPoints2D(0,0) = -1.0;
  refPoints2D(0,1) = -1.0;
  refPoints2D(1,0) =  1.0;
  refPoints2D(1,1) = -1.0;
  refPoints2D(2,0) =  1.0;
  refPoints2D(2,1) =  1.0;
  refPoints2D(3,0) = -1.0;
  refPoints2D(3,1) =  1.0;
  
  int phi_ordinals[numPoints]; // phi_i = 1 at node x_i, and 0 at other nodes
  
  double tol = 1e-15;
  BasisPtr testBasis = testOrder->getBasis(testID);
  FieldContainer<double> testValues(testBasis->getCardinality(),refPoints2D.dimension(0));
  testBasis->getValues(testValues, refPoints2D, Intrepid::OPERATOR_VALUE);
  for (int testOrdinal=0; testOrdinal<testBasis->getCardinality(); testOrdinal++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      if (abs(testValues(testOrdinal,pointIndex)-1.0) < tol) {
        phi_ordinals[pointIndex] = testOrdinal;
      }
    }
  }
  
  BasisPtr trialBasis = trialOrder->getBasis(trialID,0); // uniform: all sides should be the same
  
  numPoints = 2;
  FieldContainer<double> refPoints1D(numPoints,1); // line nodes (for linear basis)
  refPoints2D(0,0) = -1.0;
  refPoints2D(1,0) = 1.0;
  int v_ordinals[numPoints];
  
  FieldContainer<double> trialValues(trialBasis->getCardinality(),refPoints1D.dimension(0));
  trialBasis->getValues(trialValues, refPoints1D, Intrepid::OPERATOR_VALUE);
  for (int trialOrdinal=0; trialOrdinal<trialBasis->getCardinality(); trialOrdinal++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      if (abs(trialValues(trialOrdinal,pointIndex)-1.0) < tol) {
        v_ordinals[pointIndex] = trialOrdinal;
      }
    }
  }
  
  map<int,pair<int,int> > phiNodesForSide; // sideIndex --> pair( phiNodeIndex for v0, phiNodeIndex for v1)
  phiNodesForSide[0] = make_pair(0,1);
  phiNodesForSide[1] = make_pair(1,2);
  phiNodesForSide[2] = make_pair(2,3);
  phiNodesForSide[3] = make_pair(3,0);
  
  preStiff.initialize(0.0);
  int numSides = 4;
  int phiNodes[2];
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    phiNodes[0] = phiNodesForSide[sideIndex].first;
    phiNodes[1] = phiNodesForSide[sideIndex].second;
    
    double agreeValue[2], disagreeValue[2];
    
    agreeValue[0]    = 2.0 / 3.0;  // for non-multi bases, if nodes for phi and v0 line up
    disagreeValue[0] = 1.0 / 3.0;  // for non-multi bases, if nodes for phi and v0 don't line up
    agreeValue[1]    = 2.0 / 3.0;  // for non-multi bases, if nodes for phi and v1 line up
    disagreeValue[1] = 1.0 / 3.0;  // for non-multi bases, if nodes for phi and v1 don't line up
    
    bool hasMultiBasis = (brokenSides.find(sideIndex) != brokenSides.end());
    int numSubSides = hasMultiBasis ? 2 : 1; // because of our setup, max subsides is 2
        
    for (int subSideIndex=0; subSideIndex < numSubSides; subSideIndex++) {
      if (hasMultiBasis) {
        // here, the meaning of agreement: if phi belongs to the first node in the big element edge and the little edge node is the first, that's agreement 
        if (subSideIndex == 0) {
          agreeValue[0]    = 5.0 / 12.0;
          disagreeValue[0] = 1.0 / 12.0;
          agreeValue[1]    = 1.0 /  6.0;
          disagreeValue[1] = 1.0 /  3.0;
        } else {
          agreeValue[0]    = 1.0 /  6.0;
          disagreeValue[0] = 1.0 /  3.0;
          agreeValue[1]    = 5.0 / 12.0;
          disagreeValue[1] = 1.0 / 12.0;
        }
      }
      for (int nodeIndex=0; nodeIndex<2; nodeIndex++) { // loop over the line's two nodes
        int testOrdinal = phi_ordinals[ phiNodes[nodeIndex] ]; // ordinal of the test that "agrees"
        int testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
        int trialDofIndex;
        if (hasMultiBasis) {
          trialDofIndex = trialOrder->getDofIndex(trialID,nodeIndex,sideIndex,subSideIndex);
//          cout << "trialDofIndex for trialID " << trialID << ", node " << nodeIndex << ", subSideIndex " << subSideIndex << ": " << trialDofIndex << endl;
        } else {
          trialDofIndex = trialOrder->getDofIndex(trialID,nodeIndex,sideIndex);
        }
        preStiff(0,testDofIndex,trialDofIndex) += agreeValue[nodeIndex] * h_ratio * sideParities(0,sideIndex);
        testOrdinal = phi_ordinals[ phiNodes[1-nodeIndex] ]; // ordinal of the test that "disagrees"
        testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
        preStiff(0,testDofIndex,trialDofIndex) += disagreeValue[nodeIndex] * h_ratio * sideParities(0,sideIndex);
      }
    }
  }
}

void MeshRefinementTests::setup() {
  // throughout setup, we have hard-coded cellIDs.
  // this will break if the way mesh or the refinements order their elements changes.
  // not hard, though tedious, to test for the points we know should be in the elements.
  // (see the diagram in the header file for the mesh layouts, which will make clear which points
  //  should be in which elements.)
  // (It should be the case that all such hard-coding is isolated to this method.)
  
  // MultiBasis meshes:
  FieldContainer<double> quadPoints(4,2);
  
  // setup so that the big h is 1.0, small h is 0.5:
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 2.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 2.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int H1Order = 2;  // means linear for fluxes
  int delta_p = -1; // means tests will likewise be (bi-)linear
  int horizontalCells = 2; int verticalCells = 1;
  
  _h = 1.0;
  _h_small = 0.5;
  
  _fluxBilinearForm = Teuchos::rcp( new TestBilinearFormFlux() );
  
  _multiA = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _multiA->setUsePatchBasis(false);
  vector<GlobalIndexType> cellsToRefine;
  cellsToRefine.push_back(0); // hard-coding cellID: would be better to test for the point, but I'm being lazy
  _multiA->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  // get elements
  _A1multi = _multiA->getElement(1);
  _A3multi = _multiA->getElement(3);
  _A4multi = _multiA->getElement(4);
  _A5multi = _multiA->getElement(5);
  
  _multiB = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _multiB->setUsePatchBasis(false);
  
  _B1multi = _multiB->getElement(1);
  
  cellsToRefine.clear();
  
  _multiC = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _multiC->setUsePatchBasis(false);
  
  cellsToRefine.push_back(0); // hard-coding cellID: would be better to test for the point, but I'm being lazy
  _multiC->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  cellsToRefine.clear();
  cellsToRefine.push_back(1); // hard-coding cellID: would be better to test for the point, but I'm being lazy
  _multiC->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  cellsToRefine.clear();

  _C4multi = _multiC->getElement(4);
  _C5multi = _multiC->getElement(5);
  
  // PatchBasis meshes:
  _patchA = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _patchA->setUsePatchBasis(true);
  cellsToRefine.clear();
  cellsToRefine.push_back(0); // hard-coding cellID
  _patchA->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  // get elements
  _A1patch = _patchA->getElement(1);
  _A3patch = _patchA->getElement(3);
  _A4patch = _patchA->getElement(4);
  _A5patch = _patchA->getElement(5);
  
  _patchB = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _patchB->setUsePatchBasis(true);
  
  _B1patch = _patchB->getElement(1);
  
  cellsToRefine.clear();
  
  _patchC = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _patchC->setUsePatchBasis(true);
  cellsToRefine.push_back(0); // hard-coding cellID
  _patchC->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  cellsToRefine.clear();
  cellsToRefine.push_back(1); // hard-coding cellID
  _patchC->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  cellsToRefine.clear();
  
  _C4patch = _patchC->getElement(4);
  _C5patch = _patchC->getElement(5);  
}

void MeshRefinementTests::teardown() {
  
}

void MeshRefinementTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testTraceTermProjection()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testPRefinements()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testMultiBasisSideParities()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  setup();
//  if (testPatchBasisSideParities()) {
//    numTestsPassed++;
//  }
//  numTestsRun++;
//  teardown();
  
  setup();
  if (testUniformMeshStiffnessMatrices()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testMultiBasisStiffnessMatrices()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  setup();
//  if (testPatchBasisStiffnessMatrices()) {
//    numTestsPassed++;
//  }
//  numTestsRun++;
//  teardown();
}

bool MeshRefinementTests::testUniformMeshStiffnessMatrices() {
  bool success = true;
  
  double tol = 1e-14;
  double maxDiff;
  
  FieldContainer<double> expectedValues;
  FieldContainer<double> physicalCellNodes;
  FieldContainer<double> actualValues;
  FieldContainer<double> sideParities;
  
  // test that _B1{multi|patch} and _C4{multi|patch} have the expected values
  
  // B1: (large element)
  // multi:
  // determine expected values:
  ElementTypePtr elemType = _B1multi->elementType();
  sideParities = _multiB->cellSideParitiesForCell(_B1multi->cellID());
  { // check that the test can proceed:
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    int testID  =  *(testOrder->getVarIDs().begin()); // just grab the first one
    BasisPtr testBasis = testOrder->getBasis(testID);
    if (! testBasis->isConforming()) {
      static bool haveWarned = false;
      if (! haveWarned ) {
        cout << "testUniformMeshStiffnessMatrices(): test relies on conforming test basis, but we don't have one.  Skipping test with a PASS.\n";
        haveWarned = true;
      }
      return success;
    }
  }
  preStiffnessExpectedUniform(expectedValues,_h,elemType,sideParities);
  // get actual values:
  physicalCellNodes = _multiB->physicalCellNodesForCell(_B1multi->cellID());
  BilinearFormUtility::computeStiffnessMatrixForCell(actualValues, _multiB, _B1multi->cellID());

  if ( !fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Failure in uniform mesh B (with usePatchBasis=false) stiffness computation; maxDiff = " << maxDiff << endl;
    cout << "expectedValues:\n" << expectedValues;
    cout << "actualValues:\n" << actualValues;
    success = false;
  }
  
  // patch:
  // determine expected values:
  elemType = _B1patch->elementType();
  sideParities = _patchB->cellSideParitiesForCell(_B1patch->cellID());
  preStiffnessExpectedUniform(expectedValues,_h,elemType,sideParities);
  // get actual values:
  physicalCellNodes = _patchB->physicalCellNodesForCell(_B1patch->cellID());
  BilinearFormUtility::computeStiffnessMatrixForCell(actualValues, _patchB, _B1patch->cellID());
  
  if ( !fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Failure in uniform mesh B (with usePatchBasis=true) stiffness computation; maxDiff = " << maxDiff << endl;
    cout << "expectedValues:\n" << expectedValues;
    cout << "actualValues:\n" << actualValues;
    success = false;
  }
  
  // C4: (small element)
  // multi:
  // determine expected values:
  elemType = _C4multi->elementType();
  sideParities = _multiC->cellSideParitiesForCell(_C4multi->cellID());
  preStiffnessExpectedUniform(expectedValues,_h_small,elemType,sideParities);
  // get actual values:
  physicalCellNodes = _multiC->physicalCellNodesForCell(_C4multi->cellID());
  BilinearFormUtility::computeStiffnessMatrixForCell(actualValues, _multiC, _C4multi->cellID());
  
  if ( !fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Failure in uniform mesh C (with usePatchBasis=false) stiffness computation; maxDiff = " << maxDiff << endl;
    cout << "expectedValues:\n" << expectedValues;
    cout << "actualValues:\n" << actualValues;
    success = false;
  }
  
  // patch:
  // determine expected values:
  elemType = _C4patch->elementType();
  sideParities = _patchC->cellSideParitiesForCell(_C4patch->cellID());
  preStiffnessExpectedUniform(expectedValues,_h_small,elemType,sideParities);
  // get actual values:
  physicalCellNodes = _patchC->physicalCellNodesForCell(_C4patch->cellID());
  BilinearFormUtility::computeStiffnessMatrixForCell(actualValues, _patchC, _C4patch->cellID());
  
  if ( !fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Failure in uniform mesh C (with usePatchBasis=true) stiffness computation; maxDiff = " << maxDiff << endl;
    cout << "expectedValues:\n" << expectedValues;
    cout << "actualValues:\n" << actualValues;
    success = false;
  }
  
  return success;
}

bool MeshRefinementTests::testMultiBasisStiffnessMatrices() {
  bool success = true;
  
  if ( ! checkMultiElementStiffness(_multiA) ) {
    success = false;
  }
  
  if ( ! checkMultiElementStiffness(_multiB) ) {
    success = false;
  }
  
  if ( ! checkMultiElementStiffness(_multiC) ) {
    success = false;
  }
  
  return success;
}

bool MeshRefinementTests::testPatchBasisStiffnessMatrices() {
  bool success = true;
  
  if ( ! checkPatchElementStiffness(_patchA) ) {
    success = false;
  }
  
  if ( ! checkPatchElementStiffness(_patchB) ) {
    success = false;
  }
  
  if ( ! checkPatchElementStiffness(_patchC) ) {
    success = false;
  }
  
  return success;
}

bool MeshRefinementTests::testMultiBasisSideParities() {
  bool success = true;
  
  FieldContainer<double> A0_side_parities = _multiA->cellSideParitiesForCell(_A3multi->getParent()->cellID());
  FieldContainer<double> A1_side_parities = _multiA->cellSideParitiesForCell(_A1multi->cellID());
  FieldContainer<double> A3_side_parities = _multiA->cellSideParitiesForCell(_A3multi->cellID());
  FieldContainer<double> A4_side_parities = _multiA->cellSideParitiesForCell(_A4multi->cellID());
  
  // check the entry for sideIndex 1 (the patchBasis side)
  if (A0_side_parities(0,1) != A3_side_parities(0,1)) {
    success = false;
    cout << "Failure: MultiBasisSideParities: child doesn't match parent along broken side.\n";
  }
  if (A3_side_parities(0,1) != A4_side_parities(0,1)) {
    success = false;
    cout << "Failure: MultiBasisSideParities: children don't match each other along parent side.\n";
  }
  if (A3_side_parities(0,1) != - A1_side_parities(0,3)) {
    success = false;
    cout << "Failure: MultiBasisSideParities: children aren't opposite large neighbor cell parity.\n";
  }
  return success;
}

bool MeshRefinementTests::testPatchBasisSideParities() {
  bool success = true;
  
  FieldContainer<double> A0_side_parities = _patchA->cellSideParitiesForCell(_A3patch->getParent()->cellID());
  FieldContainer<double> A1_side_parities = _patchA->cellSideParitiesForCell(_A1patch->cellID());
  FieldContainer<double> A3_side_parities = _patchA->cellSideParitiesForCell(_A3patch->cellID());
  FieldContainer<double> A4_side_parities = _patchA->cellSideParitiesForCell(_A4patch->cellID());
  
  // check the entry for sideIndex 1 (the patchBasis side)
  if (A0_side_parities(0,1) != A3_side_parities(0,1)) {
    success = false;
    cout << "Failure: PatchBasisSideParities: child doesn't match parent along broken side.\n";
  }
  if (A3_side_parities(0,1) != A4_side_parities(0,1)) {
    success = false;
    cout << "Failure: PatchBasisSideParities: children don't match each other along parent side.\n";
  }
  if (A3_side_parities(0,1) != - A1_side_parities(0,3)) {
    success = false;
    cout << "Failure: PatchBasisSideParities: children aren't opposite large neighbor cell parity.\n";
  }

  return success;
}

bool cellsHaveH1Order(MeshPtr mesh, int H1Order, set<GlobalIndexType> cellIDs) {
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    int cellOrder = mesh->globalDofAssignment()->getH1Order(cellID);
    if (cellOrder != H1Order) {
      cout << "cell " << cellID << "'s H1Order " << cellOrder << " does not match expected " << H1Order << endl;
      return false;
    }
  }
  return true;
}

bool MeshRefinementTests::testPRefinements() {
  // make a few simple meshes:
  bool success = true;
  
  VarFactory vf;
  vf.fieldVar("u1");
  vf.fluxVar("un_hat");
  vf.traceVar("u_hat");
  vf.testVar("v", HGRAD);
  BFPtr emptyBF = Teuchos::rcp( new BF(vf) );
  
  vector<double> dimensions1D(1,1.0), dimensions2D(2,1.0), dimensions3D(3,1.0);
  vector<int> elements1D(1,3), elements2D(2,2), elements3D(3,1);
  
  int H1Order = 1;
  MeshPtr mesh1D = MeshFactory::rectilinearMesh(emptyBF, dimensions1D, elements1D, H1Order);
  MeshPtr mesh2D = MeshFactory::rectilinearMesh(emptyBF, dimensions2D, elements2D, H1Order);
  MeshPtr mesh3D = MeshFactory::rectilinearMesh(emptyBF, dimensions3D, elements3D, H1Order);
  
  vector< MeshPtr > meshes;
  meshes.push_back(mesh1D);
  meshes.push_back(mesh2D);
  meshes.push_back(mesh3D);
  
  for (int meshOrdinal=0; meshOrdinal < meshes.size(); meshOrdinal++) {
    MeshPtr mesh = meshes[meshOrdinal];
    
    // check that the H1Orders are right to begin with:
    set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
    if (! cellsHaveH1Order(mesh, H1Order, cellIDs)) {
      cout << "Internal test error: initial mesh order does not match expected.\n";
      success = false;
    }
    
    // first, try a uniform p-refinement:
    mesh->pRefine(cellIDs,2);
    H1Order += 2;
    if (! cellsHaveH1Order(mesh, H1Order, cellIDs)) {
      cout << "Test failure: after p-refinement (by 2), cells do not match expected order.\n";
      success = false;
    }

    // now, a uniform p-unrefinement to take us back:
    mesh->pRefine(cellIDs,-2);
    H1Order -= 2;
    if (! cellsHaveH1Order(mesh, H1Order, cellIDs)) {
      cout << "Test failure: after p-refinement (by 2), cells do not match expected order.\n";
      success = false;
    }
    
    // now, h-refine
    CellPtr cell0 = mesh->getTopology()->getCell(0);
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cell0->topology()->getKey());
    set<GlobalIndexType> set0;
    set0.insert(0);
    mesh->hRefine(set0, refPattern);
    
    cellIDs = mesh->getActiveCellIDs();
    if (! cellsHaveH1Order(mesh, H1Order, cellIDs)) {
      cout << "After h-refinement, mesh order does not match expected.\n";
      success = false;
    }
    // uniform p-refinement:
    mesh->pRefine(cellIDs,1);
    H1Order += 1;
    if (! cellsHaveH1Order(mesh, H1Order, cellIDs)) {
      cout << "After h-refinement followed by p-refinement, mesh order does not match expected.\n";
      success = false;
    }
    if (! cellsHaveH1Order(mesh, H1Order, set0)) {
      cout << "After h-refinement followed by p-refinement, cell 0 (the parent) does not have the new polynomial order.\n";
      success = false;
    }
    
    mesh->pRefine(cellIDs,-1);
    H1Order -= 1;
    if (! cellsHaveH1Order(mesh, H1Order, cellIDs)) {
      cout << "After h-refinement followed by p-refinement and p-unrefinement, mesh order does not match expected.\n";
      success = false;
    }
    if (! cellsHaveH1Order(mesh, H1Order, set0)) {
      cout << "After h-refinement followed by p-refinement and p-unrefinement, cell 0 (the parent) does not have the new polynomial order.\n";
      success = false;
    }
  }
  return success;
}

bool MeshRefinementTests::testTraceTermProjection() {
  cout << "WARNING: testTraceTermProjection not yet finished.\n";
  /*
   */
  
  bool success = true;
  
  int spaceDim = 2;
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr phi_exact; // want an exactly representable solution with non-trivial psi_n (i.e. grad phi dot n should be non-trivial)
  // for now, we just go very simple.  Linear in x,y,z.
  switch (spaceDim) {
    case 1:
      phi_exact = x;
      break;
    case 2:
      phi_exact = x + y;
      break;
    case 3:
      phi_exact = x + y + z;
      break;
    default:
      cout << "MeshRefinementTests::testTraceTermProjection(): unhandled space dimension.\n";
      break;
  }
  
//  int H1Order = 5; // debugging
  int H1Order = 2; // so field order is linear
  
  bool useConformingTraces = true;
  PoissonFormulation pf(spaceDim,useConformingTraces);
  
  BFPtr bf = pf.bf();
  
  // fields
  VarPtr phi = pf.phi();
  VarPtr psi = pf.psi();
  
  // traces
  VarPtr phi_hat = pf.phi_hat();
  VarPtr psi_n = pf.psi_n_hat();
  
  // tests
  VarPtr tau = pf.tau();
  VarPtr q = pf.q();
  
  int testSpaceEnrichment = spaceDim; //
  //  double width = 1.0, height = 1.0, depth = 1.0;
  
  vector<double> dimensions;
  for (int d=0; d<spaceDim; d++) {
    dimensions.push_back(1.0);
  }
  
  //  cout << "dimensions[0] = " << dimensions[0] << "; dimensions[1] = " << dimensions[1] << endl;
  //  cout << "numCells[0] = " << numCells[0] << "; numCells[1] = " << numCells[1] << endl;
  
  vector<int> numCells(spaceDim, 2); // one element in each spatial direction
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, numCells, H1Order, testSpaceEnrichment);
  
  // rhs = f * q, where f = \Delta phi
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f;
  switch (spaceDim) {
    case 1:
      f = phi_exact->dx()->dx();
      break;
    case 2:
      f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
      break;
    case 3:
      f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled spaceDim");
      break;
  }
  rhs->addTerm(f * q);
  
  IPPtr graphNorm = bf->graphNorm();
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  SolutionPtr solution;
  
  bc->addDirichlet(phi_hat, boundary, phi_exact);
  solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  solution->solve();
  
  FunctionPtr psi_exact = (spaceDim > 1) ? phi_exact->grad() : phi_exact->dx();
  
  FunctionPtr psi_n_exact;
  FunctionPtr n = Function::normal();
  if (spaceDim > 1) {
    psi_n_exact = psi_exact * n;
  } else {
    psi_n_exact = psi_exact;
  }

  FunctionPtr psi_soln = Function::solution(psi, solution);
  FunctionPtr psi_n_soln = Function::solution(psi_n, solution); // includes (correction for) parity weighting
  FunctionPtr phi_hat_soln = Function::solution(phi_hat, solution);
  
  FunctionPtr psi_err = psi_exact - psi_soln;
  FunctionPtr psi_n_err = psi_n_exact - psi_n_soln;
  FunctionPtr phi_hat_err = phi_exact - phi_hat_soln;
  
  double err_L2 = psi_err->l2norm(mesh);
  
  double tol = 1e-12;
  
  // SANITY CHECKS ON INITIAL SOLUTION
  // psi error first
  if (err_L2 > tol) {
    cout << "testTraceTermProjection error: psi in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }
  
  // psi_n error:
  err_L2 = psi_n_err->l2norm(mesh);
  if (err_L2 > tol) {
    cout << "testTraceTermProjection error: psi_n in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }
  
  // phi_hat error:
  err_L2 = phi_hat_err->l2norm(mesh);
  if (err_L2 > tol) {
    cout << "testTraceTermProjection error: phi_hat in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }
  
  // do a uniform refinement, then check that psi_n_soln and phi_hat_soln match the exact
  CellPtr sampleCell = mesh->getTopology()->getCell(0);
  CellTopoPtr cellTopo = sampleCell->topology();
  RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo->getKey());
  
  mesh->registerSolution(solution); // this way, solution will get the memo to project
  mesh->hRefine(mesh->getActiveCellIDs(), refPattern);
  
  err_L2 = phi_hat_err->l2norm(mesh);
  if (err_L2 > tol) {
    cout << "testTraceTermProjection failure: projected phi_hat differs from exact by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }
  
  err_L2 = psi_n_err->l2norm(mesh);
  if (err_L2 > tol) {
    cout << "testTraceTermProjection failure: projected psi_n differs from exact by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }
  
  if (success==false) { // then export
#ifdef HAVE_EPETRAEXT_HDF5
    HDF5Exporter solnExporter(mesh, "soln");
    VarFactory vf = bf->varFactory();
    solnExporter.exportSolution(solution, vf, 0, 10);
    
    HDF5Exporter fxnExporter(mesh, "fxn");
    vector<string> fxnNames;
    vector< FunctionPtr > fxns;
    // fields:
    fxnNames.push_back("psi_exact");
    fxns.push_back(psi_exact);
    fxnNames.push_back("psi_soln");
    fxns.push_back(psi_soln);
    
    fxnNames.push_back("psi_exact_x");
    fxns.push_back(psi_exact->x());
    fxnNames.push_back("psi_soln_x");
    fxns.push_back(psi_soln->x());
    
    fxnNames.push_back("psi_exact_y");
    fxns.push_back(psi_exact->y());
    fxnNames.push_back("psi_soln_y");
    fxns.push_back(psi_soln->y());
    fxnExporter.exportFunction(fxns, fxnNames, 0, 10);
    
    // traces:
    fxnNames.clear();
    fxns.clear();
    fxnNames.push_back("psi_n_exact");
    fxns.push_back(psi_n_exact);
    fxnNames.push_back("psi_n_soln");
    fxns.push_back(psi_n_soln);
    fxnNames.push_back("psi_n_err");
    fxns.push_back(psi_n_err);
    fxnExporter.exportFunction(fxns, fxnNames, 0, 10);
#endif
  }
  
  return success;
}