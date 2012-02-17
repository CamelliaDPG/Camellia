//
//  MeshRefinementTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "DofOrdering.h"
#include "MeshRefinementTests.h"

typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;

void MeshRefinementTests::preStiffnessExpectedUniform(FieldContainer<double> &preStiff, 
                                                      double h, ElementTypePtr elemType) {
  DofOrderingPtr trialOrder = elemType->trialOrderPtr;
  DofOrderingPtr testOrder = elemType->testOrderPtr;
  preStiff.resize(1,testOrder->totalDofs(),trialOrder->totalDofs());
  int testID  =  testOrder->getVarIDs()[0]; // just one
  int trialID = trialOrder->getVarIDs()[0]; // just one (the flux)
  
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
      if (abs(testValues(trialOrdinal,pointIndex)-1.0) < tol) {
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
      preStiff(0,testDofIndex,trialDofIndex) = 2.0/3.0 * h; // computed manually
      testOrdinal = phi_ordinals[ phiNodes[1-nodeIndex] ]; // ordinal of the test that "disagrees"
      testDofIndex = testOrder->getDofIndex(testID,testOrdinal);
      preStiff(0,testDofIndex,trialDofIndex) = 1.0/3.0 * h; // computed manually
    }
  }
}

void MeshRefinementTests::preStiffnessExpectedPatch(FieldContainer<double> &preStiff, double h, 
                                                    const map<int,int> &sidesWithBiggerNeighborToIndexInParentSide,
                                                    ElementTypePtr elemType) {
  
}

void MeshRefinementTests::preStiffnessExpectedMulti(FieldContainer<double> &preStiff, double h,
                                                    const set<int> &brokenSides, ElementTypePtr elemType) {
  
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
  
  _fluxBilinearForm = Teuchos::rcp( new TestBilinearFormFlux() );
  
  _multiA = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _multiA->setUsePatchBasis(false);
  vector<int> cellsToRefine;
  cellsToRefine.push_back(0); // hard-coding cellID: would be better to test for the point, but I'm being lazy
  _multiA->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  // get elements
  _A1multi = _multiA->getElement(1);
  _A4multi = _multiA->getElement(4);
  _A5multi = _multiA->getElement(5);
  
  _multiB = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _multiB->setUsePatchBasis(false);
  
  _B1multi = _multiB->getElement(1);
  
  cellsToRefine.clear();
  
  _multiC = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
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
  _patchA = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _patchA->setUsePatchBasis(true);
  cellsToRefine.clear();
  cellsToRefine.push_back(0); // hard-coding cellID
  _patchA->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  // get elements
  _A1patch = _patchA->getElement(1);
  _A4patch = _patchA->getElement(4);
  _A5patch = _patchA->getElement(5);
  
  _patchB = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
  _patchB->setUsePatchBasis(true);
  
  _B1patch = _patchB->getElement(1);
  
  cellsToRefine.clear();
  
  _patchC = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _fluxBilinearForm, H1Order, H1Order+delta_p);
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
  
  setup();
  if (testPatchBasisStiffnessMatrices()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool MeshRefinementTests::testUniformMeshStiffnessMatrices() {
  bool success = false;
  
  cout << "testUniformMeshStiffnessMatrices unimplemented.\n";
  
  return success;
}

bool MeshRefinementTests::testMultiBasisStiffnessMatrices() {
  bool success = false;
  
  cout << "testMultiBasisStiffnessMatrices unimplemented.\n";
  
  return success;  
}

bool MeshRefinementTests::testPatchBasisStiffnessMatrices() {
  bool success = false;
  
  cout << "testPatchBasisStiffnessMatrices unimplemented.\n";
  
  return success;
}