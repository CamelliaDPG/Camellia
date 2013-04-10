//
//  RHSTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RHSTests.h"
#include <Teuchos_GlobalMPISession.hpp>

// Shards includes
#include "Shards_CellTopology.hpp"
#include "DofOrderingFactory.h"

#include "TestBilinearFormDx.h"
#include "TestRHSOne.h"
#include "TestRHSLinear.h"
#include "BilinearFormUtility.h"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

#include "ConfusionBilinearForm.h"
#include "ConfusionProblemLegacy.h"

#include "InnerProductScratchPad.h"

void RHSTests::runTests(int &numTestsRun, int &numTestsPassed) {
//  setup();
//  if (testComputeRHSLegacy()) {
//    numTestsPassed++;
//  }
//  numTestsRun++;
//  teardown();
  setup();
  if (testIntegrateAgainstStandardBasis()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testRHSEasy()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testTrivialRHS()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}


void RHSTests::setup() {
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int H1Order = 3;
  int delta_p = 3; // for test functions
  int horizontalCells = 2; int verticalCells = 2;
  
  double eps = 1.0; // not really testing for sharp gradients right now--just want to see if things basically work
  double beta_x = 1.0;
  double beta_y = 1.0;
  
  Teuchos::RCP<ConfusionBilinearForm> confusionBF = Teuchos::rcp( new ConfusionBilinearForm(eps,beta_x,beta_y) );
  Teuchos::RCP<ConfusionProblemLegacy> confusionProblem = Teuchos::rcp( new ConfusionProblemLegacy(confusionBF) );
  _rhs = confusionProblem;
  _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, confusionBF, H1Order, H1Order+delta_p);
  _mesh->setUsePatchBasis(false);
  
  VarFactory varFactory; // Create test IDs that match the enum in ConfusionBilinearForm
  VarPtr tau = varFactory.testVar("\\tau",HDIV,ConfusionBilinearForm::TAU);
  VarPtr v = varFactory.testVar("v",HGRAD,ConfusionBilinearForm::V);
  
  _rhsEasy = Teuchos::rcp(new RHSEasy());
  _rhsEasy->addTerm( v );
}

//bool RHSTests::testComputeRHSLegacy() {
//  // "legacy": copied from DPGTests, not integrated with setup/teardown design
//  string myName = "testComputeRHS";
//  // since we need optTestWeights for regular RHS computation, we just do an identity matrix...
//  bool success = true;
//  int numTests = 1;
//  double tol = 1e-14;
//  int testOrder = 3;
//  
//  //cout << myName << ": testing with testOrder=" << testOrder << endl;
//  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new TestBilinearFormDx() );
//  
//  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
//  shards::CellTopology tri_3(shards::getCellTopologyData<shards::Triangle<3> >() );
//  shards::CellTopology cellTopo;
//  FieldContainer<double> quadPoints(numTests,4,2);
//  quadPoints(0,0,0) = -1.0; // x1
//  quadPoints(0,0,1) = -1.0; // y1
//  quadPoints(0,1,0) = 1.0;
//  quadPoints(0,1,1) = -1.0;
//  quadPoints(0,2,0) = 1.0;
//  quadPoints(0,2,1) = 1.0;
//  quadPoints(0,3,0) = -1.0;
//  quadPoints(0,3,1) = 1.0;
//  
//  FieldContainer<double> triPoints(numTests,3,2);
//  triPoints(0,0,0) = -1.0; // x1
//  triPoints(0,0,1) = -1.0; // y1
//  triPoints(0,1,0) = 1.0;
//  triPoints(0,1,1) = -1.0;
//  triPoints(0,2,0) = 1.0;
//  triPoints(0,2,1) = 1.0;
//  
//  FieldContainer<double> nodePoints;
//  
//  for (int numSides=4; numSides <= 4; numSides++) { // skip analytic triangle integration tests -- we need to correct the Mathematica notebook; the "expectedValues" are incorrect here!!
//    if (numSides == 3) {
//      cellTopo = tri_3;
//      nodePoints = triPoints;
//    } else {
//      cellTopo = quad_4;
//      nodePoints = quadPoints;
//    }
//    
//    DofOrderingFactory dofOrderingFactory(bilinearForm);
//    Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(testOrder, cellTopo);
//    
//    if (numSides == 4) {
//      // now that we have a Lobatto basis, we need to hard-code the basis for which we have precomputed these values...
//      testOrdering = Teuchos::rcp( new DofOrdering );
//      BasisPtr basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(testOrder,POINTTYPE_SPECTRAL)), 2, 0) );
//      testOrdering->addEntry(0, basis, 0);
//    }
//    
//    int numTrialDofs = testOrdering->totalDofs(); // suppose we're symmetric: numTrial = numTest
//    
//    FieldContainer<double> expectedRHSVector(numTests, numTrialDofs);
//    FieldContainer<double> actualRHSVector(numTests, numTrialDofs);
//    
//    FieldContainer<double> optimalTestWeights(numTests, numTrialDofs, numTrialDofs);
//    
//    for (int i=0; i<numTrialDofs; i++) {
//      optimalTestWeights(0,i,i) = 1.0;
//    }
//    
//    TestRHSOne rhs; // rhs == 1 => just integrate the test functions
//    // compute with the fake optimal test weights:
//    BilinearFormUtility::computeRHS(actualRHSVector, bilinearForm, rhs,
//                                    optimalTestWeights, testOrdering,
//                                    cellTopo, nodePoints);
//    
//    if (numSides==3) {
//      TestRHSOne::expectedRHSForCubicOnTri(expectedRHSVector);
//    } else {
//      TestRHSOne::expectedRHSForCubicOnQuad(expectedRHSVector);
//    }
//    
//    double maxDiff;
//    bool localSuccess = fcsAgree(expectedRHSVector,actualRHSVector,tol,maxDiff);
//    
//    if (! localSuccess) {
//      cout << "Failed computeRHS test with rhs=TestRHSOne. numSides: " << numSides << "; maxDiff: " << maxDiff << endl;
//      success = false;
//    }
//    
//    // now try the same thing, but with the quad (0,1)^2 (or the southeast half of it)
//    nodePoints(0,0,0) = 0.0; // x1
//    nodePoints(0,0,1) = 0.0; // y1
//    nodePoints(0,1,0) = 1.0;
//    nodePoints(0,1,1) = 0.0;
//    nodePoints(0,2,0) = 0.0;
//    nodePoints(0,2,1) = 1.0;
//    if (numSides==4) {
//      nodePoints(0,2,0) = 1.0;
//      nodePoints(0,2,1) = 1.0;
//      nodePoints(0,3,0) = 0.0;
//      nodePoints(0,3,1) = 1.0;
//    }
//    if (numSides==3) { // a quick triangle-only test for the lowest-order poly:
//      int lowOrder = 1;
//      Teuchos::RCP<DofOrdering> lowOrderTestOrdering = dofOrderingFactory.testOrdering(lowOrder, cellTopo);
//      int numLowOrderTrialDofs = lowOrderTestOrdering->totalDofs();
//      FieldContainer<double> lowOrderRHSVector(numTests, numLowOrderTrialDofs);
//      FieldContainer<double> lowOrderOptimalTestWeights(numTests, numLowOrderTrialDofs, numLowOrderTrialDofs);
//      
//      for (int i=0; i<numLowOrderTrialDofs; i++) {
//        lowOrderOptimalTestWeights(0,i,i) = 1.0;
//      }
//      BilinearFormUtility::computeRHS(lowOrderRHSVector, bilinearForm, rhs,
//                                      lowOrderOptimalTestWeights, lowOrderTestOrdering,
//                                      cellTopo, nodePoints);
//      
//      TestRHSOne::expectedRHSForLinearOnUnitTri(expectedRHSVector);
//      
//      localSuccess = fcsAgree(expectedRHSVector,lowOrderRHSVector,tol,maxDiff);
//      if (! localSuccess ) {
//        cout << "Failed lower order computeRHS test with on (half) the unit quad rhs=TestRHSOne. numSides: " << numSides  << "; maxDiff: " << maxDiff << endl;
//        success = false;
//      }
//    }
//    
//    BilinearFormUtility::computeRHS(actualRHSVector, bilinearForm, rhs,
//                                    optimalTestWeights, testOrdering,
//                                    cellTopo, nodePoints);
//    
//    if (numSides==3) {
//      TestRHSOne::expectedRHSForCubicOnUnitTri(expectedRHSVector);
//    } else {
//      TestRHSOne::expectedRHSForCubicOnUnitQuad(expectedRHSVector);
//    }
//    
//    localSuccess = fcsAgree(expectedRHSVector,actualRHSVector,tol,maxDiff);
//    if (! localSuccess ) {
//      cout << "Failed computeRHS test with on (half) the unit quad rhs=TestRHSOne. numSides: " << numSides << "; maxDiff: " << maxDiff << endl;
//      success = false;
//    }
//    
//    // reset nodePoints for the linear RHS test:
//    if (numSides == 3) {
//      nodePoints = triPoints;
//    } else {
//      nodePoints = quadPoints;
//    }
//    
//    TestRHSLinear linearRHS;
//    BilinearFormUtility::computeRHS(actualRHSVector, bilinearForm, linearRHS,
//                                    optimalTestWeights, testOrdering,
//                                    cellTopo, nodePoints);
//    
//    if (numSides==3) {
//      TestRHSLinear::expectedRHSForCubicOnTri(expectedRHSVector);
//    } else {
//      TestRHSLinear::expectedRHSForCubicOnQuad(expectedRHSVector);
//    }
//    
//    localSuccess = fcsAgree(expectedRHSVector,actualRHSVector,tol,maxDiff);
//    if (! localSuccess ) {
//      cout << "Failed computeRHS test with rhs=linearRHS. numSides: " << numSides << "; maxDiff: " << maxDiff << endl;
//      success = false;
//    }
//    
//    // now try the same thing, but with the quad (0,1)^2 (or the southeast half of it)
//    nodePoints(0,0,0) = 0.0; // x1
//    nodePoints(0,0,1) = 0.0; // y1
//    nodePoints(0,1,0) = 1.0;
//    nodePoints(0,1,1) = 0.0;
//    nodePoints(0,2,0) = 1.0;
//    nodePoints(0,2,1) = 1.0;
//    if (numSides==4) {
//      nodePoints(0,3,0) = 0.0;
//      nodePoints(0,3,1) = 1.0;
//    }
//    BilinearFormUtility::computeRHS(actualRHSVector, bilinearForm, linearRHS,
//                                    optimalTestWeights, testOrdering,
//                                    cellTopo, nodePoints);
//    
//    if (numSides==3) {
//      TestRHSLinear::expectedRHSForCubicOnUnitTri(expectedRHSVector);
//    } else {
//      TestRHSLinear::expectedRHSForCubicOnUnitQuad(expectedRHSVector);
//    }
//    
//    localSuccess = fcsAgree(expectedRHSVector,actualRHSVector,tol,maxDiff);
//    if (! localSuccess ) {
//      cout << "Failed computeRHS test with on (half) the unit quad rhs=linearRHS. numSides: " << numSides << "; maxDiff: " << maxDiff << endl;
//      success = false;
//    }
//  }
//  return success;
//}

bool RHSTests::testIntegrateAgainstStandardBasis() {
  bool success = true;
  double tol = 1e-14;

  int rank     = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
  Teuchos::RCP<ElementType> elemType = _mesh->getElement(0)->elementType();
  
  vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemType);
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemType);

  int numCells = elemsInPartitionOfType.size();
  int numTestDofs = elemType->testOrderPtr->totalDofs();
  
// set up diagonal testWeights matrices so we can reuse the existing computeRHS, and compare results…
  FieldContainer<double> testWeights(numCells,numTestDofs,numTestDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int i=0; i<numTestDofs; i++) {
      testWeights(cellIndex,i,i) = 1.0; 
    }
  }
  
  FieldContainer<double> rhsExpected(numCells,numTestDofs);
  FieldContainer<double> rhsActual(numCells,numTestDofs);
  
  // determine cellIDs
  vector<int> cellIDs;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = _mesh->cellID(elemType, cellIndex, rank);
    cellIDs.push_back(cellID);
  }
  
  if (numCells > 0) {
    // prepare basisCache and cellIDs
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType,_mesh));
    bool createSideCacheToo = true;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    _rhs->integrateAgainstStandardBasis(rhsActual, elemType->testOrderPtr, basisCache);
    _rhs->integrateAgainstOptimalTests(rhsExpected, testWeights, elemType->testOrderPtr, basisCache);
  }
  
  double maxDiff = 0.0;
  
  if ( ! fcsAgree(rhsExpected,rhsActual,tol,maxDiff) ) {
    success = false;
    cout << "Failed testIntegrateAgainstStandardBasis: maxDiff = " << maxDiff << endl;
  }
  
  // check success across MPI nodes
  return allSuccess(success);
}

bool RHSTests::testRHSEasy() {
  bool success = true;
  double tol = 1e-14;
  
  int rank     = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
  Teuchos::RCP<ElementType> elemType = _mesh->getElement(0)->elementType();
  
  vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemType);
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemType);
  
  int numCells = elemsInPartitionOfType.size();
  int numTestDofs = elemType->testOrderPtr->totalDofs();
  
  FieldContainer<double> rhsExpected(numCells,numTestDofs);
  FieldContainer<double> rhsActual(numCells,numTestDofs);
  
  // determine cellIDs
  vector<int> cellIDs;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = _mesh->cellID(elemType, cellIndex, rank);
    cellIDs.push_back(cellID);
  }
  
  // prepare basisCache and cellIDs
  if (numCells > 0) {
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType,_mesh));
    bool createSideCacheToo = true;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    
    _rhs->integrateAgainstStandardBasis(rhsActual, elemType->testOrderPtr, basisCache);
    _rhsEasy->integrateAgainstStandardBasis(rhsExpected, elemType->testOrderPtr, basisCache);
  }
  
  double maxDiff = 0.0;
  
  if ( ! fcsAgree(rhsExpected,rhsActual,tol,maxDiff) ) {
    success = false;
    cout << "Failed testRHSEasy: maxDiff = " << maxDiff << endl;
    cout << "Expected values:\n" << rhsExpected;
    cout << "Actual values:\n" << rhsActual;
    cout << "Test dof ordering:\n" << *(elemType->testOrderPtr);
  }
  
  return allSuccess(success);
}

bool RHSTests::testTrivialRHS(){
  
  bool success = true;
  double tol = 1e-14;
  
  int rank     = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
  Teuchos::RCP<ElementType> elemType = _mesh->getElement(0)->elementType();
  
  vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemType);
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemType);
  
  int numCells = elemsInPartitionOfType.size();
  int numTestDofs = elemType->testOrderPtr->totalDofs();

  // determine cellIDs
  vector<int> cellIDs;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = _mesh->cellID(elemType, cellIndex, rank);
    cellIDs.push_back(cellID);
  }
  
  if (numCells > 0) {
    // prepare basisCache and cellIDs
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType,_mesh));
    bool createSideCacheToo = true;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    
    FieldContainer<double> rhsExpected(numCells,numTestDofs);
    
    VarFactory varFactory; // Create test IDs that match the enum in ConfusionBilinearForm
    //  VarPtr tau = varFactory.testVar("\\tau",HDIV,ConfusionBilinearForm::TAU);
    VarPtr v = varFactory.testVar("v",HGRAD,ConfusionBilinearForm::V);

    FunctionPtr zero = Function::constant(0.0);
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    FunctionPtr f = zero;
    rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
    rhs->integrateAgainstStandardBasis(rhsExpected, elemType->testOrderPtr, basisCache);
    
    for (int i = 0;i<numCells;i++) {
      for (int j = 0;j<numTestDofs;j++) {
        if (abs(rhsExpected(i,j))>tol) {
          success = false;
        }
      }
    }
  }
  
  return allSuccess(success);
}

std::string RHSTests::testSuiteName() {
  return "RHSTests";
}

void RHSTests::teardown() {
  
}
