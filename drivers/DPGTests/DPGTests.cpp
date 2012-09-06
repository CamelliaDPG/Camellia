// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

#include "Intrepid_FieldContainer.hpp"

#include "BasisFactory.h"
#include "DofOrdering.h"
#include "BilinearFormUtility.h"
#include "MathInnerProduct.h"
#include "PoissonBilinearForm.h"
#include "DofOrderingFactory.h"

#include "Intrepid_FieldContainer.hpp"
//#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_I1_FEM.hpp"

#include "Epetra_SerialDenseMatrix.h"

#include "TestBilinearFormDx.h"
#include "TestBilinearFormTrace.h"
#include "TestBilinearFormAnalyticBoundaryIntegral.h"
#include "TestRHSOne.h"
#include "TestRHSLinear.h"

#include "MeshTestSuite.h"
#include "Mesh.h"
#include "VectorizedBasisTestSuite.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "DPGTests.h"
#include "BasisCacheTests.h"
#include "SolutionTests.h"
#include "MultiBasisTests.h"
#include "PatchBasisTests.h"
#include "ElementTests.h"
#include "MeshRefinementTests.h"
#include "RHSTests.h"
#include "FunctionTests.h"
#include "LinearTermTests.h"
#include "ScratchPadTests.h"

#include "Projector.h"
#include "BasisCache.h"

using namespace std;
using namespace Intrepid;

ElementTypePtr makeElemType(DofOrderingPtr trialOrdering, DofOrderingPtr testOrdering, shards::CellTopology &cellTopo) {
  Teuchos::RCP< shards::CellTopology > cellTopoPtr = Teuchos::rcp(new shards::CellTopology(cellTopo));
  return Teuchos::rcp( new ElementType( trialOrdering, testOrdering, cellTopoPtr) );
}

BasisCachePtr makeBasisCache(ElementTypePtr elemType, const FieldContainer<double> &physicalCellNodes, const vector<int> &cellIDs,
                         bool createSideCacheToo = true) {
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType) );
  basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
  return basisCache;
}

class SimpleQuadraticFunction : public AbstractFunction {
public:    
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    functionValues.resize(numCells,numPoints);
    for (int i=0;i<numCells;i++){
      for (int j=0;j<numPoints;j++){
        double x = physicalPoints(i,j,0);
        double y = physicalPoints(i,j,1);
        functionValues(i,j) = x*y + 3.0*x*x;
      }
    }  
  }
  
};

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  //rank=mpiSession.getRank();
  //numProcs=mpiSession.getNProc();
#else
#endif
  DPGTests::runTests();
}

static const int C1_FAKE_POLY_ORDER = -1;
static const int C3_FAKE_POLY_ORDER = -3;

void DPGTests::createBases() {
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  shards::CellTopology tri_3(shards::getCellTopologyData<shards::Triangle<3> >() );
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis;
  basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C1_FEM<double,FieldContainer<double> >() );
  BasisFactory::registerBasis(basis,0, C1_FAKE_POLY_ORDER, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C1_FEM<double,FieldContainer<double> >() );
  BasisFactory::registerBasis(basis,0, C1_FAKE_POLY_ORDER, tri_3.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  basis = Teuchos::rcp( new Basis_HGRAD_LINE_Cn_FEM<double,FieldContainer<double> >(3,POINTTYPE_SPECTRAL) );
  BasisFactory::registerBasis(basis,0, C3_FAKE_POLY_ORDER, line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  basis = Teuchos::rcp( new Basis_HGRAD_LINE_Cn_FEM<double,FieldContainer<double> >(1,POINTTYPE_SPECTRAL) );
  BasisFactory::registerBasis(basis,0, C1_FAKE_POLY_ORDER, line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  basis = Teuchos::rcp( new Basis_HDIV_QUAD_I1_FEM<double,FieldContainer<double> >() );
  BasisFactory::registerBasis(basis,1, C1_FAKE_POLY_ORDER, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
}

void DPGTests::runTests() {
  bool success;
  int numTestsTotal = 0;
  int numTestsPassed = 0;
  
  // set up a few special entries for BasisFactory first:
  createBases();
  
  // setup our TestSuite tests:
  vector< Teuchos::RCP< TestSuite > > testSuites;
  testSuites.push_back( Teuchos::rcp( new BasisCacheTests ) );
  testSuites.push_back( Teuchos::rcp( new LinearTermTests ) );
  testSuites.push_back( Teuchos::rcp( new MultiBasisTests ) );
  testSuites.push_back( Teuchos::rcp( new PatchBasisTests ) );
  testSuites.push_back( Teuchos::rcp( new SolutionTests ) );
  testSuites.push_back( Teuchos::rcp( new FunctionTests ) );
  testSuites.push_back( Teuchos::rcp( new ScratchPadTests ) );
  testSuites.push_back( Teuchos::rcp( new RHSTests ) );
  testSuites.push_back( Teuchos::rcp( new MeshRefinementTests ) );
  testSuites.push_back( Teuchos::rcp( new ElementTests ) );
  testSuites.push_back( Teuchos::rcp( new VectorizedBasisTestSuite ) );
  testSuites.push_back( Teuchos::rcp( new MeshTestSuite ) );
  
  for ( vector< Teuchos::RCP< TestSuite > >::iterator testSuiteIt = testSuites.begin();
       testSuiteIt != testSuites.end(); testSuiteIt++) {
    Teuchos::RCP< TestSuite > testSuite = *testSuiteIt;
    int numSuiteTests = 0, numSuiteTestsPassed = 0;
    testSuite->runTests(numSuiteTests, numSuiteTestsPassed);
    string name = testSuite->testSuiteName();
    cout << name << ": passed " << numSuiteTestsPassed << "/" << numSuiteTests << " tests." << endl;
    numTestsTotal  += numSuiteTests;
    numTestsPassed += numSuiteTestsPassed;
  }
    
  success = testOptimalStiffnessByIntegrating();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testOptimalStiffnessByIntegrating." << endl;
    //return; // just for now, exit on success    
  } else {
    cout << "Failed test testOptimalStiffnessByIntegrating." << endl;
    //return; // just for now, exit on fail
  }
  
  success = testComputeOptimalTest();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testComputeOptimalTest." << endl;
  } else {
    cout << "Failed test testComputeOptimalTest." << endl;
  }
  
  success = testMathInnerProductDx();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testMathInnerProductDx." << endl;
  } else {
    cout << "Failed test testMathInnerProductDx." << endl;
  }
  
  success = testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices." << endl;
  } else {
    cout << "Failed test testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices." << endl;
  }
  
  success = testComputeStiffnessConformingVertices();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testComputeStiffnessConformingVertices." << endl;
  } else {
    cout << "Failed test testComputeStiffnessConformingVertices." << endl;
  }
  
  
  success = testAnalyticBoundaryIntegral(false);
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testAnalyticBoundaryIntegral (non-conforming)." << endl;
  } else {
    cout << "Failed test testAnalyticBoundaryIntegral (non-conforming)." << endl;
  }
  
  success = testAnalyticBoundaryIntegral(true);
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testAnalyticBoundaryIntegral (conforming)." << endl;
  } else {
    cout << "Failed test testAnalyticBoundaryIntegral (conforming)." << endl;
  }
  
  success = testOptimalStiffnessByMultiplying();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testOptimalStiffnessByMultiplying." << endl;
  } else {
    cout << "Failed test testOptimalStiffnessByMultiplying." << endl;
  }
  
  success = testLowOrderTrialCubicTest();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testLowOrderTrialCubicTest." << endl;
  } else {
    cout << "Failed test testLowOrderTrialCubicTest." << endl;
  }
  
  success = testComputeStiffnessDx();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test ComputeStiffnessDx." << endl;
  } else {
    cout << "Failed test ComputeStiffnessDx." << endl;
  }
  
  success = testWeightBasis();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testWeightBasis." << endl;
  } else {
    cout << "Failed test testWeightBasis." << endl;
  }
  
  success = testDofOrdering();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testDofOrdering." << endl;
  } else {
    cout << "Failed test testDofOrdering." << endl;
  }
  
  success = testComputeStiffnessTrace();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test ComputeStiffnessTrace." << endl;
  } else {
    cout << "Failed test ComputeStiffnessTrace." << endl;
  }

  success = testComputeStiffnessFlux();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test ComputeStiffnessFlux." << endl;
  } else {
    cout << "Failed test ComputeStiffnessFlux." << endl;
  }
  
  success = testComputeOptimalTestPoisson();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testComputeOptimalTestPoisson." << endl;
  } else {
    cout << "Failed test testComputeOptimalTestPoisson." << endl;
  }

  success = testProjection();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    cout << "Passed test testProjection." << endl;
  } else {
    cout << "Failed test testProjection." << endl;
  }
  
  cout << "Passed " << numTestsPassed << " out of " << numTestsTotal << "." << endl;
}

bool DPGTests::testComputeStiffnessConformingVertices() {
  bool oldWarnState = BilinearFormUtility::warnAboutZeroRowsAndColumns();
  BilinearFormUtility::setWarnAboutZeroRowsAndColumns(false);
  
  bool success = true;
  
  string myName = "testComputeStiffnessConformingVertices";
  
  Teuchos::RCP<TestBilinearFormTrace> bilinearForm = Teuchos::rcp(new TestBilinearFormTrace());
  
  int polyOrder = 3; 
  Teuchos::RCP<DofOrdering> conformingOrdering,nonConformingOrdering;
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  DofOrderingFactory dofOrderingFactory(bilinearForm);
  
  conformingOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, true);
  nonConformingOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, false);
  
  Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(polyOrder, quad_4);
  
  int numSides = 4;
  int numDofsPerSide = 4;
  
  int numTrialDofsConforming = conformingOrdering->totalDofs();
  int numTrialDofsNonConforming = nonConformingOrdering->totalDofs();
  int numTestDofs = testOrdering->totalDofs();
  
  int numTests = 1;
  
  double tol = 1e-15;
  
  FieldContainer<double> conformingStiffness(numTests, numTestDofs, numTrialDofsConforming);
  FieldContainer<double> nonConformingStiffness(numTests, numTestDofs, numTrialDofsNonConforming);
  
  FieldContainer<double> expectedConformingStiffness(numTests, numTestDofs, numTrialDofsConforming);
  
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  BilinearFormUtility::computeStiffnessMatrix(conformingStiffness, bilinearForm,
                                              conformingOrdering, testOrdering,
                                              quad_4, quadPoints,cellSideParities);
  BilinearFormUtility::computeStiffnessMatrix(nonConformingStiffness, bilinearForm,
                                              nonConformingOrdering, testOrdering,
                                              quad_4, quadPoints,cellSideParities);
  
  expectedConformingStiffness.initialize(0.0);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    for (int dofOrdinal=0; dofOrdinal<numDofsPerSide; dofOrdinal++) {
      int trialDofIndexConforming = conformingOrdering->getDofIndex(0, dofOrdinal, sideIndex);
      int trialDofIndexNonConforming = nonConformingOrdering->getDofIndex(0, dofOrdinal, sideIndex);
      for (int testDofIndex=0; testDofIndex<numTestDofs; testDofIndex++) {
        expectedConformingStiffness(0,testDofIndex,trialDofIndexConforming) 
        += nonConformingStiffness(0,testDofIndex,trialDofIndexNonConforming);
      }
    }
  }
  
  success = fcsAgree(myName,expectedConformingStiffness,conformingStiffness,tol);

  BilinearFormUtility::setWarnAboutZeroRowsAndColumns(oldWarnState);
  return success;

}

bool DPGTests::testDofOrdering() {
  bool success = true;
  DofOrdering traceOrdering,testOrdering;
  
  string myName = "testDofOrdering";
  
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > traceBasis
  = 
  BasisFactory::getBasis(basisRank,C1_FAKE_POLY_ORDER,
                         line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  int numSides = 4;
  
  for (int i=0; i<numSides; i++) {
    traceOrdering.addEntry(0,traceBasis,0,i);
  }
  
  int numFieldsPerFlux = traceBasis->getCardinality();
  for (int i=0; i<numSides; i++) {
    int expectedIndex = i*numFieldsPerFlux;
    int actualIndex = traceOrdering.getDofIndex(0,0,i);
    if (expectedIndex != actualIndex) {
      cout << myName << ": expected " << expectedIndex << " but had " << actualIndex << endl;
      success = false;
    }
  }
  
  // now, test the DofPairing mechanism, and the Utility's use of it...
  int dofsPerSide = 2;
  numSides = 4;
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp(new TestBilinearFormTrace());
  int polyOrder = 1; // keep things simple
  Teuchos::RCP<DofOrdering> trialOrder;
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  DofOrderingFactory dofOrderingFactory(bilinearForm);
  trialOrder = dofOrderingFactory.trialOrdering(polyOrder, quad_4);
  
  FieldContainer<int> expectedDofIndices(numSides,dofsPerSide);
  expectedDofIndices(0,0) = 0;
  expectedDofIndices(0,1) = 1;
  expectedDofIndices(1,0) = 1;
  expectedDofIndices(1,1) = 2;
  expectedDofIndices(2,0) = 2;
  expectedDofIndices(2,1) = 3;
  expectedDofIndices(3,0) = 3;
  expectedDofIndices(3,1) = 0;
  
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    for (int dofOrdinal=0; dofOrdinal<dofsPerSide; dofOrdinal++) {
      int actualDofIndex = trialOrder->getDofIndex(0, dofOrdinal, sideIndex);
      int expectedDofIndex = expectedDofIndices(sideIndex,dofOrdinal);
      if ( ! expectedDofIndex == actualDofIndex ) {
        cout << myName << ": failed conforming vertex test for sideIndex " << sideIndex << ", dofOrdinal " << dofOrdinal << "." << endl;
        cout << "Expected " << expectedDofIndex << "; actual was " << actualDofIndex << "." << endl;
        success = false;
      }
    }
  }
  
  // now with cubics
  shards::CellTopology tri_3(shards::getCellTopologyData<shards::Triangle<3> >() );
  for (numSides=3; numSides <= 4; numSides++) {
    polyOrder = 3;
    dofsPerSide = polyOrder+1;
    if (numSides == 3) {
      trialOrder = dofOrderingFactory.trialOrdering(polyOrder, tri_3);
    } else {
      trialOrder = dofOrderingFactory.trialOrdering(polyOrder, quad_4);
    }
    
    // set up expected indices...
    expectedDofIndices.resize(numSides,dofsPerSide);
    int dofIndex = 0;
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      for (int dofOrdinal=0; dofOrdinal<dofsPerSide; dofOrdinal++) {
        if ((sideIndex > 0) && (dofOrdinal==0) ) {
          // then this vertex matches the last one on the previous side...
          expectedDofIndices(sideIndex,dofOrdinal) = expectedDofIndices(sideIndex-1,dofsPerSide-1);
        } else if ((sideIndex==numSides-1) && (dofOrdinal==dofsPerSide-1)) {
          // last one, back to 0
          expectedDofIndices(sideIndex,dofOrdinal) = 0;
        } else {
          expectedDofIndices(sideIndex,dofOrdinal) = dofIndex;
          dofIndex++;
        }
      }
    }
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      for (int dofOrdinal=0; dofOrdinal<dofsPerSide; dofOrdinal++) {
        int actualDofIndex = trialOrder->getDofIndex(0, dofOrdinal, sideIndex);
        int expectedDofIndex = expectedDofIndices(sideIndex,dofOrdinal);
        if ( ! expectedDofIndex == actualDofIndex ) {
          cout << myName << ": failed conforming vertex test for sideIndex " << sideIndex << ", dofOrdinal " << dofOrdinal << "." << endl;
          cout << "Expected " << expectedDofIndex << "; actual was " << actualDofIndex << "." << endl;
          success = false;
        }
      }
    }
  }
  return success;
}

bool DPGTests::testComputeStiffnessDx() {
  bool success = true;
  DofOrdering lowestOrderHGRADOrdering;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis 
  = BasisFactory::getBasis(basisRank,C1_FAKE_POLY_ORDER,
                           quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  lowestOrderHGRADOrdering.addEntry(0,basis,0);
  
  int numTests = 4;  // 1. ref quad
  // 2. upper right quadrant of ref quad
  // 3. vertical half-slice of ref quad
  // 4. horizontal half-slice of ref quad
  int numSides = 4;
  
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> stiffnessExpected(numTests,4,4);
  
  stiffnessExpected(0,0,0) = 1.0/3.0;
  stiffnessExpected(0,1,1) = 1.0/3.0;
  stiffnessExpected(0,2,2) = 1.0/3.0;
  stiffnessExpected(0,3,3) = 1.0/3.0;
  
  stiffnessExpected(0,0,3) = 1.0/6.0;
  stiffnessExpected(0,1,2) = 1.0/6.0;
  stiffnessExpected(0,2,1) = 1.0/6.0;
  stiffnessExpected(0,3,0) = 1.0/6.0;
  
  stiffnessExpected(0,0,1) = -1.0/3.0;
  stiffnessExpected(0,1,0) = -1.0/3.0;
  stiffnessExpected(0,2,3) = -1.0/3.0;
  stiffnessExpected(0,3,2) = -1.0/3.0;
  
  stiffnessExpected(0,0,2) = -1.0/6.0;
  stiffnessExpected(0,1,3) = -1.0/6.0;
  stiffnessExpected(0,2,0) = -1.0/6.0;
  stiffnessExpected(0,3,1) = -1.0/6.0;
  
  // repeat the above, this time with the upper-right quadrant of the ref quad
  
  quadPoints(1,0,0) = 0.0; // x1
  quadPoints(1,0,1) = 0.0; // y1
  quadPoints(1,1,0) = 1.0;
  quadPoints(1,1,1) = 0.0;
  quadPoints(1,2,0) = 1.0;
  quadPoints(1,2,1) = 1.0;
  quadPoints(1,3,0) = 0.0;
  quadPoints(1,3,1) = 1.0;
  
  // vertical half-slice of the ref quad, now
  quadPoints(2,0,0) =  0.0;  // x1
  quadPoints(2,0,1) = -1.0; // y1
  quadPoints(2,1,0) =  1.0;
  quadPoints(2,1,1) = -1.0;
  quadPoints(2,2,0) =  1.0;
  quadPoints(2,2,1) =  1.0;
  quadPoints(2,3,0) =  0.0;
  quadPoints(2,3,1) =  1.0;
  
  // vertical half-slice doubles those x derivatives' magnitudes
  // 2 for trial x 2 for test = 4
  // the cell measure is half what we had before, for net factor of 2...
  
  // horizontal half-slice
  quadPoints(3,0,0) = -1.0; // x1
  quadPoints(3,0,1) =  0.0; // y1
  quadPoints(3,1,0) =  1.0;
  quadPoints(3,1,1) =  0.0;
  quadPoints(3,2,0) =  1.0;
  quadPoints(3,2,1) =  1.0;
  quadPoints(3,3,0) =  -1.0;
  quadPoints(3,3,1) =  1.0;
  
  for (int i=0; i<stiffnessExpected.dimension(1); i++) {
    for (int j=0; j<stiffnessExpected.dimension(2); j++) {
      // for second test, just copy the first stiffness matrix
      stiffnessExpected(1,i,j) = stiffnessExpected(0,i,j);
      // for third test, multiply the first stiffness matrix by 2.0
      stiffnessExpected(2,i,j) = stiffnessExpected(0,i,j)*2.0;
      // for fourth test, multiply by 1/2
      stiffnessExpected(3,i,j) = stiffnessExpected(0,i,j)*0.5;
    }
  }
  
  // horizontal half-slice doesn't change the x derivatives at all
  // cell measure is again half the ref cell, so 1/2 the ref cell's stiffness.
  
  FieldContainer<double> stiffnessActual(numTests,4,4);
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp(new TestBilinearFormDx());
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  Teuchos::RCP<DofOrdering> lowestOrderHGRADOrderingPtr = Teuchos::rcp(&lowestOrderHGRADOrdering, false);
  
  BilinearFormUtility::computeStiffnessMatrix(stiffnessActual, bilinearForm,
                                              lowestOrderHGRADOrderingPtr, lowestOrderHGRADOrderingPtr,
                                              quad_4, quadPoints,cellSideParities);
  
  for (int testIndex = 0; testIndex < numTests; testIndex++) {
    double tol = 1e-14;
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        double diff = abs(stiffnessActual(testIndex,i,j)-stiffnessExpected(testIndex,i,j));
        if (diff > tol) {
          cout << "testComputeStiffnessTrace, testIndex=" << testIndex << ":" << endl;
          cout << "   expected and actual stiffness differ in i=" << i << ",j=" << j << "; difference: " << diff << endl;
          cout << "   expected: " << stiffnessExpected(testIndex,i,j) << endl;
          cout << "   actual:   " << stiffnessActual(testIndex,i,j) << endl;
          success = false;
        }
      }
    }
  }  
  
  return success;
}

bool DPGTests::testComputeStiffnessFlux() {
  bool success = true;
  Teuchos::RCP<DofOrdering> traceOrdering = Teuchos::rcp(new DofOrdering());
  Teuchos::RCP<DofOrdering> testOrdering = Teuchos::rcp(new DofOrdering());
  
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > traceBasis
  = 
  BasisFactory::getBasis(basisRank,C1_FAKE_POLY_ORDER,
                         line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  int numSides = 4;
  
  for (int i=0; i<numSides; i++) {
    traceOrdering->addEntry(0,traceBasis,0,i);
  }
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  Teuchos::RCP< Basis<double,FieldContainer<double> > > testBasis
  = 
  BasisFactory::getBasis(basisRank, C1_FAKE_POLY_ORDER, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);  
  testOrdering->addEntry(0,testBasis,0);
  
  int numTests = 1;  // 1. ref quad
  
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> stiffnessExpected(numTests,4,8); // 8 = 4 sides * 2 dofs per side
  FieldContainer<double> stiffnessActual(numTests,4,8);
  
  stiffnessExpected.initialize(); // there will be quite a few 0s.
  
  //cout << "WARNING: Flux test relies on assumption that CellTools does tensor-product directioning for edges (i.e. horizontal edges both point right, vertical both point up.  Is that true??" << endl;
  stiffnessExpected(0,0,0) = 2.0/3.0;
  stiffnessExpected(0,0,1) = 1.0/3.0;
  stiffnessExpected(0,0,6) = 1.0/3.0;
  stiffnessExpected(0,0,7) = 2.0/3.0;
  
  stiffnessExpected(0,1,0) = 1.0/3.0;
  stiffnessExpected(0,1,1) = 2.0/3.0;
  stiffnessExpected(0,1,2) = 2.0/3.0;
  stiffnessExpected(0,1,3) = 1.0/3.0;
  
  stiffnessExpected(0,2,2) = 1.0/3.0;
  stiffnessExpected(0,2,3) = 2.0/3.0;
  stiffnessExpected(0,2,4) = 2.0/3.0;
  stiffnessExpected(0,2,5) = 1.0/3.0;
  
  stiffnessExpected(0,3,4) = 1.0/3.0;
  stiffnessExpected(0,3,5) = 2.0/3.0;
  stiffnessExpected(0,3,6) = 2.0/3.0;
  stiffnessExpected(0,3,7) = 1.0/3.0;
  
  /*stiffnessExpected(0,0,0) = 2.0/3.0;
   stiffnessExpected(0,0,1) = 1.0/3.0;
   stiffnessExpected(0,0,6) = -1.0/3.0;
   stiffnessExpected(0,0,7) = -2.0/3.0;
   
   stiffnessExpected(0,1,0) = 1.0/3.0;
   stiffnessExpected(0,1,1) = 2.0/3.0;
   stiffnessExpected(0,1,2) = 2.0/3.0;
   stiffnessExpected(0,1,3) = 1.0/3.0;
   
   stiffnessExpected(0,2,2) = 1.0/3.0;
   stiffnessExpected(0,2,3) = 2.0/3.0;
   stiffnessExpected(0,2,4) = -2.0/3.0;
   stiffnessExpected(0,2,5) = -1.0/3.0;
   
   stiffnessExpected(0,3,4) = -1.0/3.0;
   stiffnessExpected(0,3,5) = -2.0/3.0;
   stiffnessExpected(0,3,6) = -2.0/3.0;
   stiffnessExpected(0,3,7) = -1.0/3.0;*/
  
  /*for (int i=0; i<stiffnessExpected.dimension(1); i++) {
   for (int j=0; j<stiffnessExpected.dimension(2); j++) {
   // for second test, just copy the first stiffness matrix
   stiffnessExpected(1,i,j) = stiffnessExpected(0,i,j);
   // for third test, multiply the first stiffness matrix by 2.0
   stiffnessExpected(2,i,j) = stiffnessExpected(0,i,j)*2.0;
   // for fourth test, multiply by 1/2
   stiffnessExpected(3,i,j) = stiffnessExpected(0,i,j)*0.5;
   }
   }*/
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp(new TestBilinearFormFlux());
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  BilinearFormUtility::computeStiffnessMatrix(stiffnessActual, bilinearForm,
                                              traceOrdering, testOrdering,
                                              quad_4, quadPoints, cellSideParities);
  
  for (int testIndex = 0; testIndex < numTests; testIndex++) {
    
    double tol = 1e-14;
    for (int i=0; i<stiffnessExpected.dimension(1); i++) {
      for (int j=0; j<stiffnessExpected.dimension(2); j++) {
        double diff = abs(stiffnessActual(testIndex,i,j)-stiffnessExpected(testIndex,i,j));
        if (diff > tol) {
          cout << "testComputeStiffnessFlux, testIndex=" << testIndex << ":" << endl;
          cout << "   expected and actual stiffness differ in i=" << i << ",j=" << j << "; difference: " << diff << endl;
          cout << "   expected: " << stiffnessExpected(testIndex,i,j) << endl;
          cout << "   actual:   " << stiffnessActual(testIndex,i,j) << endl;
          success = false;
        }
      }
    }
  }  
  if (!success) {
    cout << "testComputeStiffnessFlux failed; actual stiffness:" << stiffnessActual << endl;
  }
  return success;
  
}

bool DPGTests::testComputeStiffnessTrace() {
  bool success = true;
  Teuchos::RCP<DofOrdering> traceOrdering = Teuchos::rcp(new DofOrdering());
  Teuchos::RCP<DofOrdering> testOrdering = Teuchos::rcp(new DofOrdering());
  
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > traceBasis 
  = BasisFactory::getBasis(basisRank,C1_FAKE_POLY_ORDER,
                           line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  int numSides = 4;
  
  for (int i=0; i<numSides; i++) {
    traceOrdering->addEntry(0,traceBasis,0,i);
  }
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  Teuchos::RCP< Basis<double,FieldContainer<double> > > testBasis 
  = BasisFactory::getBasis(basisRank,C1_FAKE_POLY_ORDER,
                           quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
  
  testOrdering->addEntry(0,testBasis,1,0);
  
  int numTests = 1;  // 1. ref quad
  
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> stiffnessExpected(numTests,4,8); // 8 = 4 sides * 2 dofs per side
  FieldContainer<double> stiffnessActual(numTests,4,8);
  
  stiffnessExpected.initialize(); // there will be quite a few 0s.
  
  stiffnessExpected(0,0,0) = 1.0/2.0;
  stiffnessExpected(0,0,1) = 1.0/2.0;
  
  stiffnessExpected(0,1,2) = 1.0/2.0;
  stiffnessExpected(0,1,3) = 1.0/2.0;
  
  stiffnessExpected(0,2,4) = 1.0/2.0;
  stiffnessExpected(0,2,5) = 1.0/2.0;
  
  stiffnessExpected(0,3,6) = 1.0/2.0;
  stiffnessExpected(0,3,7) = 1.0/2.0;  
  
  /*for (int i=0; i<stiffnessExpected.dimension(1); i++) {
   for (int j=0; j<stiffnessExpected.dimension(2); j++) {
   // for second test, just copy the first stiffness matrix
   stiffnessExpected(1,i,j) = stiffnessExpected(0,i,j);
   // for third test, multiply the first stiffness matrix by 2.0
   stiffnessExpected(2,i,j) = stiffnessExpected(0,i,j)*2.0;
   // for fourth test, multiply by 1/2
   stiffnessExpected(3,i,j) = stiffnessExpected(0,i,j)*0.5;
   }
   }*/
  
  // horizontal half-slice doesn't change the x derivatives at all
  // cell measure is again half the ref cell, so 1/2 the ref cell's stiffness.
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp(new TestBilinearFormTrace());
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  BilinearFormUtility::computeStiffnessMatrix(stiffnessActual, bilinearForm,
                                              traceOrdering, testOrdering,
                                              quad_4, quadPoints,cellSideParities);
  
  for (int testIndex = 0; testIndex < numTests; testIndex++) {
    
    double tol = 1e-14;
    for (int i=0; i<stiffnessExpected.dimension(1); i++) {
      for (int j=0; j<stiffnessExpected.dimension(2); j++) {
        double diff = abs(stiffnessActual(testIndex,i,j)-stiffnessExpected(testIndex,i,j));
        if (diff > tol) {
          cout << "testComputeStiffnessTrace, testIndex=" << testIndex << ":" << endl;
          cout << "   expected and actual stiffness differ in i=" << i << ",j=" << j << "; difference: " << diff << endl;
          cout << "   expected: " << stiffnessExpected(testIndex,i,j) << endl;
          cout << "   actual:   " << stiffnessActual(testIndex,i,j) << endl;
          success = false;
        }
      }
    }
  }  
  
  
  if (!success) {
    cout << "testComputeStiffnessTrace failed; actual stiffness:" << stiffnessActual << endl;
  }
  
  return success;
  
}

bool DPGTests::testMathInnerProductDx() {
  int numTests = 1;
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new TestBilinearFormDx() );
  
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  FieldContainer<double> quadPoints(numTests,4,2);
  
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> innerProductExpected(numTests,4,4);
  // computed by mathematica (there's a workbook to generate the test
  //     code here, although this first example is handcoded...)
  innerProductExpected(0,0,0) = 10.0/9.0;
  innerProductExpected(0,1,1) = 10.0/9.0;
  innerProductExpected(0,2,2) = 10.0/9.0;
  innerProductExpected(0,3,3) = 10.0/9.0;
  
  innerProductExpected(0,0,3) = 1.0/18.0;
  innerProductExpected(0,1,2) = 1.0/18.0;
  innerProductExpected(0,2,1) = 1.0/18.0;
  innerProductExpected(0,3,0) = 1.0/18.0;
  
  innerProductExpected(0,0,1) = 1.0/18.0;
  innerProductExpected(0,1,0) = 1.0/18.0;
  innerProductExpected(0,2,3) = 1.0/18.0;
  innerProductExpected(0,3,2) = 1.0/18.0;
  
  innerProductExpected(0,0,2) = -2.0/9.0;
  innerProductExpected(0,1,3) = -2.0/9.0;
  innerProductExpected(0,2,0) = -2.0/9.0;
  innerProductExpected(0,3,1) = -2.0/9.0;
  
  // the following test does not pass.  The likely reason is that the
  // Mathematica code doesn't do any kind of transform, whereas the inner product
  // code here does.  We'll need to figure out a way to fix the Mathematica code
  // before we can test this properly...
  /*quadPoints(1,0,0) = -2;
   quadPoints(1,0,1) = -2;
   quadPoints(1,1,0) = 1;
   quadPoints(1,1,1) = -2;
   quadPoints(1,2,0) = 1;
   quadPoints(1,2,1) = 1;
   quadPoints(1,3,0) = -2;
   quadPoints(1,3,1) = 1;
   innerProductExpected(1,0,0) = 8.4375000000000;
   innerProductExpected(1,0,1) = -1.6875000000000;
   innerProductExpected(1,0,2) = 0;
   innerProductExpected(1,0,3) = -1.6875000000000;
   innerProductExpected(1,1,0) = -1.6875000000000;
   innerProductExpected(1,1,1) = 3.9375000000000;
   innerProductExpected(1,1,2) = -0.56250000000000;
   innerProductExpected(1,1,3) = 0;
   innerProductExpected(1,2,0) = 0;
   innerProductExpected(1,2,1) = -0.56250000000000;
   innerProductExpected(1,2,2) = 1.6875000000000;
   innerProductExpected(1,2,3) = -0.56250000000000;
   innerProductExpected(1,3,0) = -1.6875000000000;
   innerProductExpected(1,3,1) = 0;
   innerProductExpected(1,3,2) = -0.56250000000000;
   innerProductExpected(1,3,3) = 3.9375000000000;*/
  
  FieldContainer<double> innerProductActual(numTests,4,4);
  
  DofOrdering lowestOrderHGRADOrdering;
  
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = BasisFactory::getBasis(basisRank, C1_FAKE_POLY_ORDER, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  lowestOrderHGRADOrdering.addEntry(0,basis,0);
  
  ip->computeInnerProductMatrix(innerProductActual, Teuchos::rcp(&lowestOrderHGRADOrdering,false),
                                quad_4, quadPoints);
  
  string myName = "testMathInnerProductDx";
  
  return fcsAgree(myName, innerProductExpected,innerProductActual,1e-14);
}

bool DPGTests::fcsAgree(string &testName, FieldContainer<double> &expected, 
                        FieldContainer<double> &actual, double tol) {
  int numTests = expected.dimension(0);
  bool success = true;
  int diffsPrinted = 0, numDiffs = 0, numEntries = 0;
  double maxDiff=0.0;
  double expectedAtMaxDiff, actualAtMaxDiff;
  int maxDiffRow=-1, maxDiffCol=-1;
  vector<int> rowIndicesThatAgree, colIndicesThatAgree;
  vector<double> valuesThatAgree;
  for (int testIndex = 0; testIndex < numTests; testIndex++) {
    bool firstDiffForTest = true;
    for (int i=0; i<expected.dimension(1); i++) {
      if (expected.rank() <= 2) {
        numEntries++;
        double diff = abs(actual(testIndex,i)-expected(testIndex,i));
        if (diff > tol) {
          numDiffs++;
          if (diffsPrinted < 10) {
            if (firstDiffForTest)
              cout << testName << ",testIndex=" << testIndex << ":" << endl;
            cout << "   expected and actual vector differ in i=" << i << "; difference: " << diff << endl;
            cout << "   expected: " << expected(testIndex,i) << endl;
            cout << "   actual:   " << actual(testIndex,i) << endl;
            success = false;
            firstDiffForTest = false;
            if (diffsPrinted == 9) {
              cout << "(suppressing further detailed diff info.)" << endl;
            }
          }
          if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffRow = i;
            expectedAtMaxDiff = expected(testIndex,i);
            actualAtMaxDiff = actual(testIndex,i);
          }
          diffsPrinted++;
        } else {
          rowIndicesThatAgree.push_back(i);
          valuesThatAgree.push_back(actual(testIndex,i));
        }
      } else {
        for (int j=0; j<expected.dimension(2); j++) {
          numEntries++;
          double diff = abs(actual(testIndex,i,j)-expected(testIndex,i,j));
          if (diff > tol) {
            numDiffs++;
            if (diffsPrinted < 10) {
              if (firstDiffForTest)
                cout << testName << ",testIndex=" << testIndex << ":" << endl;
              cout << "   expected and actual matrix differ in i=" << i << ",j=" << j << "; difference: " << diff << endl;
              cout << "   expected: " << expected(testIndex,i,j) << endl;
              cout << "   actual:   " << actual(testIndex,i,j) << endl;
              success = false;
              firstDiffForTest = false;
              if (diffsPrinted == 9) {
                cout << "(suppressing further detailed diff info.)" << endl;
              }
            }
            if (diff > maxDiff) {
              maxDiff = diff;
              maxDiffRow = i;
              maxDiffCol = j;
              expectedAtMaxDiff = expected(testIndex,i,j);
              actualAtMaxDiff = actual(testIndex,i,j);
            }
            diffsPrinted++;
          } else {
            rowIndicesThatAgree.push_back(i);
            colIndicesThatAgree.push_back(j);
            valuesThatAgree.push_back(actual(testIndex,i,j));
          }
        }
      }
    }
  }
  if (numDiffs > 0) {
    cout << testName << ": matrices differ in " << numDiffs << " of " << numEntries << " entries." << endl;
    if (numDiffs != numEntries) {
      cout << "entries that agree are as follows:" << endl;
      for (unsigned i=0; i<rowIndicesThatAgree.size(); i++) {
        if (expected.rank() > 2) {
          cout << "(" << rowIndicesThatAgree[i] << "," << colIndicesThatAgree[i] << ") = " << valuesThatAgree[i];
          if (i<rowIndicesThatAgree.size()-1) cout << ", ";
        } else {
          cout << "(" << rowIndicesThatAgree[i] << ") = " << valuesThatAgree[i];
          if (i<rowIndicesThatAgree.size()-1) cout << ", ";
        }
      }
      cout << endl;
    }
    cout << "maxDiff: " << maxDiff << " at (" << maxDiffRow << "," << maxDiffCol << ") -- expected " << expectedAtMaxDiff << "; actual was " << actualAtMaxDiff << endl;
  }
  return success;
}

bool DPGTests::fcEqualsSDM(FieldContainer<double> &fc, int cellIndex,
                           Epetra_SerialDenseMatrix &sdm, double tol, bool transpose) {
  double maxDiff = 0.0;
  for (int i=0; i<fc.dimension(1); i++) {
    for (int j=0; j<fc.dimension(2); j++) {
      double diff;
      if (! transpose) {
        diff = abs(fc(cellIndex,i,j) - sdm(i,j));
      } else {
        diff = abs(fc(cellIndex,i,j) - sdm(j,i));
      }
      diff = max(diff,maxDiff);
    }
  }
  if (maxDiff > tol) {
    return false;
  } else {
    return true;
  }
}

bool DPGTests::testAnalyticBoundaryIntegral(bool conforming) {
  int numTests = 1;
  
  int order = 3; // cubic
  
  int numSides = 4; // quad
  
  double tol = 3.76e-14; // value determined by experimentation -- the lowest tolerance we pass...
  
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  vector<int> quadCellIDs;
  quadCellIDs.push_back(0);
  
  bool success = true;
  Teuchos::RCP<DofOrdering> trialOrdering;
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new TestBilinearFormAnalyticBoundaryIntegral() );
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  DofOrderingFactory dofOrderingFactory(bilinearForm);
  Teuchos::RCP<DofOrdering> cubicHGradOrdering = dofOrderingFactory.testOrdering(order, quad_4);
  
  // construct conforming or non-conforming trial basis:
  if ( ! conforming ) {
    shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
    int basisRank;
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis 
    = BasisFactory::getBasis(basisRank,C3_FAKE_POLY_ORDER,
                             line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    
    trialOrdering = Teuchos::rcp(new DofOrdering());
    
    trialOrdering->addEntry(0,basis,0,0);//one for each side in the quad
    trialOrdering->addEntry(0,basis,0,1);
    trialOrdering->addEntry(0,basis,0,2);
    trialOrdering->addEntry(0,basis,0,3);
  } else {
    // exercise the factory, which will choose a conforming basis for the H(GRAD) trace....
    trialOrdering = dofOrderingFactory.trialOrdering(order, quad_4);
  }
  
  FieldContainer<double> stiffnessExpected(numTests,cubicHGradOrdering->totalDofs(),
                                           trialOrdering->totalDofs() );
  FieldContainer<double> stiffnessActual(numTests,cubicHGradOrdering->totalDofs(),
                                         trialOrdering->totalDofs() );
  
  // from Mathematica:
  TestBilinearFormAnalyticBoundaryIntegral::expectedPreStiffnessForCubicsOnQuad(stiffnessExpected,conforming);
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  BilinearFormUtility::computeStiffnessMatrix(stiffnessActual, bilinearForm,
                                              trialOrdering, cubicHGradOrdering,
                                              quad_4, quadPoints, cellSideParities);
  
  string myNameStiffness = "testAnalyticBoundaryIntegral.stiffness";
  
  bool successLocal = fcsAgree(myNameStiffness, stiffnessExpected, stiffnessActual, tol);
  if (! successLocal) {
    success = false;
    cout << myNameStiffness << ": comparison of stiffnessExpected and stiffnessActual failed." << endl;
  }
  
  FieldContainer<double> ipMatrixExpected(numTests,cubicHGradOrdering->totalDofs(),
                                          cubicHGradOrdering->totalDofs() );
  FieldContainer<double> ipMatrixActual(numTests,cubicHGradOrdering->totalDofs(),
                                        cubicHGradOrdering->totalDofs() );
  
  TestBilinearFormAnalyticBoundaryIntegral::expectedIPMatrixForCubicsOnQuad(ipMatrixExpected);
  
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
  ip->computeInnerProductMatrix(ipMatrixActual,cubicHGradOrdering, quad_4, quadPoints);
  
  string myNameIPMatrix = "testAnalyticBoundaryIntegral.ipMatrix";
  successLocal = fcsAgree(myNameIPMatrix, ipMatrixExpected, ipMatrixActual, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameIPMatrix << ": comparison of ipMatrixExpected and ipMatrixActual failed." << endl;
  }
  
  FieldContainer<double> ipWeightsExpected(numTests,trialOrdering->totalDofs(),
                                           cubicHGradOrdering->totalDofs() );
  FieldContainer<double> ipWeightsActual(numTests,trialOrdering->totalDofs(),
                                         cubicHGradOrdering->totalDofs() );
  
  TestBilinearFormAnalyticBoundaryIntegral::expectedOptTestWeightsForCubicsOnQuad(ipWeightsExpected,conforming);
  
  ElementTypePtr elemType = makeElemType(trialOrdering, cubicHGradOrdering, quad_4);
  BasisCachePtr basisCache = makeBasisCache(elemType,quadPoints,quadCellIDs);
  int optSuccess = bilinearForm->optimalTestWeights(ipWeightsActual, ipMatrixExpected,
                                                    elemType, cellSideParities, basisCache);
  
  string myNameIPWeights = "testAnalyticBoundaryIntegral.ipWeights";
  
  if (optSuccess != 0) {
    cout << myNameIPWeights << ": computeOptimalTest failed." << endl;
  }
  
  successLocal = fcsAgree(myNameIPWeights, ipWeightsExpected, ipWeightsActual, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameIPWeights << ": comparison of ipWeightsExpected and ipWeightsActual failed." << endl;
  }
  
  // confirm that the optWeights actually fulfill the contract....
  // placed here mostly as sanity check for checkOptTestWeights
  if ( ! checkOptTestWeights(ipWeightsExpected,ipMatrixExpected,stiffnessExpected,tol) ) {
    cout << myNameIPWeights << ": check that optWeights == ipMatrix^(-1) * preStiffness failed." << endl;
    return false;
  }
  
  FieldContainer<double> finalStiffnessExpected(numTests,trialOrdering->totalDofs(),
                                                trialOrdering->totalDofs() );
  FieldContainer<double> finalStiffnessActual1(numTests,trialOrdering->totalDofs(),
                                               trialOrdering->totalDofs() );
  FieldContainer<double> finalStiffnessActual2(numTests,trialOrdering->totalDofs(),
                                               trialOrdering->totalDofs() );
  
  TestBilinearFormAnalyticBoundaryIntegral::expectedFinalStiffnessForCubicsOnQuad(finalStiffnessExpected,conforming);
  
  BilinearFormUtility::computeStiffnessMatrix(finalStiffnessActual1,ipMatrixExpected,ipWeightsExpected);
  
  string myNameFinalByMultiplying = "testAnalyticBoundaryIntegral.finalStiffnessByMultiplying";
  successLocal = fcsAgree(myNameFinalByMultiplying, finalStiffnessExpected, finalStiffnessActual1, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByMultiplying << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  BilinearFormUtility::computeOptimalStiffnessMatrix(finalStiffnessActual2, ipWeightsExpected,
                                                     bilinearForm,
                                                     trialOrdering, cubicHGradOrdering,
                                                     quad_4, quadPoints, cellSideParities);
  
  string myNameFinalByIntegrating = "testAnalyticBoundaryIntegral.finalStiffnessByIntegrating";
  successLocal = fcsAgree(myNameFinalByIntegrating, finalStiffnessExpected, finalStiffnessActual2, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByIntegrating << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  //cout << "ipMatrixActual:" << endl << ipMatrixActual;
  //cout << "ipWeightsActual:" << endl << ipWeightsActual;
  
  // now recompute the "final" matrices, but with the *actual* rather than expected ipMatrix and stiffness
  BilinearFormUtility::computeStiffnessMatrix(finalStiffnessActual1,ipMatrixActual,ipWeightsActual);
  
  string myNameFinalByMultiplyingUsingActual = "testAnalyticBoundaryIntegral.finalStiffnessByMultiplyingUsingActual";
  successLocal = fcsAgree(myNameFinalByMultiplyingUsingActual, finalStiffnessExpected, finalStiffnessActual1, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByMultiplyingUsingActual << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  BilinearFormUtility::computeOptimalStiffnessMatrix(finalStiffnessActual2, ipWeightsActual,
                                                     bilinearForm,
                                                     trialOrdering, cubicHGradOrdering,
                                                     quad_4, quadPoints, cellSideParities);
  
  string myNameFinalByIntegratingUsingActual = "testAnalyticBoundaryIntegral.finalStiffnessByIntegratingUsingActual";
  successLocal = fcsAgree(myNameFinalByIntegratingUsingActual, finalStiffnessExpected, finalStiffnessActual2, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByIntegratingUsingActual << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  return success;
}

bool DPGTests::testLowOrderTrialCubicTest() {
  bool oldWarnState = BilinearFormUtility::warnAboutZeroRowsAndColumns();
  BilinearFormUtility::setWarnAboutZeroRowsAndColumns(false);
  int numTests = 1;
  
  int order = 3; // cubic
  
  int numSides = 4;
  
  double tol = 3.76e-14; // value determined by experimentation -- the lowest tolerance we pass...
  
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  vector<int> quadCellIDs;
  quadCellIDs.push_back(0);
  
  bool success = true;
  Teuchos::RCP<DofOrdering> lowestOrderHGRADOrdering = Teuchos::rcp(new DofOrdering());
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  int basisRank;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis
  = 
  BasisFactory::getBasis(basisRank, C1_FAKE_POLY_ORDER, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  lowestOrderHGRADOrdering->addEntry(0,basis,0);
  
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new TestBilinearFormDx() );
  
  DofOrderingFactory dofOrderingFactory(bilinearForm);
  Teuchos::RCP<DofOrdering> cubicHGradOrdering = dofOrderingFactory.testOrdering(order, quad_4);
  
  FieldContainer<double> stiffnessExpected(numTests,cubicHGradOrdering->totalDofs(),
                                           lowestOrderHGRADOrdering->totalDofs() );
  FieldContainer<double> stiffnessActual(numTests,cubicHGradOrdering->totalDofs(),
                                         lowestOrderHGRADOrdering->totalDofs() );
  
  // from Mathematica:
  // put the assignment inside a block so I can use XCode's code folding to hide these 60 lines...
  {
    stiffnessExpected(0,0,0) = 0.08333333333333333;
    stiffnessExpected(0,0,1) = -0.08333333333333333;
    stiffnessExpected(0,0,2) = 4.163336342344337e-17;
    stiffnessExpected(0,0,3) = -4.163336342344337e-17;
    stiffnessExpected(0,1,0) = 5.551115123125783e-17;
    stiffnessExpected(0,1,1) = -5.551115123125783e-17;
    stiffnessExpected(0,1,2) = 9.251858538542975e-18;
    stiffnessExpected(0,1,3) = -9.251858538542975e-18;
    stiffnessExpected(0,2,0) = -2.7755575615628914e-17;
    stiffnessExpected(0,2,1) = 2.7755575615628914e-17;
    stiffnessExpected(0,2,2) = 1.0177044392397265e-16;
    stiffnessExpected(0,2,3) = -1.0177044392397265e-16;
    stiffnessExpected(0,3,0) = -0.08333333333333331;
    stiffnessExpected(0,3,1) = 0.08333333333333331;
    stiffnessExpected(0,3,2) = -2.3129646346357426e-17;
    stiffnessExpected(0,3,3) = 2.3129646346357426e-17;
    stiffnessExpected(0,4,0) = 0.30150283239582454;
    stiffnessExpected(0,4,1) = -0.30150283239582454;
    stiffnessExpected(0,4,2) = -0.11516383427084217;
    stiffnessExpected(0,4,3) = 0.11516383427084217;
    stiffnessExpected(0,5,0) = 0.;
    stiffnessExpected(0,5,1) = 0.;
    stiffnessExpected(0,5,2) = 0.;
    stiffnessExpected(0,5,3) = 0.;
    stiffnessExpected(0,6,0) = -1.1102230246251565e-16;
    stiffnessExpected(0,6,1) = 1.1102230246251565e-16;
    stiffnessExpected(0,6,2) = 1.1102230246251565e-16;
    stiffnessExpected(0,6,3) = -1.1102230246251565e-16;
    stiffnessExpected(0,7,0) = -0.3015028323958245;
    stiffnessExpected(0,7,1) = 0.3015028323958245;
    stiffnessExpected(0,7,2) = 0.11516383427084219;
    stiffnessExpected(0,7,3) = -0.11516383427084219;
    stiffnessExpected(0,8,0) = 0.11516383427084206;
    stiffnessExpected(0,8,1) = -0.11516383427084206;
    stiffnessExpected(0,8,2) = -0.30150283239582454;
    stiffnessExpected(0,8,3) = 0.30150283239582454;
    stiffnessExpected(0,9,0) = 5.551115123125783e-17;
    stiffnessExpected(0,9,1) = -5.551115123125783e-17;
    stiffnessExpected(0,9,2) = -3.3306690738754696e-16;
    stiffnessExpected(0,9,3) = 3.3306690738754696e-16;
    stiffnessExpected(0,10,0) = 2.220446049250313e-16;
    stiffnessExpected(0,10,1) = -2.220446049250313e-16;
    stiffnessExpected(0,10,2) = -3.3306690738754696e-16;
    stiffnessExpected(0,10,3) = 3.3306690738754696e-16;
    stiffnessExpected(0,11,0) = -0.11516383427084213;
    stiffnessExpected(0,11,1) = 0.11516383427084213;
    stiffnessExpected(0,11,2) = 0.3015028323958244;
    stiffnessExpected(0,11,3) = -0.3015028323958244;
    stiffnessExpected(0,12,0) = -5.319818659662208e-17;
    stiffnessExpected(0,12,1) = 5.319818659662208e-17;
    stiffnessExpected(0,12,2) = -0.08333333333333325;
    stiffnessExpected(0,12,3) = 0.08333333333333325;
    stiffnessExpected(0,13,0) = -1.8503717077085975e-17;
    stiffnessExpected(0,13,1) = 1.8503717077085975e-17;
    stiffnessExpected(0,13,2) = 0.;
    stiffnessExpected(0,13,3) = 0.;
    stiffnessExpected(0,14,0) = -1.6653345369377348e-16;
    stiffnessExpected(0,14,1) = 1.6653345369377348e-16;
    stiffnessExpected(0,14,2) = 2.220446049250313e-16;
    stiffnessExpected(0,14,3) = -2.220446049250313e-16;
    stiffnessExpected(0,15,0) = 6.476300976980079e-17;
    stiffnessExpected(0,15,1) = -6.476300976980079e-17;
    stiffnessExpected(0,15,2) = 0.08333333333333327;
    stiffnessExpected(0,15,3) = -0.08333333333333327;
  }
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  BilinearFormUtility::computeStiffnessMatrix(stiffnessActual, bilinearForm,
                                              lowestOrderHGRADOrdering, cubicHGradOrdering,
                                              quad_4, quadPoints, cellSideParities);
  
  string myNameStiffness = "testLowOrderTrialCubicTest.stiffness";
  
  bool successLocal = fcsAgree(myNameStiffness, stiffnessExpected, stiffnessActual, tol);
  if (! successLocal) {
    success = false;
    cout << myNameStiffness << ": comparison of stiffnessExpected and stiffnessActual failed." << endl;
  }
  
  FieldContainer<double> ipMatrixExpected(numTests,cubicHGradOrdering->totalDofs(),
                                          cubicHGradOrdering->totalDofs() );
  FieldContainer<double> ipMatrixActual(numTests,cubicHGradOrdering->totalDofs(),
                                        cubicHGradOrdering->totalDofs() );
  
  
  // put the assignment inside a block so I can use XCode's code folding to hide these 256 lines...
  {
    ipMatrixExpected(0,0,0) = 0.6394557823129254;
    ipMatrixExpected(0,1,0) = -0.2255000638541592;
    ipMatrixExpected(0,2,0) = -0.07211898376488793;
    ipMatrixExpected(0,3,0) = 0.04308390022675741;
    ipMatrixExpected(0,4,0) = -0.22550006385415977;
    ipMatrixExpected(0,5,0) = -0.25689112700183325;
    ipMatrixExpected(0,6,0) = 0.14597505668934127;
    ipMatrixExpected(0,7,0) = -0.06124543897569006;
    ipMatrixExpected(0,8,0) = -0.07211898376488846;
    ipMatrixExpected(0,9,0) = 0.14597505668934174;
    ipMatrixExpected(0,10,0) = -0.03505898637685316;
    ipMatrixExpected(0,11,0) = 0.011642264372517502;
    ipMatrixExpected(0,12,0) = 0.04308390022675734;
    ipMatrixExpected(0,13,0) = -0.06124543897569068;
    ipMatrixExpected(0,14,0) = 0.011642264372517065;
    ipMatrixExpected(0,15,0) = -0.003401360544217533;
    ipMatrixExpected(0,0,1) = -0.2255000638541592;
    ipMatrixExpected(0,1,1) = 2.2448979591836764;
    ipMatrixExpected(0,2,1) = -0.02267573696145056;
    ipMatrixExpected(0,3,1) = -0.0721189837648888;
    ipMatrixExpected(0,4,1) = -0.25689112700183325;
    ipMatrixExpected(0,5,1) = -1.4824317442707629;
    ipMatrixExpected(0,6,1) = -0.39496005112844845;
    ipMatrixExpected(0,7,1) = 0.1459750566893411;
    ipMatrixExpected(0,8,1) = 0.14597505668934063;
    ipMatrixExpected(0,9,1) = -0.005663493824473509;
    ipMatrixExpected(0,10,1) = 0.1469441781125666;
    ipMatrixExpected(0,11,1) = -0.03505898637685291;
    ipMatrixExpected(0,12,1) = -0.06124543897569151;
    ipMatrixExpected(0,13,1) = 0.05668934240362877;
    ipMatrixExpected(0,14,1) = -0.056689342403626525;
    ipMatrixExpected(0,15,1) = 0.011642264372516295;
    ipMatrixExpected(0,0,2) = -0.07211898376488793;
    ipMatrixExpected(0,1,2) = -0.02267573696145056;
    ipMatrixExpected(0,2,2) = 2.2448979591836737;
    ipMatrixExpected(0,3,2) = -0.22550006385415905;
    ipMatrixExpected(0,4,2) = 0.1459750566893433;
    ipMatrixExpected(0,5,2) = -0.39496005112845367;
    ipMatrixExpected(0,6,2) = -1.482431744270763;
    ipMatrixExpected(0,7,2) = -0.2568911270018305;
    ipMatrixExpected(0,8,2) = -0.03505898637685309;
    ipMatrixExpected(0,9,2) = 0.146944178112573;
    ipMatrixExpected(0,10,2) = -0.005663493824473287;
    ipMatrixExpected(0,11,2) = 0.1459750566893434;
    ipMatrixExpected(0,12,2) = 0.01164226437251712;
    ipMatrixExpected(0,13,2) = -0.05668934240362705;
    ipMatrixExpected(0,14,2) = 0.05668934240362833;
    ipMatrixExpected(0,15,2) = -0.06124543897569129;
    ipMatrixExpected(0,0,3) = 0.04308390022675741;
    ipMatrixExpected(0,1,3) = -0.0721189837648888;
    ipMatrixExpected(0,2,3) = -0.22550006385415905;
    ipMatrixExpected(0,3,3) = 0.6394557823129253;
    ipMatrixExpected(0,4,3) = -0.06124543897569116;
    ipMatrixExpected(0,5,3) = 0.1459750566893417;
    ipMatrixExpected(0,6,3) = -0.25689112700183203;
    ipMatrixExpected(0,7,3) = -0.22550006385415894;
    ipMatrixExpected(0,8,3) = 0.011642264372516423;
    ipMatrixExpected(0,9,3) = -0.035058986376853604;
    ipMatrixExpected(0,10,3) = 0.1459750566893424;
    ipMatrixExpected(0,11,3) = -0.07211898376488812;
    ipMatrixExpected(0,12,3) = -0.0034013605442177715;
    ipMatrixExpected(0,13,3) = 0.011642264372516323;
    ipMatrixExpected(0,14,3) = -0.061245438975690514;
    ipMatrixExpected(0,15,3) = 0.043083900226757676;
    ipMatrixExpected(0,0,4) = -0.22550006385415977;
    ipMatrixExpected(0,1,4) = -0.25689112700183325;
    ipMatrixExpected(0,2,4) = 0.1459750566893433;
    ipMatrixExpected(0,3,4) = -0.06124543897569116;
    ipMatrixExpected(0,4,4) = 2.244897959183676;
    ipMatrixExpected(0,5,4) = -1.4824317442707666;
    ipMatrixExpected(0,6,4) = -0.005663493824484278;
    ipMatrixExpected(0,7,4) = 0.05668934240363208;
    ipMatrixExpected(0,8,4) = -0.022675736961449866;
    ipMatrixExpected(0,9,4) = -0.39496005112844856;
    ipMatrixExpected(0,10,4) = 0.14694417811257454;
    ipMatrixExpected(0,11,4) = -0.05668934240362803;
    ipMatrixExpected(0,12,4) = -0.07211898376488868;
    ipMatrixExpected(0,13,4) = 0.14597505668934396;
    ipMatrixExpected(0,14,4) = -0.035058986376852674;
    ipMatrixExpected(0,15,4) = 0.011642264372516201;
    ipMatrixExpected(0,0,5) = -0.25689112700183325;
    ipMatrixExpected(0,1,5) = -1.4824317442707629;
    ipMatrixExpected(0,2,5) = -0.39496005112845367;
    ipMatrixExpected(0,3,5) = 0.1459750566893417;
    ipMatrixExpected(0,4,5) = -1.4824317442707666;
    ipMatrixExpected(0,5,5) = 6.4625850340136015;
    ipMatrixExpected(0,6,5) = -0.9070294784580233;
    ipMatrixExpected(0,7,5) = -0.00566349382448289;
    ipMatrixExpected(0,8,5) = -0.3949600511284477;
    ipMatrixExpected(0,9,5) = -0.9070294784580477;
    ipMatrixExpected(0,10,5) = -0.48185941043084535;
    ipMatrixExpected(0,11,5) = 0.14694417811257748;
    ipMatrixExpected(0,12,5) = 0.14597505668934396;
    ipMatrixExpected(0,13,5) = -0.005663493824478616;
    ipMatrixExpected(0,14,5) = 0.14694417811258426;
    ipMatrixExpected(0,15,5) = -0.03505898637685055;
    ipMatrixExpected(0,0,6) = 0.14597505668934127;
    ipMatrixExpected(0,1,6) = -0.39496005112844845;
    ipMatrixExpected(0,2,6) = -1.482431744270763;
    ipMatrixExpected(0,3,6) = -0.25689112700183203;
    ipMatrixExpected(0,4,6) = -0.005663493824484278;
    ipMatrixExpected(0,5,6) = -0.9070294784580233;
    ipMatrixExpected(0,6,6) = 6.462585034013607;
    ipMatrixExpected(0,7,6) = -1.482431744270771;
    ipMatrixExpected(0,8,6) = 0.14694417811257543;
    ipMatrixExpected(0,9,6) = -0.48185941043087666;
    ipMatrixExpected(0,10,6) = -0.9070294784580464;
    ipMatrixExpected(0,11,6) = -0.394960051128445;
    ipMatrixExpected(0,12,6) = -0.03505898637685384;
    ipMatrixExpected(0,13,6) = 0.14694417811257954;
    ipMatrixExpected(0,14,6) = -0.005663493824483723;
    ipMatrixExpected(0,15,6) = 0.14597505668934352;
    ipMatrixExpected(0,0,7) = -0.06124543897569006;
    ipMatrixExpected(0,1,7) = 0.1459750566893411;
    ipMatrixExpected(0,2,7) = -0.2568911270018305;
    ipMatrixExpected(0,3,7) = -0.22550006385415894;
    ipMatrixExpected(0,4,7) = 0.05668934240363208;
    ipMatrixExpected(0,5,7) = -0.00566349382448289;
    ipMatrixExpected(0,6,7) = -1.482431744270771;
    ipMatrixExpected(0,7,7) = 2.2448979591836755;
    ipMatrixExpected(0,8,7) = -0.056689342403627864;
    ipMatrixExpected(0,9,7) = 0.14694417811257268;
    ipMatrixExpected(0,10,7) = -0.3949600511284459;
    ipMatrixExpected(0,11,7) = -0.02267573696145231;
    ipMatrixExpected(0,12,7) = 0.011642264372517367;
    ipMatrixExpected(0,13,7) = -0.035058986376853105;
    ipMatrixExpected(0,14,7) = 0.14597505668934374;
    ipMatrixExpected(0,15,7) = -0.07211898376488879;
    ipMatrixExpected(0,0,8) = -0.07211898376488846;
    ipMatrixExpected(0,1,8) = 0.14597505668934063;
    ipMatrixExpected(0,2,8) = -0.03505898637685309;
    ipMatrixExpected(0,3,8) = 0.011642264372516423;
    ipMatrixExpected(0,4,8) = -0.022675736961449866;
    ipMatrixExpected(0,5,8) = -0.3949600511284477;
    ipMatrixExpected(0,6,8) = 0.14694417811257543;
    ipMatrixExpected(0,7,8) = -0.056689342403627864;
    ipMatrixExpected(0,8,8) = 2.2448979591836737;
    ipMatrixExpected(0,9,8) = -1.4824317442707693;
    ipMatrixExpected(0,10,8) = -0.005663493824473398;
    ipMatrixExpected(0,11,8) = 0.05668934240362656;
    ipMatrixExpected(0,12,8) = -0.22550006385416005;
    ipMatrixExpected(0,13,8) = -0.2568911270018309;
    ipMatrixExpected(0,14,8) = 0.1459750566893427;
    ipMatrixExpected(0,15,8) = -0.06124543897569008;
    ipMatrixExpected(0,0,9) = 0.14597505668934174;
    ipMatrixExpected(0,1,9) = -0.005663493824473509;
    ipMatrixExpected(0,2,9) = 0.146944178112573;
    ipMatrixExpected(0,3,9) = -0.035058986376853604;
    ipMatrixExpected(0,4,9) = -0.39496005112844856;
    ipMatrixExpected(0,5,9) = -0.9070294784580477;
    ipMatrixExpected(0,6,9) = -0.48185941043087666;
    ipMatrixExpected(0,7,9) = 0.14694417811257268;
    ipMatrixExpected(0,8,9) = -1.4824317442707693;
    ipMatrixExpected(0,9,9) = 6.462585034013606;
    ipMatrixExpected(0,10,9) = -0.9070294784580477;
    ipMatrixExpected(0,11,9) = -0.005663493824471871;
    ipMatrixExpected(0,12,9) = -0.2568911270018318;
    ipMatrixExpected(0,13,9) = -1.4824317442707597;
    ipMatrixExpected(0,14,9) = -0.39496005112844773;
    ipMatrixExpected(0,15,9) = 0.14597505668934363;
    ipMatrixExpected(0,0,10) = -0.03505898637685316;
    ipMatrixExpected(0,1,10) = 0.1469441781125666;
    ipMatrixExpected(0,2,10) = -0.005663493824473287;
    ipMatrixExpected(0,3,10) = 0.1459750566893424;
    ipMatrixExpected(0,4,10) = 0.14694417811257454;
    ipMatrixExpected(0,5,10) = -0.48185941043084535;
    ipMatrixExpected(0,6,10) = -0.9070294784580464;
    ipMatrixExpected(0,7,10) = -0.3949600511284459;
    ipMatrixExpected(0,8,10) = -0.005663493824473398;
    ipMatrixExpected(0,9,10) = -0.9070294784580477;
    ipMatrixExpected(0,10,10) = 6.462585034013593;
    ipMatrixExpected(0,11,10) = -1.4824317442707644;
    ipMatrixExpected(0,12,10) = 0.14597505668934246;
    ipMatrixExpected(0,13,10) = -0.3949600511284378;
    ipMatrixExpected(0,14,10) = -1.4824317442707633;
    ipMatrixExpected(0,15,10) = -0.25689112700183114;
    ipMatrixExpected(0,0,11) = 0.011642264372517502;
    ipMatrixExpected(0,1,11) = -0.03505898637685291;
    ipMatrixExpected(0,2,11) = 0.1459750566893434;
    ipMatrixExpected(0,3,11) = -0.07211898376488812;
    ipMatrixExpected(0,4,11) = -0.05668934240362803;
    ipMatrixExpected(0,5,11) = 0.14694417811257748;
    ipMatrixExpected(0,6,11) = -0.394960051128445;
    ipMatrixExpected(0,7,11) = -0.02267573696145231;
    ipMatrixExpected(0,8,11) = 0.05668934240362656;
    ipMatrixExpected(0,9,11) = -0.005663493824471871;
    ipMatrixExpected(0,10,11) = -1.4824317442707644;
    ipMatrixExpected(0,11,11) = 2.244897959183673;
    ipMatrixExpected(0,12,11) = -0.06124543897569033;
    ipMatrixExpected(0,13,11) = 0.14597505668934319;
    ipMatrixExpected(0,14,11) = -0.2568911270018329;
    ipMatrixExpected(0,15,11) = -0.22550006385415988;
    ipMatrixExpected(0,0,12) = 0.04308390022675734;
    ipMatrixExpected(0,1,12) = -0.06124543897569151;
    ipMatrixExpected(0,2,12) = 0.01164226437251712;
    ipMatrixExpected(0,3,12) = -0.0034013605442177715;
    ipMatrixExpected(0,4,12) = -0.07211898376488868;
    ipMatrixExpected(0,5,12) = 0.14597505668934396;
    ipMatrixExpected(0,6,12) = -0.03505898637685384;
    ipMatrixExpected(0,7,12) = 0.011642264372517367;
    ipMatrixExpected(0,8,12) = -0.22550006385416005;
    ipMatrixExpected(0,9,12) = -0.2568911270018318;
    ipMatrixExpected(0,10,12) = 0.14597505668934246;
    ipMatrixExpected(0,11,12) = -0.06124543897569033;
    ipMatrixExpected(0,12,12) = 0.6394557823129252;
    ipMatrixExpected(0,13,12) = -0.22550006385415983;
    ipMatrixExpected(0,14,12) = -0.07211898376488801;
    ipMatrixExpected(0,15,12) = 0.04308390022675686;
    ipMatrixExpected(0,0,13) = -0.06124543897569068;
    ipMatrixExpected(0,1,13) = 0.05668934240362877;
    ipMatrixExpected(0,2,13) = -0.05668934240362705;
    ipMatrixExpected(0,3,13) = 0.011642264372516323;
    ipMatrixExpected(0,4,13) = 0.14597505668934396;
    ipMatrixExpected(0,5,13) = -0.005663493824478616;
    ipMatrixExpected(0,6,13) = 0.14694417811257954;
    ipMatrixExpected(0,7,13) = -0.035058986376853105;
    ipMatrixExpected(0,8,13) = -0.2568911270018309;
    ipMatrixExpected(0,9,13) = -1.4824317442707597;
    ipMatrixExpected(0,10,13) = -0.3949600511284378;
    ipMatrixExpected(0,11,13) = 0.14597505668934319;
    ipMatrixExpected(0,12,13) = -0.22550006385415983;
    ipMatrixExpected(0,13,13) = 2.2448979591836804;
    ipMatrixExpected(0,14,13) = -0.022675736961452586;
    ipMatrixExpected(0,15,13) = -0.07211898376488801;
    ipMatrixExpected(0,0,14) = 0.011642264372517065;
    ipMatrixExpected(0,1,14) = -0.056689342403626525;
    ipMatrixExpected(0,2,14) = 0.05668934240362833;
    ipMatrixExpected(0,3,14) = -0.061245438975690514;
    ipMatrixExpected(0,4,14) = -0.035058986376852674;
    ipMatrixExpected(0,5,14) = 0.14694417811258426;
    ipMatrixExpected(0,6,14) = -0.005663493824483723;
    ipMatrixExpected(0,7,14) = 0.14597505668934374;
    ipMatrixExpected(0,8,14) = 0.1459750566893427;
    ipMatrixExpected(0,9,14) = -0.39496005112844773;
    ipMatrixExpected(0,10,14) = -1.4824317442707633;
    ipMatrixExpected(0,11,14) = -0.2568911270018329;
    ipMatrixExpected(0,12,14) = -0.07211898376488801;
    ipMatrixExpected(0,13,14) = -0.022675736961452586;
    ipMatrixExpected(0,14,14) = 2.244897959183672;
    ipMatrixExpected(0,15,14) = -0.22550006385415985;
    ipMatrixExpected(0,0,15) = -0.003401360544217533;
    ipMatrixExpected(0,1,15) = 0.011642264372516295;
    ipMatrixExpected(0,2,15) = -0.06124543897569129;
    ipMatrixExpected(0,3,15) = 0.043083900226757676;
    ipMatrixExpected(0,4,15) = 0.011642264372516201;
    ipMatrixExpected(0,5,15) = -0.03505898637685055;
    ipMatrixExpected(0,6,15) = 0.14597505668934352;
    ipMatrixExpected(0,7,15) = -0.07211898376488879;
    ipMatrixExpected(0,8,15) = -0.06124543897569008;
    ipMatrixExpected(0,9,15) = 0.14597505668934363;
    ipMatrixExpected(0,10,15) = -0.25689112700183114;
    ipMatrixExpected(0,11,15) = -0.22550006385415988;
    ipMatrixExpected(0,12,15) = 0.04308390022675686;
    ipMatrixExpected(0,13,15) = -0.07211898376488801;
    ipMatrixExpected(0,14,15) = -0.22550006385415985;
    ipMatrixExpected(0,15,15) = 0.639455782312925;
  }
  
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
  ip->computeInnerProductMatrix(ipMatrixActual,cubicHGradOrdering, quad_4, quadPoints);
  
  string myNameIPMatrix = "testLowOrderTrialCubicTest.ipMatrix";
  successLocal = fcsAgree(myNameIPMatrix, ipMatrixExpected, ipMatrixActual, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameIPMatrix << ": comparison of ipMatrixExpected and ipMatrixActual failed." << endl;
  }
  
  FieldContainer<double> ipWeightsExpected(numTests,lowestOrderHGRADOrdering->totalDofs(),
                                           cubicHGradOrdering->totalDofs() );
  FieldContainer<double> ipWeightsActual(numTests,lowestOrderHGRADOrdering->totalDofs(),
                                         cubicHGradOrdering->totalDofs() );
  
  {
    ipWeightsExpected(0,0,0) = 0.30063516261594286;
    ipWeightsExpected(0,1,0) = -0.30063516261594286;
    ipWeightsExpected(0,2,0) = -0.08015953937081523;
    ipWeightsExpected(0,3,0) = 0.08015953937081523;
    ipWeightsExpected(0,0,1) = 0.10470477387489954;
    ipWeightsExpected(0,1,1) = -0.10470477387489954;
    ipWeightsExpected(0,2,1) = -0.04486003786515369;
    ipWeightsExpected(0,3,1) = 0.04486003786515369;
    ipWeightsExpected(0,0,2) = -0.10470477387489892;
    ipWeightsExpected(0,1,2) = 0.10470477387489892;
    ipWeightsExpected(0,2,2) = 0.0448600378651512;
    ipWeightsExpected(0,3,2) = -0.0448600378651512;
    ipWeightsExpected(0,0,3) = -0.30063516261594103;
    ipWeightsExpected(0,1,3) = 0.30063516261594103;
    ipWeightsExpected(0,2,3) = 0.08015953937081155;
    ipWeightsExpected(0,3,3) = -0.08015953937081155;
    ipWeightsExpected(0,0,4) = 0.254346840394987;
    ipWeightsExpected(0,1,4) = -0.254346840394987;
    ipWeightsExpected(0,2,4) = -0.1264478615917692;
    ipWeightsExpected(0,3,4) = 0.1264478615917692;
    ipWeightsExpected(0,0,5) = 0.0942986995822699;
    ipWeightsExpected(0,1,5) = -0.0942986995822699;
    ipWeightsExpected(0,2,5) = -0.05526611215778378;
    ipWeightsExpected(0,3,5) = 0.05526611215778378;
    ipWeightsExpected(0,0,6) = -0.09429869958226969;
    ipWeightsExpected(0,1,6) = 0.09429869958226969;
    ipWeightsExpected(0,2,6) = 0.055266112157781495;
    ipWeightsExpected(0,3,6) = -0.055266112157781495;
    ipWeightsExpected(0,0,7) = -0.254346840394987;
    ipWeightsExpected(0,1,7) = 0.254346840394987;
    ipWeightsExpected(0,2,7) = 0.1264478615917675;
    ipWeightsExpected(0,3,7) = -0.1264478615917675;
    ipWeightsExpected(0,0,8) = 0.1264478615917676;
    ipWeightsExpected(0,1,8) = -0.1264478615917676;
    ipWeightsExpected(0,2,8) = -0.25434684039498795;
    ipWeightsExpected(0,3,8) = 0.25434684039498795;
    ipWeightsExpected(0,0,9) = 0.055266112157781495;
    ipWeightsExpected(0,1,9) = -0.055266112157781495;
    ipWeightsExpected(0,2,9) = -0.09429869958227025;
    ipWeightsExpected(0,3,9) = 0.09429869958227025;
    ipWeightsExpected(0,0,10) = -0.05526611215778328;
    ipWeightsExpected(0,1,10) = 0.05526611215778328;
    ipWeightsExpected(0,2,10) = 0.09429869958227027;
    ipWeightsExpected(0,3,10) = -0.09429869958227027;
    ipWeightsExpected(0,0,11) = -0.12644786159176932;
    ipWeightsExpected(0,1,11) = 0.12644786159176932;
    ipWeightsExpected(0,2,11) = 0.2543468403949879;
    ipWeightsExpected(0,3,11) = -0.2543468403949879;
    ipWeightsExpected(0,0,12) = 0.08015953937081269;
    ipWeightsExpected(0,1,12) = -0.08015953937081269;
    ipWeightsExpected(0,2,12) = -0.30063516261594275;
    ipWeightsExpected(0,3,12) = 0.30063516261594275;
    ipWeightsExpected(0,0,13) = 0.04486003786515183;
    ipWeightsExpected(0,1,13) = -0.04486003786515183;
    ipWeightsExpected(0,2,13) = -0.10470477387490001;
    ipWeightsExpected(0,3,13) = 0.10470477387490001;
    ipWeightsExpected(0,0,14) = -0.04486003786515453;
    ipWeightsExpected(0,1,14) = 0.04486003786515453;
    ipWeightsExpected(0,2,14) = 0.10470477387490108;
    ipWeightsExpected(0,3,14) = -0.10470477387490108;
    ipWeightsExpected(0,0,15) = -0.08015953937081494;
    ipWeightsExpected(0,1,15) = 0.08015953937081494;
    ipWeightsExpected(0,2,15) = 0.30063516261594364;
    ipWeightsExpected(0,3,15) = -0.30063516261594364;
  }
  
  
  ElementTypePtr elemType = makeElemType(lowestOrderHGRADOrdering, cubicHGradOrdering, quad_4);
  BasisCachePtr basisCache = makeBasisCache(elemType,quadPoints,quadCellIDs);
  int optSuccess = bilinearForm->optimalTestWeights(ipWeightsActual, ipMatrixExpected,
                                                    elemType, cellSideParities, basisCache);
  
  string myNameIPWeights = "testLowOrderTrialCubicTest.ipWeights";
  
  if (optSuccess != 0) {
    cout << myNameIPWeights << ": computeOptimalTest failed." << endl;
  }
  
  successLocal = fcsAgree(myNameIPWeights, ipWeightsExpected, ipWeightsActual, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameIPWeights << ": comparison of ipWeightsExpected and ipWeightsActual failed." << endl;
  }
  
  // confirm that the optWeights actually fulfill the contract....
  // placed here mostly as sanity check for checkOptTestWeights
  if ( ! checkOptTestWeights(ipWeightsExpected,ipMatrixExpected,stiffnessExpected,tol) ) {
    cout << myNameIPWeights << ": check that optWeights == ipMatrix^(-1) * preStiffness failed." << endl;
    return false;
  }
  
  FieldContainer<double> finalStiffnessExpected(numTests,lowestOrderHGRADOrdering->totalDofs(),
                                                lowestOrderHGRADOrdering->totalDofs() );
  FieldContainer<double> finalStiffnessActual1(numTests,lowestOrderHGRADOrdering->totalDofs(),
                                               lowestOrderHGRADOrdering->totalDofs() );
  FieldContainer<double> finalStiffnessActual2(numTests,lowestOrderHGRADOrdering->totalDofs(),
                                               lowestOrderHGRADOrdering->totalDofs() );
  {
    finalStiffnessExpected(0,0,0) = 0.23260288716853858;
    finalStiffnessExpected(0,1,0) = -0.23260288716853858;
    finalStiffnessExpected(0,2,0) = -0.14819181481821705;
    finalStiffnessExpected(0,3,0) = 0.14819181481821705;
    finalStiffnessExpected(0,0,1) = -0.23260288716853858;
    finalStiffnessExpected(0,1,1) = 0.23260288716853858;
    finalStiffnessExpected(0,2,1) = 0.14819181481821705;
    finalStiffnessExpected(0,3,1) = -0.14819181481821705;
    finalStiffnessExpected(0,0,2) = -0.14819181481821703;
    finalStiffnessExpected(0,1,2) = 0.14819181481821703;
    finalStiffnessExpected(0,2,2) = 0.23260288716853922;
    finalStiffnessExpected(0,3,2) = -0.23260288716853922;
    finalStiffnessExpected(0,0,3) = 0.14819181481821703;
    finalStiffnessExpected(0,1,3) = -0.14819181481821703;
    finalStiffnessExpected(0,2,3) = -0.23260288716853922;
    finalStiffnessExpected(0,3,3) = 0.23260288716853922;
  }
  
  BilinearFormUtility::computeStiffnessMatrix(finalStiffnessActual1,ipMatrixExpected,ipWeightsExpected);
  
  string myNameFinalByMultiplying = "testLowOrderTrialCubicTest.finalStiffnessByMultiplying";
  successLocal = fcsAgree(myNameFinalByMultiplying, finalStiffnessExpected, finalStiffnessActual1, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByMultiplying << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  BilinearFormUtility::computeOptimalStiffnessMatrix(finalStiffnessActual2, ipWeightsExpected,
                                                     bilinearForm,
                                                     lowestOrderHGRADOrdering, cubicHGradOrdering,
                                                     quad_4, quadPoints, cellSideParities);
  
  string myNameFinalByIntegrating = "testLowOrderTrialCubicTest.finalStiffnessByIntegrating";
  successLocal = fcsAgree(myNameFinalByIntegrating, finalStiffnessExpected, finalStiffnessActual2, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByIntegrating << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  //cout << "ipMatrixActual:" << endl << ipMatrixActual;
  //cout << "ipWeightsActual:" << endl << ipWeightsActual;
  
  // now recompute the "final" matrices, but with the *actual* rather than expected ipMatrix and stiffness
  BilinearFormUtility::computeStiffnessMatrix(finalStiffnessActual1,ipMatrixActual,ipWeightsActual);
  
  string myNameFinalByMultiplyingUsingActual = "testLowOrderTrialCubicTest.finalStiffnessByMultiplyingUsingActual";
  successLocal = fcsAgree(myNameFinalByMultiplyingUsingActual, finalStiffnessExpected, finalStiffnessActual1, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByMultiplyingUsingActual << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }
  
  BilinearFormUtility::computeOptimalStiffnessMatrix(finalStiffnessActual2, ipWeightsActual,
                                                     bilinearForm,
                                                     lowestOrderHGRADOrdering, cubicHGradOrdering,
                                                     quad_4, quadPoints, cellSideParities);
  
  string myNameFinalByIntegratingUsingActual = "testLowOrderTrialCubicTest.finalStiffnessByIntegratingUsingActual";
  successLocal = fcsAgree(myNameFinalByIntegratingUsingActual, finalStiffnessExpected, finalStiffnessActual2, tol);
  
  if (! successLocal) {
    success = false;
    cout << myNameFinalByIntegratingUsingActual << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }

  BilinearFormUtility::setWarnAboutZeroRowsAndColumns(oldWarnState);
  return success;

}

bool DPGTests::testOptimalStiffnessByMultiplying() {
  // verifies the multiplication 
  int numTests = 1;
  int numTestDofs = 25;
  int numTrialDofs = 10;
  double tol = 1e-13;
  string myName = "testOptimalStiffnessByMultiplying";
  
  FieldContainer<double> optWeights(numTests,numTrialDofs,numTestDofs);
  FieldContainer<double> ipMatrix(numTests,numTestDofs,numTestDofs);
  FieldContainer<double> stiffness(numTests,numTrialDofs,numTrialDofs);
  FieldContainer<double> expectedStiffness(numTests,numTrialDofs,numTrialDofs);
  
  srand(0);
  for (int i=0; i<numTrialDofs; i++) {
    for (int j=0; j<numTestDofs; j++) {
      double random = (double) rand() / RAND_MAX;
      optWeights(0,i,j) = random;
    }
  }
  for (int i=0; i<numTestDofs; i++) {
    for (int j=0; j<numTestDofs; j++) {
      double random = (double) rand() / RAND_MAX;
      ipMatrix(0,i,j) = random;
    }
  }
  
  // compute expected value:
  FieldContainer<double> intermediate(numTests,numTrialDofs,numTestDofs);
  intermediate.initialize(0.0);
  // intermediate = optWeights * ipMatrix^T
  for (int row=0; row<numTrialDofs; row++) {
    for (int col=0; col<numTestDofs; col++) {
      for (int i=0; i<numTestDofs; i++) {
        intermediate(0,row,col) += optWeights(0,row,i)*ipMatrix(0,col,i);
      }
    }
  }
  // expectedStiffness = intermediate * optWeights^T
  expectedStiffness.initialize(0.0);
  for (int row=0; row<numTrialDofs; row++) {
    for (int col=0; col<numTrialDofs; col++) {
      for (int i=0; i<numTestDofs; i++) {
        expectedStiffness(0,row,col) += intermediate(0,row,i) * optWeights(0,col,i);
      }
    }
  }
  
  BilinearFormUtility::computeStiffnessMatrix(stiffness,ipMatrix,optWeights);
  return fcsAgree(myName,expectedStiffness,stiffness,tol);
  
  /*  int numTests = 1;
   int numTestDofs = 25;
   int numTrialDofs = 1;
   double tol = 1e-14;
   double expectedValue = 0.0;
   string myName = "testOptimalStiffnessByMultiplying";
   
   FieldContainer<double> optWeights(numTests,numTrialDofs,numTestDofs);
   FieldContainer<double> ipMatrix(numTests,numTestDofs,numTestDofs);
   FieldContainer<double> stiffness(numTests,numTrialDofs,numTrialDofs);
   
   srand(0);
   for (int i=0; i<numTestDofs; i++) {
   double random = (double) rand() / RAND_MAX;
   optWeights(0,0,i) = random;
   expectedValue += random*random;
   ipMatrix(0,i,i) = 1.0; // initialize identity
   }
   
   BilinearFormUtility::computeStiffnessMatrix(stiffness,ipMatrix,optWeights);
   double diff = abs(stiffness(0,0,0) - expectedValue);
   if (diff > tol) {
   cout << myName << ": expected " << expectedValue << ", but had " << stiffness(0,0,0) << endl;
   return false;
   } else {
   return true;
   }*/
  
}

bool DPGTests::testOptimalStiffnessByIntegrating() {
  // tests the method BilinearFormUtility::computeOptimalStiffnessMatrix
  // the approach is this: fake optimal weight matrix with just a single
  // 1 entry (all 0s otherwise).  The result should be a stiffness matrix
  // whose entries are exactly the trial space functions integrated against
  // that single test function--i.e. there will be one column in the resulting matrix
  // that corresponds exactly to a column from the 
  // BilinearFormUtility::computeStiffnessMatrix(FieldContainer<double> &stiffness, BilinearForm &bilinearForm,
  //                                             Teuchos::RCP<DofOrdering> trialOrdering, DofOrdering &testOrdering, 
  //                                             shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes)
  // method.  The rest of the matrix should be all 0s.
  
  bool success = true;
  int numTests = 1;
  double tol = 1e-14;
  int order = 2;
  int testOrder = 3; // these particular choices inspired by the first Poisson test that fails...
  int numSides = 4;
  string myName = "testOptimalStiffnessByIntegrating";
  
  //cout << myName << ": testing with order=" << order << ", testOrder=" << testOrder << endl;
  Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new PoissonBilinearForm() );
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  shards::CellTopology tri_3(shards::getCellTopologyData<shards::Triangle<3> >() );
  shards::CellTopology cellTopo;
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> triPoints(numTests,3,2);
  triPoints(0,0,0) = -1.0; // x1
  triPoints(0,0,1) = -1.0; // y1
  triPoints(0,1,0) = 1.0;
  triPoints(0,1,1) = -1.0;
  triPoints(0,2,0) = 1.0;
  triPoints(0,2,1) = 1.0;
  
  FieldContainer<double> nodePoints;
  
  for (numSides=3; numSides <= 4; numSides++) {
    if (numSides == 3) {
      cellTopo = tri_3;
      nodePoints = triPoints;
    } else {
      cellTopo = quad_4;
      nodePoints = quadPoints;
    }
    
    DofOrderingFactory dofOrderingFactory(bilinearForm);
    
    Teuchos::RCP<DofOrdering> trialOrdering = dofOrderingFactory.trialOrdering(order, cellTopo);
    Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(testOrder, cellTopo);
    
    int numTrialDofs = trialOrdering->totalDofs();
    int numTestDofs = testOrdering->totalDofs();
    
    FieldContainer<double> expectedStiffness(numTests, numTrialDofs, numTrialDofs);
    FieldContainer<double> actualStiffness(numTests, numTrialDofs, numTrialDofs);
    
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
    
    FieldContainer<double> optimalTestWeights(numTests, numTrialDofs, numTestDofs);
    FieldContainer<double> summingOptimalTestWeights(numTests, numTrialDofs, numTestDofs);
    
    // cols in this matrix will match the columns we expect in the actualStiffness
    FieldContainer<double> stiffness(numTests, numTestDofs, numTrialDofs);
    
    FieldContainer<double> cellSideParities(numTests,numSides);
    cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
    
    // determine the values to expect:
    BilinearFormUtility::computeStiffnessMatrix(stiffness, bilinearForm,
                                                trialOrdering, testOrdering,
                                                cellTopo, nodePoints, cellSideParities);
    
    // for each test dof, run a test--should populate the topmost row of the actualStiffness
    // matrix with the corresponding column of the "stiffness" matrix
    // we can do this efficiently by populating exactly one row in each column of the 
    // optimal test weights--since we test each trial against every test function, we only need
    // test each row once...
    
    // repeat the above until each row has been tested....
    
    // for a test to make sure we're summing properly...
    FieldContainer<double> sumOfExpected(numTests,numTrialDofs,numTrialDofs);
    
    int testIndex = 0;
    while (testIndex < numTestDofs) {
      // set up the expected stiffness
      expectedStiffness.initialize(0.0);
      optimalTestWeights.initialize(0.0);
      for (int trialIndex=0; trialIndex < numTrialDofs; trialIndex++) {
        if (testIndex < numTestDofs) {
          optimalTestWeights(0,trialIndex,testIndex) = 1.0;
          summingOptimalTestWeights(0,trialIndex,testIndex) = 1.0; // this one doesn't get cleared...
          for (int i=0; i<numTrialDofs; i++) {
            expectedStiffness(0,trialIndex,i) = stiffness(0,testIndex,i);
            sumOfExpected(0,trialIndex,i) += stiffness(0,testIndex,i);
          }
        }
        testIndex++;
      }
      // compute with the fake optimal test weights:
      BilinearFormUtility::computeOptimalStiffnessMatrix(actualStiffness, optimalTestWeights,
                                                         bilinearForm,
                                                         trialOrdering, testOrdering,
                                                         cellTopo, nodePoints, cellSideParities);
      bool localSuccess = fcsAgree(myName,expectedStiffness,actualStiffness,tol);
      if (! localSuccess) {
        success = false;
        cout << myName << ": failed for testIndex " << testIndex << "." << endl;
      } else {
        //cout << myName << ": succeeded for testIndex " << testIndex << "." << endl;
      }
    }
    
    // final test: check that we sum things properly
    /*for (int testIndex=0; testIndex < numTestDofs; testIndex++) {
     optimalTestWeights(0,trialIndex,testIndex) = 1.0;
     }*/
    BilinearFormUtility::computeOptimalStiffnessMatrix(actualStiffness, summingOptimalTestWeights,
                                                       bilinearForm,
                                                       trialOrdering, testOrdering,
                                                       cellTopo, nodePoints,cellSideParities);
    
    bool finalSuccess = fcsAgree(myName,sumOfExpected,actualStiffness,tol);
    if (! finalSuccess) {
      success = false;
      cout << myName << ": failed for final, summing test." << endl;
    } else {
      //cout << myName << ": succeeded for final, summing test." << endl;
    }
  }
  
  return success;
}

bool DPGTests::testComputeOptimalTest() {
  bool oldWarnState = BilinearFormUtility::warnAboutZeroRowsAndColumns();
  BilinearFormUtility::setWarnAboutZeroRowsAndColumns(false);
  
  string myName = "testComputeOptimalTest";
  
  double tol = 1e-14;
  
  int numTests = 1;
  int numSides = 4;
  
  bool bSuccess = true;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  shards::CellTopology tri_3(shards::getCellTopologyData<shards::Triangle<3> >() );
  shards::CellTopology cellTopo;
  FieldContainer<double> quadPoints(numTests,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;
  
  FieldContainer<double> triPoints(numTests,3,2);
  triPoints(0,0,0) = -1.0; // x1
  triPoints(0,0,1) = -1.0; // y1
  triPoints(0,1,0) = 1.0;
  triPoints(0,1,1) = -1.0;
  triPoints(0,2,0) = 1.0;
  triPoints(0,2,1) = 1.0;
  
  vector<int> cellIDs;
  cellIDs.push_back(0);
  
  FieldContainer<double> nodePoints;
  
  for (numSides=3; numSides <= 4; numSides++) {
    if (numSides == 3) {
      cellTopo = tri_3;
      nodePoints = triPoints;
    } else {
      cellTopo = quad_4;
      nodePoints = quadPoints;
    }
    
    int basisRank;
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis
    = 
    BasisFactory::getBasis(basisRank, C1_FAKE_POLY_ORDER, cellTopo.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    
    DofOrdering lowestOrderHGRADOrdering;
    Teuchos::RCP<DofOrdering> lowestOrderHGRADOrderingPtr = Teuchos::rcp(&lowestOrderHGRADOrdering,false);
    lowestOrderHGRADOrdering.addEntry(0,basis,0);
    
    int numTrialDofs = lowestOrderHGRADOrdering.totalDofs();
    int numTestDofs = lowestOrderHGRADOrdering.totalDofs();
    
    FieldContainer<double> expectedStiffness(numTests, numTrialDofs, numTestDofs);
    
    FieldContainer<double> actualStiffness(numTests, numTrialDofs, numTestDofs);
    
    Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new TestBilinearFormDx() );
    
    Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
    
    FieldContainer<double> optimalTestWeights(numTests, numTrialDofs, numTestDofs);
    
    FieldContainer<double> cellSideParities(numTests,numSides);
    cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
    
    FieldContainer<double> ipMatrix(numTests,numTestDofs,numTestDofs);
    ip->computeInnerProductMatrix(ipMatrix,Teuchos::rcp(&lowestOrderHGRADOrdering,false), cellTopo, nodePoints);
    
    ElementTypePtr elemType = makeElemType(lowestOrderHGRADOrderingPtr, lowestOrderHGRADOrderingPtr, cellTopo);
    BasisCachePtr basisCache = makeBasisCache(elemType,nodePoints,cellIDs);
    int success = bilinearForm->optimalTestWeights(optimalTestWeights, ipMatrix,
                                                   elemType, cellSideParities, basisCache);
    
    if (success != 0) {
      cout << myName << ": computeOptimalTest failed." << endl;
      return false;
    }
    
    //cout << "optimalTestWeights:" << optimalTestWeights << endl;
    //cout << myName << ": inner product matrix-- " << endl << ipMatrix;
    
    BilinearFormUtility::computeStiffnessMatrix(actualStiffness,ipMatrix,optimalTestWeights);
    
    BilinearFormUtility::computeOptimalStiffnessMatrix(expectedStiffness, optimalTestWeights,
                                                       bilinearForm,
                                                       lowestOrderHGRADOrderingPtr, lowestOrderHGRADOrderingPtr,
                                                       cellTopo, nodePoints, cellSideParities);
    
    if (! fcsAgree(myName, expectedStiffness,actualStiffness,tol) ) {
      bSuccess = false;
      cout << "testComputeOptimalTest: failed initial test" << endl;
    }
    
    // otherwise, let's try another test...
    for (int order = 1; order <= 5; order++) {
      //cout << "testComputeOptimalTest: running symmetric test for order=" << order << endl;
      
      DofOrderingFactory dofOrderingFactory(bilinearForm);
      Teuchos::RCP<DofOrdering> highOrderHGradOrdering = dofOrderingFactory.testOrdering(order, cellTopo);
      
      numTrialDofs = highOrderHGradOrdering->totalDofs();
      numTestDofs = highOrderHGradOrdering->totalDofs();
      
      expectedStiffness.resize(numTests, numTrialDofs, numTestDofs);
      actualStiffness.resize(numTests, numTrialDofs, numTestDofs);
      optimalTestWeights.resize(numTests, numTrialDofs, numTestDofs);
      
      
      ipMatrix.resize(numTests,numTestDofs,numTestDofs);
      ip->computeInnerProductMatrix(ipMatrix,highOrderHGradOrdering, cellTopo, nodePoints);
      
      elemType = makeElemType(highOrderHGradOrdering, highOrderHGradOrdering, cellTopo);
      basisCache = makeBasisCache(elemType,nodePoints,cellIDs);
      success = bilinearForm->optimalTestWeights(optimalTestWeights, ipMatrix, 
                                                 elemType, cellSideParities, basisCache);

      if (success != 0) {
        cout << myName << ": computeOptimalTest failed." << endl;
        bSuccess = false;
      }
      
      //cout << "optimalTestWeights:" << optimalTestWeights << endl;
      
      BilinearFormUtility::computeStiffnessMatrix(actualStiffness,ipMatrix,optimalTestWeights);
      
      BilinearFormUtility::computeOptimalStiffnessMatrix(expectedStiffness, optimalTestWeights,
                                                         bilinearForm,
                                                         highOrderHGradOrdering, highOrderHGradOrdering,
                                                         cellTopo, nodePoints, cellSideParities);
      
      if (! fcsAgree(myName, expectedStiffness,actualStiffness,tol) ) {
        bSuccess = false;
      }
    }
    
    DofOrderingFactory dofOrderingFactory(bilinearForm);
    
    // otherwise, let's try another test...
    for (int testOrder = 2; testOrder <= 5; testOrder++) {
      //cout << "testComputeOptimalTest: running asymmetric test for testOrder=" << testOrder << endl;
      
      Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(testOrder, cellTopo);
      
      numTrialDofs = lowestOrderHGRADOrdering.totalDofs();
      numTestDofs = testOrdering->totalDofs();
      
      expectedStiffness.resize(numTests, numTrialDofs, numTrialDofs);
      actualStiffness.resize(numTests, numTrialDofs, numTrialDofs);
      optimalTestWeights.resize(numTests, numTrialDofs, numTestDofs);
      
      ipMatrix.resize(numTests,numTestDofs,numTestDofs);
      ip->computeInnerProductMatrix(ipMatrix,testOrdering, cellTopo, nodePoints);
      
      elemType = makeElemType(lowestOrderHGRADOrderingPtr, testOrdering, cellTopo);
      basisCache = makeBasisCache(elemType,nodePoints,cellIDs);
      success = bilinearForm->optimalTestWeights(optimalTestWeights, ipMatrix,
                                                 elemType, cellSideParities, basisCache);
      
      if (success != 0) {
        cout << myName << ": computeOptimalTest failed." << endl;
        bSuccess = false;
      }
      
      //cout << "optimalTestWeights:" << optimalTestWeights << endl;
      
      BilinearFormUtility::computeStiffnessMatrix(actualStiffness,ipMatrix,optimalTestWeights);
      
      BilinearFormUtility::computeOptimalStiffnessMatrix(expectedStiffness, optimalTestWeights,
                                                         bilinearForm,
                                                         lowestOrderHGRADOrderingPtr, testOrdering,
                                                         cellTopo, nodePoints, cellSideParities);
      
      if (! fcsAgree(myName, expectedStiffness,actualStiffness,tol) ) {
        bSuccess = false;
      }
    }
  }
  BilinearFormUtility::setWarnAboutZeroRowsAndColumns(oldWarnState);
  return bSuccess;
}

bool DPGTests::testWeightBasis() {
  int numCells = 2;
  int numBasisFields = 18;
  int offset = 8;
  int spaceDim = 3;
  FieldContainer<double> basisValues(numCells,numBasisFields,spaceDim);
  basisValues.initialize(3.0);
  FieldContainer<double> expectedBasisValues(basisValues);
  FieldContainer<double> weights(numCells,numBasisFields+offset);
  weights.initialize(1000.0);
  for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
    for (int fieldIndex=0; fieldIndex < numBasisFields; fieldIndex++) {
      weights(cellIndex,fieldIndex+offset) = cellIndex * 1.0 + fieldIndex * 1.34;
      for (int i=0; i<spaceDim; i++) {
        expectedBasisValues(cellIndex,fieldIndex,i) *= weights(cellIndex,fieldIndex+offset);
      }
    }
  }
  BilinearFormUtility::weightCellBasisValues(basisValues, weights, offset);
  bool success = true;
  for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
    for (int fieldIndex=0; fieldIndex < numBasisFields; fieldIndex++) {
      for (int i=0; i<spaceDim; i++) {
        if (expectedBasisValues(cellIndex,fieldIndex,i) != basisValues(cellIndex,fieldIndex,i)) {
          success = false;
          cout << "testWeightBasis: values differ in cellIndex=" << cellIndex << ", fieldIndex=" 
          << fieldIndex << ", spaceDim=" << i << endl;
        }
      }
    }
  }
  return success;
}

bool DPGTests::fcIsSymmetric(FieldContainer<double> &fc, double tol, int &cellOfAsymmetry, int &rowOfAsymmetry, int &colOfAsymmetry) {
  for (int cellIndex=0; cellIndex<fc.dimension(0); cellIndex++) {
    for (int i=0; i<fc.dimension(1); i++) {
      for (int j=0; j<fc.dimension(2); j++) {
        double diff = abs(fc(cellIndex,i,j) - fc(cellIndex,j,i));
        if (diff > tol) {
          cellOfAsymmetry = cellIndex;
          rowOfAsymmetry = i;
          colOfAsymmetry = j;
          return false;
        }
      }
    }
  }
  return true;
}

bool DPGTests::checkOptTestWeights(FieldContainer<double> &optWeights,
                                   FieldContainer<double> &ipMatrix,
                                   FieldContainer<double> &preStiffness, double tol) {
  string myName = "checkOptTestWeights";
  // should be the case that optWeights = ipMatrix^(-1) * stiffness, which is to say
  //    stiffness = ipMatrix * optWeights...
  bool success = true;
  int numCells = optWeights.dimension(0);
  FieldContainer<double> preStiffExpected(numCells,preStiffness.dimension(1),
                                          preStiffness.dimension(2));
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    Epetra_SerialDenseMatrix optWeightsT(Copy,
                                         &optWeights(cellIndex,0,0),
                                         optWeights.dimension(2), // stride
                                         optWeights.dimension(2),optWeights.dimension(1));
    
    Epetra_SerialDenseMatrix ipMatrixT(Copy,
                                       &ipMatrix(cellIndex,0,0),
                                       ipMatrix.dimension(2), // stride
                                       ipMatrix.dimension(2),ipMatrix.dimension(1));
    Epetra_SerialDenseMatrix preStiffExpectedSDM(preStiffness.dimension(1), // note not transposed
                                                 preStiffness.dimension(2));
    int multResult = preStiffExpectedSDM.Multiply('T','N',1.0,ipMatrixT,optWeightsT,0.0);
    if (multResult != 0) {
      cout << myName << ": multiplication failed with code: " << multResult << endl;
    }
    // copy to FieldContainer
    for (int i=0; i<preStiffExpected.dimension(1); i++) {
      for (int j=0; j<preStiffExpected.dimension(2); j++) {
        preStiffExpected(cellIndex,i,j) = preStiffExpectedSDM(i,j);
      }
    }
  }
  success = fcsAgree(myName,preStiffExpected,preStiffness,tol);
  return success;
}

bool DPGTests::testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices() {
  bool success = true;
  string myName = "testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices";
  
  FieldContainer<double> preStiffnessConforming;
  FieldContainer<double> preStiffnessNonConforming;
  TestBilinearFormAnalyticBoundaryIntegral::expectedPreStiffnessForCubicsOnQuad(preStiffnessConforming,true);
  TestBilinearFormAnalyticBoundaryIntegral::expectedPreStiffnessForCubicsOnQuad(preStiffnessNonConforming,false);
  
  TestBilinearFormAnalyticBoundaryIntegral bilinearForm;
  
  Teuchos::RCP<DofOrdering> conformingOrdering, nonConformingOrdering;
  
  DofOrderingFactory dofOrderingFactory(Teuchos::rcp(&bilinearForm,false));
  
  int polyOrder = 3;
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  conformingOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, true);
  nonConformingOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, false);
  
  Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(polyOrder, quad_4);
  
  int numSides = 4;
  int numDofsPerSide = 4;
  
  int numTrialDofsConforming = conformingOrdering->totalDofs();
  //  int numTrialDofsNonConforming = nonConformingOrdering->totalDofs();
  int numTestDofs = testOrdering->totalDofs();
  
  int numTests = 1;
  
  double tol = 1e-15;
  
  FieldContainer<double> expectedConformingStiffness(numTests, numTestDofs, numTrialDofsConforming);
  
  expectedConformingStiffness.initialize(0.0);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    for (int dofOrdinal=0; dofOrdinal<numDofsPerSide; dofOrdinal++) {
      int trialDofIndexConforming = conformingOrdering->getDofIndex(0, dofOrdinal, sideIndex);
      int trialDofIndexNonConforming = nonConformingOrdering->getDofIndex(0, dofOrdinal, sideIndex);
      for (int testDofIndex=0; testDofIndex<numTestDofs; testDofIndex++) {
        expectedConformingStiffness(0,testDofIndex,trialDofIndexConforming) 
        += preStiffnessNonConforming(0,testDofIndex,trialDofIndexNonConforming);
      }
    }
  }
  
  success = fcsAgree(myName,expectedConformingStiffness,preStiffnessConforming,tol);
  
  return success;
}

bool DPGTests::testComputeOptimalTestPoisson() {
  string myName = "testComputeOptimalTestPoisson";
  
  int numTests = 1;
  int numSides = 4; // quad
  
  double tol = 1e-12;
  
  bool bSuccess = true;
  
  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0
  
  for (int order = 2; order <= 3; order++) {    
    for (int testOrder=order+3; testOrder < order+5; testOrder++) {
      //cout << "testComputeOptimalTestPoisson: testing with order=" << order << ", testOrder=" << testOrder << endl;
      Teuchos::RCP<BilinearForm> bilinearForm = Teuchos::rcp( new PoissonBilinearForm() );
      
      DofOrderingFactory dofOrderingFactory(bilinearForm);
      
      shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
      
      Teuchos::RCP<DofOrdering> trialOrdering = dofOrderingFactory.trialOrdering(order, quad_4);
      Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(testOrder, quad_4);
      
      int numTrialDofs = trialOrdering->totalDofs();
      int numTestDofs = testOrdering->totalDofs();
      
      FieldContainer<double> expectedStiffness(numTests, numTrialDofs, numTrialDofs);
      
      FieldContainer<double> actualStiffness(numTests, numTrialDofs, numTrialDofs);
      
      Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
      
      FieldContainer<double> preStiffness(numTests, numTestDofs, numTrialDofs); // the RHS for opt test determination
      FieldContainer<double> optimalTestWeights(numTests, numTrialDofs, numTestDofs);
      
      FieldContainer<double> quadPoints(numTests,4,2);
      quadPoints(0,0,0) = -1.0; // x1
      quadPoints(0,0,1) = -1.0; // y1
      quadPoints(0,1,0) = 1.0;
      quadPoints(0,1,1) = -1.0;
      quadPoints(0,2,0) = 1.0;
      quadPoints(0,2,1) = 1.0;
      quadPoints(0,3,0) = -1.0;
      quadPoints(0,3,1) = 1.0;
      vector<int> cellIDs;
      cellIDs.push_back(0);
      
      FieldContainer<double> ipMatrix(numTests,numTestDofs,numTestDofs);
      
      ip->computeInnerProductMatrix(ipMatrix,testOrdering, quad_4, quadPoints);
      
      int i,j,cellIndex;
      if (! fcIsSymmetric(ipMatrix,tol,cellIndex,i,j) ) {
        cout << myName << ": inner product matrix asymmetric for i=" << i << ", j=" << j << endl;
        cout << "ipMatrix(i,j)=" << ipMatrix(cellIndex,i,j) << "; ipMatrix(j,i)=" << ipMatrix(cellIndex,j,i) << endl;
      }
      
      ElementTypePtr elemType = makeElemType(trialOrdering, testOrdering, quad_4);
      BasisCachePtr basisCache = makeBasisCache(elemType,quadPoints,cellIDs);
      int success = bilinearForm->optimalTestWeights(optimalTestWeights, ipMatrix,
                                                     elemType, cellSideParities, basisCache);
      if (success != 0) {
        cout << myName << ": computeOptimalTest failed." << endl;
        return false;
      }
      
      // let's try to confirm that the optWeights actually fulfill the contract....
      BilinearFormUtility::computeStiffnessMatrix(preStiffness, bilinearForm,trialOrdering, testOrdering, quad_4, quadPoints, cellSideParities);
      if ( ! checkOptTestWeights(optimalTestWeights,ipMatrix,preStiffness,tol) ) {
        cout << myName << ": check that optWeights == ipMatrix^(-1) * preStiffness failed." << endl;
        return false;
      }
      
      BilinearFormUtility::computeStiffnessMatrix(actualStiffness,ipMatrix,optimalTestWeights);
      
      if (! fcIsSymmetric(actualStiffness,tol,cellIndex,i,j) ) {
        cout << myName << ": actualStiffness matrix asymmetric for i=" << i << ", j=" << j << endl;
        cout << "ipMatrix(i,j)=" << actualStiffness(cellIndex,i,j) << "; ipMatrix(j,i)=" << actualStiffness(cellIndex,j,i) << endl;
      }
      
      BilinearFormUtility::computeOptimalStiffnessMatrix(expectedStiffness, optimalTestWeights,
                                                         bilinearForm,
                                                         trialOrdering, testOrdering,
                                                         quad_4, quadPoints, cellSideParities);
      
      bool localBSuccess = fcsAgree(myName, expectedStiffness,actualStiffness,tol);
      
      if ( ! localBSuccess ) {
        cout << "failed Poisson test for testOrder = " << testOrder << endl;
        bSuccess = false;
      }
    }
  }
  return bSuccess;
}


bool DPGTests::testProjection(){
  double tol = 1e-14;
  // reference cell physical cell nodes in counterclockwise order
  FieldContainer<double> physicalCellNodes(1,4,2);
  physicalCellNodes(0,0,0) = -1.0;
  physicalCellNodes(0,0,1) = -1.0;

  physicalCellNodes(0,1,0) = 1.0;
  physicalCellNodes(0,1,1) = -1.0; 

  physicalCellNodes(0,2,0) = 1.0;
  physicalCellNodes(0,2,1) = 1.0;

  physicalCellNodes(0,3,0) = -1.0;
  physicalCellNodes(0,3,1) = 1.0;

  EFunctionSpaceExtended fs = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  BasisFactory basisFactory;
  unsigned cellTopoKey = shards::Quadrilateral<4>::key;

  FieldContainer<double> basisCoefficients;

  int polyOrder = 5; // some large number
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = basisFactory.getBasis( polyOrder, cellTopoKey, fs);  

  // creating basisCache to compute values at certain points
  shards::CellTopology cellTopo = basis->getBaseCellTopology();
  int basisRank = BasisFactory::getBasisRank(basis);
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering());
  int ID = 0; // fake ID 
  dofOrderPtr->addEntry(ID,basis,basisRank);  
  int maxTrialDegree = dofOrderPtr->maxBasisDegree();

  BasisCache basisCache(physicalCellNodes, cellTopo, *(dofOrderPtr), maxTrialDegree, false);

  // simple function f(x,y) = x;
  Teuchos::RCP<SimpleQuadraticFunction> simpleFunction = Teuchos::rcp(new SimpleQuadraticFunction());

  Projector::projectFunctionOntoBasis(basisCoefficients, simpleFunction, basis, physicalCellNodes);      

  int numDofs = basis->getCardinality();
  EOperatorExtended op = IntrepidExtendedTypes::OP_VALUE;
  FieldContainer<double> cubPoints = basisCache.getPhysicalCubaturePoints();    
  FieldContainer<double> basisValues = *(basisCache.getTransformedValues(basis, op));
  int numPts = cubPoints.dimension(1);
  FieldContainer<double> basisSum(numPts);
  FieldContainer<double> functionValues;
  simpleFunction->getValues(functionValues,cubPoints);
  bool passedTest = true;
  for (int i=0;i<numPts;i++){
    double x = cubPoints(0,i,0);
    for (int j = 0;j<numDofs;j++){
      basisSum(i) += basisCoefficients(0,j)*basisValues(0,j,i);
    }
    if (abs(basisSum(i)-functionValues(0,i))>tol){
      passedTest = false;
    }
  }

  return passedTest;
}
