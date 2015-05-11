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

#include <Teuchos_GlobalMPISession.hpp>

#include "doubleBasisConstruction.h"

#include "SerialDenseWrapper.h"

#include "DPGTests.h"

// test suite includes
#include "CurvilinearMeshTests.h"
#include "ElementTests.h"
#include "FunctionTests.h"
#include "GDAMinimumRuleTests.h"
#include "GMGTests.h"
#include "HConvergenceStudyTests.h"
#include "IncompressibleFormulationsTests.h"
#include "LinearTermTests.h"
#include "LobattoBasisTests.h"
#include "MeshRefinementTests.h"
#include "MPIWrapperTests.h"
#include "MultiBasisTests.h"
#include "MeshTopologyTests.h"
#include "PatchBasisTests.h"
#include "ParametricCurveTests.h"
#include "RHSTests.h"
#include "ScratchPadTests.h"
#include "SerialDenseMatrixUtilityTests.h"
#include "SolutionTests.h"
#include "TensorBasis.h"

#include "MeshTools.h"

#include "Basis.h"
#include "BasisCache.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "HDF5Exporter.h"
#include "PoissonFormulation.h"
#include "Projector.h"

#include <vector>

//#include <fenv.h>

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
#include <xmmintrin.h>
#endif

using namespace std;
using namespace Intrepid;
using namespace Camellia;

ElementTypePtr makeElemType(DofOrderingPtr trialOrdering, DofOrderingPtr testOrdering, CellTopoPtr cellTopo) {
  return Teuchos::rcp( new ElementType( trialOrdering, testOrdering, cellTopo) );
}

BasisCachePtr makeBasisCache(ElementTypePtr elemType, const FieldContainer<double> &physicalCellNodes, const vector<GlobalIndexType> &cellIDs,
                         bool createSideCacheToo = true) {
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType) );
  basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
  return basisCache;
}

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
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
  BasisPtr basis;
  int rangeDimension = 2, scalarRank = 0, vectorRank = 1;
  Camellia::EFunctionSpace hgrad = Camellia::FUNCTION_SPACE_HGRAD;
  Camellia::EFunctionSpace hdiv = Camellia::FUNCTION_SPACE_HDIV;

  basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_QUAD_C1_FEM<double,FieldContainer<double> >()), rangeDimension, scalarRank, hgrad) );
  BasisFactory::basisFactory()->registerBasis(basis,0, C1_FAKE_POLY_ORDER, quad_4.getKey(), hgrad);
  basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_QUAD_C1_FEM<double,FieldContainer<double> >()), rangeDimension, scalarRank, hgrad ) );
  BasisFactory::basisFactory()->registerBasis(basis,0, C1_FAKE_POLY_ORDER, tri_3.getKey(), hgrad);
  basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_LINE_Cn_FEM<double,FieldContainer<double> >(3,POINTTYPE_SPECTRAL)), rangeDimension, scalarRank, hgrad ) );
  BasisFactory::basisFactory()->registerBasis(basis,0, C3_FAKE_POLY_ORDER, line_2.getKey(), hgrad);
  basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_LINE_Cn_FEM<double,FieldContainer<double> >(1,POINTTYPE_SPECTRAL)), rangeDimension, scalarRank, hgrad ) );
  BasisFactory::basisFactory()->registerBasis(basis,0, C1_FAKE_POLY_ORDER, line_2.getKey(), hgrad);
  basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HDIV_QUAD_I1_FEM<double,FieldContainer<double> >()), rangeDimension, vectorRank, hdiv ) );
  BasisFactory::basisFactory()->registerBasis(basis,1, C1_FAKE_POLY_ORDER, quad_4.getKey(), hdiv);
}

void DPGTests::runTests() {

  int rank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::TestForException_setEnableStacktrace(true);

//  fexcept_t flag;
//  fegetexceptflag(&flag, FE_INVALID | FE_DIVBYZERO);
//  fesetexceptflag(&flag, FE_INVALID | FE_DIVBYZERO);

//  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
//  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

  bool success;
  int numTestsTotal = 0;
  int numTestsPassed = 0;
  bool skipSlowTests = false;

  // set up a few special entries for BasisFactory first:
  createBases();

  runExceptionThrowingTest(); // placeholder for Teuchos unit tests that we copy and paste here to catch them when they fail by throwing an exception (the Teuchos framework catches them and doesn't provide a stack trace).

  // setup our TestSuite tests:
  vector< Teuchos::RCP< TestSuite > > testSuites;

  testSuites.push_back( Teuchos::rcp( new GMGTests ) );

  if (skipSlowTests) {
    if (rank==0) {
      cout << "skipping slow tests (IncompressibleFormulationsTests).\n";
    }
  } else {
    testSuites.push_back( Teuchos::rcp( new IncompressibleFormulationsTests(false) ) ); // false: turn "thorough" off
  }


  testSuites.push_back( Teuchos::rcp( new GDAMinimumRuleTests ) );

  testSuites.push_back( Teuchos::rcp( new LinearTermTests ) );

  testSuites.push_back( Teuchos::rcp( new MeshRefinementTests ) );

  testSuites.push_back( Teuchos::rcp( new SolutionTests ) );

  testSuites.push_back( Teuchos::rcp( new MeshTopologyTests ) );

  testSuites.push_back( Teuchos::rcp( new MeshTestSuite ) );

  testSuites.push_back( Teuchos::rcp( new ScratchPadTests ) );

  testSuites.push_back( Teuchos::rcp( new ElementTests ) );

  testSuites.push_back( Teuchos::rcp( new MultiBasisTests ) );

  testSuites.push_back( Teuchos::rcp( new FunctionTests ) );

  testSuites.push_back( Teuchos::rcp( new CurvilinearMeshTests) );

  testSuites.push_back( Teuchos::rcp( new SerialDenseMatrixUtilityTests) );


  testSuites.push_back( Teuchos::rcp( new MPIWrapperTests) );
  testSuites.push_back( Teuchos::rcp( new ParametricCurveTests) );
  testSuites.push_back( Teuchos::rcp( new RHSTests ) );
  testSuites.push_back( Teuchos::rcp( new VectorizedBasisTestSuite ) );

  testSuites.push_back( Teuchos::rcp( new HConvergenceStudyTests ) );

  testSuites.push_back( Teuchos::rcp( new LobattoBasisTests ) );

  //  testSuites.push_back( Teuchos::rcp( new IncompressibleFormulationsTests(true) ) ); // true: turn "thorough" on
  //  testSuites.push_back( Teuchos::rcp( new PatchBasisTests ) ); // skip until we have a proper GDAMinimumRule constructed

  int numTestSuites = testSuites.size();
  for (int testSuiteIndex = 0; testSuiteIndex < numTestSuites; testSuiteIndex++) {
    Teuchos::RCP< TestSuite > testSuite = testSuites[testSuiteIndex];
    int numSuiteTests = 0, numSuiteTestsPassed = 0;
    string name = testSuite->testSuiteName();
    if (rank==0) cout << "Running " << name << "." << endl;
    testSuite->runTests(numSuiteTests, numSuiteTestsPassed);
    if (rank==0) cout << name << ": passed " << numSuiteTestsPassed << "/" << numSuiteTests << " tests." << endl;
    numTestsTotal  += numSuiteTests;
    numTestsPassed += numSuiteTestsPassed;
    testSuites[testSuiteIndex] = Teuchos::rcp((TestSuite*) NULL); // allows memory to be reclaimed
  }

  success = testMathInnerProductDx();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testMathInnerProductDx." << endl;
  } else {
    if (rank==0) cout << "Failed test testMathInnerProductDx." << endl;
  }

  success = testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices." << endl;
  } else {
    if (rank==0) cout << "Failed test testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices." << endl;
  }

  success = testComputeStiffnessConformingVertices();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testComputeStiffnessConformingVertices." << endl;
  } else {
    if (rank==0) cout << "Failed test testComputeStiffnessConformingVertices." << endl;
  }


  success = testAnalyticBoundaryIntegral(false);
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testAnalyticBoundaryIntegral (non-conforming)." << endl;
  } else {
    if (rank==0) cout << "Failed test testAnalyticBoundaryIntegral (non-conforming)." << endl;
  }

  success = testAnalyticBoundaryIntegral(true);
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testAnalyticBoundaryIntegral (conforming)." << endl;
  } else {
    if (rank==0) cout << "Failed test testAnalyticBoundaryIntegral (conforming)." << endl;
  }

  success = testOptimalStiffnessByMultiplying();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testOptimalStiffnessByMultiplying." << endl;
  } else {
    if (rank==0) cout << "Failed test testOptimalStiffnessByMultiplying." << endl;
  }

  success = testWeightBasis();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testWeightBasis." << endl;
  } else {
    if (rank==0) cout << "Failed test testWeightBasis." << endl;
  }

  success = testDofOrdering();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test testDofOrdering." << endl;
  } else {
    if (rank==0) cout << "Failed test testDofOrdering." << endl;
  }

  success = testComputeStiffnessTrace();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test ComputeStiffnessTrace." << endl;
  } else {
    if (rank==0) cout << "Failed test ComputeStiffnessTrace." << endl;
  }

  success = testComputeStiffnessFlux();
  ++numTestsTotal;
  if (success) {
    numTestsPassed++;
    if (rank==0) cout << "Passed test ComputeStiffnessFlux." << endl;
  } else {
    if (rank==0) cout << "Failed test ComputeStiffnessFlux." << endl;
  }

  if (rank==0) cout << "Passed " << numTestsPassed << " out of " << numTestsTotal << "." << endl;
}

bool DPGTests::testComputeStiffnessConformingVertices() {
  bool oldWarnState = BilinearFormUtility<double>::warnAboutZeroRowsAndColumns();
  BilinearFormUtility<double>::setWarnAboutZeroRowsAndColumns(false);

  bool success = true;

  string myName = "testComputeStiffnessConformingVertices";

  TBFPtr<double> bilinearForm = Teuchos::rcp(new TestBilinearFormTrace());

  int polyOrder = 3;
  Teuchos::RCP<DofOrdering> conformingOrdering,nonConformingOrdering;
  CellTopoPtr quad_4 = Camellia::CellTopology::quad();

  DofOrderingFactory dofOrderingFactory(bilinearForm);

  vector<int> polyOrderVector(1,polyOrder);

  conformingOrdering = dofOrderingFactory.trialOrdering(polyOrderVector, quad_4, true);
  nonConformingOrdering = dofOrderingFactory.trialOrdering(polyOrderVector, quad_4, false);

  Teuchos::RCP<DofOrdering> testOrdering = dofOrderingFactory.testOrdering(polyOrderVector, quad_4);

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

  BilinearFormUtility<double>::computeStiffnessMatrix(conformingStiffness, bilinearForm,
                                              conformingOrdering, testOrdering,
                                              quad_4, quadPoints,cellSideParities);
  BilinearFormUtility<double>::computeStiffnessMatrix(nonConformingStiffness, bilinearForm,
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

  BilinearFormUtility<double>::setWarnAboutZeroRowsAndColumns(oldWarnState);
  return success;

}

bool DPGTests::testDofOrdering() {
  bool success = true;
  DofOrdering traceOrdering(CellTopology::line());

  string myName = "testDofOrdering";

  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );

  BasisPtr traceBasis
  =
  BasisFactory::basisFactory()->getBasis(C1_FAKE_POLY_ORDER,
                         line_2.getKey(), Camellia::FUNCTION_SPACE_HGRAD);

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
  BFPtr bilinearForm = Teuchos::rcp(new TestBilinearFormTrace());
  int polyOrder = 1; // keep things simple
  vector<int> polyOrderVector(1,polyOrder);
  Teuchos::RCP<DofOrdering> trialOrder;
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );

  DofOrderingFactory dofOrderingFactory(bilinearForm);
  trialOrder = dofOrderingFactory.trialOrdering(polyOrderVector, quad_4);

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
      if ( expectedDofIndex != actualDofIndex ) {
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
    vector<int> polyOrderVector(1,polyOrder);
    dofsPerSide = polyOrder+1;
    if (numSides == 3) {
      trialOrder = dofOrderingFactory.trialOrdering(polyOrderVector, tri_3);
    } else {
      trialOrder = dofOrderingFactory.trialOrdering(polyOrderVector, quad_4);
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
        if ( expectedDofIndex != actualDofIndex ) {
          cout << myName << ": failed conforming vertex test for sideIndex " << sideIndex << ", dofOrdinal " << dofOrdinal << "." << endl;
          cout << "Expected " << expectedDofIndex << "; actual was " << actualDofIndex << "." << endl;
          success = false;
        }
      }
    }
  }
  return success;
}

bool DPGTests::testComputeStiffnessFlux() {
  bool success = true;
  Teuchos::RCP<DofOrdering> traceOrdering = Teuchos::rcp(new DofOrdering(CellTopology::quad()));
  Teuchos::RCP<DofOrdering> testOrdering = Teuchos::rcp(new DofOrdering(CellTopology::quad()));

  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );

  BasisPtr traceBasis
  =
  BasisFactory::basisFactory()->getBasis(C1_FAKE_POLY_ORDER,
                         line_2.getKey(), Camellia::FUNCTION_SPACE_HGRAD);

  int numSides = 4;

  for (int i=0; i<numSides; i++) {
    traceOrdering->addEntry(0,traceBasis,0,i);
  }

  CellTopoPtr quad_4 = Camellia::CellTopology::quad();

  BasisPtr testBasis
  =
  BasisFactory::basisFactory()->getBasis(C1_FAKE_POLY_ORDER, quad_4, Camellia::FUNCTION_SPACE_HGRAD);
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

  BFPtr bilinearForm = Teuchos::rcp(new TestBilinearFormFlux());

  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0

  BilinearFormUtility<double>::computeStiffnessMatrix(stiffnessActual, bilinearForm,
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
  Teuchos::RCP<DofOrdering> traceOrdering = Teuchos::rcp(new DofOrdering(CellTopology::quad()));
  Teuchos::RCP<DofOrdering> testOrdering = Teuchos::rcp(new DofOrdering(CellTopology::quad()));

  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  BasisPtr traceBasis
  = BasisFactory::basisFactory()->getBasis(C1_FAKE_POLY_ORDER,
                           line_2.getKey(), Camellia::FUNCTION_SPACE_HGRAD);

  int numSides = 4;

  for (int i=0; i<numSides; i++) {
    traceOrdering->addEntry(0,traceBasis,0,i);
  }

  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  BasisPtr testBasis = BasisFactory::basisFactory()->getBasis(C1_FAKE_POLY_ORDER,
                                                              quad_4, Camellia::FUNCTION_SPACE_HDIV);

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

  // horizontal half-slice doesn't change the x derivatives at all
  // cell measure is again half the ref cell, so 1/2 the ref cell's stiffness.

  BFPtr bilinearForm = Teuchos::rcp(new TestBilinearFormTrace());

  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0

  BilinearFormUtility<double>::computeStiffnessMatrix(stiffnessActual, bilinearForm,
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

  BFPtr bilinearForm = TestBilinearFormDx::bf();

  IPPtr ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );

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

  FieldContainer<double> innerProductActual(numTests,4,4);

  DofOrdering lowestOrderHGRADOrdering(CellTopology::quad());

  BasisPtr basis = BasisFactory::basisFactory()->getBasis(C1_FAKE_POLY_ORDER, quad_4.getKey(), Camellia::FUNCTION_SPACE_HGRAD);

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
  vector<GlobalIndexType> quadCellIDs;
  quadCellIDs.push_back(0);

  bool success = true;
  Teuchos::RCP<DofOrdering> trialOrdering;

  BFPtr bilinearForm = TestBilinearFormAnalyticBoundaryIntegral::bf();

  CellTopoPtr quad_4 = Camellia::CellTopology::quad();

  Teuchos::RCP<DofOrdering> cubicHGradOrdering = TestBilinearFormAnalyticBoundaryIntegral::testOrdering(order);

  // construct conforming or non-conforming trial basis:
  trialOrdering = TestBilinearFormAnalyticBoundaryIntegral::trialOrdering(order, 4, conforming);

  FieldContainer<double> stiffnessExpected(numTests,cubicHGradOrdering->totalDofs(),
                                           trialOrdering->totalDofs() );
  FieldContainer<double> stiffnessActual(numTests,cubicHGradOrdering->totalDofs(),
                                         trialOrdering->totalDofs() );

  // from Mathematica:
  TestBilinearFormAnalyticBoundaryIntegral::expectedPreStiffnessForCubicsOnQuad(stiffnessExpected,conforming);

  FieldContainer<double> cellSideParities(numTests,numSides);
  cellSideParities.initialize(1.0); // for 1-element meshes, all side parites are 1.0

  ElementTypePtr elemType = makeElemType(trialOrdering, cubicHGradOrdering, quad_4);
  BasisCachePtr basisCache = makeBasisCache(elemType,quadPoints,quadCellIDs);

  bilinearForm->stiffnessMatrix(stiffnessActual, elemType, cellSideParities, basisCache);

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

  IPPtr ip = Teuchos::rcp( new MathInnerProduct(bilinearForm) );
  shards::CellTopology shardsTopo = quad_4->getShardsTopology();
  ip->computeInnerProductMatrix(ipMatrixActual,cubicHGradOrdering, shardsTopo, quadPoints);

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

  BilinearFormUtility<double>::computeStiffnessMatrix(finalStiffnessActual1,ipMatrixExpected,ipWeightsExpected);

  string myNameFinalByMultiplying = "testAnalyticBoundaryIntegral.finalStiffnessByMultiplying";
  successLocal = fcsAgree(myNameFinalByMultiplying, finalStiffnessExpected, finalStiffnessActual1, tol);

  if (! successLocal) {
    success = false;
    cout << myNameFinalByMultiplying << ": comparison of finalStiffnessExpected and finalStiffnessByMultiplying failed." << endl;
  }

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

  BilinearFormUtility<double>::computeStiffnessMatrix(stiffness,ipMatrix,optWeights);
  return fcsAgree(myName,expectedStiffness,stiffness,tol);
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
  BilinearFormUtility<double>::weightCellBasisValues(basisValues, weights, offset);
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
    Epetra_SerialDenseMatrix optWeightsT(::Copy,
                                         &optWeights(cellIndex,0,0),
                                         optWeights.dimension(2), // stride
                                         optWeights.dimension(2),optWeights.dimension(1));

    Epetra_SerialDenseMatrix ipMatrixT(::Copy,
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

  BFPtr bilinearForm = TestBilinearFormAnalyticBoundaryIntegral::bf();

  Teuchos::RCP<DofOrdering> conformingOrdering, nonConformingOrdering;

  DofOrderingFactory dofOrderingFactory(bilinearForm);

  int polyOrder = 3;
  vector<int> polyOrderVector(1,polyOrder);

  CellTopoPtr quad_4 = Camellia::CellTopology::quad();

  conformingOrdering = dofOrderingFactory.trialOrdering(polyOrderVector, quad_4, true);
  nonConformingOrdering = dofOrderingFactory.trialOrdering(polyOrderVector, quad_4, false);

  Teuchos::RCP<DofOrdering> testOrdering = Teuchos::rcp( new DofOrdering(quad_4) );
  int testID = 0;
  BasisPtr testBasis = Camellia::intrepidQuadHGRAD(polyOrder);
  testOrdering->addEntry(testID, testBasis, testBasis->rangeRank());

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

vector<double> makeVertex(double v0, double v1, double v2) {
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  v.push_back(v2);
  return v;
}

void DPGTests::runExceptionThrowingTest() {
//  int tensorialDegree = 1;
//  CellTopoPtr quad_x_time = CellTopology::cellTopology(CellTopology::quad(), tensorialDegree);
//  CellTopoPtr tri_x_time = CellTopology::cellTopology(CellTopology::triangle(), tensorialDegree);
//
//  // let's draw a little house
//  vector<double> v00 = makeVertex(-1,0,0);
//  vector<double> v10 = makeVertex(1,0,0);
//  vector<double> v20 = makeVertex(1,2,0);
//  vector<double> v30 = makeVertex(-1,2,0);
//  vector<double> v40 = makeVertex(0.0,3,0);
//  vector<double> v01 = makeVertex(-1,0,1);
//  vector<double> v11 = makeVertex(1,0,1);
//  vector<double> v21 = makeVertex(1,2,1);
//  vector<double> v31 = makeVertex(-1,2,1);
//  vector<double> v41 = makeVertex(0.0,3,1);
//
//  vector< vector<double> > spaceTimeVertices;
//  spaceTimeVertices.push_back(v00);
//  spaceTimeVertices.push_back(v10);
//  spaceTimeVertices.push_back(v20);
//  spaceTimeVertices.push_back(v30);
//  spaceTimeVertices.push_back(v40);
//  spaceTimeVertices.push_back(v01);
//  spaceTimeVertices.push_back(v11);
//  spaceTimeVertices.push_back(v21);
//  spaceTimeVertices.push_back(v31);
//  spaceTimeVertices.push_back(v41);
//
//  vector<unsigned> spaceTimeQuadVertexList;
//  spaceTimeQuadVertexList.push_back(0);
//  spaceTimeQuadVertexList.push_back(1);
//  spaceTimeQuadVertexList.push_back(2);
//  spaceTimeQuadVertexList.push_back(3);
//  spaceTimeQuadVertexList.push_back(5);
//  spaceTimeQuadVertexList.push_back(6);
//  spaceTimeQuadVertexList.push_back(7);
//  spaceTimeQuadVertexList.push_back(8);
//  vector<unsigned> spaceTimeTriVertexList;
//  spaceTimeTriVertexList.push_back(3);
//  spaceTimeTriVertexList.push_back(2);
//  spaceTimeTriVertexList.push_back(4);
//  spaceTimeTriVertexList.push_back(8);
//  spaceTimeTriVertexList.push_back(7);
//  spaceTimeTriVertexList.push_back(9);
//
//  vector< vector<unsigned> > spaceTimeElementVertices;
//  spaceTimeElementVertices.push_back(spaceTimeQuadVertexList);
//  spaceTimeElementVertices.push_back(spaceTimeTriVertexList);
//
//  vector< CellTopoPtr > spaceTimeCellTopos;
//  spaceTimeCellTopos.push_back(quad_x_time);
//  spaceTimeCellTopos.push_back(tri_x_time);
//
//  MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
//  MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );
//
//  ////////////////////   DECLARE VARIABLES   ///////////////////////
//  // define test variables
//  VarFactory varFactory;
//  VarPtr tau = varFactory.testVar("tau", HDIV);
//  VarPtr v = varFactory.testVar("v", HGRAD);
//
//  // define trial variables
//  VarPtr uhat = varFactory.traceVar("uhat");
//  VarPtr fhat = varFactory.fluxVar("fhat");
//  VarPtr u = varFactory.fieldVar("u");
//  VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);
//
//  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
//  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
//  // tau terms:
//  bf->addTerm(sigma, tau);
//  bf->addTerm(u, tau->div());
//  bf->addTerm(-uhat, tau->dot_normal());
//
//  // v terms:
//  bf->addTerm( sigma, v->grad() );
//  bf->addTerm( fhat, v);
//
//  ////////////////////   BUILD MESH   ///////////////////////
//  int H1Order = 3, pToAdd = 2;
//  Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );
//
//  Teuchos::RCP<Solution> spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );
//
//  FunctionPtr n = Function::normalSpaceTime();
//  FunctionPtr parity = Function::sideParity();
//  FunctionPtr f_flux = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity + Function::zn(1) * n->z() * parity;
//
//  map<int, Teuchos::RCP<Function> > functionMap;
//  functionMap[uhat->ID()] = Function::xn(1);
//  functionMap[fhat->ID()] = f_flux;
//  functionMap[u->ID()] = Function::xn(1);
//  functionMap[sigma->ID()] = Function::xn(1);
//  spaceTimeSolution->projectOntoMesh(functionMap);
//
//  double tol = 1e-14;
//  for (map<int, Teuchos::RCP<Function> >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
//    int trialID = entryIt->first;
//    VarPtr trialVar = varFactory.trial(trialID);
//    FunctionPtr f_expected = entryIt->second;
//    FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);
//
//    if (trialVar->varType() == FLUX) {
//      // then Function::solution() will have included a parity weight, basically on the idea that we're also multiplying by normals
//      // in our usage of the solution data.  (It may be that this is not the best way to do this.)
//
//      // For this test, though, we want to reverse that:
//      f_actual = parity * f_actual;
//    }
//
//    double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh);
////    TEST_COMPARE(err_L2, <, tol);
//  }
}
