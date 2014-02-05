#include "HConvergenceStudyTests.h"

#include "HConvergenceStudy.h"
#include "ExactSolution.h"
#include "NavierStokesFormulation.h"

void HConvergenceStudyTests::setup() {
  
}

void HConvergenceStudyTests::teardown() {
  
}

HConvergenceStudyTests::HConvergenceStudyTests() {
  
}

void HConvergenceStudyTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if ( testBestApproximationErrorComputation() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool HConvergenceStudyTests::testBestApproximationErrorComputation() {
  bool success = true;

  bool enrichVelocity = false; // true would be for the "compliant" norm, which isn't working well yet
  
  int minLogElements = 0, maxLogElements = minLogElements;
  int numCells1D = pow(2.0,minLogElements);
  int H1Order = 1;
  int pToAdd = 2;
  
  double tol = 1e-16;
  double Re = 40.0;
  
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  VarPtr u1_vgp = varFactory.fieldVar(VGP_U1_S);
  VarPtr u2_vgp = varFactory.fieldVar(VGP_U2_S);
  VarPtr sigma11_vgp = varFactory.fieldVar(VGP_SIGMA11_S);
  VarPtr sigma12_vgp = varFactory.fieldVar(VGP_SIGMA12_S);
  VarPtr sigma21_vgp = varFactory.fieldVar(VGP_SIGMA21_S);
  VarPtr sigma22_vgp = varFactory.fieldVar(VGP_SIGMA22_S);
  VarPtr p_vgp = varFactory.fieldVar(VGP_P_S);
  
  VGPStokesFormulation stokesForm(1/Re);
  
  int numCellsFineMesh = 20; // for computing a zero-mean pressure
  int H1OrderFineMesh = 5;
  
  // define Kovasznay domain:
  FieldContainer<double> quadPointsKovasznay(4,2);
  
  // Domain from Evans Hughes for Navier-Stokes:
  quadPointsKovasznay(0,0) =  0.0; // x1
  quadPointsKovasznay(0,1) = -0.5; // y1
  quadPointsKovasznay(1,0) =  1.0;
  quadPointsKovasznay(1,1) = -0.5;
  quadPointsKovasznay(2,0) =  1.0;
  quadPointsKovasznay(2,1) =  0.5;
  quadPointsKovasznay(3,0) =  0.0;
  quadPointsKovasznay(3,1) =  0.5;
  
  FunctionPtr zero = Function::zero();
  bool dontEnhanceFluxes = false;
  VGPNavierStokesProblem zeroProblem = VGPNavierStokesProblem(Re, quadPointsKovasznay,
                                                              numCellsFineMesh, numCellsFineMesh,
                                                              H1OrderFineMesh, pToAdd,
                                                              zero, zero, zero, enrichVelocity, dontEnhanceFluxes);
  
  FunctionPtr u1_exact, u2_exact, p_exact;
  NavierStokesFormulation::setKovasznay(Re, zeroProblem.mesh(), u1_exact, u2_exact, p_exact);
  
  
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re,quadPointsKovasznay,
                                                          numCells1D,numCells1D,
                                                          H1Order, pToAdd,
                                                          u1_exact, u2_exact, p_exact, enrichVelocity, dontEnhanceFluxes);

  HConvergenceStudy study(problem.exactSolution(),
                          problem.mesh()->bilinearForm(),
                          problem.exactSolution()->rhs(),
                          problem.backgroundFlow()->bc(),
                          problem.bf()->graphNorm(),
                          minLogElements, maxLogElements,
                          H1Order, pToAdd, false, false, false);
  study.setReportRelativeErrors(false); // we want absolute errors

  Teuchos::RCP<Mesh> mesh = problem.mesh();
  
  int cubatureDegreeEnrichment = 10;
  
  int L2Order = H1Order - 1;
  int meshCubatureDegree = L2Order + H1Order + pToAdd;

  study.setCubatureDegreeForExact(cubatureDegreeEnrichment + meshCubatureDegree);
  
  FunctionPtr f = u1_exact;
  int trialID = u1_vgp->ID();
  {
    double fIntegral = f->integrate(mesh,cubatureDegreeEnrichment);
//    cout << "testBestApproximationErrorComputation: integral of f on whole mesh = " << fIntegral << endl;
    
    double l2ErrorOfAverage = (Function::constant(fIntegral) - f)->l2norm(mesh,cubatureDegreeEnrichment);
//    cout << "testBestApproximationErrorComputation: l2 error of fIntegral: " << l2ErrorOfAverage << endl;
    
    ElementTypePtr elemType = mesh->elementTypes()[0];
    vector<GlobalIndexType> cellIDs = mesh->cellIDsOfTypeGlobal(elemType);
    
    bool testVsTest = false;
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment) );
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodesGlobal(elemType), cellIDs, false); // false: no side cache

    FieldContainer<double> projectionValues(cellIDs.size());
    f->integrate(projectionValues, basisCache);
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    
    for (int i=0; i<projectionValues.size(); i++) {
      projectionValues(i) /= cellMeasures(i);
    }
    
    // since we're not worried about the actual solution values at all, just use a single zero solution:
    vector< SolutionPtr > solutions;
    solutions.push_back( problem.backgroundFlow() );
    
    study.setSolutions(solutions); // this will call computeError()
    
    
    double approximationError = study.bestApproximationErrors()[trialID][0]; // 0: solution/mesh index
    
    // for a single-cell mesh, approximation error should be the same as the L^2 error of the average
    double diff = abs(approximationError - l2ErrorOfAverage);
  
    if (diff > tol) {
      cout << "testBestApproximationErrorComputation: diff " << diff << " exceeds tol " << tol << endl;
      success = false;
    } else {
//      cout << "testBestApproximationErrorComputation: diff " << diff << " is below tol " << tol << endl;
    }
  }
  return success;
}

string HConvergenceStudyTests::testSuiteName() {
  return "HConvergenceStudyTests";
}