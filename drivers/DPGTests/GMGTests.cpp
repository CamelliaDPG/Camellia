//
//  GMGTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/29/14.
//
//

#include "GMGTests.h"

#include "Var.h"
#include "Solution.h"

#include "BF.h"

#include "MeshFactory.h"

#include "PenaltyConstraints.h"

#include "GMGSolver.h"

#include "SerialDenseWrapper.h"

const static string S_GMGTests_U1 = "u_1";
const static string S_GMGTests_U2 = "u_2";
const static string S_GMGTests_PHI = "\\phi";
const static string S_GMGTests_PHI_HAT = "\\widehat{\\phi}";

FunctionPtr GMGTests::getPhiExact(int spaceDim) {
  if (spaceDim==1) {
    return Function::xn(1);
  } else if (spaceDim==2) {
    return Function::xn(1) * Function::yn(1);
  } else if (spaceDim==3) {
    return Function::xn(1) * Function::yn(1) * Function::zn(1);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled spaceDim");
}

SolutionPtr GMGTests::poissonExactSolution(int numCells_x, int H1Order, FunctionPtr phi_exact, bool useH1Traces) {
  vector<int> numCells;
  numCells.push_back(numCells_x);
  return poissonExactSolution(numCells, H1Order, phi_exact, useH1Traces);
}

SolutionPtr GMGTests::poissonExactSolution(vector<int> numCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces) {
  int spaceDim = numCells.size();
  
  VarFactory varFactory;
  Space tauSpace = (spaceDim > 1) ? HDIV : HGRAD;
  VarPtr tau = varFactory.testVar("\\tau_1", tauSpace);
  VarPtr q = varFactory.testVar("q", HGRAD);
  
  Space phi_hat_space = useH1Traces ? HGRAD : L2;
  VarPtr phi_hat;
  if (spaceDim > 1) {
    phi_hat = varFactory.traceVar(S_GMGTests_PHI_HAT, phi_hat_space);
  } else {
    // for spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
    phi_hat = varFactory.fluxVar(S_GMGTests_PHI_HAT, phi_hat_space);
  }
  VarPtr psi_n = varFactory.fluxVar("\\widehat{\\psi}_{n}");
  //  VarPtr phi_hat = varFactory.traceVar(S_GDAMinimumRuleTests_PHI_HAT, L2);
  //  cout << "WARNING: temporarily using L^2 discretization for \\widehat{\\phi}.\n";
  
  VarPtr phi = varFactory.fieldVar(S_GMGTests_PHI, L2);
  Space psiSpace = (spaceDim > 1) ? VECTOR_L2 : L2;
  VarPtr psi = varFactory.fieldVar("\\psi_1", psiSpace);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  if (spaceDim==1) {
    // for spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
    bf->addTerm(phi, tau->dx());
    bf->addTerm(psi, tau);
    bf->addTerm(-phi_hat, tau);
    
    bf->addTerm(-psi, q->dx());
    bf->addTerm(psi_n, q);
  } else {
    bf->addTerm(phi, tau->div());
    bf->addTerm(psi, tau);
    bf->addTerm(-phi_hat, tau->dot_normal());
    
    bf->addTerm(-psi, q->grad());
    bf->addTerm(psi_n, q);
  }
  
  int testSpaceEnrichment = spaceDim; //
//  double width = 1.0, height = 1.0, depth = 1.0;
  
  vector<double> dimensions;
  for (int d=0; d<spaceDim; d++) {
    dimensions.push_back(1.0);
  }
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, numCells, H1Order, testSpaceEnrichment);
  
  // rhs = f * v, where f = \Delta phi
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
  
  return solution;
}

//// copied verbatim from GDAMinimumRuleTests:
//SolutionPtr GMGTests::poissonExactSolution1D(int horizontalCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces) {
//  bool usePenaltyBCs = false;
//  
//  VarFactory varFactory;
//  VarPtr tau = varFactory.testVar("\\tau_1", HGRAD);
//  VarPtr q = varFactory.testVar("q", HGRAD);
//  
//  Space phi_hat_space = useH1Traces ? HGRAD : L2; // should not matter
//  VarPtr phi_hat = varFactory.fluxVar(S_GMGTests_PHI_HAT, phi_hat_space);
//  VarPtr psi_hat = varFactory.fluxVar("\\widehat{\\psi}");
//  
//  VarPtr phi = varFactory.fieldVar(S_GMGTests_PHI, L2);
//  VarPtr psi1 = varFactory.fieldVar("\\psi_1", L2);
//  
//  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
//  
//  bf->addTerm(phi, tau->dx());
//  bf->addTerm(psi1, tau);
//  bf->addTerm(-phi_hat, tau);
//  
//  bf->addTerm(-psi1, q->dx());
//  bf->addTerm(psi_hat, q);
//  
//  int testSpaceEnrichment = 1; //
//  double width = 3.14159;
//  
//  vector<double> dimensions;
//  dimensions.push_back(width);
//  
//  vector<int> elementCounts;
//  elementCounts.push_back(horizontalCells);
//  
//  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, testSpaceEnrichment);
//  
//  //  cout << "entities for 1D mesh:\n";
//  //  mesh->getTopology()->printAllEntities();
//  
//  // rhs = f * v, where f = \Delta phi
//  RHSPtr rhs = RHS::rhs();
//  FunctionPtr f = phi_exact->dx()->dx();
//  rhs->addTerm(f * q);
//  
//  IPPtr graphNorm = bf->graphNorm();
//  
//  BCPtr bc = BC::bc();
//  SpatialFilterPtr boundary = SpatialFilter::allSpace();
//  SolutionPtr solution;
//  if (!usePenaltyBCs) {
//    FunctionPtr n = Function::normal_1D(); // normal function (-1 or 1)
//    bc->addDirichlet(phi_hat, boundary, phi_exact * n);
//    solution = Solution::solution(mesh, bc, rhs, graphNorm);
//  } else {
//    solution = Solution::solution(mesh, bc, rhs, graphNorm);
//    SpatialFilterPtr entireBoundary = SpatialFilter::allSpace();
//    
//    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
//    pc->addConstraint(phi_hat==phi_exact,entireBoundary);
//    
//    solution->setFilter(pc);
//  }
//  
//  return solution;
//}

void GMGTests::setup() {
  
}

void GMGTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testGMGSolverTwoGrid()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testGMGSolverIdentity()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testGMGOperatorIdentity()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool GMGTests::testGMGOperatorIdentity() {
  bool success = true;
  
  /***   1D TESTS    ***/
  vector<int> cellCounts;
  cellCounts.push_back(1);
  cellCounts.push_back(2);
  cellCounts.push_back(4);
  
  for (int spaceDim=1; spaceDim<=3; spaceDim++) {
    for (int i=0; i<cellCounts.size(); i++) {
      if ((spaceDim==3) && (i==cellCounts.size()-1)) continue; // skip the 4x4x4 case, in interest of time.
      vector<int> cellCount;
      for (int d=0; d<spaceDim; d++) {
        cellCount.push_back(cellCounts[i]);
      }
      
      int H1Order = 2;
      bool useH1Traces = false;
      FunctionPtr phiExact = getPhiExact(spaceDim);
      SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);

      BCPtr poissonBC = exactPoissonSolution->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr mesh = exactPoissonSolution->mesh();
      BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      // as a first test, do "multi" grid between mesh and itself.  Solution should match phiExact.
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
      GMGOperator gmgOperator(zeroBCs, mesh, graphNorm, mesh, exactPoissonSolution->getPartitionMap(), coarseSolver);

      GlobalIndexType cellID = 0; // fine and coarse both
      GlobalIndexType expectedCoarseCellID = cellID;
      GlobalIndexType actualCoarseCellID = gmgOperator.getCoarseCellID(cellID);
      if (actualCoarseCellID != expectedCoarseCellID) {
        cout << "actualCoarseCellID does not match expected.\n";
        success = false;
      }
      LocalDofMapperPtr dofMapper = gmgOperator.getLocalCoefficientMap(cellID);
      
      DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
      int numTrialDofs = trialOrder->totalDofs();
      FieldContainer<double> localData(numTrialDofs,numTrialDofs);
      // just some arbitrary data:
      for (int i=0; i<numTrialDofs; i++) {
        for (int j=0; j<numTrialDofs; j++) {
          localData(i,j) = i + 5 * j;
          if (i==j) localData(i,j) = 100 * localData(i,j); // make diagonally dominant
        }
      }
      FieldContainer<double> mappedData = dofMapper->mapLocalData(localData, false); // true would mean "fittable" GlobalDofs only
      FieldContainer<double> expectedMappedData = localData;
      
      double tol = 1e-12;
      double maxDiff = 0;
      if (mappedData.size() != expectedMappedData.size()) {
        cout << "FAILURE: For cellCount = " << cellCounts[i] << " in " << spaceDim << "D, ";
        cout << "mapped data differs in dimension from expected: " << mappedData.size() << " != " << expectedMappedData.size() << endl;
        success = false;
        SerialDenseWrapper::writeMatrixToMatlabFile("/tmp/mappedData.dat", mappedData);
        SerialDenseWrapper::writeMatrixToMatlabFile("/tmp/expectedMappedData.dat", expectedMappedData);
        
      } else if (! fcsAgree(mappedData, expectedMappedData, tol, maxDiff)) {
        cout << "FAILURE: For cellCount = " << cellCounts[i] << " in " << spaceDim << "D, ";
        cout << "mapped data differs from expected by as much as " << maxDiff << "; tol = " << tol << endl;
        success = false;
        SerialDenseWrapper::writeMatrixToMatlabFile("/tmp/mappedData.dat", mappedData);
        SerialDenseWrapper::writeMatrixToMatlabFile("/tmp/expectedMappedData.dat", expectedMappedData);
      }
    }
    
  }
  /***   2D TESTS    ***/
  
  /***   3D TESTS    ***/
  
  return success;
}

bool GMGTests::testGMGSolverIdentity() {
  bool success = true;
  
  vector<int> cellCounts;
  cellCounts.push_back(1);
  cellCounts.push_back(2);
  cellCounts.push_back(4);

  for (int spaceDim=1; spaceDim<=3; spaceDim++) {
    for (int i=0; i<cellCounts.size(); i++) {
      if ((spaceDim==3) && (i==cellCounts.size()-1)) continue; // skip the 4x4x4 case, in interest of time.
      vector<int> cellCount;
      for (int d=0; d<spaceDim; d++) {
        cellCount.push_back(cellCounts[i]);
      }

      int H1Order = 2;
      bool useH1Traces = false;
      FunctionPtr phiExact = getPhiExact(spaceDim);
      SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      
      BCPtr poissonBC = exactPoissonSolution->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr mesh = exactPoissonSolution->mesh();
      BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();

      // as a first test, do "multi" grid between mesh and itself.  Solution should match phiExact.
      double iter_tol = 1e-14;
      bool applySmoothing = true;
      int maxIters = applySmoothing ? 100 : 1; // if smoothing not applied, then GMG should recover exactly the direct solution, in 1 iteration
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                                                      exactPoissonSolution->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver) );

      gmgSolver->setApplySmoothingOperator(applySmoothing);
      
      Teuchos::RCP<Solver> fineSolver = gmgSolver;
      
//      exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A.dat");
//      exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
      
      exactPoissonSolution->solve(coarseSolver);
      
      actualPoissonSolution->solve(fineSolver);
      
      VarFactory vf = bf->varFactory();
      VarPtr phi = vf.fieldVar(S_GMGTests_PHI);
      
      FunctionPtr exactPhiSoln = Function::solution(phi, exactPoissonSolution);
      FunctionPtr actualPhiSoln = Function::solution(phi, actualPoissonSolution);
      
      double l2_diff = (exactPhiSoln-actualPhiSoln)->l2norm(mesh);
      
      double tol = iter_tol * 10;
      if (l2_diff > tol) {
        success = false;
        cout << "FAILURE: For cellCount = " << cellCounts[i] << " in " << spaceDim << "D, ";

        cout << "GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";
      }
    }
  }
  /***   2D TESTS    ***/
  
  /***   3D TESTS    ***/
  
  return success;
}

bool GMGTests::testGMGSolverTwoGrid() {
  bool success = true;
  vector<int> cellCounts;
  cellCounts.push_back(1);
  
  for (int spaceDim=1; spaceDim<=3; spaceDim++) {
    for (int i=0; i<cellCounts.size(); i++) {
//      if ((spaceDim==3) && (i==cellCounts.size()-1)) continue; // skip the 4x4x4 case, in interest of time.
      vector<int> cellCount;
      for (int d=0; d<spaceDim; d++) {
        cellCount.push_back(cellCounts[i]);
      }
      
      int H1Order = 2;
      bool useH1Traces = false;
      FunctionPtr phiExact = getPhiExact(spaceDim);
      
      MeshPtr coarseMesh = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces)->mesh();
      
      SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      
      MeshPtr exactMesh = exactPoissonSolution->mesh();
      MeshPtr fineMesh = actualPoissonSolution->mesh();
      
      // refine uniformly once in both exact and actual:
      CellTopoPtr cellTopo = coarseMesh->getTopology()->getCell(0)->topology();
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo->getKey());
      exactMesh->hRefine(exactMesh->getActiveCellIDs(), refPattern);
      fineMesh->hRefine(fineMesh->getActiveCellIDs(), refPattern);
      
      BCPtr poissonBC = exactPoissonSolution->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr mesh = exactPoissonSolution->mesh();
      BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      double iter_tol = 1e-6;
      bool applySmoothing = true;
      int maxIters = 200;
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, fineMesh,
                                                                      exactPoissonSolution->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver) );
      
      gmgSolver->setApplySmoothingOperator(applySmoothing);
      
      Teuchos::RCP<Solver> fineSolver = gmgSolver;
      
      //      exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A.dat");
      //      exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
      
      exactPoissonSolution->solve(coarseSolver);
      
      actualPoissonSolution->solve(fineSolver);
      
      VarFactory vf = bf->varFactory();
      VarPtr phi = vf.fieldVar(S_GMGTests_PHI);
      
      FunctionPtr exactPhiSoln = Function::solution(phi, exactPoissonSolution);
      FunctionPtr actualPhiSoln = Function::solution(phi, actualPoissonSolution);
      
      double l2_diff = (exactPhiSoln-actualPhiSoln)->l2norm(mesh);
      
      double tol = iter_tol * 10;
      if (l2_diff > tol) {
        success = false;
        cout << "FAILURE: For cellCount = " << cellCounts[i] << " in " << spaceDim << "D, ";
        
        cout << "two grid GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";
      }
    }
  }
  return success;
}

string GMGTests::testSuiteName() {
  return "GMGTests";
}
