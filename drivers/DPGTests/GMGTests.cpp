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

#include "GDAMinimumRule.h"
#include "SerialDenseWrapper.h"

#include "BasisSumFunction.h"

#include "CamelliaDebugUtility.h"

#include "GnuPlotUtil.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

const static string S_GMGTests_U1 = "u_1";
const static string S_GMGTests_U2 = "u_2";
const static string S_GMGTests_PHI = "\\phi";
const static string S_GMGTests_PHI_HAT = "\\widehat{\\phi}";

FunctionPtr GMGTests::getPhiExact(int spaceDim) {
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  if (spaceDim==1) {
    return x * x + 1;
  } else if (spaceDim==2) {
    return x * y + x * x;
  } else if (spaceDim==3) {
    return x * y * z + z * z * x;
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
  
//  cout << "dimensions[0] = " << dimensions[0] << "; dimensions[1] = " << dimensions[1] << endl;
//  cout << "numCells[0] = " << numCells[0] << "; numCells[1] = " << numCells[1] << endl;
  
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

SolutionPtr GMGTests::poissonExactSolutionRefined(int H1Order, FunctionPtr phi_exact, bool useH1Traces, int refinementSetOrdinal) {
  vector<int> numCells;
  numCells.push_back(2);
  numCells.push_back(2);
  SolutionPtr soln = poissonExactSolution(numCells, H1Order, phi_exact, useH1Traces);
  
  MeshPtr mesh = soln->mesh();
  
  set<GlobalIndexType> cellIDs;
  switch (refinementSetOrdinal) {
    case 0: // no refinements
      break;
    case 1: // one refinement
      cellIDs.insert(3);
      mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),true);
      break;
    case 2:
      cellIDs.insert(3);
      mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);
      cellIDs.clear();
      
      cellIDs.insert(6);
      cellIDs.insert(7);
      mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);
      cellIDs.clear();
      
      cellIDs.insert(1);
      mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),true);
      cellIDs.clear();
      break;
      
  case 3:
    cellIDs.insert(1);
    cellIDs.insert(3);
    mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);
    
    cellIDs.clear();
    cellIDs.insert(6);
    cellIDs.insert(7);
    cellIDs.insert(8);
    cellIDs.insert(10);
    cellIDs.insert(11);
    mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);
    
    cellIDs.clear();
    cellIDs.insert(2);
    mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);
    
    cellIDs.clear();
    cellIDs.insert(4);
    cellIDs.insert(9);
    cellIDs.insert(12);
    cellIDs.insert(14);
    cellIDs.insert(19);
    cellIDs.insert(26);
    cellIDs.insert(31);
    mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),false);
    
    cellIDs.clear();
    cellIDs.insert(0);
    cellIDs.insert(5);
    mesh->hRefine(cellIDs,RefinementPattern::regularRefinementPatternQuad(),true);
    break;
  }
  
  return soln;
}

void GMGTests::setup() {
  
}

void GMGTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testGMGOperatorIdentityRHSMap()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testGMGOperatorIdentityLocalCoefficientMap()) {
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
  if (testGMGOperatorP()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  setup();
//  if (testGMGSolverThreeGrid()) {
//    numTestsPassed++;
//  }
//  numTestsRun++;
//  teardown();
  
  setup();
  if (testGMGSolverTwoGrid()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool GMGTests::testGMGOperatorIdentityLocalCoefficientMap() {
  bool success = true;
  
  vector<bool> useStaticCondensationValues;
  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
  /***   1D-3D TESTS    ***/
  vector<int> cellCounts;
  cellCounts.push_back(1);
  cellCounts.push_back(2);
  cellCounts.push_back(4);
  
  for (  vector<bool>::iterator useStaticCondensationIt = useStaticCondensationValues.begin();
       useStaticCondensationIt != useStaticCondensationValues.end(); useStaticCondensationIt++) {
    bool useStaticCondensation = *useStaticCondensationIt;
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
        GMGOperator gmgOperator(zeroBCs, mesh, graphNorm, mesh, exactPoissonSolution->getDofInterpreter(),
                                exactPoissonSolution->getPartitionMap(), coarseSolver, useStaticCondensation);

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
  }
  
  success = TestSuite::allSuccess(success);
  
  return success;
}

bool GMGTests::testGMGOperatorIdentityRHSMap() {
  bool success = true;
  
  vector<bool> useStaticCondensationValues;
  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
  // some 2D-specific tests on refined meshes
  for (  vector<bool>::iterator useStaticCondensationIt = useStaticCondensationValues.begin();
       useStaticCondensationIt != useStaticCondensationValues.end(); useStaticCondensationIt++) {
    bool useStaticCondensation = *useStaticCondensationIt;
    for (int refinementOrdinal=-1; refinementOrdinal<4; refinementOrdinal++) {
      int spaceDim = 2;
      
      int H1Order_coarse = 2, H1Order = 2;
      
      FunctionPtr phiExact = getPhiExact(spaceDim);
      
      bool useH1Traces = false; // false is the more forgiving; a place to start testing
      SolutionPtr solnCoarse, solnFine;
      if (refinementOrdinal == -1) { // simple as it gets: unrefined, single-element, and H^1 order = 1
        H1Order = 1, H1Order_coarse = 1;
        vector<int> numCells(2,1);
        solnCoarse = poissonExactSolution(numCells, H1Order_coarse, phiExact, useH1Traces);
        solnFine = poissonExactSolution(numCells, H1Order, phiExact, useH1Traces);
      } else {
        solnCoarse = poissonExactSolutionRefined(H1Order_coarse, phiExact, useH1Traces, refinementOrdinal);
        solnFine = poissonExactSolutionRefined(H1Order, phiExact, useH1Traces, refinementOrdinal);
      }
      solnCoarse->setUseCondensedSolve(useStaticCondensation);
      solnFine->setUseCondensedSolve(useStaticCondensation);
      
      BCPtr poissonBC = solnFine->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr fineMesh = solnFine->mesh();
      BF* bf = dynamic_cast< BF* >( fineMesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      // as a first test, do "multi" grid between mesh and itself.  Solution should match phiExact.
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
      
      GMGOperator gmgOperator(zeroBCs, fineMesh, graphNorm, fineMesh, solnFine->getDofInterpreter(), solnFine->getPartitionMap(), coarseSolver, useStaticCondensation);
      
      if (useStaticCondensation) {
        // need to populate local stiffness matrices before dealing with the RHS.
        solnFine->initializeLHSVector();
        solnFine->initializeStiffnessAndLoad();
        solnFine->populateStiffnessAndLoad();
      }
      
      //      GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/fineMesh", fineMesh, true); // true: label cells
      
      //    GDAMinimumRule* fineGDA = dynamic_cast< GDAMinimumRule*>(fineMesh->globalDofAssignment().get());
      
      //    fineGDA->printConstraintInfo(12);
      //    fineGDA->printConstraintInfo(15);
      //    fineGDA->printConstraintInfo(18);
      
      solnFine->initializeStiffnessAndLoad();
      Teuchos::RCP<Epetra_FEVector> rhsVector = solnFine->getRHSVector();
      
      // fill rhsVector with some arbitrary data
      rhsVector->PutScalar(0);
      
      int minLID = rhsVector->Map().MinLID();
      int numLIDs = rhsVector->Map().NumMyElements();
      for (int lid=minLID; lid < minLID + numLIDs; lid++ ) {
        GlobalIndexTypeToCast gid = rhsVector->Map().GID(lid);
        (*rhsVector)[0][lid] = (double) gid; // arbitrary data
      }
      
      // GMGOperator on rhsVector should be identity.
      Epetra_FEVector mappedRHSVector(rhsVector->Map());
      gmgOperator.setCoarseRHSVector(*rhsVector, mappedRHSVector);
      
      double tol = 1e-14;
      for (int lid=minLID; lid < minLID + numLIDs; lid++ ) {
        double expected = (*rhsVector)[0][lid];
        double actual = mappedRHSVector[0][lid];
        
        double diff = abs(expected-actual);
        if (diff > tol) {
          GlobalIndexTypeToCast gid = rhsVector->Map().GID(lid);
          
          cout << "Failure: in rhs mapping for refinement sequence " << refinementOrdinal;
          cout << " and gid " << gid << ", expected = " << expected << "; actual = " << actual << endl;
          success = false;
        }
      }
    }
  }
  success = TestSuite::allSuccess(success);
  
  return success;
}

bool GMGTests::testGMGOperatorP() {
  bool success = true;
  
  int spaceDim = 2;
  FunctionPtr phiExact = getPhiExact(spaceDim);
  int H1Order_coarse = 1;
  int H1Order = 5;
  
  int whichRefinement = 1;
  bool useStaticCondensation = false;
  
  bool useH1Traces = true; // false is the more forgiving; a place to start testing
  SolutionPtr solnCoarse = poissonExactSolutionRefined(H1Order_coarse, phiExact, useH1Traces, whichRefinement);
  SolutionPtr solnFine = poissonExactSolutionRefined(H1Order, phiExact, useH1Traces, whichRefinement);
  solnFine->setUseCondensedSolve(useStaticCondensation);
  
  MeshPtr coarseMesh = solnCoarse->mesh();
  MeshPtr fineMesh = solnFine->mesh();
  
  BCPtr poissonBC = solnFine->bc();
  BCPtr zeroBCs = poissonBC->copyImposingZero();
  BF* bf = dynamic_cast< BF* >( fineMesh->bilinearForm().get() );
  IPPtr graphNorm = bf->graphNorm();

  int maxIters = 100;
  double iter_tol = 1e-6;
  
  Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
  
  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, fineMesh,
                                                                  solnFine->getDofInterpreter(),
                                                                  solnFine->getPartitionMap(),
                                                                  maxIters, iter_tol, coarseSolver, useStaticCondensation) );
  
  GMGOperator* gmgOperator = &gmgSolver->gmgOperator();
  
  GDAMinimumRule* coarseGDA = dynamic_cast< GDAMinimumRule*>(coarseMesh->globalDofAssignment().get());

  // idea is this: for each cell in the coarse mesh, there exist global dofs mapped by that cell's local dofs
  //               for each of these, construct a set of basis coefficients (0's and one 1)
  //               map to the local coefficients on coarse mesh.
  //               map to the local coefficients on fine mesh.
  //               using these coefficients, the corresponding functions on the two meshes should be the same.
  
  set<GlobalIndexType> cellIDs = coarseMesh->cellIDsInPartition();
  
  set<GlobalIndexType> coarseFieldIndices = coarseGDA->partitionOwnedGlobalFieldIndices();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(fineMesh, cellID); // since the mesh geometry is the same, fineMesh's cache will work for both fine and coarse cell.
    
    LocalDofMapperPtr fineToCoarseMapper = gmgOperator->getLocalCoefficientMap(cellID); // this is a local-to-local mapping between the meshes
    
    DofOrderingPtr fineOrdering = fineMesh->getElementType(cellID)->trialOrderPtr;
    DofOrderingPtr coarseOrdering = coarseMesh->getElementType(cellID)->trialOrderPtr;
    
    FieldContainer<double> coarseCoefficients(coarseOrdering->totalDofs());
    
    set<int> varIDs = coarseOrdering->getVarIDs();
    for (set<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++) {
      int varID = *varIDIt;
      int sideCount = coarseOrdering->getNumSidesForVarID(varID);
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        BasisPtr coarseBasis = coarseOrdering->getBasis(varID,sideOrdinal);
        BasisPtr fineBasis = fineOrdering->getBasis(varID,sideOrdinal);
        FieldContainer<double> coarseBasisCoefficients(coarseBasis->getCardinality());
        FieldContainer<double> fineBasisCoefficients(fineBasis->getCardinality());
        
        for (int coarseBasisOrdinal=0; coarseBasisOrdinal < coarseBasis->getCardinality(); coarseBasisOrdinal++) {
          int coarseDofIndex = coarseOrdering->getDofIndex(varID, coarseBasisOrdinal, sideOrdinal);
          coarseBasisCoefficients.initialize(0);
          coarseCoefficients.initialize(0);
          fineBasisCoefficients.initialize(0);
          coarseBasisCoefficients[coarseBasisOrdinal] = 1.0;
          coarseCoefficients[coarseDofIndex] = 1.0;

          FieldContainer<double> fineLocalCoefficients = fineToCoarseMapper->mapGlobalCoefficients(coarseCoefficients);
          
          for (int basisOrdinal=0; basisOrdinal < fineBasis->getCardinality(); basisOrdinal++) {
            int fineDofIndex = fineOrdering->getDofIndex(varID, basisOrdinal, sideOrdinal);
            fineBasisCoefficients[basisOrdinal] = fineLocalCoefficients[fineDofIndex];
          }
          
          FunctionPtr fineBasisSumFunction = NewBasisSumFunction::basisSumFunction(fineBasis, fineBasisCoefficients);
          FunctionPtr coarseBasisSumFunction = NewBasisSumFunction::basisSumFunction(coarseBasis, coarseBasisCoefficients);
          FunctionPtr diffFxn = fineBasisSumFunction - coarseBasisSumFunction;
          
          BasisCachePtr basisCacheForIntegration = (coarseOrdering->getNumSidesForVarID(varID) == 1) ? basisCache : basisCache->getSideBasisCache(sideOrdinal);
          
          double l2diff = sqrt( (diffFxn * diffFxn)->integrate(basisCacheForIntegration) );
          
          double tol = 1e-14;
          if (l2diff > tol) {
            success = false;
            cout << "Test Failure: on cell " << cellID << ", for variable " << varID;
            if (coarseOrdering->getNumSidesForVarID(varID) > 1) cout << " on side " << sideOrdinal << " ";
            cout << " for coarse basis ordinal " << coarseBasisOrdinal << ", ";
            cout << "the L^2 norm of difference between fine mesh representation and coarse representation exceeds tol: ";
            cout << l2diff << " > " << tol << "\n";
            break;
          }
        }
      }
    }
  }

  return TestSuite::allSuccess(success);
}

bool GMGTests::testGMGSolverIdentity() {
  bool success = true;
  
  bool useStaticCondensation = false;
  
  // some 2D-specific tests on refined meshes
  for (int refinementOrdinal=1; refinementOrdinal<2; refinementOrdinal++) {
    int spaceDim = 2;
    
    int H1Order_coarse = 5, H1Order = 5;
    
    FunctionPtr phiExact = getPhiExact(spaceDim);
    
    bool useH1Traces = false; // false is the more forgiving; a place to start testing
    SolutionPtr solnCoarse = poissonExactSolutionRefined(H1Order_coarse, phiExact, useH1Traces, refinementOrdinal);
    SolutionPtr solnFine = poissonExactSolutionRefined(H1Order, phiExact, useH1Traces, refinementOrdinal);
    solnFine->setUseCondensedSolve(useStaticCondensation);
    
    BCPtr poissonBC = solnFine->bc();
    BCPtr zeroBCs = poissonBC->copyImposingZero();
    MeshPtr fineMesh = solnFine->mesh();
    BF* bf = dynamic_cast< BF* >( fineMesh->bilinearForm().get() );
    IPPtr graphNorm = bf->graphNorm();
    
    // as a first test, do "multi" grid between mesh and itself.  Solution should match phiExact.
    double iter_tol = 1e-14;
    bool applySmoothing = false;
    int maxIters = applySmoothing ? 100 : 1; // if smoothing not applied, then GMG should recover exactly the direct solution, in 1 iteration
    Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
    
//    solnFine->setWriteMatrixToFile(true, "/tmp/A_fine.dat");
    
    {
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, fineMesh, graphNorm, fineMesh,
                                                                      solnFine->getDofInterpreter(),
                                                                      solnFine->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );

//      GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/fineMesh", fineMesh, true); // true: label cells
      
      // before we test the solve proper, let's check that with smoothing off, ApplyInverse acts just like the standard solve
//      solnFine->setWriteMatrixToFile(true, "/tmp/A_direct.dat");
//      solnFine->setWriteRHSToMatrixMarketFile(true, "/tmp/b_direct.dat");
      solnFine->initializeLHSVector();
      solnFine->initializeStiffnessAndLoad();
      solnFine->populateStiffnessAndLoad();
      Teuchos::RCP<Epetra_MultiVector> rhsVector = solnFine->getRHSVector();
      // since I'm not totally sure that the KluSolver won't clobber the rhsVector, make a copy:
      Epetra_MultiVector rhsVectorCopy(*rhsVector);
      solnFine->solve();
      Teuchos::RCP<Epetra_MultiVector> lhsVector = solnFine->getLHSVector();
//      EpetraExt::MultiVectorToMatlabFile("/tmp/x_direct.dat",*lhsVector);
      
      Epetra_MultiVector gmg_lhsVector(rhsVectorCopy.Map(), 1); // lhs has same distribution structure as rhs
      gmgSolver->setApplySmoothingOperator(false); // turn off for the next test:
      gmgSolver->gmgOperator().ApplyInverse(rhsVectorCopy, gmg_lhsVector);
      
      double tol = 1e-10;
      int minLID = gmg_lhsVector.Map().MinLID();
      int numLIDs = gmg_lhsVector.Map().NumMyElements();
      for (int lid=minLID; lid < minLID + numLIDs; lid++ ) {
        double direct_val = (*lhsVector)[0][lid];
        double gmg_val = gmg_lhsVector[0][lid];
        double diff = abs(direct_val - gmg_val);
        if (diff > tol) {
          GlobalIndexType gid = gmg_lhsVector.Map().GID(lid);
          cout << "FAILURE: For refinement sequence " << refinementOrdinal << " in " << spaceDim << "D, ";
          cout << "GMG ApplyInverse and direct solve differ for gid " << gid << " with difference = " << diff << ".\n";
          success = false;
        }
      }
    }

    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, fineMesh, graphNorm, fineMesh,
                                                                    solnFine->getDofInterpreter(),
                                                                    solnFine->getPartitionMap(),
                                                                    maxIters, iter_tol, coarseSolver, useStaticCondensation) );
    
    gmgSolver->setApplySmoothingOperator(applySmoothing);
    
    Teuchos::RCP<Solver> fineSolver = gmgSolver;

    solnFine->solve(coarseSolver);
    
    solnCoarse->solve(fineSolver);
    
    VarFactory vf = bf->varFactory();
    VarPtr phi = vf.fieldVar(S_GMGTests_PHI);
    
    FunctionPtr directPhiSoln = Function::solution(phi, solnFine);
    FunctionPtr iterativePhiSoln = Function::solution(phi, solnCoarse);
    
    double l2_diff = (directPhiSoln-iterativePhiSoln)->l2norm(fineMesh);
    
    double tol = iter_tol * 10;
    if (l2_diff > tol) {
      success = false;
      cout << "FAILURE: For refinement sequence " << refinementOrdinal << " in " << spaceDim << "D, ";
      
      cout << "GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";
    }
  }
  
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

      // do "multi" grid between mesh and itself.  Solution should match phiExact.
      double iter_tol = 1e-14;
      bool applySmoothing = true;
      int maxIters = applySmoothing ? 100 : 1; // if smoothing not applied, then GMG should recover exactly the direct solution, in 1 iteration
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                                                      exactPoissonSolution->getDofInterpreter(),
                                                                      exactPoissonSolution->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );

      {
        // before we test the solve proper, let's check that with smoothing off, ApplyInverse acts just like the standard solve
//        exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A_direct.dat");
//        exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b_direct.dat");
        exactPoissonSolution->initializeLHSVector();
        exactPoissonSolution->initializeStiffnessAndLoad();
        exactPoissonSolution->populateStiffnessAndLoad();
        Teuchos::RCP<Epetra_MultiVector> rhsVector = exactPoissonSolution->getRHSVector();
        // since I'm not totally sure that the KluSolver won't clobber the rhsVector, make a copy:
        Epetra_MultiVector rhsVectorCopy(*rhsVector);
        exactPoissonSolution->solve();
        Teuchos::RCP<Epetra_MultiVector> lhsVector = exactPoissonSolution->getLHSVector();
//        EpetraExt::MultiVectorToMatlabFile("/tmp/x_direct.dat",*lhsVector);
        
        Epetra_MultiVector gmg_lhsVector(rhsVectorCopy.Map(), 1); // lhs has same distribution structure as rhs
        gmgSolver->setApplySmoothingOperator(false); // turn off for the next test:
        gmgSolver->gmgOperator().ApplyInverse(rhsVectorCopy, gmg_lhsVector);
        
        double tol = 1e-10;
        int minLID = gmg_lhsVector.Map().MinLID();
        int numLIDs = gmg_lhsVector.Map().NumMyElements();
        for (int lid=minLID; lid < minLID + numLIDs; lid++ ) {
          double direct_val = (*lhsVector)[0][lid];
          double gmg_val = gmg_lhsVector[0][lid];
          double diff = abs(direct_val - gmg_val);
          if (diff > tol) {
            GlobalIndexType gid = gmg_lhsVector.Map().GID(lid);
            cout << "FAILURE: For cellCount = " << cellCounts[i] << " in " << spaceDim << "D, ";
            cout << "GMG ApplyInverse and direct solve differ for gid " << gid << " with difference = " << diff << ".\n";
            success = false;
          }
        }
      }
      
      success = TestSuite::allSuccess(success);
      
      gmgSolver->setApplySmoothingOperator(applySmoothing);
      
      Teuchos::RCP<Solver> fineSolver = gmgSolver;
      
//      exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A.dat");
//      exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
      
//      exactPoissonSolution->solve(coarseSolver);
      
      actualPoissonSolution->solve(fineSolver);
      exactPoissonSolution->solve(coarseSolver);
      
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
  
  return success;
}

bool GMGTests::testGMGSolverTwoGrid() {
  bool success = true;
  vector<int> cellCounts;
  cellCounts.push_back(1);
  
  bool useStaticCondensation = false;
  
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
      
      MeshPtr coarseMesh = poissonExactSolution(cellCount, H1Order-1, phiExact, useH1Traces)->mesh();
      
      SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      
      MeshPtr exactMesh = exactPoissonSolution->mesh();
      MeshPtr fineMesh = actualPoissonSolution->mesh();
      
      // refine uniformly once in both exact and actual:
      CellTopoPtrLegacy cellTopo = coarseMesh->getTopology()->getCell(0)->topology();
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo->getKey());
      exactMesh->hRefine(exactMesh->getActiveCellIDs(), refPattern);
      fineMesh->hRefine(fineMesh->getActiveCellIDs(), refPattern);
      
      BCPtr poissonBC = exactPoissonSolution->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr mesh = exactPoissonSolution->mesh();
      BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      double iter_tol = 1e-8;
      bool applySmoothing = true;
      int maxIters = 200;
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, fineMesh,
                                                                      exactPoissonSolution->getDofInterpreter(),
                                                                      exactPoissonSolution->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );
      
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

bool GMGTests::testGMGSolverThreeGrid() {
  bool success = true;
  vector<int> cellCounts;
  cellCounts.push_back(1);
  
  bool useStaticCondensation = false;
  
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

      SolutionPtr coarsestSolution = poissonExactSolution(cellCount, H1Order-1, phiExact, useH1Traces);
      MeshPtr coarsestMesh = coarsestSolution->mesh();
      SolutionPtr coarseSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      MeshPtr coarseMesh = coarseSolution->mesh();
      
      SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
      
      MeshPtr exactMesh = exactPoissonSolution->mesh();
      MeshPtr fineMesh = actualPoissonSolution->mesh();
      
      // refine uniformly once in both exact and actual:
      CellTopoPtrLegacy cellTopo = coarseMesh->getTopology()->getCell(0)->topology();
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
      Teuchos::RCP<Solver> coarsestSolver = Teuchos::rcp( new KluSolver );
      
      Teuchos::RCP<GMGSolver> coarseSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarsestMesh, graphNorm, coarseMesh,
                                                                         coarseSolution->getDofInterpreter(),
                                                                         coarseSolution->getPartitionMap(),
                                                                         maxIters, iter_tol / 10, coarsestSolver, useStaticCondensation) );
      
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, fineMesh,
                                                                      exactPoissonSolution->getDofInterpreter(),
                                                                      exactPoissonSolution->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );
      
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
        
        cout << "three grid GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";
      }
    }
  }
  return success;
}

string GMGTests::testSuiteName() {
  return "GMGTests";
}
