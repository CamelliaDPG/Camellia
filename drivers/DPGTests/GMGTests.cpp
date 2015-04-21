//
//  GMGTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/29/14.
//
//

#include "GMGTests.h"

#include "BasisSumFunction.h"
#include "BF.h"
#include "CamelliaDebugUtility.h"
#include "GnuPlotUtil.h"
#include "GDAMinimumRule.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"
#include "Var.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

const static string S_GMGTests_U1 = "u_1";
const static string S_GMGTests_U2 = "u_2";

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

VarPtr GMGTests::getPoissonPhi(int spaceDim) {
  bool conformingTraces = true; // doesn't affect variable labeling
  PoissonFormulation pf(spaceDim,conformingTraces);
  return pf.phi();
}

VarPtr GMGTests::getPoissonPsi(int spaceDim) {
  bool conformingTraces = true; // doesn't affect variable labeling
  PoissonFormulation pf(spaceDim,conformingTraces);
  return pf.psi();
}

VarPtr GMGTests::getPoissonPhiHat(int spaceDim) {
  bool conformingTraces = true; // doesn't affect variable labeling
  PoissonFormulation pf(spaceDim,conformingTraces);
  return pf.phi_hat();
}

VarPtr GMGTests::getPoissonPsi_n(int spaceDim) {
  bool conformingTraces = true; // doesn't affect variable labeling
  PoissonFormulation pf(spaceDim,conformingTraces);
  return pf.psi_n_hat();
}

SolutionPtr GMGTests::poissonExactSolution(int numCells_x, int H1Order, FunctionPtr phi_exact, bool useH1Traces) {
  vector<int> numCells;
  numCells.push_back(numCells_x);
  return poissonExactSolution(numCells, H1Order, phi_exact, useH1Traces);
}

SolutionPtr GMGTests::poissonExactSolution(vector<int> numCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces) {
  int spaceDim = numCells.size();
  
  PoissonFormulation pf(spaceDim,useH1Traces);
  
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
  if (testProlongationOperator()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testGMGSolverIdentityUniformMeshes()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testGMGSolverIdentity2DRefinedMeshes()) {
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
  if (testGMGSolverTwoGrid()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testGMGSolverThreeGrid()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool GMGTests::testGMGOperatorIdentityLocalCoefficientMap() {
  bool success = true;
  
  /***   1D-3D TESTS    ***/
  vector<int> cellCounts;
  cellCounts.push_back(1);
  cellCounts.push_back(2);
  cellCounts.push_back(4);
  
  vector<bool> useStaticCondensationValues;
  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
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
        Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
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
    for (int refinementOrdinal=-1; refinementOrdinal<3; refinementOrdinal++) { // refinementOrdinal=3 is a slow one, so we skip it...
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
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );

      if (useStaticCondensation) {
        // need to populate local stiffness matrices before dealing with the RHS.
        solnFine->initializeLHSVector();
        solnFine->initializeStiffnessAndLoad();
        solnFine->populateStiffnessAndLoad();
      }
      
      GMGOperator gmgOperator(zeroBCs, fineMesh, graphNorm, fineMesh, solnFine->getDofInterpreter(), solnFine->getPartitionMap(), coarseSolver, useStaticCondensation);
      
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
  vector<bool> useStaticCondensationValues;
  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
  for (  vector<bool>::iterator useStaticCondensationIt = useStaticCondensationValues.begin();
       useStaticCondensationIt != useStaticCondensationValues.end(); useStaticCondensationIt++) {
    bool useStaticCondensation = *useStaticCondensationIt;
  
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
    
    Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
    
    if (useStaticCondensation) {
      // need to populate local stiffness matrices for sake of condensed solver
      solnFine->initializeLHSVector();
      solnFine->initializeStiffnessAndLoad();
      solnFine->populateStiffnessAndLoad();
    }
    
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
      
      vector< VarPtr > fieldVars = bf->varFactory().fieldVars();
      set<int> fieldIDs;
      for (vector< VarPtr >::iterator fieldIt = fieldVars.begin(); fieldIt != fieldVars.end(); fieldIt++) {
        fieldIDs.insert((*fieldIt)->ID());
      }
      
      int sideCount = fineMesh->getElementType(cellID)->cellTopoPtr->getSideCount();
      
      set<int> varIDs = coarseOrdering->getVarIDs();
      for (set<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++) {
        int varID = *varIDIt;
        if ((fieldIDs.find(varID) != fieldIDs.end()) && useStaticCondensation) continue; // skip field test for static condensation: these guys are mapped in that case...
        for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
          if (coarseOrdering->hasBasisEntry(varID, sideOrdinal) != fineOrdering->hasBasisEntry(varID, sideOrdinal)) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarse and fine orderings disagree on whether varID is defined on side sideOrdinal");
          }
          if (! coarseOrdering->hasBasisEntry(varID, sideOrdinal)) continue;
          
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
            
            FunctionPtr fineBasisSumFunction = BasisSumFunction::basisSumFunction(fineBasis, fineBasisCoefficients);
            FunctionPtr coarseBasisSumFunction = BasisSumFunction::basisSumFunction(coarseBasis, coarseBasisCoefficients);
            FunctionPtr diffFxn = fineBasisSumFunction - coarseBasisSumFunction;
            
            BasisCachePtr basisCacheForIntegration = (coarseOrdering->getSidesForVarID(varID).size() == 1) ? basisCache : basisCache->getSideBasisCache(sideOrdinal);
            
            double l2diff = sqrt( (diffFxn * diffFxn)->integrate(basisCacheForIntegration) );
            
            double tol = 1e-14;
            if (l2diff > tol) {
              success = false;
              cout << "Test Failure: on cell " << cellID << ", for variable " << varID;
              if (coarseOrdering->getSidesForVarID(varID).size() > 1) cout << " on side " << sideOrdinal << " ";
              cout << " for coarse basis ordinal " << coarseBasisOrdinal << ", ";
              cout << "the L^2 norm of difference between fine mesh representation and coarse representation exceeds tol: ";
              cout << l2diff << " > " << tol << "\n";
              break;
            }
          }
        }
      }
    }
  }

  return TestSuite::allSuccess(success);
}

bool GMGTests::testGMGSolverIdentity2DRefinedMeshes() {
  bool success = true;
  
  vector<bool> useStaticCondensationValues;
  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
  for (  vector<bool>::iterator useStaticCondensationIt = useStaticCondensationValues.begin();
       useStaticCondensationIt != useStaticCondensationValues.end(); useStaticCondensationIt++) {
    bool useStaticCondensation = *useStaticCondensationIt;
    
    // some 2D-specific tests on refined meshes
    for (int refinementOrdinal=0; refinementOrdinal<4; refinementOrdinal++) {
      int spaceDim = 2;
      
      int H1Order_coarse = 2, H1Order = 2;
      
      FunctionPtr phiExact = getPhiExact(spaceDim);
      
      bool useH1Traces = false; // false is the more forgiving; a place to start testing
      SolutionPtr solnCoarse = poissonExactSolutionRefined(H1Order_coarse, phiExact, useH1Traces, refinementOrdinal);
      SolutionPtr solnFine = poissonExactSolutionRefined(H1Order, phiExact, useH1Traces, refinementOrdinal);
      solnFine->setUseCondensedSolve(useStaticCondensation);
      solnCoarse->setUseCondensedSolve(useStaticCondensation);
      
      BCPtr poissonBC = solnFine->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr fineMesh = solnFine->mesh();
      BF* bf = dynamic_cast< BF* >( fineMesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      // as a first test, do "multi" grid between mesh and itself.  Solution should match phiExact.
      double iter_tol = 1e-11;
      bool applySmoothing = false;
      int maxIters = applySmoothing ? 100 : 1; // if smoothing not applied, then GMG should recover exactly the direct solution, in 1 iteration
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
      
      //    solnFine->setWriteMatrixToFile(true, "/tmp/A_fine.dat");
      
      if (useStaticCondensation) {
        // need to populate local stiffness matrices before dealing with the RHS.
        solnFine->initializeLHSVector();
        solnFine->initializeStiffnessAndLoad();
        solnFine->populateStiffnessAndLoad();
      }
      
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, fineMesh, graphNorm, fineMesh,
                                                                      solnFine->getDofInterpreter(),
                                                                      solnFine->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );
      gmgSolver->setComputeConditionNumberEstimate(false);
      
      {
        solnFine->initializeLHSVector();
        solnFine->initializeStiffnessAndLoad();
        solnFine->populateStiffnessAndLoad();
        Teuchos::RCP<Epetra_MultiVector> rhsVector = solnFine->getRHSVector();
        // since I'm not totally sure that the KluSolver won't clobber the rhsVector, make a copy:
        Epetra_MultiVector rhsVectorCopy(*rhsVector);
        solnFine->solve();
        Teuchos::RCP<Epetra_MultiVector> lhsVector = solnFine->getLHSVector();
        //      EpetraExt::MultiVectorToMatlabFile("/tmp/x_direct.dat",*lhsVector);
        
        vector<bool> fineDiagonalScalingValues;
        fineDiagonalScalingValues.push_back(false);
        fineDiagonalScalingValues.push_back(true);
        
        // since we may change the RHS vector below, let's make a copy and use that
        Epetra_MultiVector rhsVectorCopy2(rhsVectorCopy);
        
        Epetra_CrsMatrix *A = solnFine->getStiffnessMatrix().get();
        const Epetra_BlockMap* map = &A->Map();
        Teuchos::RCP<Epetra_Vector> diagA = Teuchos::rcp( new Epetra_Vector(*map) );
        A->ExtractDiagonalCopy(*diagA);
        
        Teuchos::RCP<Epetra_Vector> diagA_inv = Teuchos::rcp( new Epetra_Vector(*map, 1) );
        Teuchos::RCP<Epetra_Vector> diagA_sqrt_inv = Teuchos::rcp( new Epetra_Vector(*map, 1) );
        diagA_inv->Reciprocal(*diagA);
        if (map->NumMyElements() > 0) {
          for (int lid = map->MinLID(); lid <= map->MaxLID(); lid++) {
            (*diagA_sqrt_inv)[lid] = 1.0 / sqrt((*diagA)[lid]);
          }
        }
        Teuchos::RCP<Epetra_Vector> diagA_sqrt = Teuchos::rcp( new Epetra_Vector(*map, 1) );
        diagA_sqrt->Reciprocal(*diagA_sqrt_inv);
        
        gmgSolver->gmgOperator().setSmootherType(GMGOperator::POINT_JACOBI);
        gmgSolver->gmgOperator().computeCoarseStiffnessMatrix(A);
        
        for (int i=0; i<fineDiagonalScalingValues.size(); i++) {
          bool fineSolverUsesDiagonalScaling = fineDiagonalScalingValues[i];
          Epetra_MultiVector gmg_lhsVector(rhsVectorCopy.Map(), 1); // lhs has same distribution structure as rhs
          gmgSolver->gmgOperator().setSmootherType(GMGOperator::NONE); // turn off for this next test, comparing direct solve to ApplyInverse
          gmgSolver->gmgOperator().setFineSolverUsesDiagonalScaling(fineSolverUsesDiagonalScaling);
          
          if (fineSolverUsesDiagonalScaling) {
            // then we need to imitate GMGSolver: first, communicate the diagonal
            gmgSolver->gmgOperator().setStiffnessDiagonal(diagA);
            
            // then, apply the inverse sqrt diagonal operator to the RHS we pass to GMG Operator.
            rhsVectorCopy2.Multiply(1.0, rhsVectorCopy2, *diagA_sqrt_inv, 0);
//            EpetraExt::MultiVectorToMatlabFile("/tmp/scaled_rhs.dat",rhsVectorCopy2);
          }
          
          gmgSolver->gmgOperator().ApplyInverse(rhsVectorCopy2, gmg_lhsVector);
          
          // determine the expected value
          Epetra_MultiVector directValue(*lhsVector); // x
          if (applySmoothing && !fineSolverUsesDiagonalScaling) {
            // x + D^-1 b
            directValue.Multiply(1.0, rhsVectorCopy2, *diagA_inv, 1.0);
          } else if (applySmoothing && fineSolverUsesDiagonalScaling) {
            // D^1/2 x + D^-1/2 b
            directValue.Multiply(1.0, directValue, *diagA_sqrt, 0.0);
            directValue.Multiply(1.0, rhsVectorCopy2, *diagA_sqrt_inv, 1.0);
          } else if (!applySmoothing && fineSolverUsesDiagonalScaling) {
            // D^1/2 x
            directValue.Multiply(1.0, directValue, *diagA_sqrt, 0.0);
          }
          
          double tol = 1e-10;
          int minLID = gmg_lhsVector.Map().MinLID();
          int numLIDs = gmg_lhsVector.Map().NumMyElements();
          for (int lid=minLID; lid < minLID + numLIDs; lid++ ) {
            double direct_val = directValue[0][lid];
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
      }
      
//      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, fineMesh, graphNorm, fineMesh,
//                                                                      solnFine->getDofInterpreter(),
//                                                                      solnFine->getPartitionMap(),
//                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );
//      
      gmgSolver->setApplySmoothingOperator(applySmoothing);
      
      Teuchos::RCP<Solver> fineSolver = gmgSolver;
      
      solnFine->solve(coarseSolver);
      
      solnCoarse->solve(fineSolver);
      
      VarPtr phi = getPoissonPhi(spaceDim);
      
      FunctionPtr directPhiSoln = Function::solution(phi, solnFine);
      FunctionPtr iterativePhiSoln = Function::solution(phi, solnCoarse);
      
      double l2_diff = (directPhiSoln-iterativePhiSoln)->l2norm(fineMesh);
      
      double tol = iter_tol * 100;
      if (l2_diff > tol) {
        success = false;
        cout << "FAILURE: For refinement sequence " << refinementOrdinal << " in " << spaceDim << "D, ";
        
        cout << "GMG solver and direct differ with L^2 norm of the difference = " << l2_diff << ".\n";
      }
    }
  }
  
  return allSuccess(success);
}

bool GMGTests::testGMGSolverIdentityUniformMeshes() {
  
  bool success = true;
  
  vector<bool> useStaticCondensationValues;
  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
  for (  vector<bool>::iterator useStaticCondensationIt = useStaticCondensationValues.begin();
       useStaticCondensationIt != useStaticCondensationValues.end(); useStaticCondensationIt++) {
    bool useStaticCondensation = *useStaticCondensationIt;
    
    vector<int> cellCounts;
    cellCounts.push_back(1);
    cellCounts.push_back(2);
//    cellCounts.push_back(4);

    for (int spaceDim=1; spaceDim<=3; spaceDim++) {
      for (int i=0; i<cellCounts.size(); i++) {
//        if ((spaceDim==3) && (i==cellCounts.size()-1)) continue; // skip the 4x4x4 case, in interest of time.
        vector<int> cellCount;
        for (int d=0; d<spaceDim; d++) {
          cellCount.push_back(cellCounts[i]);
        }

        int H1Order = 1;
        bool useH1Traces = false;
        FunctionPtr phiExact = getPhiExact(spaceDim);
        SolutionPtr exactPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
        SolutionPtr actualPoissonSolution = poissonExactSolution(cellCount, H1Order, phiExact, useH1Traces);
        
        exactPoissonSolution->setUseCondensedSolve(useStaticCondensation);
        actualPoissonSolution->setUseCondensedSolve(useStaticCondensation);
        
        BCPtr poissonBC = exactPoissonSolution->bc();
        BCPtr zeroBCs = poissonBC->copyImposingZero();
        MeshPtr mesh = exactPoissonSolution->mesh();
        BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
        IPPtr graphNorm = bf->graphNorm();

        if (useStaticCondensation) {
          // need to populate local stiffness matrices for condensed dof interpreter
          exactPoissonSolution->initializeLHSVector();
          exactPoissonSolution->initializeStiffnessAndLoad();
          exactPoissonSolution->populateStiffnessAndLoad();
        }
        
        Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
        int maxIters = 100;
        double iter_tol = 1e-14;
        Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                                                        exactPoissonSolution->getDofInterpreter(),
                                                                        exactPoissonSolution->getPartitionMap(),
                                                                        maxIters, iter_tol, coarseSolver, useStaticCondensation) );
        gmgSolver->setComputeConditionNumberEstimate(false);

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

          vector<bool> applySmoothingValues;
          applySmoothingValues.push_back(false);
          applySmoothingValues.push_back(true);
          
          for (int j=0; j<applySmoothingValues.size(); j++) {
            bool applySmoothing = applySmoothingValues[j];
          
            vector<bool> fineDiagonalScalingValues;
            fineDiagonalScalingValues.push_back(false);
            fineDiagonalScalingValues.push_back(true);
            
            for (int j=0; j<fineDiagonalScalingValues.size(); j++) {
              bool fineSolverUsesDiagonalScaling = fineDiagonalScalingValues[j];
              Epetra_MultiVector gmg_lhsVector(rhsVectorCopy.Map(), 1); // lhs has same distribution structure as rhs
              gmgSolver->setApplySmoothingOperator(applySmoothing);
              gmgSolver->gmgOperator().setFineSolverUsesDiagonalScaling(fineSolverUsesDiagonalScaling);
              
              // since we may change the RHS vector below, let's make a copy and use that
              Epetra_MultiVector rhsVectorCopy2(rhsVectorCopy);
              
              Epetra_CrsMatrix *A = exactPoissonSolution->getStiffnessMatrix().get();
              
              const Epetra_Map* map = &A->RowMatrixRowMap();
              
              Teuchos::RCP<Epetra_Vector> diagA = Teuchos::rcp( new Epetra_Vector(*map) );
              A->ExtractDiagonalCopy(*diagA);
              
              gmgSolver->gmgOperator().setStiffnessDiagonal(diagA);
              
              Teuchos::RCP<Epetra_Vector> diagA_inv = Teuchos::rcp( new Epetra_Vector(*map, 1) );
              Teuchos::RCP<Epetra_Vector> diagA_sqrt_inv = Teuchos::rcp( new Epetra_Vector(*map, 1) );
              diagA_inv->Reciprocal(*diagA);
              if (map->NumMyElements() > 0) {
                for (int lid = map->MinLID(); lid <= map->MaxLID(); lid++) {
                  (*diagA_sqrt_inv)[lid] = 1.0 / sqrt((*diagA)[lid]);
                }
              }
              Teuchos::RCP<Epetra_Vector> diagA_sqrt = Teuchos::rcp( new Epetra_Vector(*map, 1) );
              diagA_sqrt->Reciprocal(*diagA_sqrt_inv);

//              EpetraExt::RowMatrixToMatlabFile("/tmp/A.dat",*A);
//              EpetraExt::MultiVectorToMatlabFile("/tmp/rhs.dat",rhsVectorCopy2);
              
              // determine the expected value
              Epetra_MultiVector directValue(*lhsVector); // x
              if (applySmoothing && !fineSolverUsesDiagonalScaling) {
                // x + D^-1 b
                directValue.Multiply(1.0, rhsVectorCopy2, *diagA_inv, 1.0);
              } else if (applySmoothing && fineSolverUsesDiagonalScaling) {
                // D^1/2 x + D^-1/2 b
                directValue.Multiply(1.0, directValue, *diagA_sqrt, 0.0);
                directValue.Multiply(1.0, rhsVectorCopy2, *diagA_sqrt_inv, 1.0);
              } else if (!applySmoothing && fineSolverUsesDiagonalScaling) {
                // D^1/2 x
                directValue.Multiply(1.0, directValue, *diagA_sqrt, 0.0);
              }
              
              if (fineSolverUsesDiagonalScaling) {
                // then, apply the inverse sqrt diagonal operator to the RHS we pass to GMG Operator.
                rhsVectorCopy2.Multiply(1.0, rhsVectorCopy2, *diagA_sqrt_inv, 0);
//                EpetraExt::MultiVectorToMatlabFile("/tmp/scaled_rhs.dat",rhsVectorCopy2);
              }

              // if applySmoothing = false, then we expect exact agreement between direct solution and iterative.
              // If applySmoothing = true,  then we expect iterative = exact + D^-1 b
              
              if (applySmoothing) {
                gmgSolver->gmgOperator().setSmootherType(GMGOperator::POINT_JACOBI);
              } else {
                gmgSolver->gmgOperator().setSmootherType(GMGOperator::NONE);
              }
              
              gmgSolver->gmgOperator().computeCoarseStiffnessMatrix(A);
              gmgSolver->gmgOperator().ApplyInverse(rhsVectorCopy2, gmg_lhsVector);

              double tol = 1e-10;
              int minLID = gmg_lhsVector.Map().MinLID();
              int numLIDs = gmg_lhsVector.Map().NumMyElements();
              for (int lid=minLID; lid < minLID + numLIDs; lid++ ) {
                double direct_val = directValue[0][lid];
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
          }
        }
        success = TestSuite::allSuccess(success);
        
        vector<bool> applySmoothingValues;
        applySmoothingValues.push_back(false);
        applySmoothingValues.push_back(true);
        
        for (int j=0; j<applySmoothingValues.size(); j++) {
          bool applySmoothing = applySmoothingValues[j];
        
          // do "multi" grid between mesh and itself.  Solution should match phiExact.
          int maxIters = applySmoothing ? 100 : 1; // if smoothing not applied, then GMG should recover exactly the direct solution, in 1 iteration

          if (useStaticCondensation) {
            // need to populate local stiffness matrices in the 
            actualPoissonSolution->initializeLHSVector();
            actualPoissonSolution->initializeStiffnessAndLoad();
            actualPoissonSolution->populateStiffnessAndLoad();
          }
          
          Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                                                          actualPoissonSolution->getDofInterpreter(),
                                                                          actualPoissonSolution->getPartitionMap(),
                                                                          maxIters, iter_tol, coarseSolver, useStaticCondensation) );

          gmgSolver->setComputeConditionNumberEstimate(false);
          gmgSolver->setApplySmoothingOperator(applySmoothing);
          
          Teuchos::RCP<Solver> fineSolver = gmgSolver;
          
    //      exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A.dat");
    //      exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
          
    //      exactPoissonSolution->solve(coarseSolver);
          
          actualPoissonSolution->solve(fineSolver);
          exactPoissonSolution->solve(coarseSolver);
          
          VarPtr phi = getPoissonPhi(spaceDim);
          
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
    }
  }
  return TestSuite::allSuccess(success);
}

bool GMGTests::testGMGSolverTwoGrid() {
  bool success = true;
  vector<int> cellCounts;
  cellCounts.push_back(1);
  
  bool useStaticCondensation = false; // static condensation is not yet supported for h-multigrid.
  
  for (int spaceDim=1; spaceDim<=3; spaceDim++) {
    for (int i=0; i<cellCounts.size(); i++) {
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
      CellTopoPtr cellTopo = coarseMesh->getTopology()->getCell(0)->topology();
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo->getKey());
      exactMesh->hRefine(exactMesh->getActiveCellIDs(), refPattern);
      fineMesh->hRefine(fineMesh->getActiveCellIDs(), refPattern);
      
      BCPtr poissonBC = exactPoissonSolution->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr mesh = exactPoissonSolution->mesh();
      BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      double iter_tol = 1e-12;
      bool applySmoothing = true;
      int maxIters = 200;
      Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
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
      
      VarPtr phi = getPoissonPhi(spaceDim);
      
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
  
  vector<bool> useStaticCondensationValues; // static condensation is not yet supported for h-multigrid.
//  useStaticCondensationValues.push_back(true);
  useStaticCondensationValues.push_back(false);
  
  for (  vector<bool>::iterator useStaticCondensationIt = useStaticCondensationValues.begin();
       useStaticCondensationIt != useStaticCondensationValues.end(); useStaticCondensationIt++) {
    bool useStaticCondensation = *useStaticCondensationIt;
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
      CellTopoPtr cellTopo = coarseMesh->getTopology()->getCell(0)->topology();
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo->getKey());
      exactMesh->hRefine(exactMesh->getActiveCellIDs(), refPattern);
      fineMesh->hRefine(fineMesh->getActiveCellIDs(), refPattern);
      
      BCPtr poissonBC = exactPoissonSolution->bc();
      BCPtr zeroBCs = poissonBC->copyImposingZero();
      MeshPtr mesh = exactPoissonSolution->mesh();
      BF* bf = dynamic_cast< BF* >( mesh->bilinearForm().get() );
      IPPtr graphNorm = bf->graphNorm();
      
      double iter_tol = 1e-10;
      bool applySmoothing = true;
      int maxIters = 200;
      Teuchos::RCP<Solver> coarsestSolver = Teuchos::rcp( new Amesos2Solver(true) );
      
      coarseSolution->setUseCondensedSolve(useStaticCondensation);
      exactPoissonSolution->setUseCondensedSolve(useStaticCondensation);
      actualPoissonSolution->setUseCondensedSolve(useStaticCondensation);
      
      if (useStaticCondensation) {
        // then for coarseSolution->getDofInterpreter() and exactPoissonSolution->getDofInterpreter() to work,
        // need to populate their local stiffness matrices
        coarseSolution->initializeLHSVector();
        coarseSolution->initializeStiffnessAndLoad();
        coarseSolution->populateStiffnessAndLoad();
        
        exactPoissonSolution->initializeLHSVector();
        exactPoissonSolution->initializeStiffnessAndLoad();
        exactPoissonSolution->populateStiffnessAndLoad();
      }
      
      Teuchos::RCP<GMGSolver> coarseSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarsestMesh, graphNorm, coarseMesh,
                                                                         coarseSolution->getDofInterpreter(),
                                                                         coarseSolution->getPartitionMap(),
                                                                         maxIters, iter_tol * 10, coarsestSolver, useStaticCondensation) );
      
      Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, fineMesh,
                                                                      exactPoissonSolution->getDofInterpreter(),
                                                                      exactPoissonSolution->getPartitionMap(),
                                                                      maxIters, iter_tol, coarseSolver, useStaticCondensation) );
      
      coarseSolver->setComputeConditionNumberEstimate(false);
      gmgSolver->setComputeConditionNumberEstimate(false);
      
      gmgSolver->setApplySmoothingOperator(applySmoothing);
      
      Teuchos::RCP<Solver> fineSolver = gmgSolver;
      
      //      exactPoissonSolution->setWriteMatrixToFile(true, "/tmp/A.dat");
      //      exactPoissonSolution->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
      
      exactPoissonSolution->solve(coarsestSolver);
      
      actualPoissonSolution->solve(fineSolver);
      
      VarPtr phi = getPoissonPhi(spaceDim);
      
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
  }
  return success;
}

bool GMGTests::testProlongationOperator() {
  /*
   This is not a great test, just a quick check against some hand-determined values.
   */
  
  bool success = true;
  
  int H1Order = 2;
  bool useH1Traces = false;
  int spaceDim = 1;
  int coarseCellCount = 1;
  
  FunctionPtr phiExact = getPhiExact(spaceDim);
  
  MeshPtr coarseMesh = poissonExactSolution(coarseCellCount, H1Order, phiExact, useH1Traces)->mesh();
  
  SolutionPtr exactPoissonSolution = poissonExactSolution(coarseCellCount, H1Order, phiExact, useH1Traces);
  SolutionPtr actualPoissonSolution = poissonExactSolution(coarseCellCount, H1Order, phiExact, useH1Traces);
  
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
  
  bool useStaticCondensation = false;
  
  double iter_tol = 1e-8;
  int maxIters = 200;
  Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new Amesos2Solver(true) );
  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, fineMesh,
                                                                  exactPoissonSolution->getDofInterpreter(),
                                                                  exactPoissonSolution->getPartitionMap(),
                                                                  maxIters, iter_tol, coarseSolver, useStaticCondensation) );

  gmgSolver->gmgOperator().constructLocalCoefficientMaps();
  GlobalIndexType cellID = 1; // the left cell in the fine mesh
  LocalDofMapperPtr localCoefficientMap = gmgSolver->gmgOperator().getLocalCoefficientMap(cellID);
  
  bool traceToFieldsEnabled = true; // have not yet implemented the thing that will map fine traces to coarse fields
  
  // just hard-coding the values we expect right now.  Rows are fine cell 1 local dofs.  Columns are coarse cell 0 dofs.
  
  /*
   Since our field variables are linear, for v_coarse that has corresponding v_fine in fine mesh, we expect
   coeff(v_coarse) = coeff(v_fine)
   
   When v_fine doesn't have a corresponding coarse dof, there will be v_coarse_left and v_coarse_right with:
   coeff(v_fine) = 0.5 * ( coeff(v_coarse_left) + coeff(v_coarse_right) )
   */
  
  DofOrderingPtr trialPtr = fineMesh->getElementType(cellID)->trialOrderPtr;
  VarPtr phi = getPoissonPhi(spaceDim);
  VarPtr psi = getPoissonPsi(spaceDim);
  VarPtr psi_n = getPoissonPsi_n(spaceDim);
  VarPtr phi_hat = getPoissonPhiHat(spaceDim);
  
  int basisOrdinalLeft = 0, basisOrdinalRight = 1;
  int leftSide = 0, rightSide = 1;
  int pointBasisOrdinal = 0;
  
  int phiLeft = trialPtr->getDofIndex(phi->ID(), basisOrdinalLeft);
  int phiRight = trialPtr->getDofIndex(phi->ID(), basisOrdinalRight);
  
  int psiLeft = trialPtr->getDofIndex(psi->ID(), basisOrdinalLeft);
  int psiRight = trialPtr->getDofIndex(psi->ID(), basisOrdinalRight);
  
  // the fine
  int phiHatLeft = trialPtr->getDofIndex(phi_hat->ID(), pointBasisOrdinal, leftSide);
  int phiHatRight = trialPtr->getDofIndex(phi_hat->ID(), pointBasisOrdinal, rightSide);
  
  int psi_n_left = trialPtr->getDofIndex(psi_n->ID(), pointBasisOrdinal, leftSide);
  int psi_n_right = trialPtr->getDofIndex(psi_n->ID(), pointBasisOrdinal, rightSide);
  
  FieldContainer<double> expectedValues(8,8);
  
  if (traceToFieldsEnabled) {
    // phiHatLeft maps to the corresponding trace on the coarse element; hence no weight for (phiHatLeft,phiLeft)
    expectedValues(phiHatRight,phiLeft) = 0.5;
    
    expectedValues(phiHatLeft,phiRight) = 0.0;
    expectedValues(phiHatRight,phiRight) = 0.5;
  
    // psi_n_left maps to the corresponding flux on the coarse element; hence no weight for (psi_n_left,psiLeft)
    expectedValues(psi_n_right,psiLeft) = 0.5;
    
    expectedValues(psi_n_left,psiRight) = 0.0;
    expectedValues(psi_n_right,psiRight) = 0.5;
  }
  
  expectedValues(phiHatLeft,phiHatLeft) = 1.0;
  
  expectedValues(psi_n_left,psi_n_left) = 1.0;
  
  expectedValues(phiLeft,phiLeft) = 1.0;
  
  expectedValues(phiRight,phiLeft) = 0.5;
  expectedValues(phiRight,phiRight) = 0.5;
  
  expectedValues(psiLeft,psiLeft) = 1.0;

  expectedValues(psiRight,psiLeft) = 0.5;
  expectedValues(psiRight,psiRight) = 0.5;
  
  FieldContainer<double> actualValues(8,8);
  for (int i=0; i<8; i++) {
    FieldContainer<double> localData(8);
    localData(i) = 1.0;
    FieldContainer<double> globalData = localCoefficientMap->mapLocalData(localData, false);
    for (int j=0; j<8; j++) {
      actualValues(i,j) = globalData(j);
    }
  }
  
  double tol = 1e-12;
  double maxDiff;
  if (! fcsAgree(expectedValues, actualValues, tol, maxDiff)) {
    cout << "left child: expected differs from actual; maxDiff " << maxDiff << endl;
    success = false;
    
    cout << "actualValues:\n" << actualValues;
    cout << "expectedValues:\n" << expectedValues;
    
    cout << "key:\n";
    cout << "psi_n_left: " << psi_n_left << endl;
    cout << "psi_n_right: " << psi_n_right << endl;
    cout << "phiHatLeft: " << phiHatLeft << endl;
    cout << "phiHatRight: " << phiHatRight << endl;
  }
  
  cellID = 2; // the right cell in the fine mesh
  localCoefficientMap = gmgSolver->gmgOperator().getLocalCoefficientMap(cellID);
  
  trialPtr = fineMesh->getElementType(cellID)->trialOrderPtr;
  
  expectedValues.initialize(0.0);
  
  if (traceToFieldsEnabled) { // in 1D, traces and fluxes are the same, and both need to be parity weighted....
    expectedValues(phiHatLeft,phiLeft) = -0.5;
    expectedValues(phiHatRight,phiLeft) = 0.0;

    expectedValues(phiHatLeft,phiRight) = -0.5;
    // phiHatRight maps to the corresponding trace on the coarse element; hence no weight for (phiHatRight,phiRight)
  
    expectedValues(psi_n_left,psiLeft) = -0.5;
    expectedValues(psi_n_right,psiLeft) = 0.0;

    expectedValues(psi_n_left,psiRight) = -0.5;
    // psi_n_right maps to the corresponding flux on the coarse element; hence no weight for (psi_n_right,psiRight)
  }
  
  expectedValues(phiHatRight,phiHatRight) = 1.0;
  
  expectedValues(psi_n_right,psi_n_right) = 1.0;
  
  expectedValues(phiRight,phiRight) = 1.0;
  
  expectedValues(phiLeft,phiLeft) = 0.5;
  expectedValues(phiLeft,phiRight) = 0.5;
  
  expectedValues(psiRight,psiRight) = 1.0;
  
  expectedValues(psiLeft,psiLeft) = 0.5;
  expectedValues(psiLeft,psiRight) = 0.5;
  
  actualValues.initialize(0.0);
  for (int i=0; i<8; i++) {
    FieldContainer<double> localData(8);
    localData(i) = 1.0;
    FieldContainer<double> globalData = localCoefficientMap->mapLocalData(localData, false);
    for (int j=0; j<8; j++) {
      actualValues(i,j) = globalData(j);
    }
  }
  
  maxDiff = 0;
  if (! fcsAgree(expectedValues, actualValues, tol, maxDiff)) {
    cout << "right child: expected differs from actual; maxDiff " << maxDiff << endl;
    success = false;
    
    cout << "actualValues:\n" << actualValues;
    cout << "expectedValues:\n" << expectedValues;
    
    cout << "key:\n";
    cout << "psi_n_left: " << psi_n_left << endl;
    cout << "psi_n_right: " << psi_n_right << endl;
    cout << "phiHatLeft: " << phiHatLeft << endl;
    cout << "phiHatRight: " << phiHatRight << endl;
  }
  
  return success;
}

string GMGTests::testSuiteName() {
  return "GMGTests";
}
