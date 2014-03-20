//
//  GDAMinimumRuleTests.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/26/14.
//
//

#include "GDAMinimumRuleTests.h"

#include "MeshFactory.h"

#include "BF.h"

#include "Solution.h"
#include "PreviousSolutionFunction.h"

#include "Epetra_SerialComm.h"

#include "GnuPlotUtil.h"

const static string S_GDAMinimumRule_U1 = "u_1";
const static string S_GDAMinimumRule_U2 = "u_2";

// boundary value for u
class GDAMinimumRuleTests_U0 : public Function {
public:
  GDAMinimumRuleTests_U0() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        // solution with a boundary layer (section 5.2 in DPG Part II)
        // for x = 1, y = 1: u = 0
        if ( ( abs(x-1.0) < tol ) || (abs(y-1.0) < tol ) ) {
          values(cellIndex,ptIndex) = 0;
        } else if ( abs(x) < tol ) { // for x=0: u = 1 - y
          values(cellIndex,ptIndex) = 1.0 - y;
        } else { // for y=0: u=1-x
          values(cellIndex,ptIndex) = 1.0 - x;
        }
        
      }
    }
  }
};

class GDAMinimumRuleTests_TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(y-1.0) < tol);
  }
};

class GDAMinimumRuleTests_RampBoundaryFunction_U1 : public SimpleFunction {
  double _eps; // ramp width
public:
  GDAMinimumRuleTests_RampBoundaryFunction_U1(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    if ( (abs(x) < _eps) ) { // top left
      return x / _eps;
    } else if ( abs(1.0-x) < _eps) { // top right
      return (1.0-x) / _eps;
    } else { // top middle
      return 1;
    }
  }
};


GDAMinimumRuleTests::GDAMinimumRuleTests() {
  
}

SolutionPtr GDAMinimumRuleTests::quadMeshSolutionConfusion(bool useMinRule, int horizontalCells, int verticalCells, int H1Order) {
  double eps = 1e-1;
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta_const * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  MeshPtr mesh;
  
  int pToAddTest = 2;
  double width = 1.0, height = 1.0;
  if (useMinRule) {
    mesh = MeshFactory::quadMeshMinRule(confusionBF, H1Order, pToAddTest, width, height, horizontalCells, verticalCells);
  } else {
    mesh = MeshFactory::quadMesh(confusionBF, H1Order, pToAddTest, width, height, horizontalCells, verticalCells);
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Function::zero();
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  FunctionPtr u0 = Teuchos::rcp( new GDAMinimumRuleTests_U0 );
  bc->addDirichlet(uhat, SpatialFilter::allSpace(), u0);
  
  IPPtr ip = confusionBF->graphNorm();
  
  SolutionPtr soln = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  return soln;
}

SolutionPtr GDAMinimumRuleTests::quadMeshSolutionStokesCavityFlow(bool useMinRule, int horizontalCells, int verticalCells, int H1Order) {
  VarFactory varFactory;
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV);  // tau_1
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV);  // tau_2
  VarPtr v1 = varFactory.testVar("v1", HGRAD); // v_1
  VarPtr v2 = varFactory.testVar("v2", HGRAD); // v_2
  VarPtr q = varFactory.testVar("q", HGRAD); // q
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr t1_n = varFactory.fluxVar("\\widehat{t}_{1n}");
  VarPtr t2_n = varFactory.fluxVar("\\widehat{t}_{2n}");
  
  VarPtr u1 = varFactory.fieldVar(S_GDAMinimumRule_U1, L2);
  VarPtr u2 = varFactory.fieldVar(S_GDAMinimumRule_U2, L2);
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2", VECTOR_L2);
  VarPtr p = varFactory.fieldVar("p");
  
  double mu = 1;
  
  BFPtr stokesBF = Teuchos::rcp( new BF(varFactory) );
  // tau1 terms:
  stokesBF->addTerm(u1, tau1->div());
  stokesBF->addTerm(sigma1, tau1); // (sigma1, tau1)
  stokesBF->addTerm(-u1hat, tau1->dot_normal());
  
  // tau2 terms:
  stokesBF->addTerm(u2, tau2->div());
  stokesBF->addTerm(sigma2, tau2);
  stokesBF->addTerm(-u2hat, tau2->dot_normal());
  
  // v1:
  stokesBF->addTerm(mu * sigma1, v1->grad()); // (mu sigma1, grad v1)
  stokesBF->addTerm( - p, v1->dx() );
  stokesBF->addTerm( t1_n, v1);
  
  // v2:
  stokesBF->addTerm(mu * sigma2, v2->grad()); // (mu sigma2, grad v2)
  stokesBF->addTerm( - p, v2->dy());
  stokesBF->addTerm( t2_n, v2);
  
  // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  FunctionPtr n = Function::normal();
  stokesBF->addTerm(u1hat * n->x() + u2hat * n->y(), q);
  
  int testSpaceEnrichment = 2; //
  double width = 1.0, height = 1.0;
  
  MeshPtr mesh;
  if (useMinRule) {
    mesh = MeshFactory::quadMeshMinRule(stokesBF, H1Order, testSpaceEnrichment,
                                        width, height,
                                        horizontalCells, verticalCells);
  } else {
    mesh = MeshFactory::quadMesh(stokesBF, H1Order, testSpaceEnrichment,
                                 width, height,
                                 horizontalCells, verticalCells);
  }
  
  RHSPtr rhs = RHS::rhs(); // zero
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr topBoundary = Teuchos::rcp( new GDAMinimumRuleTests_TopBoundary );
  SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);
  
  // top boundary:
  FunctionPtr u1_bc_fxn = Teuchos::rcp( new GDAMinimumRuleTests_RampBoundaryFunction_U1(1.0/64) );
  FunctionPtr zero = Function::zero();
  bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
  bc->addDirichlet(u2hat, topBoundary, zero);
  
  // everywhere else:
  bc->addDirichlet(u1hat, otherBoundary, zero);
  bc->addDirichlet(u2hat, otherBoundary, zero);
  
  bc->addZeroMeanConstraint(p);
  
  IPPtr graphNorm = stokesBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  return solution;
}

SolutionPtr GDAMinimumRuleTests::quadMeshSolutionStokesExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order,
                                                                     FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact) {
  // assumes that div u = 0, and p has zero average on the domain (a unit square).
  
  VarFactory varFactory;
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV);  // tau_1
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV);  // tau_2
  VarPtr v1 = varFactory.testVar("v1", HGRAD); // v_1
  VarPtr v2 = varFactory.testVar("v2", HGRAD); // v_2
  VarPtr q = varFactory.testVar("q", HGRAD); // q
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr t1_n = varFactory.fluxVar("\\widehat{t}_{1n}");
  VarPtr t2_n = varFactory.fluxVar("\\widehat{t}_{2n}");
  
  VarPtr u1 = varFactory.fieldVar(S_GDAMinimumRule_U1, L2);
  VarPtr u2 = varFactory.fieldVar(S_GDAMinimumRule_U2, L2);
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2", VECTOR_L2);
  VarPtr p = varFactory.fieldVar("p");
  
  double mu = 1;
  
  BFPtr stokesBF = Teuchos::rcp( new BF(varFactory) );
  // tau1 terms:
  stokesBF->addTerm(u1, tau1->div());
  stokesBF->addTerm(sigma1, tau1); // (sigma1, tau1)
  stokesBF->addTerm(-u1hat, tau1->dot_normal());
  
  // tau2 terms:
  stokesBF->addTerm(u2, tau2->div());
  stokesBF->addTerm(sigma2, tau2);
  stokesBF->addTerm(-u2hat, tau2->dot_normal());
  
  // v1:
  stokesBF->addTerm(mu * sigma1, v1->grad()); // (mu sigma1, grad v1)
  stokesBF->addTerm( - p, v1->dx() );
  stokesBF->addTerm( t1_n, v1);
  
  // v2:
  stokesBF->addTerm(mu * sigma2, v2->grad()); // (mu sigma2, grad v2)
  stokesBF->addTerm( - p, v2->dy());
  stokesBF->addTerm( t2_n, v2);
  
  // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  FunctionPtr n = Function::normal();
  stokesBF->addTerm(u1hat * n->x() + u2hat * n->y(), q);
  
  int testSpaceEnrichment = 2; //
  double width = 1.0, height = 1.0;
  
  MeshPtr mesh;
  if (useMinRule) {
    mesh = MeshFactory::quadMeshMinRule(stokesBF, H1Order, testSpaceEnrichment,
                                        width, height,
                                        horizontalCells, verticalCells);
  } else {
    mesh = MeshFactory::quadMesh(stokesBF, H1Order, testSpaceEnrichment,
                                 width, height,
                                 horizontalCells, verticalCells);
  }
  
  // rhs = f * v, where f = mu * \Delta u - grad p
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f1 = mu * u1_exact->dx()->dx() + u1_exact->dy()->dy() - p_exact->dx();
  FunctionPtr f2 = mu * u2_exact->dx()->dx() + u2_exact->dy()->dy() - p_exact->dy();
  rhs->addTerm(f1 * v1 + f2 * v2);
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(u1hat, boundary, u1_exact);
  bc->addDirichlet(u2hat, boundary, u2_exact);
  
  bc->addZeroMeanConstraint(p);
  
  IPPtr graphNorm = stokesBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  return solution;
}


void GDAMinimumRuleTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testMultiCellMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testMultiCellMesh complete.\n";
  
  setup();
  if (testHangingNode()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testHangingNode complete.\n";
  
  setup();
  if (testSingleCellMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testSingleCellMesh complete.\n";
  
  setup();
  if (testHRefinements()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testHRefinements complete.\n";
  
  setup();
  if (testGlobalToLocalToGlobalConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

//  cout << "testGlobalToLocalToGlobalConsistency complete.\n";
  
  setup();
  if (testLocalInterpretationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testLocalInterpretationConsistency complete.\n";
}
void GDAMinimumRuleTests::setup() {
  
}

void GDAMinimumRuleTests::teardown() {
  
}

bool GDAMinimumRuleTests::subTestCompatibleSolutionsAgree(int horizontalCells, int verticalCells, int H1Order, int numUniformRefinements) {
  bool success = true;
  
  // up to a permutation of the rows/columns, everything should be identical for min and max rule on a single cell mesh.
  SolutionPtr minRuleConfusion = quadMeshSolutionConfusion(true, horizontalCells, verticalCells, H1Order);
  SolutionPtr maxRuleConfusion = quadMeshSolutionConfusion(false, horizontalCells, verticalCells, H1Order);
  
  SolutionPtr minRuleStokes = quadMeshSolutionStokesCavityFlow(true, horizontalCells, verticalCells, H1Order);
  SolutionPtr maxRuleStokes = quadMeshSolutionStokesCavityFlow(false, horizontalCells, verticalCells, H1Order);
  
  
  // DEBUGGING output to file:
  {
//    int numProcs = Teuchos::GlobalMPISession::getNProc();
//    if (numProcs == 1) {
//      minRuleConfusion->setWriteMatrixToFile(true, "confusionMinRuleStiffness_1_proc");
//      minRuleConfusion->setWriteRHSToMatrixMarketFile(true, "confusionMinRuleRHS_1_proc");
//    } else if (numProcs == 2) {
//      minRuleConfusion->setWriteMatrixToFile(true, "confusionMinRuleStiffness_2_proc");
//      minRuleConfusion->setWriteRHSToMatrixMarketFile(true, "confusionMinRuleRHS_2_proc");
//    } else if (numProcs == 4) {
//      minRuleConfusion->setWriteMatrixToFile(true, "confusionMinRuleStiffness_4_proc");
//      minRuleConfusion->setWriteRHSToMatrixMarketFile(true, "confusionMinRuleRHS_4_proc");
//    }
//    if (numProcs == 1) {
//      minRuleStokes->setWriteMatrixToFile(true, "stokesMinRuleStiffness_1_proc");
//      minRuleStokes->setWriteRHSToMatrixMarketFile(true, "stokesMinRuleRHS_1_proc");
//    } else if (numProcs == 2) {
//      minRuleStokes->setWriteMatrixToFile(true, "stokesMinRuleStiffness_2_proc");
//      minRuleStokes->setWriteRHSToMatrixMarketFile(true, "stokesMinRuleRHS_2_proc");
//    } else if (numProcs == 4) {
//      minRuleStokes->setWriteMatrixToFile(true, "stokesMinRuleStiffness_4_proc");
//      minRuleStokes->setWriteRHSToMatrixMarketFile(true, "stokesMinRuleRHS_4_proc");
//    }
//    
//    GnuPlotUtil::writeComputationalMeshSkeleton("stokesMesh", minRuleStokes->mesh());
  }

  for (int testIndex=1; testIndex<2; testIndex++) { // just to distinguish between Stokes and Confusion
    
    bool isStokes = (testIndex == 1);

    SolutionPtr minRuleSoln = isStokes ? minRuleStokes : minRuleConfusion;
    SolutionPtr maxRuleSoln = isStokes ? maxRuleStokes : maxRuleConfusion;
    
    if (minRuleSoln->mesh()->numGlobalDofs() != maxRuleSoln->mesh()->numGlobalDofs()) {
      cout << "subTestCompatibleSolutionsAgree() failure: min rule mesh doesn't have the same # of global dofs as max rule mesh.  For a compatible mesh, these should be identical.\n";
      success = false;
    }
    
    GlobalIndexType cellID = 0; // sample cell
    
    DofOrderingPtr trialOrderingMaxRule = maxRuleSoln->mesh()->getElementType(cellID)->trialOrderPtr;
    DofOrderingPtr trialOrderingMinRule = minRuleSoln->mesh()->getElementType(cellID)->trialOrderPtr;
    
    for (int ref=0; ref<numUniformRefinements; ref++) {
      set<GlobalIndexType> cellIDsMinRule = minRuleSoln->mesh()->getActiveCellIDs();
      minRuleSoln->mesh()->hRefine(cellIDsMinRule, RefinementPattern::regularRefinementPatternQuad());
      
      set<GlobalIndexType> cellIDsMaxRule = maxRuleSoln->mesh()->getActiveCellIDs();
      maxRuleSoln->mesh()->hRefine(cellIDsMaxRule, RefinementPattern::regularRefinementPatternQuad());
    }
    
    minRuleSoln->mesh()->bilinearForm()->setUseSPDSolveForOptimalTestFunctions(true); // trying something... (we're seeing a floating point exception when this is false, presumably due to a division by zero...)
    
    minRuleSoln->solve();

    maxRuleSoln->mesh()->bilinearForm()->setUseSPDSolveForOptimalTestFunctions(true); // trying something... (we're seeing a floating point exception when this is false, presumably due to a division by zero...)
    
    maxRuleSoln->solve();
    
    VarFactory vf = maxRuleSoln->mesh()->bilinearForm()->varFactory();
    
    set<int> varIDs = trialOrderingMaxRule->getVarIDs();
    
    double tol=1e-9; // OK, this is a very loose tolerance...
    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
      int varID = *varIt;
      VarPtr var = vf.trial(varID);
      string varName = var->name();
      
      FunctionPtr maxRuleSolnFxn = Function::solution(var, maxRuleSoln); //Teuchos::rcp( new PreviousSolutionFunction( maxRuleSoln, var ));
      FunctionPtr minRuleSolnFxn = Function::solution(var, minRuleSoln); //Teuchos::rcp( new PreviousSolutionFunction( minRuleSoln, var ));
      
      //    ((PreviousSolutionFunction*) maxRuleSolnFxn.get())->setOverrideMeshCheck(true);
      //    ((PreviousSolutionFunction*) minRuleSolnFxn.get())->setOverrideMeshCheck(true);
      
      double l2diff = (maxRuleSolnFxn - minRuleSolnFxn)->l2norm(maxRuleSoln->mesh());
      
      if (l2diff > tol) {
        cout << "Max and min rule solution for var " << varName << " differ; L^2 norm of difference: " << l2diff << endl;
        success = false;
      }
    }
//    cout << "In subTestCompatibleSolutionsAgree, paused to give a chance to examine the stiffness matrix and RHS files. Press any key to continue.\n";
//    cin.ignore();
//    cin.get();
  }
  return success;
}

bool GDAMinimumRuleTests::testGlobalToLocalToGlobalConsistency() {
  bool success = true;
  
  int H1Order = 3;
  SolutionPtr minRuleSoln = quadMeshSolutionConfusion(true, 1, 2, H1Order);
  
  MeshPtr mesh = minRuleSoln->mesh();
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
  
  GlobalDofAssignmentPtr gda = minRuleSoln->mesh()->globalDofAssignment();

  int numGlobalDofs = gda->globalDofCount();
  FieldContainer<double> globalData(numGlobalDofs);
  for (int i=0; i<numGlobalDofs; i++) {
    globalData(i) = 2*i + 1; // arbitrary data
  }
  FieldContainer<double> globalDataExpected = globalData;
  FieldContainer<double> globalDataActual(numGlobalDofs);
  FieldContainer<double> localData;
  
  Epetra_SerialComm Comm;
  Epetra_BlockMap map(numGlobalDofs, 1, 0, Comm);
  Epetra_Vector globalDataVector(map);
  for (int i=0; i<numGlobalDofs; i++) {
    globalDataVector[i] = globalData(i);
  }
  
  cellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    gda->interpretGlobalData(cellID, localData, globalDataVector,false); // false: don't accumulate
    FieldContainer<GlobalIndexType> globalDofIndices;
    FieldContainer<double> globalDataForCell;
    gda->interpretLocalData(cellID, localData, globalDataForCell, globalDofIndices, false);
    for (int dofOrdinal=0; dofOrdinal < globalDofIndices.size(); dofOrdinal++) {
      GlobalIndexType dofIndex = globalDofIndices(dofOrdinal);
      globalDataActual(dofIndex) = globalDataForCell(dofOrdinal);
    }
  }
  
  double tol=1e-12;
  double maxDiff;
  if (TestSuite::fcsAgree(globalDataActual, globalDataExpected, tol, maxDiff)) {
    //    cout << "global data actual and expected AGREE; max difference is " << maxDiff << endl;
    //    cout << "globalDataActual:\n" << globalDataActual;
  } else {
    cout << "global data actual and expected DISAGREE; max difference is " << maxDiff << endl;
    success = false;
    cout << "Expected:\n" << globalDataExpected;
    cout << "Actual:\n" << globalDataActual;
  }
  
  return success;
}

bool GDAMinimumRuleTests::testLocalInterpretationConsistency() {
  bool success = true;
  int H1Order = 3;
  SolutionPtr minRuleSoln = quadMeshSolutionConfusion(true, 1, 2, H1Order);

  // do a uniform refinement
  MeshPtr mesh = minRuleSoln->mesh();
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
  
  GlobalDofAssignmentPtr gda = minRuleSoln->mesh()->globalDofAssignment();
  
  int globalDofCount = mesh->numGlobalDofs();
  cellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    DofOrderingPtr trialOrdering = gda->elementType(cellID)->trialOrderPtr;
    FieldContainer<double> localData(trialOrdering->totalDofs());
    // initialize with dummy data
    for (int dofOrdinal=0; dofOrdinal<localData.size(); dofOrdinal++) {
      localData(dofOrdinal) = dofOrdinal;
    }
    FieldContainer<double> globalDataExpected(globalDofCount);
    FieldContainer<GlobalIndexType> globalDofIndices;
    FieldContainer<double> globalDataForCell;
    gda->interpretLocalData(cellID, localData, globalDataForCell, globalDofIndices);
    for (int dofOrdinal=0; dofOrdinal < globalDofIndices.size(); dofOrdinal++) {
      GlobalIndexType dofIndex = globalDofIndices(dofOrdinal);
      globalDataExpected(dofIndex) += globalDataForCell(dofOrdinal);
    }
    
    FieldContainer<double> globalDataActual(globalDataExpected.size());
    set<int> varIDs = trialOrdering->getVarIDs();
    
    set<int> indicesForMappedData;
    
    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
      int varID = *varIt;
      int sideCount = trialOrdering->getNumSidesForVarID(varID);
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        BasisPtr basis = trialOrdering->getBasis(varID,sideOrdinal);
        FieldContainer<double> basisData(basis->getCardinality());
        // on the assumption that minimum rule does *not* enforce conformity locally, we can consistently pull basis data
        // from the local data without worrying about "double dipping"
        vector<int> dofIndices = trialOrdering->getDofIndices(varID, sideOrdinal);
        int basisOrdinal = 0;
        for (vector<int>::iterator dofIndexIt = dofIndices.begin(); dofIndexIt != dofIndices.end(); dofIndexIt++, basisOrdinal++) {
          int dofIndex = *dofIndexIt;
          
          if (indicesForMappedData.find(dofIndex) != indicesForMappedData.end()) {
            cout << "Error: local dofIndex " << dofIndex << " belongs to multiple bases.  This is not expected in the design of testLocalInterpretationConsistency().\n";
            success = false;
          }
          
          basisData(basisOrdinal) = localData(dofIndex);
          indicesForMappedData.insert(dofIndex);
        }
        FieldContainer<double> globalDataForBasis;
        FieldContainer<GlobalIndexType> globalDofIndicesForBasis;
        gda->interpretLocalBasisData(cellID, varID, sideOrdinal, basisData, globalDataForBasis, globalDofIndicesForBasis);
        int globalEntryCount = globalDofIndicesForBasis.size();
        for (int entryOrdinal=0; entryOrdinal<globalEntryCount; entryOrdinal++) {
          GlobalIndexType entryIndex = globalDofIndicesForBasis(entryOrdinal);
          globalDataActual(entryIndex) += globalDataForBasis(entryOrdinal);
        }
      }
    }
    
    double tol=1e-12;
    double maxDiff;
    if (TestSuite::fcsAgree(globalDataActual, globalDataExpected, tol, maxDiff)) {
  //    cout << "global data actual and expected AGREE; max difference is " << maxDiff << endl;
  //    cout << "globalDataActual:\n" << globalDataActual;
    } else {
      cout << "global data actual and expected DISAGREE; max difference is " << maxDiff << endl;
      success = false;
    }
}

  return success;
}

bool GDAMinimumRuleTests::testHangingNode() {
  bool success = true;
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2;
  int H1Order = 3;
  
  // exact solution: for now, we just use a linear u, zero p
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u1_exact = x + y;
  FunctionPtr u2_exact = -x - y;
  
  FunctionPtr p_exact = Function::zero();
  
  SolutionPtr soln = quadMeshSolutionStokesExactSolution(true, horizontalCellsInitialMesh, verticalCellsInitialMesh, H1Order,
                                                         u1_exact, u2_exact, p_exact);
  
  MeshPtr mesh = soln->mesh();
  
  set<GlobalIndexType> cellIDs;
  cellIDs.insert(0);
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
  soln->solve();
  
  VarFactory vf = soln->mesh()->bilinearForm()->varFactory();
  VarPtr u1 = vf.fieldVar(S_GDAMinimumRule_U1);
  VarPtr u2 = vf.fieldVar(S_GDAMinimumRule_U2);
  
  FunctionPtr u1_soln = Function::solution(u1, soln);
  FunctionPtr u2_soln = Function::solution(u2, soln);
  
  FunctionPtr u1_err = u1_soln - u1_exact;
  FunctionPtr u2_err = u2_soln - u2_exact;
  
  double tol = 1e-13;
  double u1_err_l2 = u1_err->l2norm(mesh);
  if (u1_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for mesh with hanging node and exactly recoverable solution, u1 error is " << u1_err_l2 << endl;
  }
  
  double u2_err_l2 = u2_err->l2norm(mesh);
  if (u2_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for mesh with hanging node and exactly recoverable solution, u2 error is " << u2_err_l2 << endl;
  }
  
  return success;
}

bool GDAMinimumRuleTests::testHRefinements() {
  bool success = true;
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2;
  int H1Order = 3;
  SolutionPtr soln = quadMeshSolutionConfusion(true, horizontalCellsInitialMesh, verticalCellsInitialMesh, H1Order);
  MeshPtr mesh = soln->mesh();
  int numRefs = 3;
  // this test is mostly about managing to do the h-refinements without failing some assertion along the way
  // Once the refinement is done, we just check that it has the right number of active cells.
  int activeCellCountExpected = horizontalCellsInitialMesh * verticalCellsInitialMesh;
  for (int ref=0; ref<numRefs; ref++) {
    activeCellCountExpected *= 4;
    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
    mesh->hRefine(activeCellIDs, RefinementPattern::regularRefinementPatternQuad());
    int activeCellCount = mesh->getActiveCellIDs().size();
    if (activeCellCount != activeCellCountExpected) {
      cout << "After refinement # " << ref << ", expected " << activeCellCountExpected;
      cout << " active cells, but had " << activeCellCount << endl;
      success = false;
    }
  }
  return success;
}

bool GDAMinimumRuleTests::testMultiCellMesh() {
  bool success = true;
  
  typedef pair<int, int> MeshDimensions;
  typedef pair<MeshDimensions, int> MeshToTest;
  
  vector< MeshToTest > testList;
  
  for (int polyOrder = 3; polyOrder < 5; polyOrder++) { // starting at 3 for now because that's the first one where we are getting a floating point error (divide by zero)
    int horizontalCells = 1, verticalCells = 2;
    MeshToTest meshParams = make_pair( make_pair(horizontalCells, verticalCells), polyOrder+1);
//    testList.push_back(meshParams); // skipping this one for now
    horizontalCells = 4;
    verticalCells = 2;
    meshParams = make_pair( make_pair(horizontalCells, verticalCells), polyOrder+1);
    testList.push_back(meshParams);
  }
  
  for (vector< MeshToTest >::iterator meshParamIt = testList.begin(); meshParamIt != testList.end(); meshParamIt++) {
    MeshToTest meshParams = *meshParamIt;
    MeshDimensions dim = meshParams.first;
    int horizontalCells = dim.first;
    int verticalCells = dim.second;
    int H1Order = meshParams.second;
    
    for (int numRefs = 1; numRefs < 2; numRefs++) {
      cout << "About to run test for " << horizontalCells << " x " << verticalCells;
      cout << ", k=" << H1Order - 1 << " mesh with " << numRefs << " refinements.\n";
      if (! subTestCompatibleSolutionsAgree(horizontalCells, verticalCells, H1Order, numRefs) ) {
        cout << "For unrefined (compatible) " << horizontalCells << " x " << verticalCells;
        cout << " mesh with H1Order = " << H1Order;
        cout << " after " << numRefs << " refinements, max and min rules disagree.\n";
        success = false;
      }
      cout << "Completed test for " << horizontalCells << " x " << verticalCells;
      cout << ", k=" << H1Order - 1 << " mesh with " << numRefs << " refinements.\n";
    }
  }
  return success;
}

bool GDAMinimumRuleTests::testSingleCellMesh() {
  int horizontalCells = 1, verticalCells = 1;
  int numUniformRefinements = 0;
  int H1Order = 4;
  return subTestCompatibleSolutionsAgree(horizontalCells, verticalCells, H1Order, numUniformRefinements);
}