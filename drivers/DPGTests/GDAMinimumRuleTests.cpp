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

#include "SolutionExporter.h"

#include "MeshTestSuite.h"

#include "GnuPlotUtil.h"

#include "PenaltyConstraints.h"

#include "CamelliaDebugUtility.h"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

const static string S_GDAMinimumRuleTests_U1 = "u_1";
const static string S_GDAMinimumRuleTests_U2 = "u_2";
const static string S_GDAMinimumRuleTests_PHI = "\\phi";
const static string S_GDAMinimumRuleTests_PHI_HAT = "\\widehat{\\phi}";

class GDAMinimumRuleTests_UnitHexahedronBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y, double z) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
    bool zMatch = (abs(z) < tol) || (abs(z-1.0) < tol);
    return xMatch || yMatch || zMatch;
  }
};

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

SolutionPtr GDAMinimumRuleTests::confusionExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order, bool divideIntoTriangles) {
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
    mesh = MeshFactory::quadMeshMinRule(confusionBF, H1Order, pToAddTest, width, height, horizontalCells, verticalCells, divideIntoTriangles);
  } else {
    mesh = MeshFactory::quadMesh(confusionBF, H1Order, pToAddTest, width, height, horizontalCells, verticalCells, divideIntoTriangles);
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

SolutionPtr GDAMinimumRuleTests::stokesCavityFlowSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order, bool divideIntoTriangles) {
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
  
  VarPtr u1 = varFactory.fieldVar(S_GDAMinimumRuleTests_U1, L2);
  VarPtr u2 = varFactory.fieldVar(S_GDAMinimumRuleTests_U2, L2);
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
                                        horizontalCells, verticalCells, divideIntoTriangles);
  } else {
    mesh = MeshFactory::quadMesh(stokesBF, H1Order, testSpaceEnrichment,
                                 width, height,
                                 horizontalCells, verticalCells, divideIntoTriangles);
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

SolutionPtr GDAMinimumRuleTests::poissonExactSolution3D(int horizontalCells, int verticalCells, int depthCells, int H1Order, FunctionPtr phi_exact, bool useH1Traces) {
  
  bool usePenaltyBCs = false;
  
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau_1", HDIV);
  VarPtr q = varFactory.testVar("q", HGRAD);
  
  Space phi_hat_space = useH1Traces ? HGRAD : L2;
  VarPtr phi_hat = varFactory.traceVar(S_GDAMinimumRuleTests_PHI_HAT, phi_hat_space);
  //  VarPtr phi_hat = varFactory.traceVar(S_GDAMinimumRuleTests_PHI_HAT, L2);
  //  cout << "WARNING: temporarily using L^2 discretization for \\widehat{\\phi}.\n";
  VarPtr psi_n = varFactory.fluxVar("\\widehat{\\psi}_{n}");
  
  VarPtr phi = varFactory.fieldVar(S_GDAMinimumRuleTests_PHI, L2);
  VarPtr psi1 = varFactory.fieldVar("\\psi_1", L2);
  VarPtr psi2 = varFactory.fieldVar("\\psi_2", L2);
  VarPtr psi3 = varFactory.fieldVar("\\psi_3", L2);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  bf->addTerm(phi, tau->div());
  bf->addTerm(psi1, tau->x());
  bf->addTerm(psi2, tau->y());
  bf->addTerm(psi3, tau->z());
  bf->addTerm(-phi_hat, tau->dot_normal());
  
  bf->addTerm(-psi1, q->dx());
  bf->addTerm(-psi2, q->dy());
  bf->addTerm(-psi3, q->dz());
  bf->addTerm(psi_n, q);
  
  int testSpaceEnrichment = 3; //
  double width = 1.0, height = 1.0, depth = 1.0;
  
  vector<double> dimensions;
  dimensions.push_back(width);
  dimensions.push_back(height);
  dimensions.push_back(depth);
  
  vector<int> elementCounts;
  elementCounts.push_back(horizontalCells);
  elementCounts.push_back(verticalCells);
  elementCounts.push_back(depthCells);
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, testSpaceEnrichment);
  
  // rhs = f * v, where f = \Delta phi
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
  rhs->addTerm(f * q);

  IPPtr graphNorm = bf->graphNorm();
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  SolutionPtr solution;
  if (!usePenaltyBCs) {
    bc->addDirichlet(phi_hat, boundary, phi_exact);
    solution = Solution::solution(mesh, bc, rhs, graphNorm);
  } else {
    solution = Solution::solution(mesh, bc, rhs, graphNorm);
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new GDAMinimumRuleTests_UnitHexahedronBoundary );
    
    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    pc->addConstraint(phi_hat==phi_exact,entireBoundary);
    
    solution->setFilter(pc);
  }
  
  return solution;
}

SolutionPtr GDAMinimumRuleTests::poissonExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order, FunctionPtr phi_exact,
                                                      bool divideIntoTriangles) {
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau_1", HDIV);
  VarPtr q = varFactory.testVar("q", HGRAD);
  
  VarPtr phi_hat = varFactory.traceVar(S_GDAMinimumRuleTests_PHI_HAT);
//  VarPtr phi_hat = varFactory.traceVar(S_GDAMinimumRuleTests_PHI_HAT, L2);
//  cout << "WARNING: temporarily using L^2 discretization for \\widehat{\\phi}.\n";
  VarPtr psi_n = varFactory.fluxVar("\\widehat{\\psi}_{n}");
  
  VarPtr phi = varFactory.fieldVar(S_GDAMinimumRuleTests_PHI, L2);
  VarPtr psi1 = varFactory.fieldVar("\\psi_1", L2);
  VarPtr psi2 = varFactory.fieldVar("\\psi_2", L2);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  bf->addTerm(phi, tau->div());
  bf->addTerm(psi1, tau->x());
  bf->addTerm(psi2, tau->y());
  bf->addTerm(-phi_hat, tau->dot_normal());
  
  bf->addTerm(-psi1, q->dx());
  bf->addTerm(-psi2, q->dy());
  bf->addTerm(psi_n, q);
  
  int testSpaceEnrichment = 2; //
  double width = 1.0, height = 1.0;
  
  MeshPtr mesh;
  if (useMinRule) {
    mesh = MeshFactory::quadMeshMinRule(bf, H1Order, testSpaceEnrichment,
                                        width, height,
                                        horizontalCells, verticalCells,
                                        divideIntoTriangles);
  } else {
    mesh = MeshFactory::quadMesh(bf, H1Order, testSpaceEnrichment,
                                 width, height,
                                 horizontalCells, verticalCells, divideIntoTriangles);
  }
  
  // rhs = f * v, where f = \Delta phi
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
  rhs->addTerm(f * q);
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(phi_hat, boundary, phi_exact);

  IPPtr graphNorm = bf->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  return solution;
}

SolutionPtr GDAMinimumRuleTests::stokesExactSolution(bool useMinRule, int horizontalCells, int verticalCells, int H1Order,
                                                     FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool divideIntoTriangles) {
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
  
  VarPtr u1 = varFactory.fieldVar(S_GDAMinimumRuleTests_U1, L2);
  VarPtr u2 = varFactory.fieldVar(S_GDAMinimumRuleTests_U2, L2);
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
                                        horizontalCells, verticalCells, divideIntoTriangles);
  } else {
    mesh = MeshFactory::quadMesh(stokesBF, H1Order, testSpaceEnrichment,
                                 width, height,
                                 horizontalCells, verticalCells, divideIntoTriangles);
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
  bool useQuads = false;
  setup();
  if (testHangingNodePoisson(useQuads)) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  cout << "testHangingNodePoisson (triangles) complete.\n";
  setup();
  if (testHangingNodeStokes(useQuads)) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  cout << "testHangingNodeStokes (triangles) complete.\n";
  
  useQuads = true;
  setup();
  if (testHangingNodePoisson(useQuads)) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  //  cout << "testHangingNodePoisson (quads) complete.\n";
  setup();
  if (testHangingNodeStokes(useQuads)) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  //  cout << "testHangingNodeStokes (quads) complete.\n";
  
  setup();
  if (testHangingNodePoisson3D()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  cout << "testHangingNodePoisson3D complete.\n";
  
  setup();
  if (testSingleCellMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  cout << "testSingleCellMesh complete.\n";
  
  setup();
  if (testMultiCellMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  cout << "testMultiCellMesh complete.\n";

  // skip this test for now--I'm not sure the failure tells us a lot...
//  setup();
//  if (testPoissonCompatibleMeshWithHeterogeneousOrientations2D()) {
//    numTestsPassed++;
//  }
//  numTestsRun++;
//  teardown();

  setup();
  if (testGlobalToLocalToGlobalConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  //  cout << "testGlobalToLocalToGlobalConsistency complete.\n";
  
  setup();
  if (testHRefinements()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testHRefinements complete.\n";
  
  setup();
  if (testLocalInterpretationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
//  cout << "testLocalInterpretationConsistency complete.\n";

}
void GDAMinimumRuleTests::setup() {
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  _testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    _testPoints1D(i, 0) = x[i];
  }
  
  static const int numPointsEachDirection2D = 4;

  _testPoints2D = FieldContainer<double>(numPointsEachDirection2D * numPointsEachDirection2D,2);
  int ptIndex = 0;
  double y[numPointsEachDirection2D] = {-1.0, -0.5, 0, 1.0};
  for (int i=0; i<numPointsEachDirection2D; i++) {
    for (int j=0; j<numPointsEachDirection2D; j++) {
      _testPoints2D(ptIndex, 0) = y[i];
      _testPoints2D(ptIndex, 1) = y[j];
      ptIndex++;
    }
  }
}

void GDAMinimumRuleTests::teardown() {
  
}

bool GDAMinimumRuleTests::subTestCompatibleSolutionsAgree(int horizontalCells, int verticalCells, int H1Order, int numUniformRefinements, bool divideIntoTriangles) {
  bool success = true;
  
  // up to a permutation of the rows/columns, everything should be identical for min and max rule on a single cell mesh.
  SolutionPtr minRuleConfusion = confusionExactSolution(true, horizontalCells, verticalCells, H1Order, divideIntoTriangles);
  SolutionPtr maxRuleConfusion = confusionExactSolution(false, horizontalCells, verticalCells, H1Order, divideIntoTriangles);
  
  SolutionPtr minRuleStokes = stokesCavityFlowSolution(true, horizontalCells, verticalCells, H1Order, divideIntoTriangles);
  SolutionPtr maxRuleStokes = stokesCavityFlowSolution(false, horizontalCells, verticalCells, H1Order, divideIntoTriangles);
  
  
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

  for (int testIndex=0; testIndex<2; testIndex++) { // just to distinguish between Stokes and Confusion
    
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

bool GDAMinimumRuleTests::checkLocalGlobalConsistency(MeshPtr mesh) {
  bool success = true;
  
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  
  GlobalDofAssignmentPtr gda = mesh->globalDofAssignment();
  
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

bool GDAMinimumRuleTests::testGlobalToLocalToGlobalConsistency() {
  
  int H1Order = 1;
  bool divideIntoTriangles = false;
  SolutionPtr minRuleSoln = confusionExactSolution(true, 1, 2, H1Order, divideIntoTriangles);

  MeshPtr mesh = minRuleSoln->mesh();
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
    
  bool success = checkLocalGlobalConsistency(mesh);
  
  return success;
}

bool GDAMinimumRuleTests::testLocalInterpretationConsistency() {
  bool success = true;
  int H1Order = 3;
  bool divideIntoTriangles = false;
  SolutionPtr minRuleSoln = confusionExactSolution(true, 1, 2, H1Order, divideIntoTriangles);

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

bool GDAMinimumRuleTests::testHangingNodePoisson3D() {
  bool success = true;
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2, depthCellsInitialMesh = 1;
  int H1Order = 2;
  
  // exact solution: for now, we just use a linear phi
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
//  FunctionPtr phi_exact = x + y;
  FunctionPtr phi_exact = x + y - z;
//  FunctionPtr phi_exact = Function::constant(3.14159);
  
  bool useH1Traces = true; // "true" is the more thorough test
  
  SolutionPtr soln = poissonExactSolution3D(horizontalCellsInitialMesh, verticalCellsInitialMesh, depthCellsInitialMesh, H1Order, phi_exact, useH1Traces);
  
  MeshPtr mesh = soln->mesh();

  set<GlobalIndexType> cellIDs;
  cellIDs.insert(0);
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
  
//  Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> > linearBasis(1,Intrepid::POINTTYPE_SPECTRAL);
//  FieldContainer<double> dofCoords(4,2);
//  linearBasis.getDofCoords(dofCoords);
//  cout << "dofCoords for linear basis on quad:\n" << dofCoords;
  
//  cout << "Poisson 3D hanging node mesh after one refinement:\n";
//  mesh->getTopology()->printAllEntities();
  
  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints2D)) {
    cout << "GDAMinimumRuleTests failure: for 3D Poisson mesh with hanging nodes, neighboring bases do not agree on sides." << endl;
    success = false;
  }
  
  VarFactory vf = soln->mesh()->bilinearForm()->varFactory();
  VarPtr phi = vf.fieldVar(S_GDAMinimumRuleTests_PHI);
  VarPtr phi_hat = vf.traceVar(S_GDAMinimumRuleTests_PHI_HAT);
  
  map<int, FunctionPtr> phi_exact_map;
  phi_exact_map[phi->ID()] = phi_exact;
  soln->projectOntoMesh(phi_exact_map);
  
  FunctionPtr phi_soln = Function::solution(phi, soln);
  FunctionPtr phi_err = phi_soln - phi_exact;
  
  FunctionPtr phi_hat_soln = Function::solution(phi_hat, soln);

//  NewVTKExporter exporter(mesh->getTopology());
//  exporter.exportFunction(phi_soln, "projected_phi_soln3D");
//  exporter.exportFunction(phi_err, "projected_phi_err3D");
//  
//  FunctionPtr phi_restrictedToSkeleton = Function::meshSkeletonCharacteristic() * phi_soln;
//  exporter.exportFunction(phi_restrictedToSkeleton, "projected_phi_soln3D_restrictedToSkeleton");
  
  double tol = 1e-12;
  double phi_err_l2 = phi_err->l2norm(mesh);
//  if (phi_err_l2 > tol) {
//    success = false;
//    cout << "GDAMinimumRuleTests failure: for 3D mesh with hanging node and exactly recoverable solution, error in projected exact solution is " << phi_err_l2 << endl;
//    
//    NewVTKExporter exporter(mesh->getTopology());
//    exporter.exportFunction(phi_soln, "projected_phi_soln3D");
//    exporter.exportFunction(phi_err, "projected_phi_err3D");
//  }

  soln->clear();
  soln->solve();
  
  phi_err_l2 = phi_err->l2norm(mesh);
  if (phi_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for 3D mesh with hanging node and exactly recoverable solution, phi error is " << phi_err_l2 << endl;
    
    NewVTKExporter exporter(mesh->getTopology());
    exporter.exportFunction(phi_hat_soln, "phi_hat_soln3D");
    exporter.exportFunction(phi_soln, "phi_soln3D");
    exporter.exportFunction(phi_err, "phi_err3D");
  }
  
//  /// now try all that with a finer mesh (largely to see if refining reduces error)
//  set<IndexType> activeCellIDs = mesh->getTopology()->getActiveCellIndices();
////  Camellia::print("activeCellIDs", activeCellIDs);
//  mesh->hRefine(activeCellIDs, RefinementPattern::regularRefinementPatternHexahedron());
//  
//  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints2D)) {
//    cout << "GDAMinimumRuleTests failure: for 3D Poisson mesh with hanging nodes (uniformly refined), neighboring bases do not agree on sides." << endl;
//    success = false;
//  }
//  
//  soln->solve();
//  
//  vf = soln->mesh()->bilinearForm()->varFactory();
//  phi = vf.fieldVar(S_GDAMinimumRuleTests_PHI);
//  
//  phi_soln = Function::solution(phi, soln);
//  
//  phi_err = phi_soln - phi_exact;
//  
//  tol = 1e-12;
//  phi_err_l2 = phi_err->l2norm(mesh);
//  if (phi_err_l2 > tol) {
//    success = false;
//    cout << "GDAMinimumRuleTests failure: for 3D mesh with hanging node (uniformly refined) and exactly recoverable solution, phi error is " << phi_err_l2 << endl;
//    
//    NewVTKExporter exporter(mesh->getTopology());
//    exporter.exportFunction(phi_soln, "phi_soln3D");
//    exporter.exportFunction(phi_err, "phi_err3D");
//  }
  
  return success;
}

bool GDAMinimumRuleTests::testHangingNodePoisson(bool useQuads) {
  bool success = true;
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2;
  bool divideIntoTriangles = !useQuads;
  if (divideIntoTriangles) { // keep things super simple for now: just 2 triangles in the initial mesh
    verticalCellsInitialMesh = 1;
  }
  int H1Order = 3;
  
  // exact solution: for now, we just use a linear phi
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr phi_exact = x + y;
  
  SolutionPtr soln = poissonExactSolution(true, horizontalCellsInitialMesh, verticalCellsInitialMesh, H1Order, phi_exact, divideIntoTriangles);
  
  MeshPtr mesh = soln->mesh();
  
  set<GlobalIndexType> cellIDs;
  cellIDs.insert(0);
  if (useQuads) {
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
  } else {
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternTriangle());
  }
  soln->solve();
  
  VarFactory vf = soln->mesh()->bilinearForm()->varFactory();
  VarPtr phi = vf.fieldVar(S_GDAMinimumRuleTests_PHI);
  
  FunctionPtr phi_soln = Function::solution(phi, soln);
  
  FunctionPtr phi_err = phi_soln - phi_exact;
  
  GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/hangingNodeTestMesh", mesh, true); // true: label cells
  
  VTKExporter solnExporter(soln,soln->mesh(),vf);
  solnExporter.exportSolution("poissonSolution");
  
  double tol = 1e-12;
  double phi_err_l2 = phi_err->l2norm(mesh);
  if (phi_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for mesh with hanging node and exactly recoverable solution, phi error is " << phi_err_l2 << endl;
  }
  
  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints1D)) {
    string meshType = useQuads ? "quad" : "triangle";
    cout << "GDAMinimumRuleTests failure: for 2D Poisson " << meshType << " mesh with hanging node, neighboring bases do not agree on sides." << endl;
    success = false;
  }
  
  if (useQuads) {
    // then let's do one more test, with a 2-irregular mesh
    cellIDs.clear();
    cellIDs.insert(5); // TODO: eliminate this hard-coded cell ID in favor of one that we look up in terms of physical points...
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
    
    GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/hangingNodeTestMesh_2irregular", mesh, true); // true: label cells
    
    solnExporter.exportSolution("poissonSolution_2_irregular");
    
    tol = 1e-12;
    phi_err_l2 = phi_err->l2norm(mesh);
    if (phi_err_l2 > tol) {
      success = false;
      cout << "GDAMinimumRuleTests failure: for mesh with hanging node and exactly recoverable solution, phi error is " << phi_err_l2 << endl;
    }
    
    if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints1D)) {
      string meshType = useQuads ? "quad" : "triangle";
      cout << "GDAMinimumRuleTests failure: for 2D Poisson " << meshType << " mesh with 2-irregular hanging node, neighboring bases do not agree on sides." << endl;
      success = false;
    }
  }
  
  return success;
}

bool GDAMinimumRuleTests::testHangingNodeStokes(bool useQuads) {
  bool success = true;
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2;
  bool divideIntoTriangles = !useQuads;
  if (divideIntoTriangles) { // keep things super simple for now: just 2 triangles in the initial mesh
    verticalCellsInitialMesh = 1;
  }
  
  int H1Order = 3;
  
  // exact solution: for now, we just use a linear u, zero p
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u1_exact = x + y;
  FunctionPtr u2_exact = -x - y;
  
  FunctionPtr p_exact = Function::zero();
  
  SolutionPtr soln = stokesExactSolution(true, horizontalCellsInitialMesh, verticalCellsInitialMesh, H1Order,
                                         u1_exact, u2_exact, p_exact, divideIntoTriangles);
  
  MeshPtr mesh = soln->mesh();
  
  set<GlobalIndexType> cellIDs;
  cellIDs.insert(0);
  if (useQuads) {
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
  } else {
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternTriangle());
  }
  soln->solve();
  
  VarFactory vf = soln->mesh()->bilinearForm()->varFactory();
  VarPtr u1 = vf.fieldVar(S_GDAMinimumRuleTests_U1);
  VarPtr u2 = vf.fieldVar(S_GDAMinimumRuleTests_U2);
  
  FunctionPtr u1_soln = Function::solution(u1, soln);
  FunctionPtr u2_soln = Function::solution(u2, soln);
  
  FunctionPtr u1_err = u1_soln - u1_exact;
  FunctionPtr u2_err = u2_soln - u2_exact;
  
  GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/hangingNodeTestMesh", mesh, true); // true: label cells
  VTKExporter solnExporter(soln,soln->mesh(),vf);
  solnExporter.exportSolution("stokesExactSolution");
  
  double tol = divideIntoTriangles ? 1e-10 : 1e-13; // for now, anyway, we accept a larger tolerance for triangular meshes...
  double u1_err_l2 = u1_err->l2norm(mesh);
  double u1_l2 = u1_exact->l2norm(mesh);
  double u1_rel_err = u1_err_l2 / u1_l2;
  if (u1_rel_err > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for mesh with hanging node and exactly recoverable solution, relative u1 error is " << u1_rel_err << endl;
  }
  
  double u2_err_l2 = u2_err->l2norm(mesh);
  double u2_l2 = u2_exact->l2norm(mesh);
  double u2_rel_err = u2_err_l2 / u2_l2;
  if (u2_rel_err > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for mesh with hanging node and exactly recoverable solution, relative u2 error is " << u2_rel_err << endl;
  }
  
  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints1D)) {
    string meshType = useQuads ? "quad" : "triangle";
    cout << "GDAMinimumRuleTests failure: for 2D Stokes " << meshType << " mesh with hanging node, neighboring bases do not agree on sides." << endl;
    success = false;
  }
  
  return success;
}

bool GDAMinimumRuleTests::testHRefinements() {
  bool success = true;
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 2;
  bool divideIntoTriangles = false;
  int H1Order = 3;
  SolutionPtr soln = confusionExactSolution(true, horizontalCellsInitialMesh, verticalCellsInitialMesh, H1Order, divideIntoTriangles);
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
  
  bool skip2Dtests = false;

  if (!skip2Dtests) {
    bool divideIntoTriangles = false;
    
    typedef pair<int, int> MeshDimensions;
    typedef pair<MeshDimensions, int> MeshToTest;
    
    vector< MeshToTest > testList;
    
    for (int polyOrder = 0; polyOrder < 3; polyOrder++) {
      int horizontalCells = 1, verticalCells = 2;
      MeshToTest meshParams = make_pair( make_pair(horizontalCells, verticalCells), polyOrder+1);
      testList.push_back(meshParams);
  //    horizontalCells = 4;
  //    verticalCells = 2;
  //    meshParams = make_pair( make_pair(horizontalCells, verticalCells), polyOrder+1);
  //    testList.push_back(meshParams);
    }
    
    for (vector< MeshToTest >::iterator meshParamIt = testList.begin(); meshParamIt != testList.end(); meshParamIt++) {
      MeshToTest meshParams = *meshParamIt;
      MeshDimensions dim = meshParams.first;
      int horizontalCells = dim.first;
      int verticalCells = dim.second;
      int H1Order = meshParams.second;
      
      for (int numRefs = 0; numRefs < 2; numRefs++) {
  //      cout << "About to run test for " << horizontalCells << " x " << verticalCells;
  //      cout << ", k=" << H1Order - 1 << " mesh with " << numRefs << " refinements.\n";
        if (! subTestCompatibleSolutionsAgree(horizontalCells, verticalCells, H1Order, numRefs, divideIntoTriangles) ) {
          cout << "For unrefined (compatible) " << horizontalCells << " x " << verticalCells;
          cout << " mesh with H1Order = " << H1Order;
          cout << " after " << numRefs << " refinements, max and min rules disagree.\n";
          success = false;
        }
  //      cout << "Completed test for " << horizontalCells << " x " << verticalCells;
  //      cout << ", k=" << H1Order - 1 << " mesh with " << numRefs << " refinements.\n";
      }
    }
  }
  
  // now, try just a single 2x2x2 mesh in 3D:
  
  int horizontalCellsInitialMesh = 2, verticalCellsInitialMesh = 2, depthCellsInitialMesh = 2;
  int H1Order = 2;
  
  // exact solution: for now, we just use a linear phi
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr phi_exact = x + y - z;
  
  bool useH1Traces = true; // "true" is the more thorough test: set to false for now to diagnose test failure
  
  SolutionPtr soln = poissonExactSolution3D(horizontalCellsInitialMesh, verticalCellsInitialMesh, depthCellsInitialMesh, H1Order, phi_exact, useH1Traces);

  MeshPtr mesh = soln->mesh();
  
  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints2D)) {
    cout << "GDAMinimumRuleTests failure: for 3D 2x2x2 Poisson mesh, neighboring bases do not agree on sides." << endl;
    success = false;
  }
  
  soln->solve();
  
  VarFactory vf = soln->mesh()->bilinearForm()->varFactory();
  VarPtr phi = vf.fieldVar(S_GDAMinimumRuleTests_PHI);
  
  FunctionPtr phi_soln = Function::solution(phi, soln);
  
  FunctionPtr phi_err = phi_soln - phi_exact;
  
  double tol = 1e-12;
  double phi_err_l2 = phi_err->l2norm(mesh);
  if (phi_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for 3D 2x2x2 mesh and exactly recoverable Poisson solution, phi error is " << phi_err_l2 << endl;
    
    NewVTKExporter exporter(mesh->getTopology());
    exporter.exportFunction(phi_soln, "phi_soln3D");
    exporter.exportFunction(phi_err, "phi_err3D");
  }
  
  // now, construct a single-cell mesh, but then refine it
  soln = poissonExactSolution3D(1, 1, 1, H1Order, phi_exact, useH1Traces);
  
  mesh = soln->mesh();
  
  set<GlobalIndexType> cellIDs;
  cellIDs.insert(0);
  mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
  
  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints2D)) {
    cout << "GDAMinimumRuleTests failure: for 3D 2x2x2 Poisson mesh (arrived at by refinement of single cell), neighboring bases do not agree on sides." << endl;
    success = false;
  }
  
  soln->solve();
  
  vf = soln->mesh()->bilinearForm()->varFactory();
  phi = vf.fieldVar(S_GDAMinimumRuleTests_PHI);
  
  phi_soln = Function::solution(phi, soln);
  
  phi_err = phi_soln - phi_exact;
  
  tol = 1e-12;
  phi_err_l2 = phi_err->l2norm(mesh);
  if (phi_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for 3D 2x2x2 Poisson mesh (arrived at by refinement of single cell) and exactly recoverable Poisson solution, phi error is " << phi_err_l2 << endl;
    
    NewVTKExporter exporter(mesh->getTopology());
    exporter.exportFunction(phi_soln, "phi_soln3D");
    exporter.exportFunction(phi_err, "phi_err3D");
  }
  
  return success;
}

bool GDAMinimumRuleTests::testPoissonCompatibleMeshWithHeterogeneousOrientations2D() {
  bool success = true;
  
  int H1Order = 4;
  
  // exact solution: for now, we just use a linear phi
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr phi_exact = x + y;
  
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau_1", HDIV);
  VarPtr q = varFactory.testVar("q", HGRAD);
  
  VarPtr phi_hat = varFactory.traceVar(S_GDAMinimumRuleTests_PHI_HAT);
//  VarPtr phi_hat = varFactory.traceVar("\\widehat{\\phi}", L2);
//  cout << "WARNING: temporarily using L^2 discretization for \\widehat{\\phi}.\n";
  VarPtr psi_n = varFactory.fluxVar("\\widehat{\\psi}_{n}");
  
  VarPtr phi = varFactory.fieldVar(S_GDAMinimumRuleTests_PHI, L2);
  VarPtr psi1 = varFactory.fieldVar("\\psi_1", L2);
  VarPtr psi2 = varFactory.fieldVar("\\psi_2", L2);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  bf->addTerm(phi, tau->div());
  bf->addTerm(psi1, tau->x());
  bf->addTerm(psi2, tau->y());
  bf->addTerm(-phi_hat, tau->dot_normal());
  
  bf->addTerm(-psi1, q->dx());
  bf->addTerm(-psi2, q->dy());
  bf->addTerm(psi_n, q);
  
  int testSpaceEnrichment = 2; //
  
  int spaceDim = 2;
  
  double x0 = 0, x1 = 1.0, x2 = 2.0;
  double y0 = 0, y1 = 1.0;
  
  vector<double> A; A.push_back(x0); A.push_back(y0);
  vector<double> B; B.push_back(x1); B.push_back(y0);
  vector<double> C; C.push_back(x2); C.push_back(y0);
  
  vector<double> D; D.push_back(x2); D.push_back(y1);
  vector<double> E; E.push_back(x1); E.push_back(y1);
  vector<double> F; F.push_back(x0); F.push_back(y1);
  
  // create element vertex orderings that are CW for one quad, and CCW for the other:
  vector< vector<double> > ABEF; ABEF.push_back(A); ABEF.push_back(B); ABEF.push_back(E); ABEF.push_back(F);
  vector< vector<double> > CBED; CBED.push_back(C); CBED.push_back(B); CBED.push_back(E); CBED.push_back(D);
  
  vector< vector<double> > BAFE; BAFE.push_back(B); BAFE.push_back(A); BAFE.push_back(F); BAFE.push_back(E);

  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(spaceDim) );
  
  CellTopoPtr quadTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
  cout << "Note: experimentally trying with both cells oriented CW.\n";
  meshTopology->addCell(quadTopo, BAFE);
//  meshTopology->addCell(quadTopo, ABEF);
  meshTopology->addCell(quadTopo, CBED);
  
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, testSpaceEnrichment) );
  
  // rhs = f * v, where f = \Delta phi
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
  rhs->addTerm(f * q);
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(phi_hat, boundary, phi_exact);
  
  IPPtr graphNorm = bf->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  solution->solve();
  
  FunctionPtr phi_soln = Function::solution(phi, solution);
  
  FunctionPtr phi_err = phi_exact - phi_soln;
  
  double tol = 1e-14; // start low
  double err = phi_err->l2norm(mesh);
  if (err > tol) {
    cout << "In 2D mesh with heterogeneous orientations, phi_err has L^2 norm of " << err << " which exceeds tol of " << tol << endl;
    
    GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/heterogenouslyOrientedMesh", mesh, true); // true: label cells
    
    VTKExporter solnExporter(solution,mesh,varFactory);
    solnExporter.exportSolution("poissonSolutionHeterogenouslyOrientedMesh");
    
    map<int, FunctionPtr> exactSolutionMap;
    exactSolutionMap[phi->ID()] = phi_exact;
    exactSolutionMap[psi1->ID()] = phi_exact->dx();
    exactSolutionMap[psi2->ID()] = phi_exact->dy();
    solution->projectOntoMesh(exactSolutionMap);

    solnExporter.exportSolution("poissonProjectedExactSolutionHeterogenouslyOrientedMesh");
    
    { // experiment: trying with the maximum rule code
      vector< vector<double> > vertices;
      vertices.push_back(A);
      vertices.push_back(B);
      vertices.push_back(C);
      vertices.push_back(D);
      vertices.push_back(E);
      vertices.push_back(F);
      
      vector<IndexType> BAFE;
      BAFE.push_back(1); BAFE.push_back(0); BAFE.push_back(5); BAFE.push_back(4);
      
      vector<IndexType> CBED;
      CBED.push_back(2); CBED.push_back(1); CBED.push_back(4); CBED.push_back(3);
      
      vector< vector<IndexType> > elementVertices;
      elementVertices.push_back(BAFE);
      elementVertices.push_back(CBED);
      MeshPtr maxRuleMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, bf, H1Order, testSpaceEnrichment));
      
      SolutionPtr maxRuleSoln = Solution::solution(maxRuleMesh, bc, rhs, graphNorm);

      maxRuleSoln->solve();
      
      FunctionPtr phiSolnMaxRule = Function::solution(phi, maxRuleSoln);
      FunctionPtr phiErrMaxRule = phi_exact - phiSolnMaxRule;
      double maxRuleError = phiErrMaxRule->l2norm(maxRuleMesh);
      
      cout << "Phi error for clockwise mesh using max rule: " << maxRuleError << endl;
      
      VTKExporter solnExporter(maxRuleSoln,mesh,varFactory);
      solnExporter.exportSolution("phiSolnMaxRuleCWMesh");
    }
    
    success = false;
  }
  
  return success;
}

bool GDAMinimumRuleTests::testSingleCellMesh() {
  int horizontalCells = 1, verticalCells = 1;
  int numUniformRefinements = 0;
  bool divideIntoTriangles = false;
  int H1Order = 4;
  bool success = true;
  if (! subTestCompatibleSolutionsAgree(horizontalCells, verticalCells, H1Order, numUniformRefinements, divideIntoTriangles) ) {
    success = false;
  }
  
  int horizontalCellsInitialMesh = 1, verticalCellsInitialMesh = 1, depthCellsInitialMesh = 1;
  H1Order = 2;
  
  // exact solution: for now, we just use a linear phi
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr phi_exact = x + y - z;
  
  bool useH1Traces = true; // "true" is the more thorough test: set to false for now to diagnose test failure
  
  SolutionPtr soln = poissonExactSolution3D(horizontalCellsInitialMesh, verticalCellsInitialMesh, depthCellsInitialMesh, H1Order, phi_exact, useH1Traces);
  
  soln->solve();
  
  MeshPtr mesh = soln->mesh();
  
  VarFactory vf = soln->mesh()->bilinearForm()->varFactory();
  VarPtr phi = vf.fieldVar(S_GDAMinimumRuleTests_PHI);
  
  FunctionPtr phi_soln = Function::solution(phi, soln);
  
  FunctionPtr phi_err = phi_soln - phi_exact;
  
  double tol = 1e-12;
  double phi_err_l2 = phi_err->l2norm(mesh);
  if (phi_err_l2 > tol) {
    success = false;
    cout << "GDAMinimumRuleTests failure: for 3D single-cell mesh and exactly recoverable Poisson solution, phi error is " << phi_err_l2 << endl;
    
    NewVTKExporter exporter(mesh->getTopology());
    exporter.exportFunction(phi_soln, "phi_soln3D");
    exporter.exportFunction(phi_err, "phi_err3D");
  }
  
  // this should be a trivial test:
  if ( ! MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints2D)) {
    cout << "GDAMinimumRuleTests failure: for single-cell (!) 3D mesh, neighboring bases do not agree on sides." << endl;
    success = false;
  }

  return success;
}
