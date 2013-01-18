#include "TransientTests.h"

double dt = 0.5;
int numTimeSteps = 40;
double halfWidth = 1.0;
int H1Order = 3, pToAdd = 2;

class ScalarParamFunction : public Function {
  double _a;
  public:
  ScalarParamFunction(double a) : Function(0){
    _a = a;
  }
  void set_param(double a){
    _a = a;
  }
  double get_param(){
    return _a;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    values.initialize(_a);
  }
};

class LeftBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      return (abs(x) < tol);
    }
};

// boundary value for sigma_n
class InletBC : public Function {
  public:
    InletBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        double xCenter = 0;
        double yCenter = 0;
        int nPts = 0;
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          xCenter += x;
          yCenter += y;
          nPts++;
        }
        xCenter /= nPts;
        yCenter /= nPts;
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (abs(y) <= halfWidth && abs(yCenter) < halfWidth)
            values(cellIndex, ptIndex) = 1.0;
          else
            values(cellIndex, ptIndex) = 0.0;
        }
      }
    }
};

void TransientTests::SetUp()
{
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  beta_n_u_hat = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  u = varFactory.fieldVar("u");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  
  ////////////////////   BUILD MESH   ///////////////////////
  bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  
  meshBoundary(0,0) =  0.0; // x1
  meshBoundary(0,1) = -2.0; // y1
  meshBoundary(1,0) =  4.0;
  meshBoundary(1,1) = -2.0;
  meshBoundary(2,0) =  4.0;
  meshBoundary(2,1) =  2.0;
  meshBoundary(3,0) =  0.0;
  meshBoundary(3,1) =  2.0;

  int horizontalCells = 4, verticalCells = 4;
  
  // create a pointer to a new mesh:
  mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                bf, H1Order, H1Order+pToAdd);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  flowResidual = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  

  FunctionPtr u_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u) );
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    
  
  // v terms:
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_hat, v);

  // transient terms
  bf->addTerm( u, invDt*v );
  rhs->addTerm( u_prev_time * invDt * v );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  FunctionPtr u1 = Teuchos::rcp( new InletBC );
  bc->addDirichlet(beta_n_u_hat, lBoundary, -u1);

  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(prevTimeFlow);
  mesh->registerSolution(flowResidual);

  // ==================== SET INITIAL GUESS ==========================
  double u_free = 0.0;
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()] = Teuchos::rcp( new ConstantScalarFunction(u_free) );

  // prevTimeFlow->projectOntoMesh(functionMap);
  
}

void TransientTests::stepToSteadyState()
{
  ////////////////////   SOLVE & REFINE   ///////////////////////
  int timestepCount = 0;
  double time_tol = 1e-9;
  double L2_time_residual = 1e9;
  // cout << L2_time_residual <<" "<< time_tol << timestepCount << numTimeSteps << endl;
  while((L2_time_residual > time_tol) && (timestepCount < numTimeSteps))
  {
    solution->solve(false);
    // Subtract solutions to get residual
    flowResidual->setSolution(solution);
    flowResidual->addSolution(prevTimeFlow, -1.0);       
    L2_time_residual = flowResidual->L2NormOfSolutionGlobal(u->ID());
    cout << "L2_time_residual " << L2_time_residual << endl;

    prevTimeFlow->setSolution(solution); // reset previous time solution to current time sol
    timestepCount++;
  }
}

// TEST_F(TransientTests, TestConverged)
// {
//   VTKExporter exporter(solution, mesh, varFactory);
//   FunctionPtr u_exact = Teuchos::rcp( new InletBC );
//   FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
//   FunctionPtr u_sqr = (u_soln - u_exact)*(u_soln - u_exact);
//   double L2_error = sqrt(u_sqr->integrate(mesh));
//   EXPECT_LT(L2_error, 1e-5) << "Converged error greater than expected";
//   // VTKExporter exporter(solution, mesh, varFactory);
//   // exporter.exportFunction(u_exact, "u_exact", "u_exact");
//   exporter.exportFunction(u_soln, "u_soln1");
//   // exporter.exportFunction(u_diff, "u_diff", "u_diff");
//   double energyThreshold = 0.2; // for mesh refinements
//   RefinementStrategy refinementStrategy( solution, energyThreshold );
//   refinementStrategy.refine(true); // print to console on commRank 0
//   // solution->solve(false);
//   u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
//   u_sqr = (u_soln - u_exact)*(u_soln - u_exact);
//   exporter.exportFunction(u_soln, "u_soln2");
//   // exporter.exportFunction(u_sqr, "u_sqr");
//   double L2_error_ref = sqrt(u_sqr->integrate(mesh));
//   cout << "L2 before " << L2_error << endl << "L2 after " << L2_error_ref << endl;
// }

TEST_F(TransientTests, TestProjection)
{
  FunctionPtr u_exact = Teuchos::rcp( new InletBC );
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()] = u_exact;

  solution->projectOntoMesh(functionMap);
  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  FunctionPtr u_sqr = (u_prev - u_exact)*(u_prev - u_exact);
  double L2_error_before_ref = sqrt(u_sqr->integrate(mesh));

  bool savePlots = false;
  VTKExporter exporter(solution, mesh, varFactory);
  if (savePlots)
    exporter.exportFunction(u_prev, "u_prev");

  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  vector<int> cellsToRefine;
  cellsToRefine.push_back(1);
  cellsToRefine.push_back(2);
  refinementStrategy.hRefineCells(mesh, cellsToRefine);

  if (savePlots)
    exporter.exportFunction(u_prev, "u_ref");

  u_sqr = (u_prev - u_exact)*(u_prev - u_exact);
  double L2_error_after_ref = sqrt(u_sqr->integrate(mesh));
  double tol = 1e-8;
  EXPECT_NEAR(L2_error_after_ref, L2_error_before_ref, tol) << "Refinement increases error";
}
