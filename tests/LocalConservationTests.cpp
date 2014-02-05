#include "LocalConservationTests.h"
#include "CamelliaConfig.h"
#include "CheckConservation.h"
#include "MeshFactory.h"

#ifdef USE_VTK
#include "SolutionExporter.h"
#endif

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
  private:
    double halfWidth;
  public:
    InletBC(double _halfWidth) : Function(0), halfWidth(_halfWidth) {}
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

void OneTermConservationTests::SetUp()
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
  int H1Order = 3, pToAdd = 2;
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
  mesh = MeshFactory::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  
  // v terms:
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_hat, v);
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  f = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  rhs->addTerm( f * v );
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  FunctionPtr u1 = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  bc->addDirichlet(beta_n_u_hat, lBoundary, -u1);

  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
}

void MixedTermConservationTests::SetUp()
{
  double dt = 0.25;

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
  int H1Order = 2, pToAdd = 2;
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

  int horizontalCells = 2, verticalCells = 1;
  
  // create a pointer to a new mesh:
  mesh = MeshFactory::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    
  
  // v terms:
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_hat, v);
  bf->addTerm( invDt*u, v );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v );
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  FunctionPtr u1 = Teuchos::rcp( new InletBC(2.0) );
  bc->addDirichlet(beta_n_u_hat, lBoundary, -u1);

  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
}

TEST_F(OneTermConservationTests, TestStandardDPG)
{
  solution->solve(false);
// #ifdef USE_VTK
//   VTKExporter exporter(solution, mesh, varFactory);
//   exporter.exportSolution("Conservation");
// #endif
  FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
  Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, f, varFactory, mesh);
  // cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
  //   << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
  EXPECT_LT(fluxImbalances[0], 1e-14) << "Maximum flux imbalance is too large";
}

TEST_F(OneTermConservationTests, TestConservativeDPG)
{
  solution->lagrangeConstraints()->addConstraint(beta_n_u_hat == f);
  solution->solve(false);
// #ifdef USE_VTK
//   VTKExporter exporter(solution, mesh, varFactory);
//   exporter.exportSolution("Conservation");
// #endif
  FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
  Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, f, varFactory, mesh);
  // cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
  //   << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
  EXPECT_LT(fluxImbalances[0], 1e-14) << "Maximum flux imbalance is too large";
}

TEST_F(MixedTermConservationTests, TestStandardDPG)
{
  solution->solve(false);
#ifdef USE_VTK
  // VTKExporter exporter(solution, mesh, varFactory);
  // exporter.exportSolution("mixed");
#endif
  FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
  FunctionPtr source = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  source = -invDt * source;
  Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, source, varFactory, mesh, 0);
  // cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
  //   << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
}

TEST_F(MixedTermConservationTests, TestConservativeDPG)
{

  LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(1.0, beta_n_u_hat) );
  LinearTermPtr sourcePart = Teuchos::rcp<LinearTerm>( new LinearTerm(invDt, u) );
  conservedQuantity->addTerm(sourcePart, true);
  // solution->setWriteMatrixToMatrixMarketFile(true, "stiff");
  solution->lagrangeConstraints()->addConstraint(conservedQuantity == f);
  solution->solve(false);
#ifdef USE_VTK
  // VTKExporter exporter(solution, mesh, varFactory);
  // exporter.exportSolution("mixed_conservative");
#endif
  FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
  FunctionPtr source = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  source = -invDt * source;
  Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, source, varFactory, mesh, 0);
  // cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
  //   << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
}
