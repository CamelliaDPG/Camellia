#include "FunctionTests.h"
#include "MeshFactory.h"

void FunctionTests::SetUp()
{
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
  
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  double eps = 1e-2;
  
  // standard confusion bilinear form
  _confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  _confusionBF->addTerm(sigma1 / eps, tau->x());
  _confusionBF->addTerm(sigma2 / eps, tau->y());
  _confusionBF->addTerm(u, tau->div());
  _confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  _confusionBF->addTerm( sigma1, v->dx() );
  _confusionBF->addTerm( sigma2, v->dy() );
  _confusionBF->addTerm( beta_const * u, - v->grad() );
  _confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
    
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
  int H1Order = 1, pToAdd = 0;
  int horizontalCells = 1, verticalCells = 1;
  
  // create a pointer to a new mesh:
  _spectralConfusionMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                               _confusionBF, H1Order, H1Order+pToAdd);
  
  // some 2D test points:
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {-1.0,-0.8,-0.6,-.4,-.2,0,0.2,0.4,0.6,0.8};
  double y[NUM_POINTS_1D] = {-0.8,-0.6,-.4,-.2,0,0.2,0.4,0.6,0.8,1.0};
  
  _testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[j];
    }
  }
  
  _elemType = _spectralConfusionMesh->getElement(0)->elementType();
  vector<GlobalIndexType> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  _basisCache->setRefCellPoints(_testPoints);
  
  _basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), cellIDs, true );
}

void FunctionTests::checkFunctionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache) {
  ASSERT_EQ(f1->rank(), f2->rank())
    << "f1->rank() " << f1->rank() << " != f2->rank() " << f2->rank() << endl;

  int rank = f1->rank();
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  Teuchos::Array<int> dim;
  dim.append(numCells);
  dim.append(numPoints);
  for (int i=0; i<rank; i++) {
    dim.append(spaceDim);
  }
  FieldContainer<double> f1Values(dim);
  FieldContainer<double> f2Values(dim);
  f1->values(f1Values,basisCache);
  f2->values(f2Values,basisCache);
  
  double tol = 1e-14;
  double maxDiff;
  EXPECT_TRUE(fcsAgree(f1Values,f2Values,tol,maxDiff))
    << "Test failed: f1 and f2 disagree; maxDiff " << maxDiff << ".\n"
    << "f1Values: \n" << f1Values
    << "f2Values: \n" << f2Values;
}

TEST_F(FunctionTests, TestThatLikeFunctionsAgree)
{
  FunctionPtr u_prev = Teuchos::rcp( new UPrev );
  
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  
  checkFunctionsAgree(e2 * u_prev, 
      Teuchos::rcp( new ConstantVectorFunction( e2 ) ) * u_prev,
      _basisCache);
  
  FunctionPtr e1_f = Teuchos::rcp( new ConstantVectorFunction( e1 ) );
  FunctionPtr e2_f = Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  FunctionPtr one  = Teuchos::rcp( new ConstantScalarFunction( 1.0 ) );
  checkFunctionsAgree( Teuchos::rcp( new ProductFunction(e1_f, (e1_f + e2_f)) ), // e1_f * (e1_f + e2_f)
      one,
      _basisCache);
  
  vector<double> e1_div2 = e1;
  e1_div2[0] /= 2.0;
  
  checkFunctionsAgree(u_prev_squared_div2, 
      (e1_div2 * beta) * u_prev,
      _basisCache);
  
  checkFunctionsAgree(e1 * u_prev_squared_div2, 
      (e1_div2 * beta * e1) * u_prev,
      _basisCache);
  
  checkFunctionsAgree(e1 * u_prev_squared_div2 + e2 * u_prev, 
      (e1_div2 * beta * e1 + Teuchos::rcp( new ConstantVectorFunction( e2 ) )) * u_prev,
      _basisCache);
}

TEST_F(FunctionTests, TestPolarizedFunctions)
{
  // take f = r cos theta.  Then: f==x, df/dx == 1, df/dy == 0
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
  
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1) );
  FunctionPtr zero = Function::zero();
  
  FunctionPtr f = Teuchos::rcp( new PolarizedFunction( x * cos_y ) );
  
  FunctionPtr df_dx = f->dx();
  FunctionPtr df_dy = f->dy();
  
  // f == x
  checkFunctionsAgree(f, x, _basisCache);
  
  // df/dx == 1
  checkFunctionsAgree(df_dx, one, _basisCache);
  
  // df/dy == 0
  checkFunctionsAgree(df_dy, zero, _basisCache);
  
  // take f = r sin theta.  Then: f==y, df/dx == 0, df/dy == 1
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  f = Teuchos::rcp( new PolarizedFunction( x * sin_y ) );
  df_dx = f->dx();
  df_dy = f->dy();
  
  // f == x
  checkFunctionsAgree(f, y, _basisCache);
  
  // df/dx == 0
  checkFunctionsAgree(df_dx, zero, _basisCache);
  
  // df/dy == 1
  checkFunctionsAgree(df_dy, one, _basisCache);
  
  // Something a little more complicated: f(x) = x^2
  // take f = r^2 cos^2 theta.  Then: f==x^2, df/dx == 2x, df/dy == 0

  f = Teuchos::rcp( new PolarizedFunction( x * cos_y * x * cos_y ) );
  df_dx = f->dx();
  df_dy = f->dy();
  
  // f == x^2
  checkFunctionsAgree(f, x * x, _basisCache);
  
  // df/dx == 2x
  checkFunctionsAgree(df_dx, 2 * x, _basisCache);
  
  // df/dy == 0
  checkFunctionsAgree(df_dy, zero, _basisCache);
}

TEST_F(FunctionTests, TestProductRule)
{
  // take f = x^2 * exp(x).  f' = 2 x * exp(x) + f
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  
  FunctionPtr f = x2 * exp_x;
  FunctionPtr f_prime = f->dx();
  
  FunctionPtr f_prime_expected = 2.0 * x * exp_x + f;
  
  checkFunctionsAgree(f_prime, f_prime_expected,
      _basisCache);
}

TEST_F(FunctionTests, TestQuotientRule)
{
  // take f = exp(x) / x^2.  f' = f - 2 * x * exp(x) / x^4
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  
  FunctionPtr f = exp_x / x2;
  FunctionPtr f_prime = f->dx();
  
  FunctionPtr f_prime_expected = f - 2 * x * exp_x / (x2 * x2);
  
  checkFunctionsAgree(f_prime, f_prime_expected,
      _basisCache);
}
