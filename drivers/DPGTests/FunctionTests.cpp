//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/9/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "FunctionTests.h"
#include "SpatiallyFilteredFunction.h"
#include "Solution.h"
#include "BasisSumFunction.h"

#include "MeshFactory.h"
#include "StokesFormulation.h"

// "previous solution" value for u -- what Burgers would see, according to InitialGuess.h, in first linear step
class UPrev : public Function {
public:
  UPrev() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = 1 - 2*x;
      }
    }
  }
};


class BoundaryLayerFunction : public SimpleFunction {
  double _eps;
public:
  BoundaryLayerFunction(double eps) {
    _eps = eps;
  }
  double value(double x, double y){
    return exp(x/_eps);
  }
};
void FunctionTests::setup() {
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
  _spectralConfusionMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
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
  vector<int> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  _basisCache->setRefCellPoints(_testPoints);
  
  _basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), cellIDs, true );
}

void FunctionTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testComponentFunction()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVectorFunctionValuesOrdering()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testJacobianOrdering()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testBasisSumFunction()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testValuesDottedWithTensor()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testJumpIntegral()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testIntegrate()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testAdaptiveIntegrate()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testThatLikeFunctionsAgree()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testProductRule()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testQuotientRule()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testPolarizedFunctions()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
}

bool FunctionTests::testBasisSumFunction() {
  bool success = true;
  // on a single-element mesh, the BasisSumFunction should be identical to
  // the Solution with those coefficients

  // define a new mesh: more interesting if we're not on the ref cell
  int spaceDim = 2;
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 2.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  int H1Order = 1, pToAdd = 0;
  int horizontalCells = 1, verticalCells = 1;
  
  // create a pointer to a new mesh:
  MeshPtr spectralConfusionMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                      _confusionBF, H1Order, H1Order+pToAdd);
  
  BCPtr bc = Teuchos::rcp( new BCEasy );
  SolutionPtr soln = Teuchos::rcp( new Solution(spectralConfusionMesh, bc) );
  
  int cellID = 0;
  double tol = 1e-16; // overly restrictive, just for now.
  
  DofOrderingPtr trialSpace = spectralConfusionMesh->getElement(cellID)->elementType()->trialOrderPtr;
  set<int> trialIDs = trialSpace->getVarIDs();
  
  BasisCachePtr volumeCache = BasisCache::basisCacheForCell(spectralConfusionMesh, cellID);
  
  for (set<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    int numSides = trialSpace->getNumSidesForVarID(trialID);
    bool boundaryValued = numSides != 1;
    // note that for volume trialIDs, sideIndex = 0, and numSides = 1…
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      BasisCachePtr sideCache = volumeCache->getSideBasisCache(sideIndex);
      BasisPtr basis = trialSpace->getBasis(trialID, sideIndex);
      int basisCardinality = basis->getCardinality();
      for (int basisOrdinal = 0; basisOrdinal<basisCardinality; basisOrdinal++) {
        FieldContainer<double> basisCoefficients(basisCardinality);
        basisCoefficients(basisOrdinal) = 1.0;
        soln->setSolnCoeffsForCellID(basisCoefficients, cellID, trialID, sideIndex);
        
        VarPtr v = Var::varForTrialID(trialID, spectralConfusionMesh->bilinearForm());
        FunctionPtr solnFxn = Function::solution(v, soln);
        FunctionPtr basisSumFxn = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients, OP_VALUE, boundaryValued) );
        if (!boundaryValued) {
          double l2diff = (solnFxn - basisSumFxn)->l2norm(spectralConfusionMesh);
//          cout << "l2diff = " << l2diff << endl;
          if (l2diff > tol) {
            success = false;
            cout << "testBasisSumFunction: l2diff of " << l2diff << " exceeds tol of " << tol << endl;
            cout << "l2norm of basisSumFxn: " << basisSumFxn->l2norm(spectralConfusionMesh) << endl;
            cout << "l2norm of solnFxn: " << solnFxn->l2norm(spectralConfusionMesh) << endl;
          }
          l2diff = (solnFxn->dx() - basisSumFxn->dx())->l2norm(spectralConfusionMesh);
          //          cout << "l2diff = " << l2diff << endl;
          if (l2diff > tol) {
            success = false;
            cout << "testBasisSumFunction: l2diff of dx() " << l2diff << " exceeds tol of " << tol << endl;
            cout << "l2norm of basisSumFxn->dx(): " << basisSumFxn->dx()->l2norm(spectralConfusionMesh) << endl;
            cout << "l2norm of solnFxn->dx(): " << solnFxn->dx()->l2norm(spectralConfusionMesh) << endl;
          }
          
          // test that the restriction to a side works
          for (int i=0; i<volumeCache->cellTopology().getSideCount(); i++) {
            BasisCachePtr mySideCache = volumeCache->getSideBasisCache(i);
            if (! solnFxn->equals(basisSumFxn, mySideCache, tol)) {
              success = false;
              cout << "testBasisSumFunction: on side 0, l2diff of " << l2diff << " exceeds tol of " << tol << endl;
              reportFunctionValueDifferences(solnFxn, basisSumFxn, mySideCache, tol);
            }
            if (! solnFxn->grad(spaceDim)->equals(basisSumFxn->grad(spaceDim), mySideCache, tol)) {
              success = false;
              cout << "testBasisSumFunction: on side 0, l2diff of dx() " << l2diff << " exceeds tol of " << tol << endl;
              reportFunctionValueDifferences(solnFxn->grad(spaceDim), basisSumFxn->grad(spaceDim), mySideCache, tol);
            }
          }
        } else {
          FieldContainer<double> cellIntegral(1);
          // compute l2 diff of integral along the one side where we can legitimately assert equality:
          FunctionPtr diffFxn = solnFxn - basisSumFxn;
          (diffFxn*diffFxn)->integrate(cellIntegral, sideCache);
          double l2diff = sqrt(cellIntegral(0));
          if (l2diff > tol) {
            success = false;
            cout << "testBasisSumFunction: on side " << sideIndex << ", l2diff of " << l2diff << " exceeds tol of " << tol << endl;
            
            int numCubPoints = sideCache->getPhysicalCubaturePoints().dimension(1);
            FieldContainer<double> solnFxnValues(1,numCubPoints);
            FieldContainer<double> basisFxnValues(1,numCubPoints);
            solnFxn->values(solnFxnValues, sideCache);
            basisSumFxn->values(basisFxnValues, sideCache);
            cout << "solnFxnValues:\n" << solnFxnValues;
            cout << "basisFxnValues:\n" << basisFxnValues;
          } else {
//            cout << "testBasisSumFunction: on side " << sideIndex << ", l2diff of " << l2diff << " is within tol of " << tol << endl;
          }
        }
      }
    }
  }
  
  return success;
}

bool FunctionTests::testThatLikeFunctionsAgree() {
  bool success = true;
  
  FunctionPtr u_prev = Teuchos::rcp( new UPrev );
  
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr beta = e1 * u_prev + Function::constant( e2 );
  
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  
  if (! functionsAgree(e2 * u_prev, 
                       Function::constant( e2 ) * u_prev,
                       _basisCache) ) {
    cout << "two like functions differ...\n";
    success = false;
  }
  
  FunctionPtr e1_f = Function::constant( e1 );
  FunctionPtr e2_f = Function::constant( e2 );
  FunctionPtr one  = Function::constant( 1.0 );
  if (! functionsAgree( Teuchos::rcp( new ProductFunction(e1_f, (e1_f + e2_f)) ), // e1_f * (e1_f + e2_f)
                       one,
                       _basisCache) ) {
    cout << "two like functions differ...\n";
    success = false;
  }
  
  vector<double> e1_div2 = e1;
  e1_div2[0] /= 2.0;
  
  if (! functionsAgree(u_prev_squared_div2, 
                       (e1_div2 * beta) * u_prev,
                       _basisCache) ) {
    cout << "two like functions differ...\n";
    success = false;
  }
  
  if (! functionsAgree(e1 * u_prev_squared_div2, 
                       (e1_div2 * beta * e1) * u_prev,
                       _basisCache) ) {
    cout << "two like functions differ...\n";
    success = false;
  }
  
  if (! functionsAgree(e1 * u_prev_squared_div2 + e2 * u_prev, 
                       (e1_div2 * beta * e1 + Teuchos::rcp( new ConstantVectorFunction( e2 ) )) * u_prev,
                       _basisCache) ) {
    cout << "two like functions differ...\n";
    success = false;
  }
  
  return success;
}

bool FunctionTests::testComponentFunction() {
  FunctionPtr one = Function::constant(1);
  FunctionPtr two = Function::constant(2);
  
  FunctionPtr vector = Function::vectorize(one, two);
  FunctionPtr xPart = Function::xPart(vector);
  FunctionPtr yPart = Function::yPart(vector);
  
  bool success = true;
  if (! functionsAgree(one, xPart, _basisCache)) {
    success = false;
    cout << "xPart != one";
  }
  if (! functionsAgree(two, yPart, _basisCache)) {
    success = false;
    cout << "yPart != two";
  }
  return success;
}

bool FunctionTests::functionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache) {
  if (f2->rank() != f1->rank() ) {
    cout << "f1->rank() " << f1->rank() << " != f2->rank() " << f2->rank() << endl;
    return false;
  }
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
  bool functionsAgree = TestSuite::fcsAgree(f1Values,f2Values,tol,maxDiff);
  if ( ! functionsAgree ) {
    functionsAgree = false;
    cout << "Test failed: f1 and f2 disagree; maxDiff " << maxDiff << ".\n";
    cout << "f1Values: \n" << f1Values;
    cout << "f2Values: \n" << f2Values;
  } else {
//    cout << "f1 and f2 agree!" << endl;
  }
  return functionsAgree;
}

bool FunctionTests::testPolarizedFunctions() {
  bool success = true;
  
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
  if (! functionsAgree(f, x, _basisCache) ) {
    cout << "f != x...\n";
    success = false;
  }
  
  // df/dx == 1
  if (! functionsAgree(df_dx, one, _basisCache) ) {
    cout << "df/dx != 1...\n";
    success = false;
  }
  
  // df/dy == 0
  if (! functionsAgree(df_dy, zero, _basisCache) ) {
    cout << "df/dy != 0...\n";
    success = false;
  }
  
  // take f = r sin theta.  Then: f==y, df/dx == 0, df/dy == 1
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  f = Teuchos::rcp( new PolarizedFunction( x * sin_y ) );
  df_dx = f->dx();
  df_dy = f->dy();
  
  // f == x
  if (! functionsAgree(f, y, _basisCache) ) {
    cout << "f != y...\n";
    success = false;
  }
  
  // df/dx == 0
  if (! functionsAgree(df_dx, zero, _basisCache) ) {
    cout << "df/dx != 0...\n";
    success = false;
  }
  
  // df/dy == 0
  if (! functionsAgree(df_dy, one, _basisCache) ) {
    cout << "df/dy != 1...\n";
    success = false;
  }
  
  // Something a little more complicated: f(x) = x^2
  // take f = r^2 cos^2 theta.  Then: f==x^2, df/dx == 2x, df/dy == 0

  f = Teuchos::rcp( new PolarizedFunction( x * cos_y * x * cos_y ) );
  df_dx = f->dx();
  df_dy = f->dy();
  
  // f == x^2
  if (! functionsAgree(f, x * x, _basisCache) ) {
    cout << "f != x^2...\n";
    success = false;
  }
  
  // df/dx == 2x
  if (! functionsAgree(df_dx, 2 * x, _basisCache) ) {
    cout << "df/dx != 2x...\n";
    success = false;
  }
  
  // df/dy == 0
  if (! functionsAgree(df_dy, zero, _basisCache) ) {
    cout << "df/dy != 0...\n";
    success = false;
  }
  
  return success;
}

bool FunctionTests::testProductRule() {
  bool success = true;
  
  // take f = x^2 * exp(x).  f' = 2 x * exp(x) + f
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  
  FunctionPtr f = x2 * exp_x;
  FunctionPtr f_prime = f->dx();
  
  FunctionPtr f_prime_expected = 2.0 * x * exp_x + f;
  
  if (! functionsAgree(f_prime, f_prime_expected,
                       _basisCache) ) {
    cout << "Product rule: expected and actual derivatives differ...\n";
    success = false;
  }

  return success;
}

bool FunctionTests::testQuotientRule() {
  bool success = true;
  // take f = exp(x) / x^2.  f' = f - 2 * x * exp(x) / x^4
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  
  FunctionPtr f = exp_x / x2;
  FunctionPtr f_prime = f->dx();
  
  FunctionPtr f_prime_expected = f - 2 * x * exp_x / (x2 * x2);
  
  if (! functionsAgree(f_prime, f_prime_expected,
                       _basisCache) ) {
    cout << "Quotient rule: expected and actual derivatives differ...\n";
    success = false;
  }
  
  return success;
  
}

bool FunctionTests::testIntegrate(){
  bool success = true;

  FunctionPtr x = Function::xn(1);
  double value = x->integrate(_spectralConfusionMesh);
  double expectedValue = 0.0; // odd function in x on (-1,1)^2
  double tol = 1e-11;
  if (abs(value-expectedValue)>tol){
    success = false;
    cout << "failed testIntegrate() on function x" << endl;
  }
  
  // now, let's try for the integral of the dot product of vector-valued functions
  FunctionPtr y = Function::yn(1);
  FunctionPtr f1 = 1 * Function::vectorize(x, y); // 1 * to trigger creation of a ProductFunction
  
  value = (f1 * f1)->integrate(_spectralConfusionMesh,1); // enrich cubature to handle quadratics
  expectedValue = 8.0 / 3.0; // integral of x^2 + y^2 on (-1,1)^2
  if (abs(value-expectedValue)>tol){
    success = false;
    cout << "failing testIntegrate() on function (x,y) dot (x,y)" << endl;
  }
  
  return success;
}

bool FunctionTests::testAdaptiveIntegrate(){
  bool success = true;

  // we must create our own basisCache here because _basisCache
  // has had its ref cell points set, which basically means it's
  // opted out of having any help with integration.
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  vector<int> cellIDs;
  cellIDs.push_back(0);
  basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(0), cellIDs, true );
  
  double eps = .1; // 
  FunctionPtr boundaryLayerFunction = Teuchos::rcp( new BoundaryLayerFunction(eps) );
  int numCells = basisCache->cellIDs().size();
  FieldContainer<double> integrals(numCells);
  double quadtol = 1e-2;
  double computedIntegral = boundaryLayerFunction->integrate(_spectralConfusionMesh,quadtol);
  double trueIntegral = (eps*(exp(1/eps) - exp(-1/eps)))*2.0;
  double diff = trueIntegral-computedIntegral;
  double relativeError = abs(diff)/abs(trueIntegral); // relative error
  
  double tol = 1e-2;
  if (relativeError > tol){
    success = false;
    cout << "failing testAdaptiveIntegrate() with computed integral " << computedIntegral << " and true integral " << trueIntegral << endl;
  }
  return success;
}

bool FunctionTests::testJacobianOrdering() {
  bool success = true;
  
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  
  FunctionPtr f = Function::vectorize(y, Function::zero());
  
  // test 1: Jacobian ordering is f_i,j
  int spaceDim = 2;
  int cellID = 0;
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(_spectralConfusionMesh, cellID);
  
  FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  
  FieldContainer<double> expectedValues(numCells, numPoints, spaceDim, spaceDim);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      expectedValues(cellIndex,ptIndex,0,0) = 0;
      expectedValues(cellIndex,ptIndex,0,1) = 1;
      expectedValues(cellIndex,ptIndex,1,0) = 0;
      expectedValues(cellIndex,ptIndex,1,1) = 0;
    }
  }
  
  FieldContainer<double> values(numCells, numPoints, spaceDim, spaceDim);
  f->grad(spaceDim)->values(values, basisCache);
  
  double maxDiff = 0;
  double tol = 1e-14;
  if (! fcsAgree(expectedValues, values, tol, maxDiff)) {
    cout << "expectedValues does not match values in testJacobianOrdering().\n";
    reportFunctionValueDifferences(physicalPoints, expectedValues, values, tol);
    success = false;
  }
  
  // test 2: ordering of VectorizedBasis agrees
  // (actually implemented where it belongs, in Vectorized_BasisTestSuite)
  
  // test 3: ordering of CellTools::getJacobian
  FieldContainer<double> nodes(1,4,2);
  nodes(0,0,0) =  1;
  nodes(0,0,1) = -2;
  nodes(0,1,0) =  1;
  nodes(0,1,1) =  2;
  nodes(0,2,0) = -1;
  nodes(0,2,1) =  2;
  nodes(0,3,0) = -1;
  nodes(0,3,1) = -2;
  
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  int cubDegree = 4;
  BasisCachePtr rotatedCache = Teuchos::rcp( new BasisCache(nodes, quad_4, cubDegree) );
  
  physicalPoints = rotatedCache->getPhysicalCubaturePoints();
  numCells = physicalPoints.dimension(0);
  numPoints = physicalPoints.dimension(1);
  
  FieldContainer<double> expectedJacobian(numCells,numPoints,spaceDim,spaceDim);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      expectedJacobian(cellIndex,ptIndex,0,0) = 0;
      expectedJacobian(cellIndex,ptIndex,0,1) = -1;
      expectedJacobian(cellIndex,ptIndex,1,0) = 2;
      expectedJacobian(cellIndex,ptIndex,1,1) = 0;
    }
  }
  
  FieldContainer<double> jacobianValues = rotatedCache->getJacobian();
  
  maxDiff = 0;
  if (! fcsAgree(expectedJacobian, jacobianValues, tol, maxDiff)) {
    cout << "expectedJacobian does not match jacobianValues in testJacobianOrdering().\n";
    reportFunctionValueDifferences(physicalPoints, expectedJacobian, jacobianValues, tol);
    success = false;
  }
  
  return success;
}

class CellIDFilteredFunction : public Function {
  FunctionPtr _fxn;
  set<int> _cellIDs;
public:
  CellIDFilteredFunction(FunctionPtr fxn, set<int> cellIDs) : Function(fxn->rank()) {
    _fxn = fxn;
    _cellIDs = cellIDs;
  }
  CellIDFilteredFunction(FunctionPtr fxn, int cellID) : Function(fxn->rank()) {
    _fxn = fxn;
    _cellIDs.insert(cellID);
  }
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // not the most efficient implementation
    _fxn->values(values,basisCache);
    vector<int> contextCellIDs = basisCache->cellIDs();
    int cellIndex=0; // keep track of index into values
    
    int entryCount = values.size();
    int numCells = values.dimension(0);
    int numEntriesPerCell = entryCount / numCells;
    
    for (vector<int>::iterator cellIt = contextCellIDs.begin(); cellIt != contextCellIDs.end(); cellIt++) {
      int cellID = *cellIt;
      if (_cellIDs.find(cellID) == _cellIDs.end()) {
        // clear out the associated entries
        for (int j=0; j<numEntriesPerCell; j++) {
          values[cellIndex*numEntriesPerCell + j] = 0;
        }
      }
      cellIndex++;
    }
  }
};

bool FunctionTests::testJumpIntegral() {
  bool success = true;
  double tol = 1e-14;
  
  // define nodes for mesh
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  int H1Order = 1, pToAdd = 0;
  int horizontalCells = 2, verticalCells = 2;
  int numSides = 4;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                _confusionBF, H1Order, H1Order+pToAdd);
  
  FieldContainer<double> points(1,2);
  // southwest center:
  points(0,0) = 0.25; points(0,1) = 0.25;
  vector< Teuchos::RCP<Element> > elements = mesh->elementsForPoints(points);
  
  int swCellID = elements[0]->cellID();
  
  double val = 1.0;
  FunctionPtr valFxn = Function::constant(val);
  FunctionPtr valOnSWCell = Teuchos::rcp( new CellIDFilteredFunction(valFxn,swCellID) );
  
  // the jump for this should be 1 along the two interior edges, each of length 0.5
  double sideLength = 0.5;
  
  int cubEnrichment = 0;
  
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    double actualValue = valOnSWCell->integralOfJump(mesh, swCellID, sideIndex, cubEnrichment);
    double expectedValue;
    if (mesh->boundary().boundaryElement(swCellID, sideIndex)) {
      expectedValue = 0;
    } else {
      double sideParity = mesh->parityForSide(swCellID, sideIndex);
      expectedValue = sideParity * val * sideLength;
    }
    
    double diff = abs(actualValue-expectedValue);
    if (diff > tol) {
      cout << "testJumpFunction(): expected " << expectedValue << " but actualValue was " << actualValue << endl;
      success = false;
    }
  }
  
  return success;
}

bool FunctionTests::testValuesDottedWithTensor() {
  bool success = true;
  
  vector< FunctionPtr > vectorFxns;
  
  double xValue = 3, yValue = 4;
  FunctionPtr simpleVector = Function::vectorize(Function::constant(xValue), Function::constant(yValue));
  vectorFxns.push_back(simpleVector);
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  vectorFxns.push_back( Function::vectorize(x*x, x*y) );
  
  VGPStokesFormulation vgpStokes = VGPStokesFormulation(1.0);
  BFPtr bf = vgpStokes.bf();
  
  int h1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, h1Order);
  
  int cellID=0; // the only cell
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

  for (int i=0; i<vectorFxns.size(); i++) {
    FunctionPtr vectorFxn_i = vectorFxns[i];
    for (int j=0; j<vectorFxns.size(); j++) {
      FunctionPtr vectorFxn_j = vectorFxns[j];
      FunctionPtr dotProduct = vectorFxn_i * vectorFxn_j;
      FunctionPtr expectedDotProduct = vectorFxn_i->x() * vectorFxn_j->x() + vectorFxn_i->y() * vectorFxn_j->y();
      if (! expectedDotProduct->equals(dotProduct, basisCache)) {
        cout << "testValuesDottedWithTensor() failed: expected dot product does not match dotProduct.\n";
        success = false;
        double tol = 1e-14;
        reportFunctionValueDifferences(dotProduct, expectedDotProduct, basisCache, tol);
      }
    }
  }
  
  // now, let's try the same thing, but for a LinearTerm dot product
  VarFactory vf;
  VarPtr v = vf.testVar("v", HGRAD);
  
  DofOrderingPtr dofOrdering = Teuchos::rcp( new DofOrdering );
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  BasisPtr basis = BasisFactory::getBasis(h1Order, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  dofOrdering->addEntry(v->ID(), basis, v->rank());
  
  int numCells = 1;
  int numFields = basis->getCardinality();
  
  for (int i=0; i<vectorFxns.size(); i++) {
    FunctionPtr f_i = vectorFxns[i];
    LinearTermPtr lt_i = f_i * v;
    LinearTermPtr lt_i_x = f_i->x() * v;
    LinearTermPtr lt_i_y = f_i->y() * v;
    for (int j=0; j<vectorFxns.size(); j++) {
      FunctionPtr f_j = vectorFxns[j];
      LinearTermPtr lt_j = f_j * v;
      LinearTermPtr lt_j_x = f_j->x() * v;
      LinearTermPtr lt_j_y = f_j->y() * v;
      FieldContainer<double> values(numCells,numFields,numFields);
      lt_i->integrate(values, dofOrdering, lt_j, dofOrdering, basisCache);
      FieldContainer<double> values_expected(numCells,numFields,numFields);
      lt_i_x->integrate(values_expected,dofOrdering,lt_j_x,dofOrdering,basisCache);
      lt_i_y->integrate(values_expected,dofOrdering,lt_j_y,dofOrdering,basisCache);
      double tol = 1e-14;
      double maxDiff = 0;
      if (!fcsAgree(values, values_expected, tol, maxDiff)) {
        cout << "FunctionTests::testValuesDottedWithTensor: ";
        cout << "dot product and sum of the products of scalar components differ by maxDiff " << maxDiff;
        cout << " in LinearTerm::integrate().\n";
        success = false;
      }
    }
  }
  
//  // finally, let's try the same sort of thing, but now with a vector-valued basis
//  BasisPtr vectorBasisTemp = BasisFactory::getBasis(h1Order, quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD);
//  VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<double, FieldContainer<double> > *)vectorBasisTemp.get(),false);
//
//  BasisPtr compBasis = vectorBasis->getComponentBasis();
//  
//  // create a new v, and a new dofOrdering
//  VarPtr v_vector = vf.testVar("v_vector", VECTOR_HGRAD);
//  dofOrdering = Teuchos::rcp( new DofOrdering );
//  dofOrdering->addEntry(v_vector->ID(), vectorBasis, v_vector->rank());
//
//  DofOrderingPtr dofOrderingComp = Teuchos::rcp( new DofOrdering );
//  dofOrderingComp->addEntry(v->ID(), compBasis, v->rank());
//  
    
  return success;
}

bool FunctionTests::testVectorFunctionValuesOrdering() {
  bool success = true;
  
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr x_vector = Function::vectorize(x, Function::zero());
  
  BasisCachePtr basisCache = BasisCache::parametricQuadCache(10);
  
  FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
  int numCells = points.dimension(0);
  int numPoints = points.dimension(1);
  int spaceDim = points.dimension(2);
  FieldContainer<double> values(numCells,numPoints,spaceDim);
  
  x_vector->values(values, basisCache);
  
//  cout << "(x,0) function values:\n" << values;
  
  double tol = 1e-14;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double xValueExpected = points(cellIndex,ptIndex,0);
      double yValueExpected = 0;
      double xValue = values(cellIndex,ptIndex,0);
      double yValue = values(cellIndex,ptIndex,1);
      double xErr = abs(xValue-xValueExpected);
      double yErr = abs(yValue-yValueExpected);
      if ( (xErr > tol) || (yErr > tol) ) {
        success = false;
        cout << "testVectorFunctionValuesOrdering(): vectorized function values incorrect (presumably out of order).\n";
        cout << "x: " << xValueExpected << " ≠ " << xValue << endl;
        cout << "y: " << yValueExpected << " ≠ " << yValue << endl;
      }
    }
  }
  
  return success;
}

std::string FunctionTests::testSuiteName() {
  return "FunctionTests";
}
