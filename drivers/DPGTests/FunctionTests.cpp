//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/9/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "FunctionTests.h"

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

class Exp_x : public SimpleFunction {
public:
  double value(double x, double y) {
    return exp(x);
  }
  FunctionPtr dx() {
    return Teuchos::rcp( new Exp_x );
  }
  FunctionPtr dy() {
    return Function::zero();
  }
};

class Y : public SimpleFunction {
public:
  double value(double x, double y) {
    return y;
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return Teuchos::rcp( new ConstantScalarFunction(1.0) );
  }
};

class Cos_y : public SimpleFunction {
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
};

class Sin_y : public SimpleFunction {
  double value(double x, double y) {
    return sin(y);
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return Teuchos::rcp( new Cos_y );
  }
};

double Cos_y::value(double x, double y) {
  return cos(y);
}
FunctionPtr Cos_y::dx() {
  return Function::zero();
}
FunctionPtr Cos_y::dy() {
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  return - sin_y;
}

class Xn : public SimpleFunction {
  int _n;
public:
  Xn(int n) {
    _n = n;
  }
  double value(double x, double y) {
    double val = 1.0;
    // NOTE: inefficient for large n!
    for (int i=0; i<_n; i++) {
      val *= x;
    }
    return val;
  }
  FunctionPtr dx() {
    if (_n == 0) {
      return Function::zero();
    }
    FunctionPtr x_n_minus = Teuchos::rcp( new Xn(_n-1) );
    return _n * x_n_minus;
  }
  FunctionPtr dy() {
    return Function::zero();
  }  
};

class Yn : public SimpleFunction {
  int _n;
public:
  Yn(int n) {
    _n = n;
  }
  double value(double x, double y) {
    double val = 1.0;
    // NOTE: inefficient for large n!
    for (int i=0; i<_n; i++) {
      val *= y;
    }
    return val;
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    if (_n == 0) {
      return Function::zero();
    }
    FunctionPtr y_n_minus = Teuchos::rcp( new Yn(_n-1) );
    return _n * y_n_minus;
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
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[i];
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
  
}

bool FunctionTests::testThatLikeFunctionsAgree() {
  bool success = true;
  
  FunctionPtr u_prev = Teuchos::rcp( new UPrev );
  
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  
  if (! functionsAgree(e2 * u_prev, 
                       Teuchos::rcp( new ConstantVectorFunction( e2 ) ) * u_prev,
                       _basisCache) ) {
    cout << "two like functions differ...\n";
    success = false;
  }
  
  FunctionPtr e1_f = Teuchos::rcp( new ConstantVectorFunction( e1 ) );
  FunctionPtr e2_f = Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  FunctionPtr one  = Teuchos::rcp( new ConstantScalarFunction( 1.0 ) );
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

std::string FunctionTests::testSuiteName() {
  return "FunctionTests";
}
