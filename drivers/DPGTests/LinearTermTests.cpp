//
//  LinearTermTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/31/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "LinearTermTests.h"
#include "BF.h"
#include "IP.h"

#include "IP.h"
#include "BCEasy.h"
#include "RHSEasy.h"

#include "PreviousSolutionFunction.h"

typedef pair< FunctionPtr, VarPtr > LinearSummand;

class Sine_x : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = sin(x);
      }
    }
  }
};


class Cosine_y : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = cos(y);
      }
    }
  }
};

void LinearTermTests::setup() {
  
//  VarPtr v1, v2, v3; // HGRAD members (test variables)
//  VarPtr q1, q2, q3; // HDIV members (test variables)
//  VarPtr u1, u2, u3; // L2 members (trial variables)
//  VarPtr u1_hat, u2_hat; // trace variables
//  VarPtr u3_hat_n; // flux variable
//  
//  FunctionPtr sine_x;

  sine_x = Teuchos::rcp( new Sine_x );
  cos_y = Teuchos::rcp( new Cosine_y );
  
  q1 = varFactory.testVar("q_1", HDIV);
  q2 = varFactory.testVar("q_2", HDIV);
  q3 = varFactory.testVar("q_3", HDIV);
  
  v1 = varFactory.testVar("v_1", HGRAD);
  v2 = varFactory.testVar("v_2", HGRAD);
  v3 = varFactory.testVar("v_3", HGRAD);
  
  u1 = varFactory.fieldVar("u_1", HGRAD);
  u2 = varFactory.fieldVar("u_2", HGRAD);
  u3 = varFactory.fieldVar("u_3", HGRAD);
  
  u1_hat = varFactory.traceVar("\\widehat{u}_1");
  u2_hat = varFactory.traceVar("\\widehat{u}_2");
  
  u3_hat_n = varFactory.fluxVar("\\widehat{u}_3n");
  
  bf = Teuchos::rcp(new BF(varFactory)); // made-up bf for Mesh + previous solution tests
  
  bf->addTerm(u1_hat, q1->dot_normal());
  bf->addTerm(u1, q1->x());
  bf->addTerm(u2, q1->y());
  
  bf->addTerm(u3_hat_n, v1);
  bf->addTerm(u3, v1);
  
//  DofOrderingFactory discreteSpaceFactory(bf);
  
  int polyOrder = 3, testToAdd = 2;
  Teuchos::RCP<shards::CellTopology> quadTopoPtr;
//  quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
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
  int horizontalElements = 2, verticalElements = 2;
  
  mesh = Mesh::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bf, polyOrder+1, polyOrder+1+testToAdd);
    
  ElementTypePtr elemType = mesh->getElement(0)->elementType();
  trialOrder = elemType->trialOrderPtr;
  testOrder = elemType->testOrderPtr;
  
  basisCache = Teuchos::rcp(new BasisCache(elemType, mesh));
  
  vector<int> cellIDs;
  cellIDs.push_back(0); 
  cellIDs.push_back(1);
  cellIDs.push_back(2);
  cellIDs.push_back(3);
  bool createSideCacheToo = true;
  
  basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);
}

void LinearTermTests::teardown() {
  
}

void LinearTermTests::runTests(int &numTestsRun, int &numTestsPassed) { 
  setup();
  if (testBoundaryPlusVolumeTerms()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testSums()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testIntegration()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testEnergyNorm()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}
  
bool LinearTermTests::testSums() {
  bool success = true;
  
  LinearTermPtr sum = v1 + v2;
  
  if (sum->summands().size() != 2) {
    success = false;
    cout << "sum has the wrong number of summands\n";
    return success;
  }
  
  LinearSummand first_summand = sum->summands()[0];
  LinearSummand second_summand = sum->summands()[1];
  
  VarPtr first_var = first_summand.second;
  EOperatorExtended first_op = first_var->op();
  
  VarPtr second_var = second_summand.second;
  EOperatorExtended second_op = second_var->op();
  
  if (first_var->ID() != v1->ID()) {
    success = false;
    cout << "first summand isn't v1.\n";
  }
  
  if (first_var->op() != OP_VALUE) {
    success = false;
    cout << "first op isn't VALUE.\n";
  }
  
  if (second_var->ID() != v2->ID()) {
    success = false;
    cout << "second summand isn't v2 (is named " << second_var->name() << ").\n";
  }
  
  if (second_var->op() != OP_VALUE) {
    success = false;
    cout << "second op isn't VALUE.\n";
  }
  
  // check that sum reports having both varIDs
  if (sum->varIDs().find(v1->ID()) == sum->varIDs().end()) {
    cout << "sum->varIDs() doesn't include v1.\n";
    success = false;
  }
  
  if (sum->varIDs().find(v2->ID()) == sum->varIDs().end()) {
    cout << "sum->varIDs() doesn't include v2.\n";
    success = false;
  }
  
  if (sum->varIDs().size() != 2) {
    cout << "sum->varIDs() doesn't have the expected size (expected 2; is " << sum->varIDs().size() << ").\n";
    success = false;
  }
  
  // TODO: check that the sum is correct
  
  return success;
}

bool checkLTSumConsistency(LinearTermPtr a, LinearTermPtr b, DofOrderingPtr dofOrdering, BasisCachePtr basisCache) {
  double tol = 1e-14;
  
  int numCells = basisCache->cellIDs().size();
  int numDofs = dofOrdering->totalDofs();
  bool forceBoundaryTerm = false;
  FieldContainer<double> aValues(numCells,numDofs), bValues(numCells,numDofs), sumValues(numCells,numDofs);
  a->integrate(aValues,dofOrdering,basisCache,forceBoundaryTerm);
  b->integrate(bValues,dofOrdering,basisCache,forceBoundaryTerm);
  (a+b)->integrate(sumValues, dofOrdering, basisCache, forceBoundaryTerm);
  
  int size = aValues.size();
  
  for (int i=0; i<size; i++) {
    double expectedValue = aValues[i] + bValues[i];
    double diff = abs( expectedValue - sumValues[i] );
    if (diff > tol) {
      return false;
    }
  }
  return true;
}

bool LinearTermTests::testIntegration() {  
  // for now, we just check the consistency: for LinearTerm a = b + c, does a->integrate
  // give the same values as b->integrate + c->integrate ?
  bool success = true;
  
  //  VarPtr v1, v2, v3; // HGRAD members (test variables)
  //  VarPtr q1, q2, q3; // HDIV members (test variables)
  //  VarPtr u1, u2, u3; // L2 members (trial variables)
  //  VarPtr u1_hat, u2_hat; // trace variables
  //  VarPtr u3_hat_n; // flux variable
  //  
  //  FunctionPtr sine_x;
  
  if ( ! checkLTSumConsistency(1 * v1, 1 * v2, testOrder, basisCache) ) {
    cout << "(v1 + v2)->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  if ( ! checkLTSumConsistency(sine_x * v1, 1 * v2, testOrder, basisCache) ) {
    cout << "(sine_x * v1 + v2)->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  if ( ! checkLTSumConsistency(1 * q1->div(), 1 * q2->x(), testOrder, basisCache) ) {
    cout << "(q1->div() + q2->x())->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  if ( ! checkLTSumConsistency(1 * u1, 1 * u2, trialOrder, basisCache) ) {
    cout << "(u1 + u2)->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  if ( ! checkLTSumConsistency(1 * u1, sine_x * u2, trialOrder, basisCache) ) {
    cout << "(u1 + sine_x * u2)->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  // now, same thing, but with boundary-value-only functions in the mix:
  // this next is a fairly complex test; may want to add a more granular one above...
  IPPtr ip = Teuchos::rcp(new IP);
  Teuchos::RCP<RHS> rhs = Teuchos::rcp(new RHSEasy);
  Teuchos::RCP<BC> bc = Teuchos::rcp(new BCEasy);
  SolutionPtr solution = Teuchos::rcp( new Solution(mesh,bc,rhs,ip) );
  // project some functions onto solution, so that something interesting is there:
  FunctionPtr u1_proj = sine_x;
  FunctionPtr u2_proj = cos_y;
  FunctionPtr u3_proj = u1_proj * u2_proj;
  map<int, FunctionPtr> solnToProject;
  solnToProject[u1->ID()] = u1_proj;
  solnToProject[u2->ID()] = u2_proj;
  solnToProject[u3->ID()] = u3_proj;
  solnToProject[u1_hat->ID()] = u1_proj;
  solnToProject[u2_hat->ID()] = u2_proj;
  // u3_hat_n isn't too much like a 'real' bilinear form, in that u3 itself is a scalar
  // this is just a test, so I'm not worried about it...
  solnToProject[u3_hat_n->ID()] = u3_proj;
  
  solution->projectOntoMesh(solnToProject);
  
  LinearTermPtr bfTestFunctional = bf->testFunctional(solution);
  
  // bf->addTerm(u1, q1->x());
  // bf->addTerm(u2, q1->y());
  // bf->addTerm(u3, v1);
  
//  bf->addTerm(u1_hat, q1->dot_normal());
//  bf->addTerm(u3_hat_n, v1);
  
  LinearTermPtr testFunctionalNoBoundaryValues = u1_proj * q1->x() + u2_proj * q1->y() + u3_proj * v1;
  
  FunctionPtr u1_hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u1_hat) );
  FunctionPtr u2_hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u2_hat) );
  FunctionPtr u3_hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u3_hat_n) );
  LinearTermPtr testFunctionalBoundaryValues = u1_hat_prev * q1->dot_normal() + u3_hat_prev * v1;
  
  if ( ! checkLTSumConsistency(testFunctionalNoBoundaryValues, testFunctionalBoundaryValues, 
                               testOrder, basisCache) ) {
    cout << "bfTestFunctional->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  if ( ! checkLTSumConsistency(testFunctionalBoundaryValues, bfTestFunctional - testFunctionalBoundaryValues, 
                               testOrder, basisCache) ) {
    cout << "bfTestFunctional->integrate not consistent with sum of summands integration.\n";
    success = false;
  }
  
  if ( ! checkLTSumConsistency(testFunctionalNoBoundaryValues, bfTestFunctional - testFunctionalNoBoundaryValues, 
                               testOrder, basisCache) ) {
    cout << "bfTestFunctional->integrate not consistent with sum of summands integration.\n";
    success = false;
  }

  return success;
}

bool LinearTermTests::testBoundaryPlusVolumeTerms() {
  bool success = true;
  
  // notion is integration by parts:
  // (div f, v) = < f * n, v > - (f, grad v)
  
  // We perform two subtests for each test: first we try with a particular
  // function substituted for the variable.  Second, we integrate over the
  // basis for the mesh (i.e. we test a whole bunch of functions, whose
  // precise definition is a bit complicated).
  
  /////////////   FIRST TEST  ////////////////
  
  // start simply: define f to be (x, 0)
  // (div f, v) = (1, v)
  // < f * n, v > - (f, grad v) = < x n1, v > - ( x, v->dx() )
  
  FunctionPtr x = Teuchos::rcp( new Xn(1) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr y2 = Teuchos::rcp( new Yn(2) );
  FunctionPtr x3 = Teuchos::rcp( new Xn(3) );
  FunctionPtr y3 = Teuchos::rcp( new Yn(3) );
  
  FunctionPtr vector_fxn = Function::vectorize( x, Function::zero() ); // div of this = 1
  LinearTermPtr lt_v = vector_fxn->div()*v1;
  
  // part a: substitute v1 = y^2
  
  FunctionPtr v1_value = y2;
  map< int, FunctionPtr > var_values;
  var_values[v1->ID()] = v1_value;
  
  double expectedValue = lt_v->evaluate(var_values, false)->integrate(mesh);
  
  FunctionPtr n = Function::normal();
  
  LinearTermPtr ibp = vector_fxn * n * v1 - vector_fxn * v1->grad();
  
  double boundaryIntegralSum = ibp->evaluate(var_values,true)->integrate(mesh);
  double volumeIntegralSum   = ibp->evaluate(var_values,false)->integrate(mesh);
  double actualValue = boundaryIntegralSum + volumeIntegralSum;
  
  double tol = 1e-14;
  if (abs(expectedValue - actualValue)>tol){
    success = false;
  }

  // part b: integrate the bases over each of the cells:
  FieldContainer<double> integrals_expected( mesh->numElements(), testOrder->totalDofs() );
  FieldContainer<double> integrals_actual( mesh->numElements(), testOrder->totalDofs() );
  
  lt_v->integrate(integrals_expected,testOrder,basisCache);
  ibp->integrate(integrals_actual,testOrder,basisCache);
  
  double maxDiff = 0;
  if (! fcsAgree(integrals_actual, integrals_expected, tol, maxDiff) ) {
    cout << "LT integrated by parts does not agree with the original; maxDiff: " << maxDiff << endl;
    success = false;
  }
  
  // just on the odd chance that ordering makes a difference, repeat this test with the opposite order in ibp:
  ibp =  - vector_fxn * v1->grad() + vector_fxn * n * v1;
  ibp->integrate(integrals_actual,testOrder,basisCache);
  
  maxDiff = 0;
  if (! fcsAgree(integrals_actual, integrals_expected, tol, maxDiff) ) {
    cout << "LT integrated by parts does not agree with the original; maxDiff: " << maxDiff << endl;
    success = false;
  }
  
  /////////////   SECOND TEST    ////////////////
  
  vector_fxn = Function::vectorize( x2 / 6.0, x2 * y / 2.0 ); // div of this = x / 3 + x^2 / 2
  
  lt_v = vector_fxn->div()*v1;
  
  // part a: substitute v1 = y^2
  expectedValue = lt_v->evaluate(var_values, false)->integrate(mesh);
  
  // integrating x2v by parts:
  ibp = vector_fxn * n * v1 - vector_fxn * v1->grad();
  boundaryIntegralSum = ibp->evaluate(var_values,true)->integrate(mesh);
  volumeIntegralSum   = ibp->evaluate(var_values,false)->integrate(mesh);
  actualValue = boundaryIntegralSum + volumeIntegralSum;
  
  tol = 1e-14;
  if (abs(expectedValue - actualValue)>tol){
    success = false;
  }
  
  // part b: integrate the bases over each of the cells:
  integrals_expected.initialize(0.0);
  integrals_actual.initialize(0.0);
  
  lt_v->integrate(integrals_expected,testOrder,basisCache);
  ibp->integrate(integrals_actual,testOrder,basisCache);
  
  maxDiff = 0;
  if (! fcsAgree(integrals_actual, integrals_expected, tol, maxDiff) ) {
    cout << "LT integrated by parts does not agree with the original; maxDiff: " << maxDiff << endl;
    success = false;
  }
  
  // just on the odd chance that ordering makes a difference, repeat this test with the opposite order in ibp:
  ibp =  - vector_fxn * v1->grad() + vector_fxn * n * v1;
  ibp->integrate(integrals_actual,testOrder,basisCache);
  
  maxDiff = 0;
  if (! fcsAgree(integrals_actual, integrals_expected, tol, maxDiff) ) {
    cout << "LT integrated by parts does not agree with the original; maxDiff: " << maxDiff << endl;
    success = false;
  }
  
  return success;
}

bool LinearTermTests::testEnergyNorm() {
  bool success = true;
  
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(v1); // L^2 on an HGrad var
  ip->addTerm(q1); // L^2 on Hdiv var

  FunctionPtr one = Function::constant(1);
  LinearTermPtr identity = one*v1; 

  double norm = identity->energyNormTotal(mesh,ip); // should be equal to the sqrt of the measure of the domain [-1,1]^2

  double tol = 1e-15;
  if (abs(norm-2.0)>tol){
    success = false;
  }
  return success;
}

std::string LinearTermTests::testSuiteName() {
  return "LinearTermTests";
}
