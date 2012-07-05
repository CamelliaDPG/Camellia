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

void LinearTermTests::setup() {
  
//  VarPtr v1, v2, v3; // HGRAD members (test variables)
//  VarPtr q1, q2, q3; // HDIV members (test variables)
//  VarPtr u1, u2, u3; // L2 members (trial variables)
//  VarPtr u1_hat, u2_hat; // trace variables
//  VarPtr u3_hat_n; // flux variable
//  
//  FunctionPtr sine_x;

  sine_x = Teuchos::rcp( new Sine_x );
  
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
  
  BFPtr bf = Teuchos::rcp(new BF(varFactory)); // we don't actually *use* the bf -- just for the DofOrderingFactory
  
  DofOrderingFactory discreteSpaceFactory(bf);
  
  int polyOrder = 3, testToAdd = 2;
  Teuchos::RCP<shards::CellTopology> quadTopoPtr;
  quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
  trialOrder = discreteSpaceFactory.trialOrdering(polyOrder, *(quadTopoPtr.get()));
  testOrder = discreteSpaceFactory.testOrdering(polyOrder + testToAdd, *(quadTopoPtr.get()));
  
  ElementTypePtr elemType = Teuchos::rcp( new ElementType( trialOrder, testOrder, quadTopoPtr ) );
  
  basisCache = Teuchos::rcp(new BasisCache(elemType));
  
  // define nodes for "mesh"
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
  // create a pointer to a new mesh:  
  int horizontalCells = 2;
  int verticalCells = 2;

  mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, 
			     verticalCells, bf, polyOrder, 
			     polyOrder+testToAdd, false);
  
  quadPoints.resize(1,4,2); // 1 is the num cells...

  vector<int> cellIDs; cellIDs.push_back(0);
  bool createSideCacheToo = true;
  
  basisCache->setPhysicalCellNodes(quadPoints, cellIDs, createSideCacheToo);
}

void LinearTermTests::teardown() {
  
}

void LinearTermTests::runTests(int &numTestsRun, int &numTestsPassed) { 
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
  cout << "LinearTermTests::testIntegration() not yet implemented.\n";
  
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
  
  return success;
}

bool LinearTermTests::testEnergyNorm() {
  bool success = true;
  
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(v1); // L^2 on an HGrad var
  ip->addTerm(q1); // L^2 on Hdiv var

  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
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
