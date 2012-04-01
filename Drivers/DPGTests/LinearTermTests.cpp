//
//  LinearTermTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/31/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "LinearTermTests.h"

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
}
  
bool LinearTermTests::testSums() {
  bool success = true;
  
  LinearTermPtr sum = v1 + v2;
  
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
  
  // TODO: check that the sum is correct
  
  return success;
}

bool LinearTermTests::testIntegration() {
  cout << "LinearTermTests::testIntegration() not yet implemented.\n";
  return false;
}
  
std::string LinearTermTests::testSuiteName() {
  return "LinearTermTests";
}