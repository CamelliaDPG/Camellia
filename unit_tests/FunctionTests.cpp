//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/20/15.
//
//

#include "Function.h"

#include "Teuchos_UnitTestHarness.hpp"
namespace {
  TEUCHOS_UNIT_TEST( Function, VectorMultiply )
  {
    FunctionPtr x2 = Function::xn(2);
    FunctionPtr y4 = Function::yn(4);
    vector<double> weight(2);
    weight[0] = 3; weight[1] = 2;
    FunctionPtr g = Function::vectorize(x2,y4);
    double x0 = 2, y0 = 3;
    double expectedValue = weight[0] * x0 * x0 + weight[1] * y0 * y0 * y0 * y0;
    double actualValue = Function::evaluate(g * weight, x0, y0);
    double tol = 1e-14;
    TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
  }
  TEUCHOS_UNIT_TEST( Function, MinAndMaxFunctions )
  {
    FunctionPtr one = Function::constant(1);
    FunctionPtr two = Function::constant(2);
    FunctionPtr minFcn = min(one,two);
    FunctionPtr maxFcn = max(one,two);
    double x0 = 0, y0 = 0;
    double expectedValue = 1.0;
    double actualValue = Function::evaluate(minFcn, x0, y0);
    double tol = 1e-14;
    TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
    expectedValue = 2.0;
    actualValue = Function::evaluate(maxFcn, x0, y0);
    TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
  }
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace
