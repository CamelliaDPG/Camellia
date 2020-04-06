//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/20/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "ConvectionDiffusionReactionFormulation.h"
#include "MeshFactory.h"
#include "SimpleSolutionFunction.h"
#include "Solution.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  // test variable in H(div) with the operator specified
  void testDivOp(Camellia::EOperator op, Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = 2;
    FunctionPtr beta = Function::constant(std::vector<double>({1.0,1.0}));
    const double epsilon = 1e-2;
    const double alpha   = 1.0;
    ConvectionDiffusionReactionFormulation form(ConvectionDiffusionReactionFormulation::LEAST_SQUARES, spaceDim,
                                                beta, epsilon, alpha);
    int H1Order = 1;
    
    std::vector<int> elemCounts = {2,2};
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {2.0,2.0}, elemCounts, H1Order);
    SolutionPtr solution = Solution::solution(form.bf(), mesh);
    
    VarPtr divVar = form.sigma();
    // sanity check
    TEST_EQUALITY(divVar->space(), Camellia::Space::HDIV);
    
    FunctionPtr exactSolution = Function::vectorize(Function::xn(1), Function::yn(1)); // (x,y)
    FunctionPtr exactOpValue  = Function::op(exactSolution, op);
    
    using Scalar = double;
    std::map< int,TFunctionPtr<Scalar> > exactSolutionMap;
    exactSolutionMap[divVar->ID()] = exactSolution;
    
    const SolutionComponent solutionComponent = STANDARD_SOLUTION;
    const int solutionStage = 0;
    const int solutionOrdinal = solution->solutionOrdinal(solutionComponent, solutionStage);
    solution->projectOntoMesh(exactSolutionMap, solutionOrdinal);
    
    bool weightFluxesBySideParity = false; // not applicable here
    SimpleSolutionFunction<Scalar> simpleSolutionFunction = SimpleSolutionFunction<Scalar>(divVar, solution, weightFluxesBySideParity,
                                                                                           solutionComponent, solutionStage);
    
    FunctionPtr f = Teuchos::rcp(&simpleSolutionFunction, false); // false: does not own memory
    // check that our projection succeeded:
    double l2_error = (f-exactSolution)->l2norm(mesh);
    double tol = 1e-14;
    TEST_COMPARE(l2_error, <, tol);
    
    auto actualOpValue = Function::op(f, op);
    l2_error = (actualOpValue - exactOpValue)->l2norm(mesh);
    TEST_COMPARE(l2_error, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value )
  {
    Camellia::EOperator op = OP_VALUE;
    testDivOp(op, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Div )
  {
    Camellia::EOperator op = OP_DIV;
    testDivOp(op, out, success);
  }
} // namespace
