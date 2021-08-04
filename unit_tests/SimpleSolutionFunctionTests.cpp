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
#include "PoissonFormulation.h"
#include "SimpleSolutionFunction.h"
#include "Solution.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  void testDevicePathValues(ValueModificationType modType, Teuchos::FancyOStream &out, bool &success)
  {
    const double tol = 1e-15;
    const int spaceDim = 1;
    const int elementWidth = 1;
    const int H1Order = 1;
    const int testSpaceEnrichment = 1;
    bool useConformingTraces = true; // does not matter in 1D
    
    vector<double> dimensions(spaceDim,1.0);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr poissonMesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, {elementWidth}, H1Order, testSpaceEnrichment);
    
    VarPtr u = poissonForm.u();
    const double two = 2.0;
    FunctionPtr two_Function = Function::constant(two);
    SolutionPtr solution = Solution::solution(poissonMesh);
    solution->projectOntoMesh({{u->ID(), two_Function}});
    
    bool weightFluxesBySideParity = false; // does not matter for fields
    // this simple solution function has value 2 everywhere:
    SimpleSolutionFunction<double> two_SimpleSolutionFunction(u, solution, weightFluxesBySideParity);
    
    double startingValue = 3.141592; // value to initialized the input view with
    double expectedValue = startingValue;
    assignValue(expectedValue, modType, two);
    
    const auto & myCellIDs = poissonMesh->cellIDsInPartition();
    for (auto cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(poissonMesh, cellID);
      
      const int oneCell = 1;
      const int numPoints = basisCache->getPhysicalCubaturePointsFC().dimension(1); // points have shape (C,P,D)
      Camellia::FieldContainer<double> values(oneCell, numPoints);
      
      values.initialize(startingValue);
      auto dvValues = values.checkoutDeviceView();
      two_SimpleSolutionFunction.values(dvValues, modType, basisCache);
      
      values.checkinDeviceView(true); // deep copies back to host, so we can do the comparison on host
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        TEST_FLOATING_EQUALITY(expectedValue, values(0,pointOrdinal), tol);
      }
    }
  }
  
  // test variable in H(div) with the operator specified
  void testHDivOp(Camellia::EOperator op, Teuchos::FancyOStream &out, bool &success)
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
    testHDivOp(op, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Div )
  {
    Camellia::EOperator op = OP_DIV;
    testHDivOp(op, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_ASSIGN )
  {
    ValueModificationType modType = ASSIGN;
    testDevicePathValues(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_ADD )
  {
    ValueModificationType modType = ADD;
    testDevicePathValues(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_DIVIDE )
  {
    ValueModificationType modType = DIVIDE;
    testDevicePathValues(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_MULTIPLY )
  {
    ValueModificationType modType = MULTIPLY;
    testDevicePathValues(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_SUBTRACT )
  {
    ValueModificationType modType = SUBTRACT;
    testDevicePathValues(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_MINIMUM )
  {
    ValueModificationType modType = MINIMUM;
    testDevicePathValues(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SimpleSolutionFunction, Value_DevicePathValues_MAXIMUM )
  {
    ValueModificationType modType = MAXIMUM;
    testDevicePathValues(modType, out, success);
  }
} // namespace
