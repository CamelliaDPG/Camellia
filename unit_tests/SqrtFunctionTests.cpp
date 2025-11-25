//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SqrtFunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/25.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "Function.h"
#include "MonomialFunctions.h"
#include "TrigFunctions.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( SqrtFunction, DataPathMatches1D1V )
  {
    const double tol=1e-15;
    
    const int baseSpaceDim = 1;
    const int numTensorExtrusions = 1;
    std::vector<int> H1Orders(1 + numTensorExtrusions, 5);
    
    // make sure we have variation in x and y
    MeshPtr mesh = makeSingleElementExtrudedTestMesh(baseSpaceDim, numTensorExtrusions, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (const auto & cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      FunctionPtr f1;

      {
        FunctionPtr v = MonomialFunction::xi_n(1, 1);
        f1 = Function::sqrtFunction(v * v + 1.);
      }
          
      std::vector<FunctionPtr> fxnsToTest {f1};
      
      for (auto fxn : fxnsToTest)
      {
        out << "testing fxn of class " << fxn->getClassName() << " with displayString " << fxn->displayString() << std::endl;
        testFunctionDataPathMatchesViewPath1V(fxn, basisCache, tol, out, success);
      }
    }
  }

  TEUCHOS_UNIT_TEST( SqrtFunction, DataPathMatches1D1T1V )
  {
    const double tol=1e-15;
    
    const int baseSpaceDim = 1;
    const int numTensorExtrusions = 2;
    std::vector<int> H1Orders(1 + numTensorExtrusions, 5);
    
    // make sure we have variation in x and y
    MeshPtr mesh = makeSingleElementExtrudedTestMesh(baseSpaceDim, numTensorExtrusions, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (const auto & cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      FunctionPtr v = MonomialFunction::xi_n(2, 1);
      FunctionPtr f = Function::sqrtFunction(v * v + 1.);
      
      testFunctionDataPathMatchesViewPath1T1V(f, basisCache, tol, out, success);
    }
  }

  TEUCHOS_UNIT_TEST( SqrtFunction, DataPathMatches1D3V )
  {
    const double tol=1e-15;
    
    const int baseSpaceDim = 1;
    const int numTensorExtrusions = 3;
    std::vector<int> H1Orders(1 + numTensorExtrusions, 5);
    
    // make sure we have variation in x and y
    MeshPtr mesh = makeSingleElementExtrudedTestMesh(baseSpaceDim, numTensorExtrusions, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (const auto & cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      FunctionPtr vz = MonomialFunction::xi_n(3, 1);
      
      FunctionPtr f = Function::sqrtFunction(vz * vz + 1.);
      
      testFunctionDataPathMatchesViewPath3V(f, basisCache, tol, out, success);
    }
  }
} // namespace
