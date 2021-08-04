//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ProductFunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 5/20/21.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "Function.h"
#include "MonomialFunctions.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( ProductFunction, DataPathMatches1D1V )
  {
    const double tol=1e-15;
    
    const int baseSpaceDim = 1;
    const int numTensorExtrusions = 1;
    std::vector<int> H1Orders(1 + numTensorExtrusions, 1);
    
    // make sure we have variation in x and y
    MeshPtr mesh = makeSingleElementExtrudedTestMesh(baseSpaceDim, numTensorExtrusions, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (const auto & cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      FunctionPtr x = Function::xn(1);
      FunctionPtr y = Function::yn(1);
      
      testFunctionDataPathMatchesViewPath1V(x * y, basisCache, tol, out, success);
    }
  }

  TEUCHOS_UNIT_TEST( ProductFunction, DataPathMatches1D3V )
  {
    const double tol=1e-15;
    
    const int baseSpaceDim = 1;
    const int numTensorExtrusions = 3;
    std::vector<int> H1Orders(1 + numTensorExtrusions, 1);
    
    // make sure we have variation in x and y
    MeshPtr mesh = makeSingleElementExtrudedTestMesh(baseSpaceDim, numTensorExtrusions, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (const auto & cellID : myCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      FunctionPtr x   = Function::xn(1);
      FunctionPtr v_x = MonomialFunction::xi_n(1, 1);
      FunctionPtr v_y = MonomialFunction::xi_n(2, 1);
      FunctionPtr v_z = MonomialFunction::xi_n(3, 1);
      
      FunctionPtr f = x * v_x * v_y * v_z;
      
      testFunctionDataPathMatchesViewPath3V(f, basisCache, tol, out, success);
    }
  }
} // namespace
