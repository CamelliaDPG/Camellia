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
#include "TrigFunctions.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( SumFunction, DataPathMatches1D1V )
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
      
      FunctionPtr f1, f2;

      {
        FunctionPtr x = Function::xn(1);
        FunctionPtr y = Function::yn(1);
        f1 = x + y;
      }
      
      {
        const double k_x = 0.2;
        FunctionPtr cos_kx = Teuchos::rcp(new Cos_ax(k_x));
        const double gamma = 1e-3;
        f2 = 1. + gamma * cos_kx;
      }
          
      std::vector<FunctionPtr> fxnsToTest {f1, f2};
      
      for (auto fxn : fxnsToTest)
      {
        out << "testing fxn of class " << fxn->getClassName() << " with displayString " << fxn->displayString() << std::endl;
        testFunctionDataPathMatchesViewPath1V(fxn, basisCache, tol, out, success);
      }
    }
  }

  TEUCHOS_UNIT_TEST( SumFunction, DataPathMatches1D3V )
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
      
      FunctionPtr f = x + v_x + v_y + v_z;
      
      testFunctionDataPathMatchesViewPath3V(f, basisCache, tol, out, success);
    }
  }
} // namespace
