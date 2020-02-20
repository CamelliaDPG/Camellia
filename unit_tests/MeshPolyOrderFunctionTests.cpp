//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MeshPolyOrderFunctionTests
//  Camellia
//
//  Created by Stefan Henneking on 1/15/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "MeshPolyOrderFunction.h"

#include "BasisCache.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( MeshPolyOrderFunction, PolyOrder_values )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    int H1Order = 2;
    vector<int> elemCounts = {2,2};
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, elemCounts, H1Order);
    auto myCellIDs = mesh->cellIDsInPartition();
    auto f = MeshPolyOrderFunction(mesh);

    double expectedValue = H1Order-1.0;
    double tol = 1.0e-14;
    
    for (int cellID : myCellIDs) {
      auto fc = FieldContainer<double>(1,1);
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      f.TFunction<double>::values(fc, basisCache);
      TEST_FLOATING_EQUALITY(expectedValue,fc(0,0),tol);
    }
  }
} // namespace

