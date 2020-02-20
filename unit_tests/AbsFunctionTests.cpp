//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  AbsFunctionTests
//  Camellia
//
//  Created by Stefan Henneking on 1/15/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "AbsFunction.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( AbsFunction, Abs_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f = TFunction<double>::constant(-1.0);
    auto f_abs = AbsFunction(f);
    BasisCachePtr nullBasisCache;
    f_abs.TFunction<double>::values(fc, nullBasisCache);
    double expectedValue = 1.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }

} // namespace

