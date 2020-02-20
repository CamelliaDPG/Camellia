//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MinMaxFunctionsTests
//  Camellia
//
//  Created by Stefan Henneking on 1/15/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "MinMaxFunctions.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( MinMaxFunctions, Min_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f1 = TFunction<double>::constant(1.0);
    auto f2 = TFunction<double>::constant(2.0);
    auto f_min = MinFunction(f1,f2);
    BasisCachePtr nullBasisCache;
    f_min.Function::values(fc, nullBasisCache);
    double expectedValue = 1.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }

  TEUCHOS_UNIT_TEST( MinMaxFunctions, Max_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f1 = TFunction<double>::constant(1.0);
    auto f2 = TFunction<double>::constant(2.0);
    auto f_max = MaxFunction(f1,f2);
    BasisCachePtr nullBasisCache;
    f_max.Function::values(fc, nullBasisCache);
    double expectedValue = 2.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }
} // namespace

