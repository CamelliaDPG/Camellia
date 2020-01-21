//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ExpFunctionTests
//  Camellia
//
//  Created by Stefan Henneking on 1/16/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "TrigFunctions.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( TrigFunctions, Sin_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f = TFunction<double>::constant(1.0);
    auto f_sin = TrigFunctions<double>::sin(f);
    
    BasisCachePtr voidBasisCache;
    f_sin->values(fc, voidBasisCache);
    const double sin1 = 0.8414709848078965;
    const double tol = 1.0e-13;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY( fc[i], sin1, tol );
    }
  }

  TEUCHOS_UNIT_TEST( TrigFunctions, Cos_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f = TFunction<double>::constant(1.0);
    auto f_cos = TrigFunctions<double>::cos(f);
    
    BasisCachePtr voidBasisCache;
    f_cos->values(fc, voidBasisCache);
    const double cos1 = 0.5403023058681397;
    const double tol = 1.0e-13;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY( fc[i], cos1, tol );
    }
  }
} // namespace

