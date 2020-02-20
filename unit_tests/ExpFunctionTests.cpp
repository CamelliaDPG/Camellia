//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ExpFunctionTests
//  Camellia
//
//  Created by Stefan Henneking on 1/15/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "ExpFunction.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( ExpFunctions, Exp_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f = TFunction<double>::constant(2.0);
    auto f_exp = Exp<double>(f);
    const double x0 = 0.0, x1 = 1.0;
    const int quadratureDegree = 2;
    BasisCachePtr simpleBasisCache = BasisCache::basisCache1D(x0, x1, quadratureDegree);
    f_exp.Function::values(fc, simpleBasisCache);
    const double e2 = 7.3890560989306502;
    const double tol = 1.0e-13;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY( fc[i], e2, tol );
    }
  }

  TEUCHOS_UNIT_TEST( ExpFunctions, Ln_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f = TFunction<double>::constant(2.0);
    auto f_ln = Ln<double>(f);
    const double x0 = 0.0, x1 = 1.0;
    const int quadratureDegree = 2;
    BasisCachePtr simpleBasisCache = BasisCache::basisCache1D(x0, x1, quadratureDegree);
    f_ln.Function::values(fc, simpleBasisCache);
    const double ln2 = 0.6931471805599453;
    const double tol = 1.0e-13;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY( fc[i], ln2, tol );
    }
  }
} // namespace

