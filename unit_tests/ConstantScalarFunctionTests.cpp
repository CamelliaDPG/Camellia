//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ConstantScalarFunctionTests
//  Camellia
//
//  Created by Stefan Henneking on 1/13/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "ConstantScalarFunction.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_isZero )
  {
    auto f0 = ConstantScalarFunction<double>(0.0);
    TEST_EQUALITY( f0.isZero(), true );
    //
    auto f1 = ConstantScalarFunction<double>(1.0);
    TEST_EQUALITY( f1.isZero(), false );
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_displayString )
  {
    string str = "UNIT_TEST_ConstantScalarFunction_displayString";
    auto f = ConstantScalarFunction<double>(0.0, str);
    TEST_EQUALITY( str.compare(f.displayString()), 0 );
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_values )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f = ConstantScalarFunction<double>(1.0);
    BasisCachePtr voidBasisCache;
    f.values(fc, voidBasisCache);
    double expectedValue = 1.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_multiply )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f1 = ConstantScalarFunction<double>(1.0);
    auto f2 = ConstantScalarFunction<double>(2.0);
    BasisCachePtr voidBasisCache;
    f1.values(fc, voidBasisCache);
    f2.scalarMultiplyFunctionValues(fc, voidBasisCache);
    double expectedValue = 2.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_divide )
  {
    const int dim0 = 5, dim1 = 2;
    auto fc = FieldContainer<double>(dim0,dim1);
    auto f1 = ConstantScalarFunction<double>(1.0);
    auto f2 = ConstantScalarFunction<double>(2.0);
    BasisCachePtr voidBasisCache;
    f1.values(fc, voidBasisCache);
    f2.scalarDivideFunctionValues(fc, voidBasisCache);
    double expectedValue = 0.5;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }

} // namespace

