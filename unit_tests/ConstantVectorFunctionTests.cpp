//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ConstantVectorFunctionTests
//  Camellia
//
//  Created by Stefan Henneking on 1/15/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "ConstantVectorFunction.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( ConstantVectorFunction, Vector_isZero )
  {
    std::vector<double> value_2D = {0.0,0.0};
    auto f0_2D = ConstantVectorFunction<double>(value_2D);
    TEST_EQUALITY( f0_2D.isZero(), true );
    
    value_2D[1] = 1.0;
    auto f1_2D = ConstantVectorFunction<double>(value_2D);
    TEST_EQUALITY( f1_2D.isZero(), false );
    
    std::vector<double> value_3D = {0.0,0.0,0.0};
    auto f0_3D = ConstantVectorFunction<double>(value_3D);
    TEST_EQUALITY( f0_3D.isZero(), true );
    
    value_3D[2] = 1.0;
    auto f1_3D = ConstantVectorFunction<double>(value_3D);
    TEST_EQUALITY( f1_3D.isZero(), false );
  }

  TEUCHOS_UNIT_TEST( ConstantVectorFunction, Vector_values2D )
  {
    const int dim0 = 5, dim1 = 2, dim2 = 2;
    auto fc = FieldContainer<double>(dim0,dim1,dim2);
    std::vector<double> value = {0.0,1.0};
    auto f = ConstantVectorFunction<double>(value);
    BasisCachePtr voidBasisCache;
    f.values(fc, voidBasisCache);
    double expectedValue0 = 0.0;
    double expectedValue1 = 1.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1*dim2; i += 2) {
      TEST_FLOATING_EQUALITY(expectedValue0,fc[i  ],tol);
      TEST_FLOATING_EQUALITY(expectedValue1,fc[i+1],tol);
    }
  }

  TEUCHOS_UNIT_TEST( ConstantVectorFunction, Vector_values3D )
  {
    const int dim0 = 5, dim1 = 2, dim2 = 3;
    auto fc = FieldContainer<double>(dim0,dim1,dim2);
    std::vector<double> value = {0.0,1.0,2.0};
    auto f = ConstantVectorFunction<double>(value);
    BasisCachePtr voidBasisCache;
    f.values(fc, voidBasisCache);
    double expectedValue0 = 0.0;
    double expectedValue1 = 1.0;
    double expectedValue2 = 2.0;
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1*dim2; i += 3) {
      TEST_FLOATING_EQUALITY(expectedValue0,fc[i  ],tol);
      TEST_FLOATING_EQUALITY(expectedValue1,fc[i+1],tol);
      TEST_FLOATING_EQUALITY(expectedValue2,fc[i+2],tol);
    }
  }

} // namespace

