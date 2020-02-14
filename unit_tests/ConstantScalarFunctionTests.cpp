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
  void testDevicePath(const ValueModificationType modType, Teuchos::FancyOStream &out, bool &success)
  {
    // tests the "device path" values implementation
    const int dim0 = 5, dim1 = 2;
    const double existingValue = 3.141592; // the value already in the view prior to calling values()
    const double constantValue = 1.5;      // function value
    double expectedValue;
    switch (modType)
    {
      case Camellia::ASSIGN:
        expectedValue = constantValue;
        break;
      case Camellia::ADD:
        expectedValue = existingValue + constantValue;
        break;
      case Camellia::SUBTRACT:
        expectedValue = existingValue - constantValue;
        break;
      case Camellia::MULTIPLY:
        expectedValue = existingValue * constantValue;
        break;
      case Camellia::DIVIDE:
        expectedValue = existingValue / constantValue;
        break;
      case Camellia::MINIMUM:
        expectedValue = std::min(existingValue, constantValue);
        break;
      case Camellia::MAXIMUM:
        expectedValue = std::max(existingValue, constantValue);
        break;
    }
    auto fc = FieldContainer<double>(dim0,dim1);
    auto fcView = fc.checkoutDeviceView(false); // false: don't copy from host
    Kokkos::deep_copy( fcView, existingValue);
    
    auto f = ConstantScalarFunction<double>(constantValue);
    BasisCachePtr nullBasisCache;
    f.values(fcView, modType, nullBasisCache); // device path
    fc.checkinDeviceView(true); // true: copy back to host
    double tol = 1.0e-14;
    for (int i = 0; i < dim0*dim1; i++) {
      TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
    }
  }
  
  void testScalarModify(const ValueModificationType modType, Teuchos::FancyOStream &out, bool &success)
  {
    if ((modType != DIVIDE) && (modType != MULTIPLY))
    {
      out << "ERROR: test does not support this modType.\n";
      success = false;
      return;
    }
    for (int fcRank=2; fcRank<=6; fcRank++)
    {
      Teuchos::Array<int> fcDim(fcRank);
      // set first dimensions to 1, last 2 dimensions to 5, 2
      for (int i=0; i<fcRank-2; i++)
      {
        fcDim[i] = 1;
      }
      fcDim[fcRank-2] = 5;
      fcDim[fcRank-1] = 2;
      const double value1 = 3.5;
      const double value2 = 1.4;
      double expectedValue = (modType == MULTIPLY) ? value1 * value2 : value1 / value2;
      auto fc = FieldContainer<double>(fcDim);
      for (int i = 0; i < fc.size(); i++) {
        fc[i] = value1;
      }
      auto f = ConstantScalarFunction<double>(value2);
      BasisCachePtr voidBasisCache;
      if (modType == MULTIPLY)
        f.scalarMultiplyFunctionValues(fc, voidBasisCache);
      else
        f.scalarDivideFunctionValues(fc, voidBasisCache);
      double tol = 1.0e-14;
      for (int i = 0; i < fc.size(); i++) {
        TEST_FLOATING_EQUALITY(expectedValue,fc[i],tol);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_isZero )
  {
    auto f0 = ConstantScalarFunction<double>(0.0);
    TEST_EQUALITY( f0.isZero(), true );
    //
    auto f1 = ConstantScalarFunction<double>(1.0);
    TEST_EQUALITY( f1.isZero(), false );
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, DevicePath_ASSIGN )
  {
    const ValueModificationType modType = ASSIGN;
    testDevicePath(modType, out, success);
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, DevicePath_ADD )
  {
    const ValueModificationType modType = ADD;
    testDevicePath(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConstantScalarFunction, DevicePath_SUBTRACT )
  {
    const ValueModificationType modType = SUBTRACT;
    testDevicePath(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConstantScalarFunction, DevicePath_MULTIPLY )
  {
    const ValueModificationType modType = MULTIPLY;
    testDevicePath(modType, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ConstantScalarFunction, DevicePath_DIVIDE )
  {
    const ValueModificationType modType = DIVIDE;
    testDevicePath(modType, out, success);
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
    testScalarModify(MULTIPLY, out, success);
  }

  TEUCHOS_UNIT_TEST( ConstantScalarFunction, Scalar_divide )
  {
    testScalarModify(DIVIDE, out, success);
  }

} // namespace

