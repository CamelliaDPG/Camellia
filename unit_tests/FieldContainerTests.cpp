//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FieldContainerTests
//  Camellia
//
//  Created by Nate Roberts on 10/3/19.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "Camellia_FieldContainer.hpp"
#include "SerialDenseWrapper.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( FieldContainer, AssignmentOperator )
  {
    const int numCells  = 1;
    const int numPoints = 1;
    const int spaceDim  = 1;
    const double x = 1.0;
    Camellia::FieldContainer<double> points(numCells,numPoints,spaceDim);
    points(0,0,0) = x;
    
    Camellia::FieldContainer<double> myPoints = points;
    TEST_EQUALITY(myPoints(0,0,0), x);
    
    auto myPointsDeviceView = myPoints.checkoutDeviceView();
    double xDevice;
    const int cellOrdinal = 0;
    const int d = 0;
    Kokkos::parallel_reduce("x device reduction", numPoints, KOKKOS_LAMBDA (const int pointOrdinal, double& lvalue)
    {
      lvalue += myPointsDeviceView(cellOrdinal,pointOrdinal,d);
    }, xDevice);
    
    TEST_EQUALITY(x, xDevice);
    myPoints.checkinDeviceView(false);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostAllocationRecorded )
  {
    const auto initialCount               = fcAllocationRecord.allocationCount;
    const auto initialBytesAllocatedTotal = fcAllocationRecord.bytesAllocatedTotal;
    const auto initialNetBytesAllocated   = fcAllocationRecord.bytesAllocatedNet;
    
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialBytesAllocatedTotal + newBytesExpected, fcAllocationRecord.bytesAllocatedTotal);
    TEST_EQUALITY(initialNetBytesAllocated   + newBytesExpected, fcAllocationRecord.bytesAllocatedNet);
    TEST_EQUALITY(initialCount + 1, fcAllocationRecord.allocationCount);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, DeviceAllocationRecorded )
  {
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    const auto initialCount               = deviceViewAllocationRecord.allocationCount     + deviceViewFakeAllocationRecord.allocationCount;
    const auto initialBytesAllocatedTotal = deviceViewAllocationRecord.bytesAllocatedTotal + deviceViewFakeAllocationRecord.bytesAllocatedTotal;
    const auto initialNetBytesAllocated   = deviceViewAllocationRecord.bytesAllocatedNet   + deviceViewFakeAllocationRecord.bytesAllocatedNet;
   
    auto dv = values.checkoutDeviceView();  // DeviceView is initialized lazily on checkout
    
    // Camellia::Allocation uses SharedScratchStorage in checkoutMirrorView, so the checkout may not result in a new allocation
    const int newCount = deviceViewAllocationRecord.allocationCount + deviceViewFakeAllocationRecord.allocationCount;
    int newBytesExpected;
    if (newCount == initialCount)
    {
      // reuse case: all counts should match initial
      newBytesExpected = 0;
    }
    else
    {
      // new allocation
      TEST_EQUALITY( initialCount + 1,  newCount );
      newBytesExpected = values.size() * sizeof(Scalar);
    }
    
    const int newBytesAllocatedTotal = deviceViewAllocationRecord.bytesAllocatedTotal + deviceViewFakeAllocationRecord.bytesAllocatedTotal;
    const int newNetBytesAllocated   = deviceViewAllocationRecord.bytesAllocatedNet + deviceViewFakeAllocationRecord.bytesAllocatedNet;
    TEST_EQUALITY(initialBytesAllocatedTotal + newBytesExpected, newBytesAllocatedTotal);
    TEST_EQUALITY(initialNetBytesAllocated   + newBytesExpected, newNetBytesAllocated  );
        
    values.checkinDeviceView(false); // false: don't deep copy to host
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, DeviceToHostDeepCopyRecorded )
  {
    const auto initialCount      = deviceToHostDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = deviceToHostDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(false); // false: don't deep copy from host to device (unnecessary here)
    values.checkinDeviceView(true); // true: do deep copy from device to host
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, deviceToHostDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, deviceToHostDeepCopyRecord.deepCopyCount);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, GetDeviceViewAllocatesSeparateStorage )
  {
    // test that Camellia::Allocation<Scalar>::allocateDeviceViewMatchingView() actually allocates new storage, rather than returning a separate view object pointing to same data
    // TODO: create AllocationTests.cpp, move this test there, and put it in an "Allocation" test group
    const double tol = 1e-15;
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    auto dvValues = values.checkoutDeviceView();
    Kokkos::deep_copy(dvValues, 1.0);
    auto dvValuesScratch = Camellia::Allocation<Scalar>::allocateDeviceViewMatchingView(dvValues);
    Kokkos::deep_copy(dvValuesScratch, 2.0); // should not change dvValues
    values.checkinDeviceView(true); // deep copies dvValues back to host
    
    double expectedValue = 1.0;
    for (int i=0; i<dim0; i++)
    {
      TEST_FLOATING_EQUALITY(expectedValue, values[i], tol);
    }
    Camellia::Allocation<Scalar>::recordDeviceViewDeallocation(dvValuesScratch);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostToDeviceDeepCopyRecordedConstCase )
  {
    const auto initialCount      = hostToDeviceDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = hostToDeviceDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    const Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(); // first checkout should copy
    values.checkinDeviceView(false);  // false: don't deep copy to host
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, hostToDeviceDeepCopyRecord.deepCopyCount);
    
    auto dvSecondCheckout = values.checkoutDeviceView(); // no copy here
    values.checkinDeviceView(false); // false: don't deep copy to host
    
    // the counts should be the same as after the first checkout
    TEST_EQUALITY(initialByteCount + newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, hostToDeviceDeepCopyRecord.deepCopyCount);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostToDeviceDeepCopyRecorded )
  {
    const auto initialCount      = hostToDeviceDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = hostToDeviceDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(true); // true: deep copy
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, hostToDeviceDeepCopyRecord.deepCopyCount);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, ViewKeptInSync )
  {
    Camellia::FieldContainer<double> testContainer(3,3,3);
    testContainer[4] = 300.0;
    testContainer.resize(3);
    
    // TODO: finish this test
  }
} // namespace
