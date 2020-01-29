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

#include "Epetra_SerialDenseMatrix.h"
#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "Camellia_FieldContainer.hpp"
#include "SerialDenseWrapper.h"

using namespace Camellia;

namespace
{
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
    
    const auto initialCount               = deviceViewAllocationRecord.allocationCount;
    const auto initialBytesAllocatedTotal = deviceViewAllocationRecord.bytesAllocatedTotal;
    const auto initialNetBytesAllocated   = deviceViewAllocationRecord.bytesAllocatedNet;
   
    auto dv = values.checkoutDeviceView();  // DeviceView is initialized lazily on checkout
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialBytesAllocatedTotal + newBytesExpected, deviceViewAllocationRecord.bytesAllocatedTotal);
    TEST_EQUALITY(initialNetBytesAllocated   + newBytesExpected, deviceViewAllocationRecord.bytesAllocatedNet);
    TEST_EQUALITY(initialCount + 1, deviceViewAllocationRecord.allocationCount);
    
    values.checkinDeviceView();
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
    // test that FieldContainer's getDeviceView actually allocates new storage, rather than returning a separate view object pointing to same data
    const double tol = 1e-15;
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    auto dvValues = values.checkoutDeviceView();
    Kokkos::deep_copy(dvValues, 1.0);
    auto dvValuesScratch = Camellia::FieldContainer<Scalar>::getDeviceView(dvValues);
    Camellia::FieldContainer<Scalar>::recordDeviceViewDeallocation(dvValuesScratch);
    Kokkos::deep_copy(dvValuesScratch, 2.0); // should not change dvValues
    values.checkinDeviceView(); // deep copies dvValues back to host
    
    double expectedValue = 1.0;
    for (int i=0; i<dim0; i++)
    {
      TEST_FLOATING_EQUALITY(expectedValue, values[i], tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostToDeviceDeepCopyRecordedConstCase )
  {
    const auto initialCount      = hostToDeviceDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = hostToDeviceDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    const Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(); // first checkout should copy
    values.checkinDeviceView();
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, hostToDeviceDeepCopyRecord.deepCopyCount);
    
    auto dvSecondCheckout = values.checkoutDeviceView(); // no copy here
    values.checkinDeviceView();
    
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
  
  
  TEUCHOS_UNIT_TEST( FieldContainer, convertSDMToFC_DataOwnership )
  {
    // here, we are attempting to isolate an issue in which (it appears) the optimized build
    // does something different with respect to data ownership during a call to solveSPDSystemMultipleRHS()
    // -- the optimized version does not have the final FieldContainer owning the data, even though it was
    // produced via a copy.  The somewhat convoluted construction below is an artifact of trying to replicate the
    // code path that leads to this issue.
    const int M = 40;
    const int N = 15;
    Epetra_SerialDenseMatrix dataSDM(M,N);
    
    Teuchos::Array<int> dataDim(2);
    dataDim[0] = N;
    dataDim[1] = M;
    
    Camellia::FieldContainer<double> data(1,N,M);
    for (int i=0; i<M; i++)
    {
      for (int j=0; j<N; j++)
      {
        dataSDM(i,j) = i*j;
      }
    }
    bool shouldDeepCopy = false;
    bool shouldOwnData = false;
    
    Camellia::FieldContainer<double> dataFC(dataDim, &data(0,0,0), shouldDeepCopy, shouldOwnData); // this is the line we had to change to get things to pass.  We had been using 'false' (default) values for the latter two arguments.  I'm not entirely sure why this should matter (i.e., there may be a bug lurking in the FC construction), but the extra deep copy here probably won't hurt performance too much, and may avoid a memory leak.
    
    bool ownsMemory = dataFC.getData().has_ownership();
    TEST_EQUALITY(ownsMemory, shouldOwnData);
    
    SerialDenseWrapper::convertSDMToFC(dataFC, dataSDM);
    
    ownsMemory = dataFC.getData().has_ownership();
    TEST_EQUALITY(ownsMemory, shouldOwnData);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, ViewKeptInSync )
  {
    Camellia::FieldContainer<double> testContainer(3,3,3);
    testContainer[4] = 300.0;
    testContainer.resize(3);
    
    // TODO: finish this test
  }
} // namespace
