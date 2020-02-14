// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  Camellia_KokkosAllocations.cpp
 *
 *  Created by Nathan Roberts on 1/29/20.
 *
 */

#include "Camellia_KokkosAllocations.h"
#include "SharedScratchStorage.h"

#include <set>
#include <sstream>
#include <string>

namespace Camellia
{
  AllocationRecord fcAllocationRecord;
  AllocationRecord deviceViewAllocationRecord;
  AllocationRecord deviceViewFakeAllocationRecord;
  
  std::set<long long> outstandingFCAllocationNumbers;
  bool recordOutstandingFCAllocationNumbers = false; // roughly 8% performance penalty for all of runTests if this is true, but very useful for tracking down unfreed allocations.
  
  DeepCopyRecord deviceToHostDeepCopyRecord;
  DeepCopyRecord hostToDeviceDeepCopyRecord;
  
  std::string allocationReport(AllocationRecord &record)
  {
    std::ostringstream allocationReportStream;
    
    double totalInMB  = double(record.bytesAllocatedTotal) / 1024.0 / 1024.0;
    double netInMB    = double(record.bytesAllocatedNet) / 1024.0 / 1024.0;
    double netHWMInMB = double(record.bytesAllocatedNetHighWaterMark) / 1024.0 / 1024.0;
    
    allocationReportStream << record.allocationCount << " Views/FieldContainers have been allocated.\n";
    allocationReportStream << "Total bytes allocated:                " << record.bytesAllocatedTotal << " (" << totalInMB << " MB)\n";
    allocationReportStream << "Net bytes allocated:                  " << record.bytesAllocatedNet   << " (" << netInMB   << " MB)\n";
    allocationReportStream << "Net bytes allocated, high water mark: " << record.bytesAllocatedNetHighWaterMark << " (" << netHWMInMB << " MB)\n";
    return allocationReportStream.str();
  }
  
  template<typename Scalar>
  DeviceView<Scalar> Allocation<Scalar>::allocateDeviceView(const std::vector<int> &dims, const std::string &label)
  {
    DeviceView<Scalar> view;
    switch (dims.size())
    {
      case 0:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label));
        break;
      case 1:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0]);
        break;
      case 2:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1]);
        break;
      case 3:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2]);
        break;
      case 4:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3]);
        break;
      case 5:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4]);
        break;
      case 6:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
        break;
      case 7:
        view = DeviceView<Scalar>(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
    }
    recordAllocation<Scalar>(deviceViewAllocationRecord, view.size());
    return view;
  }
  
  template<typename Scalar>
  DeviceView<Scalar> Allocation<Scalar>::allocateDeviceViewMatchingView(DeviceView<Scalar> &dv)
  {
    DeviceView<Scalar> deviceView = Kokkos::create_mirror(DeviceMemorySpace(), dv);
    Camellia::recordAllocation<Scalar>(deviceViewAllocationRecord, deviceView.size());
    return deviceView;
  }
  
  // this struct effectively allows us to implement different versions of Allocation<Scalar>::checkoutMirrorView() on host and device
  template<typename Scalar, bool memorySpacesMatch>
  struct CheckoutMirrorView {
    static DeviceView<Scalar> impl(HostView<Scalar> &hostView);
  };
  
  template<typename Scalar>
  struct CheckoutMirrorView<Scalar, true> // true: memory spaces match
  {
    static DeviceView<Scalar> impl(HostView<Scalar> &hostView)
    {
      // record the "fake" allocation
      recordAllocation<Scalar>(deviceViewFakeAllocationRecord, hostView.size());
      return Kokkos::create_mirror_view(hostView);
    }
  };
  
  template<typename Scalar>
  struct CheckoutMirrorView<Scalar, false> // false: memory spaces don't match
  {
    static DeviceView<Scalar> impl(HostView<Scalar> &hostView)
    {
      auto deviceView = SharedScratchStorage<Scalar>::sharedScratchStorage().checkoutScratchStorageMatchingHostView(hostView);
      return deviceView;
    }
  };
  
  template<typename Scalar>
  DeviceView<Scalar> Allocation<Scalar>::checkoutMirrorView(HostView<Scalar> &hostView) // true: memory spaces match
  {
    constexpr bool memorySpacesMatch = std::is_same<HostMemorySpace, DeviceMemorySpace>::value;
    return CheckoutMirrorView<Scalar, memorySpacesMatch>::impl(hostView);
  }
  
  template<typename Scalar>
  DeviceView<Scalar> Allocation<Scalar>::checkoutDeviceView(DeviceView<Scalar> &viewForSizing)
  {
    auto deviceView = SharedScratchStorage<Scalar>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(viewForSizing);
    return deviceView;
  }
  
  template<typename Scalar>
  DeviceView<Scalar> Allocation<Scalar>::checkoutDeviceView(const std::vector<int> &dims)
  {
    auto deviceView = SharedScratchStorage<Scalar>::sharedScratchStorage().checkoutScratchStorage(dims);
    return deviceView;
  }
  
  template<typename Scalar>
  void Allocation<Scalar>::checkinDeviceView(DeviceView<Scalar> &deviceView)
  {
    SharedScratchStorage<Scalar>::sharedScratchStorage().checkinScratchStorage(deviceView);
  }
  
  template<typename Scalar>
  void Allocation<Scalar>::checkinMirrorView(DeviceView<Scalar> &deviceView)
  {
    constexpr bool memorySpacesMatch = std::is_same<HostMemorySpace, DeviceMemorySpace>::value;
    const bool deviceViewIsNotEmpty = (deviceView.size() > 0); // we allow calling checkin on empty views that weren't checked out...
    if (deviceViewIsNotEmpty)
    {
      // if memory spaces match, we never actually checked anything out...
      if (!memorySpacesMatch)
      {
        SharedScratchStorage<Scalar>::sharedScratchStorage().checkinScratchStorage(deviceView);
      }
      else
      {
        // record the "fake" deallocation
        recordDeallocation<Scalar>(deviceViewFakeAllocationRecord, deviceView.size());
      }
    }
  }
  
  template<typename Scalar>
  void Allocation<Scalar>::recordDeviceViewDeallocation(DeviceView<Scalar> &dv) {
    Camellia::recordDeallocation<Scalar>(deviceViewAllocationRecord, dv.size());
  }
  
  // ETI
  template class Allocation<double>;  // for now, we only support double, int, bool.  Could add support for long long, complex, float, etc.
  template class Allocation<int>;
  template class Allocation<bool>;
} // namespace Camellia
