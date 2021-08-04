// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  Camellia_KokkosAllocations.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/29/20.
//

#ifndef Camellia_KokkosAllocations_h
#define Camellia_KokkosAllocations_h

#include <set>
#include "TypeDefs.h"

#include "Intrepid2_ScalarView.hpp"

namespace Camellia
{
  //! AllocationRecord allows us to track allocations.
  //! We store two instances of this:
  //! - one that tracks FieldContainer allocations
  //! - one that tracks "Device View" allocations (which are no-ops when Device memory space is the same as Host)
  struct AllocationRecord
  {
    int allocationCount = 0;
    long long bytesAllocatedTotal = 0;
    long long bytesAllocatedNet = 0;
    long long bytesAllocatedNetHighWaterMark = 0;
  };
  
  extern AllocationRecord fcAllocationRecord;
  extern AllocationRecord deviceViewAllocationRecord;
  extern AllocationRecord deviceViewFakeAllocationRecord; // records calls to create_mirror_view() that resolve without a real allocation, so that we can see what *would* happen on CUDA even when running on host.
  
  extern std::set<long long> outstandingFCAllocationNumbers;
  extern bool recordOutstandingFCAllocationNumbers; // if false, we skip tracking of the outstanding FC numbers, which does have some performance cost (about 8% penalty for the test suite as a whole)
  
  struct DeepCopyRecord
  {
    int deepCopyCount = 0;
    long long bytesDeepCopied = 0;
  };
  
  extern DeepCopyRecord deviceToHostDeepCopyRecord;
  extern DeepCopyRecord hostToDeviceDeepCopyRecord;
  
  template<typename Scalar>
  void recordDeepCopy(DeepCopyRecord &record, long long entryCount)
  {
    record.deepCopyCount++;
    record.bytesDeepCopied += entryCount * sizeof(Scalar);
  }
  
  template<typename Scalar>
  void recordAllocation(AllocationRecord &record, long long entryCount)
  {
    record.allocationCount++;
    long long bytes = sizeof(Scalar) * entryCount;
    record.bytesAllocatedNet   += bytes;
    record.bytesAllocatedTotal += bytes;
    if (record.bytesAllocatedNet > record.bytesAllocatedNetHighWaterMark)
    {
      record.bytesAllocatedNetHighWaterMark = record.bytesAllocatedNet;
    }
  }
  
  template<typename Scalar>
  void recordDeallocation(AllocationRecord &record, long long entryCount)
  {
    long long bytes = sizeof(Scalar) * entryCount;
    record.bytesAllocatedNet -= bytes;
  }
  
  template<typename Scalar>
  class Allocation {
    template< bool B, class T = void >
    using enable_if_t = typename std::enable_if<B,T>::type;
    
    using Intrepid2View = Intrepid2::ScalarView<Scalar,Camellia::DeviceType>;
  public:
    static DeviceView<Scalar> allocateDeviceView(const std::vector<int> &dims, const std::string &label);
    static DeviceView<Scalar> allocateDeviceViewMatchingView(DeviceView<Scalar> &dv);
    static void recordDeviceViewDeallocation(DeviceView<Scalar> &dv);
    
    static Intrepid2View allocateIntrepid2DeviceViewMatchingView(DeviceView<Scalar> &dv);
    static Intrepid2View allocateIntrepid2DeviceView(const std::vector<int> &dims, const std::string &label);
    
    static DeviceView<Scalar> checkoutMirrorView(HostView<Scalar> &hostView, const std::string &description); // does not result in an allocation if HostMemorySpace is same as DeviceMemorySpace
    
    static void checkinMirrorView(DeviceView<Scalar> &deviceView);
    
    static DeviceView<Scalar> checkoutDeviceView(const std::vector<int> &dims, const std::string &description);
    static DeviceView<Scalar> checkoutDeviceView(DeviceView<Scalar> &deviceView, const std::string &description); // does not result in an allocation if HostMemorySpace is same as DeviceMemorySpace
    static void checkinDeviceView(DeviceView<Scalar> &deviceView);
    
    template<class ViewType>
    static void recordViewDeallocation(ViewType &dv) {
      Camellia::recordDeallocation<Scalar>(deviceViewAllocationRecord, dv.size());
    }
    
    // TODO: consider implementing, using these:
//    static Intrepid2View checkoutIntrepid2DeviceView(const std::vector<int> &dims);
//    static Intrepid2View checkoutIntrepid2DeviceView(DeviceView<Scalar> &deviceView);
//    static void checkinIntrepid2DeviceView(Intrepid2::ScalarView<Scalar,Camellia::DeviceType> &intrepid2DeviceView);
  };
}

#endif /* Camellia_KokkosAllocations_h */
