// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  SharedScratchStorage.cpp
 *
 *  Created by Nathan Roberts on 1/28/20.
 *
 */

#include "SharedScratchStorage.h"

#include "Camellia_FieldContainer.hpp"
#include "Camellia_KokkosAllocations.h"

namespace Camellia
{
  static const std::string SHARED_SCRATCH_STORAGE_LABEL("Camellia Shared Scratch storage");
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::allocateView(const ScratchStorageDimensions &dims, const std::string &label) const
  {
    typename SharedScratchStorage<Scalar>::DeviceView view;
    switch (dims.size())
    {
      case 0:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label));
        break;
      case 1:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0]);
        break;
      case 2:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1]);
        break;
      case 3:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2]);
        break;
      case 4:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3]);
        break;
      case 5:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4]);
        break;
      case 6:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
        break;
      case 7:
        view = DeviceView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
    }
    recordAllocation<Scalar>(deviceViewAllocationRecord, view.size());
    return view;
  }
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::checkoutScratchStorage(const ScratchStorageDimensions &dim)
  {
    if (_scratchStorageCache.find(dim) == _scratchStorageCache.end())
    {
      // no views of these dimensions have been requested before
      // then create a single
      DeviceView scratchView = allocateView(dim, SHARED_SCRATCH_STORAGE_LABEL);
      ScratchStorageEntry entry {scratchView,false}; // false: not checked out
      _scratchStorageCache[dim] = std::vector< ScratchStorageEntry > {entry};
    }
    auto & scratchStorageVector = _scratchStorageCache.find(dim)->second;
    for (auto & entry : scratchStorageVector)
    {
      bool & checkedOut = entry.second;
      if (!checkedOut)
      {
        checkedOut = true;
        _hitCount++;
        return entry.first;
      }
    }
    // if we get here, all matching entries are checked out; create a new one
    DeviceView scratchView = allocateView(dim, SHARED_SCRATCH_STORAGE_LABEL);
    ScratchStorageEntry entry {scratchView,true}; // true: is checked out (will be, on return)
    scratchStorageVector.push_back(entry);
    return entry.first;
  }
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::checkoutScratchStorageMatchingDeviceView(SharedScratchStorage<Scalar>::DeviceView &dv)
  {
    ScratchStorageDimensions dim = viewDimensions(dv);
    return checkoutScratchStorage(dim);
  }
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::checkoutScratchStorageMatchingHostView(SharedScratchStorage<Scalar>::HostView &hostView)
  {
    ScratchStorageDimensions dim = viewDimensions(hostView);
    return checkoutScratchStorage(dim);
  }
  
  template<typename Scalar>
  void SharedScratchStorage<Scalar>::checkinScratchStorage(DeviceView &scratchView)
  {
    ScratchStorageDimensions dim(scratchView.rank());
    for (int i=0; i<scratchView.rank(); i++)
    {
      dim[i] = scratchView.extent_int(i);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(_scratchStorageCache.find(dim) == _scratchStorageCache.end(), std::invalid_argument, "checkin of unrecognized scratch view");
    auto & scratchStorageVector = _scratchStorageCache.find(dim)->second;
    for (auto & entry : scratchStorageVector)
    {
      if (entry.first == scratchView)
      {
        entry.second = false; // no longer checked out
        return;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "checkin of unrecognized scratch view");
  }
  
  template<typename Scalar>
  void SharedScratchStorage<Scalar>::clear() // should be called during finalize()
  {
    for (auto & mapEntry : _scratchStorageCache)
    {
      auto & storageCacheVector = mapEntry.second;
      for (ScratchStorageEntry &entry : storageCacheVector)
      {
        // all scratch views should have been checked back in; assert this
        const bool isCheckedOut = entry.second;
        TEUCHOS_TEST_FOR_EXCEPTION(isCheckedOut, std::invalid_argument, "Scratch view was checked out without being checked back in");
        Allocation<Scalar>::recordDeviceViewDeallocation(entry.first);
      }
    }
    _scratchStorageCache.clear();
  }
  
  template<typename Scalar>
  long SharedScratchStorage<Scalar>::hitCount()
  {
    return _hitCount;
  }
  
  template<typename Scalar>
  SharedScratchStorage<Scalar> & SharedScratchStorage<Scalar>::sharedScratchStorage() // returns singleton instance
  {
    static SharedScratchStorage<Scalar> sharedStorage;
    return sharedStorage;
  }
  
  template class SharedScratchStorage<double>;  // for now, we only support double, int, bool.  Could add support for long long, complex, float, etc.
  template class SharedScratchStorage<int>;
  template class SharedScratchStorage<bool>;
  
  void clearSharedScratchStorage()
  {
    SharedScratchStorage<double>::sharedScratchStorage().clear();
    SharedScratchStorage<int>   ::sharedScratchStorage().clear();
    SharedScratchStorage<bool>  ::sharedScratchStorage().clear();
  }
} // namespace Camellia
