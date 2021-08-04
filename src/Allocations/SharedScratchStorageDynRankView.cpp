// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  SharedScratchStorageDynRankView.cpp
 *
 *  Created by Nathan Roberts on 1/28/20.
 *
 */

#include "SharedScratchStorageDynRankView.h"

#include "Camellia_FieldContainer.hpp"
#include "Camellia_KokkosAllocations.h"

namespace Camellia
{
  static const std::string SHARED_SCRATCH_STORAGE_DYNRANKVIEW_LABEL("Camellia Shared Scratch storage (DynRankView)");
  
  template<typename Scalar>
  typename SharedScratchStorageDynRankView<Scalar>::DynRankView
  SharedScratchStorageDynRankView<Scalar>::allocateView(const ScratchStorageDimensions &dims, const std::string &label) const
  {
    typename SharedScratchStorageDynRankView<Scalar>::DynRankView view;
    switch (dims.size())
    {
      case 0:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label));
        break;
      case 1:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0]);
        break;
      case 2:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1]);
        break;
      case 3:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2]);
        break;
      case 4:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3]);
        break;
      case 5:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4]);
        break;
      case 6:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
        break;
      case 7:
        view = DynRankView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
    }
    recordAllocation<Scalar>(deviceViewAllocationRecord, view.size());
    return view;
  }

  template<typename Scalar>
  typename SharedScratchStorageDynRankView<Scalar>::DynRankView
  SharedScratchStorageDynRankView<Scalar>::checkoutScratchStorage(const ScratchStorageDimensions &dim, std::string description)
  {
    if (description == "")
    {
      std::cout << "checkoutScratchStorage() called without a description specified.\n";
    }
    using std::get;
    if (_scratchStorageCache.find(dim) == _scratchStorageCache.end())
    {
      // no views of these dimensions have been requested before
      // then create a single
      DynRankView scratchView = allocateView(dim, SHARED_SCRATCH_STORAGE_DYNRANKVIEW_LABEL);
      ScratchStorageEntry entry {scratchView,false,description}; // false: not checked out
      _scratchStorageCache[dim] = std::vector< ScratchStorageEntry > {entry};
    }
    auto & scratchStorageVector = _scratchStorageCache.find(dim)->second;
    // for size-0 entries (i.e., those that are zero length in some dimension), we only store a single, empty view for each unique dimension vector.
    size_t viewSize = get<0>(scratchStorageVector[0]).size();
    bool checkoutValueOnReturn = (0 != viewSize) ? true : false;
    for (auto & entry : scratchStorageVector)
    {
      bool & checkedOut = get<1>(entry);
      if (!checkedOut)
      {
        checkedOut = checkoutValueOnReturn;
        get<2>(entry) = description;
        _hitCount++;
        return get<0>(entry);
      }
    }
    // if we get here, all matching entries are checked out; create a new one
    DynRankView scratchView = allocateView(dim, SHARED_SCRATCH_STORAGE_DYNRANKVIEW_LABEL);
    ScratchStorageEntry entry {scratchView,checkoutValueOnReturn,description};
    scratchStorageVector.push_back(entry);
    return get<0>(entry);
  }
  
  template<typename Scalar>
  typename SharedScratchStorageDynRankView<Scalar>::DynRankView
  SharedScratchStorageDynRankView<Scalar>::checkoutScratchStorageMatchingDynRankView(SharedScratchStorageDynRankView<Scalar>::DynRankView &dv, std::string description)
  {
    ScratchStorageDimensions dim = viewDimensions(dv);
    return checkoutScratchStorage(dim, description);
  }
  
  template<typename Scalar>
  typename SharedScratchStorageDynRankView<Scalar>::DynRankView
  SharedScratchStorageDynRankView<Scalar>::checkoutScratchStorageMatchingHostView(SharedScratchStorageDynRankView<Scalar>::HostView &hostView, std::string description)
  {
    ScratchStorageDimensions dim = viewDimensions(hostView);
    return checkoutScratchStorage(dim, description);
  }
  
  template<typename Scalar>
  void SharedScratchStorageDynRankView<Scalar>::checkinScratchStorage(DynRankView &scratchView)
  {
    if (scratchView.size() == 0)
    {
      // the checkout flag for size-0 views is always set to false; they can be reused without conflict
      // because there's no place to store data.
      return;
    }
    
    using std::get;
    ScratchStorageDimensions dim(scratchView.rank());
    for (int i=0; i<scratchView.rank(); i++)
    {
      dim[i] = scratchView.extent_int(i);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(_scratchStorageCache.find(dim) == _scratchStorageCache.end(), std::invalid_argument, "checkin of unrecognized scratch view");
    auto & scratchStorageVector = _scratchStorageCache.find(dim)->second;
    for (auto & entry : scratchStorageVector)
    {
      if (get<0>(entry) == scratchView)
      {
        get<1>(entry) = false; // no longer checked out
        return;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "checkin of unrecognized scratch view");
  }
  
  template<typename Scalar>
  void SharedScratchStorageDynRankView<Scalar>::clear() // should be called during finalize()
  {
    using std::get;
    for (auto & mapEntry : _scratchStorageCache)
    {
      auto & storageCacheVector = mapEntry.second;
      for (ScratchStorageEntry &entry : storageCacheVector)
      {
        // all scratch views should have been checked back in; assert this
        const bool & isCheckedOut = get<1>(entry);
        const std::string & description = get<2>(entry);
        TEUCHOS_TEST_FOR_EXCEPTION(isCheckedOut, std::invalid_argument, "Scratch view was checked out without being checked back in; last checked out with description '" + description + "'");
        Allocation<Scalar>::recordDeviceViewDeallocation(get<0>(entry));
      }
    }
    _scratchStorageCache.clear();
  }
  
  template<typename Scalar>
  long SharedScratchStorageDynRankView<Scalar>::hitCount()
  {
    return _hitCount;
  }
  
  template<typename Scalar>
  SharedScratchStorageDynRankView<Scalar> & SharedScratchStorageDynRankView<Scalar>::sharedScratchStorage() // returns singleton instance
  {
    static SharedScratchStorageDynRankView<Scalar> sharedStorage;
    return sharedStorage;
  }
  
  template class SharedScratchStorageDynRankView<double>;  // for now, we only support double, int, bool.  Could add support for long long, complex, float, etc.
  template class SharedScratchStorageDynRankView<int>;
  template class SharedScratchStorageDynRankView<bool>;
  
  void clearSharedScratchStorageDynRankView()
  {
    SharedScratchStorageDynRankView<double>::sharedScratchStorage().clear();
    SharedScratchStorageDynRankView<int>   ::sharedScratchStorage().clear();
    SharedScratchStorageDynRankView<bool>  ::sharedScratchStorage().clear();
  }
} // namespace Camellia
