// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  SharedScratchStorageFixedRankView.cpp
 *
 *  Created by Nathan Roberts on 1/28/20.
 *
 */

#include "SharedScratchStorageFixedRankView.h"

#include "Camellia_FieldContainer.hpp"
#include "Camellia_KokkosAllocations.h"

namespace Camellia
{
  static const std::string SHARED_SCRATCH_STORAGE_LABEL("Camellia Shared Scratch storage");
  
  template<typename Scalar, int rank>
  typename SharedScratchStorageFixedRankView<Scalar,rank>::FixedView
  SharedScratchStorageFixedRankView<Scalar,rank>::allocateView(const ScratchStorageDimensions &dims, const std::string &label) const
  {
    typename SharedScratchStorageFixedRankView<Scalar,rank>::FixedView view;
    switch (dims.size())
    {
      case 0:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label));
        break;
      case 1:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0]);
        break;
      case 2:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1]);
        break;
      case 3:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2]);
        break;
      case 4:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3]);
        break;
      case 5:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4]);
        break;
      case 6:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]);
        break;
      case 7:
        view = FixedView(Kokkos::ViewAllocateWithoutInitializing(label),dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported rank");
    }
    recordAllocation<Scalar>(deviceViewAllocationRecord, view.size());
    return view;
  }

  template<typename Scalar, int rank>
  typename SharedScratchStorageFixedRankView<Scalar,rank>::FixedView
  SharedScratchStorageFixedRankView<Scalar,rank>::checkoutScratchStorage(const ScratchStorageDimensions &dim, std::string description)
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
      FixedView scratchView = allocateView(dim, SHARED_SCRATCH_STORAGE_LABEL);
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
    FixedView scratchView = allocateView(dim, SHARED_SCRATCH_STORAGE_LABEL);
    ScratchStorageEntry entry {scratchView,checkoutValueOnReturn,description}; 
    scratchStorageVector.push_back(entry);
    return get<0>(entry);
  }
  
  template<typename Scalar, int rank>
  typename SharedScratchStorageFixedRankView<Scalar,rank>::FixedView
  SharedScratchStorageFixedRankView<Scalar,rank>::checkoutScratchStorageMatchingFixedView(const SharedScratchStorageFixedRankView<Scalar,rank>::FixedView &dv, std::string description)
  {
    ScratchStorageDimensions dim = viewDimensions(dv);
    return checkoutScratchStorage(dim, description);
  }
  
  template<typename Scalar, int rank>
  typename SharedScratchStorageFixedRankView<Scalar,rank>::FixedView
  SharedScratchStorageFixedRankView<Scalar,rank>::checkoutScratchStorageMatchingHostView(const SharedScratchStorageFixedRankView<Scalar,rank>::HostView &hostView, std::string description)
  {
    ScratchStorageDimensions dim = viewDimensions(hostView);
    return checkoutScratchStorage(dim, description);
  }
  
  template<typename Scalar, int rank>
  void SharedScratchStorageFixedRankView<Scalar,rank>::checkinScratchStorage(const FixedView &scratchView)
  {
    if (scratchView.size() == 0)
    {
      // the checkout flag for size-0 views is always set to false; they can be reused without conflict
      // because there's no place to store data.
      return;
    }
    
    using std::get;
    ScratchStorageDimensions dim(rank);
    for (int i=0; i<rank; i++)
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
  
  template<typename Scalar, int rank>
  void SharedScratchStorageFixedRankView<Scalar,rank>::clear() // should be called during finalize()
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
        Allocation<Scalar>::recordViewDeallocation(get<0>(entry));
      }
    }
    _scratchStorageCache.clear();
  }
  
  template<typename Scalar, int rank>
  long SharedScratchStorageFixedRankView<Scalar,rank>::hitCount()
  {
    return _hitCount;
  }
  
  template class SharedScratchStorageFixedRankView<double, 1>;  // for now, we only support double, int, bool, long long.  Could add support for complex, float, etc.
  template class SharedScratchStorageFixedRankView<double, 2>;
  template class SharedScratchStorageFixedRankView<double, 3>;
  template class SharedScratchStorageFixedRankView<double, 4>;
  template class SharedScratchStorageFixedRankView<double, 5>;
  template class SharedScratchStorageFixedRankView<double, 6>;
  template class SharedScratchStorageFixedRankView<double, 7>;

  template class SharedScratchStorageFixedRankView<int, 1>;
  template class SharedScratchStorageFixedRankView<int, 2>;
  template class SharedScratchStorageFixedRankView<int, 3>;
  template class SharedScratchStorageFixedRankView<int, 4>;
  template class SharedScratchStorageFixedRankView<int, 5>;
  template class SharedScratchStorageFixedRankView<int, 6>;
  template class SharedScratchStorageFixedRankView<int, 7>;

  template class SharedScratchStorageFixedRankView<bool, 1>;
  template class SharedScratchStorageFixedRankView<bool, 2>;
  template class SharedScratchStorageFixedRankView<bool, 3>;
  template class SharedScratchStorageFixedRankView<bool, 4>;
  template class SharedScratchStorageFixedRankView<bool, 5>;
  template class SharedScratchStorageFixedRankView<bool, 6>;
  template class SharedScratchStorageFixedRankView<bool, 7>;

  template class SharedScratchStorageFixedRankView<long long, 1>;
  template class SharedScratchStorageFixedRankView<long long, 2>;
  template class SharedScratchStorageFixedRankView<long long, 3>;
  template class SharedScratchStorageFixedRankView<long long, 4>;
  template class SharedScratchStorageFixedRankView<long long, 5>;
  template class SharedScratchStorageFixedRankView<long long, 6>;
  template class SharedScratchStorageFixedRankView<long long, 7>;
} // namespace Camellia
