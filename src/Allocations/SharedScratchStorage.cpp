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
//using ScratchDataDimensions = std::vector<Intrepid2::DimensionInfo>;
//Data checkoutScratchData(const ScratchDataDimensions &dataDims, std::string description="");
//void checkinScratchData(Data &data);
  
  template<int rank, typename FixedRankViewType>
  Intrepid2::Data<typename FixedRankViewType::value_type, Camellia::DeviceType>
  constructData(FixedRankViewType view, std::vector<int> extents, std::vector<Intrepid2::DataVariationType> variationTypes,
                int blockPlusDiagonalLastNonDiagonal)
  {
    Kokkos::Array<int,rank> extentsArray;
    Kokkos::Array<Intrepid2::DataVariationType,rank> variationTypesArray;
    for (int d=0; d<rank; d++)
    {
      extentsArray[d]        = extents[d];
      variationTypesArray[d] = variationTypes[d];
    }
    using Data = Intrepid2::Data<typename FixedRankViewType::value_type, Camellia::DeviceType>;
    return Data(view, extentsArray, variationTypesArray, blockPlusDiagonalLastNonDiagonal);
  }

  template<typename FixedRankViewType>
  Intrepid2::Data<typename FixedRankViewType::value_type, Camellia::DeviceType>
  constructData(FixedRankViewType view, std::vector<int> extents, std::vector<Intrepid2::DataVariationType> variationTypes,
                int blockPlusDiagonalLastNonDiagonal)
  {
    const int rank = variationTypes.size();
    switch (rank)
    {
      case 1: return constructData<1,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      case 2: return constructData<2,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      case 3: return constructData<3,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      case 4: return constructData<4,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      case 5: return constructData<5,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      case 6: return constructData<6,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      case 7: return constructData<7,FixedRankViewType>(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
    }
  }

  template<typename Scalar>
  void SharedScratchStorage<Scalar>::checkinScratchData(const typename SharedScratchStorage<Scalar>::Data &data)
  {
    const int rank = data.getUnderlyingViewRank();
    switch (rank) {
      case 1: _fixedRank1.checkinScratchStorage(data.template getUnderlyingView<1>()); break;
      case 2: _fixedRank2.checkinScratchStorage(data.template getUnderlyingView<2>()); break;
      case 3: _fixedRank3.checkinScratchStorage(data.template getUnderlyingView<3>()); break;
      case 4: _fixedRank4.checkinScratchStorage(data.template getUnderlyingView<4>()); break;
      case 5: _fixedRank5.checkinScratchStorage(data.template getUnderlyingView<5>()); break;
      case 6: _fixedRank6.checkinScratchStorage(data.template getUnderlyingView<6>()); break;
      case 7: _fixedRank7.checkinScratchStorage(data.template getUnderlyingView<7>()); break;
        
      default:
        break;
    }
  }

  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::Data
  SharedScratchStorage<Scalar>::checkoutScratchData(const ScratchDataDimensions &dim, std::string description)
  {
    std::vector<int> underlyingViewDims;
        
    std::vector<int> extents;
    std::vector<Intrepid2::DataVariationType> variationTypes;
    int blockPlusDiagonalLastNonDiagonal = -1;
    for (const auto & dimInfo : dim)
    {
      bool haveHitBlockPlusDiagonal = false;
      switch (dimInfo.variationType)
      {
        case Intrepid2::GENERAL:
        case Intrepid2::MODULAR:
          underlyingViewDims.push_back(dimInfo.dataExtent);
          break;
        case Intrepid2::BLOCK_PLUS_DIAGONAL:
          if (!haveHitBlockPlusDiagonal)
          {
            haveHitBlockPlusDiagonal = true;
            blockPlusDiagonalLastNonDiagonal = dimInfo.blockPlusDiagonalLastNonDiagonal;
            underlyingViewDims.push_back(dimInfo.dataExtent);
          }
        case Intrepid2::CONSTANT:
          // skip this dimension; it does not enter the underlying view
          break;
      }
      extents.push_back(dimInfo.logicalExtent);
      variationTypes.push_back(dimInfo.variationType);
    }
    if (underlyingViewDims.size() == 0)
    {
      // constant everywhere: want a rank 1, single-entry view
      underlyingViewDims.push_back(1);
    }
    const int nominalRank = dim.size();
    const int dataRank = underlyingViewDims.size();
    switch (dataRank) {
      case 1:
      {
        auto view = _fixedRank1.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      case 2:
      {
        auto view = _fixedRank2.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      case 3:
      {
        auto view = _fixedRank3.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      case 4:
      {
        auto view = _fixedRank4.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      case 5:
      {
        auto view = _fixedRank5.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      case 6:
      {
        auto view = _fixedRank6.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      case 7:
      {
        auto view = _fixedRank7.checkoutScratchStorage(underlyingViewDims, description);
        return constructData(view, extents, variationTypes, blockPlusDiagonalLastNonDiagonal);
      }
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
    }
  }
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::checkoutScratchStorage(const ScratchStorageDimensions &dim, std::string description)
  {
    return _dynRank.checkoutScratchStorage(dim, description);
  }
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::checkoutScratchStorageMatchingDeviceView(SharedScratchStorage<Scalar>::DeviceView &dv, std::string description)
  {
    return _dynRank.checkoutScratchStorageMatchingDynRankView(dv, description);
  }
  
  template<typename Scalar>
  typename SharedScratchStorage<Scalar>::DeviceView
  SharedScratchStorage<Scalar>::checkoutScratchStorageMatchingHostView(SharedScratchStorage<Scalar>::HostView &hostView, std::string description)
  {
    return _dynRank.checkoutScratchStorageMatchingHostView(hostView, description);
  }
  
  template<typename Scalar>
  void SharedScratchStorage<Scalar>::checkinScratchStorage(DeviceView &scratchView)
  {
    _dynRank.checkinScratchStorage(scratchView);
  }
  
  template<typename Scalar>
  void SharedScratchStorage<Scalar>::clear() // should be called during finalize()
  {
    _fixedRank1.clear();
    _fixedRank2.clear();
    _fixedRank3.clear();
    _fixedRank4.clear();
    _fixedRank5.clear();
    _fixedRank6.clear();
    _fixedRank7.clear();
    
    _dynRank.clear();
  }
  
  template<typename Scalar>
  long SharedScratchStorage<Scalar>::hitCount()
  {
    long hitCount = 0;
    hitCount += _fixedRank1.hitCount();
    hitCount += _fixedRank2.hitCount();
    hitCount += _fixedRank3.hitCount();
    hitCount += _fixedRank4.hitCount();
    hitCount += _fixedRank5.hitCount();
    hitCount += _fixedRank6.hitCount();
    hitCount += _fixedRank7.hitCount();
    hitCount += _dynRank.hitCount();
    return hitCount;
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
