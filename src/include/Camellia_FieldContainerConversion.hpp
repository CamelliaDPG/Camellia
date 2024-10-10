// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//
//  Camellia_FieldContainerConversion.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/29/21.
//

#ifndef Camellia_FieldContainerConversion_h
#define Camellia_FieldContainerConversion_h

#include "Camellia_FieldContainer.hpp"
#include "Camellia_KokkosAllocations.h"
#include "Camellia_KokkosHelpers.h"

#include "Intrepid2_ScalarView.hpp" // TODO: in Intrepid2, add this include to Intrepid2_TensorPoints.hpp
#include "Intrepid2_Types.hpp"      // TODO: in Intrepid2, add this include to Intrepid2_TensorPoints.hpp

#include "Intrepid2_BasisValues.hpp"
#include "Intrepid2_CubatureTensor.hpp"
#include "Intrepid2_FunctorIterator.hpp"
#include "Intrepid2_TensorData.hpp"
#include "Intrepid2_TensorPoints.hpp"
#include "Intrepid2_Utils.hpp"
#include "Intrepid2_ViewIterator.hpp"

namespace Camellia {
  template<class Scalar>
  class FieldContainerConversion
  {
    using BasisValues   = Intrepid2::BasisValues<Scalar,DeviceType>;
    using TensorData    = Intrepid2::TensorData<Scalar,DeviceType>;
    using Data          = Intrepid2::Data<Scalar,DeviceType>;
    using TensorPoints  = Intrepid2::TensorPoints<Scalar,DeviceType>;
    using DynRankView   = Kokkos::DynRankView<Scalar,DeviceType>;
  private:
    template<class ViewLikeContainer, int rank>
    static Camellia::FieldContainer<Scalar, 0> fromDeviceContainerWithRank(ViewLikeContainer &viewLikeContainer)
    {
      using FunctorIterator = Intrepid2::FunctorIterator<ViewLikeContainer,Scalar,rank>;
      using DeviceView      = Camellia::DeviceView<Scalar>;
      using ViewIterator    = Intrepid2::ViewIterator<DeviceView,Scalar>;
      Teuchos::Array<int> dimensions(rank);
      for (int d=0; d<rank; d++)
      {
        dimensions[d] = viewLikeContainer.extent_int(d);
      }
      Camellia::FieldContainer<Scalar, 0> fc(dimensions);
      
      auto deviceView = fc.checkoutDeviceView(false); // false: don't deep copy from host
      
      deepCopy<rank,DeviceView,ViewLikeContainer>(deviceView, viewLikeContainer);
      fc.checkinDeviceView(true); // true: do deep copy to host
      return fc;
    }
    
    template<class ViewLikeContainer>
    static Camellia::FieldContainer<Scalar, 0> fromDeviceContainer(ViewLikeContainer &viewLikeContainer)
    {
      const int rank = Intrepid2::getFunctorRank(viewLikeContainer);
      switch (rank)
      {
        case 1: return fromDeviceContainerWithRank<ViewLikeContainer,1>(viewLikeContainer);
        case 2: return fromDeviceContainerWithRank<ViewLikeContainer,2>(viewLikeContainer);
        case 3: return fromDeviceContainerWithRank<ViewLikeContainer,3>(viewLikeContainer);
        case 4: return fromDeviceContainerWithRank<ViewLikeContainer,4>(viewLikeContainer);
        case 5: return fromDeviceContainerWithRank<ViewLikeContainer,5>(viewLikeContainer);
        case 6: return fromDeviceContainerWithRank<ViewLikeContainer,6>(viewLikeContainer);
        case 7: return fromDeviceContainerWithRank<ViewLikeContainer,7>(viewLikeContainer);
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
      }
    }
  public:
    static Camellia::FieldContainer<Scalar, 0> fromData(Data &data)
    {
      if (!data.isValid())
      {
        // then return an empty FC
        return Camellia::FieldContainer<Scalar, 0>();
      }
      
      // TensorData can be rank 0, 1, 2 or 3
      const int rank = Intrepid2::getFunctorRank(data);
      if (rank == 0)
      {
        return Camellia::FieldContainer<Scalar, 0>();
      }
      else if (rank == 1)
      {
        return fromDeviceContainerWithRank<Data,1>(data);
      }
      else if (rank == 2)
      {
        return fromDeviceContainerWithRank<Data,2>(data);
      }
      else if (rank == 3)
      {
        return fromDeviceContainerWithRank<Data,3>(data);
      }
      else if (rank == 4)
      {
        return fromDeviceContainerWithRank<Data,4>(data);
      }
      else if (rank == 5)
      {
        return fromDeviceContainerWithRank<Data,5>(data);
      }
      else if (rank == 6)
      {
        return fromDeviceContainerWithRank<Data,6>(data);
      }
      else if (rank == 7)
      {
        return fromDeviceContainerWithRank<Data,7>(data);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
    }
    
    static Camellia::FieldContainer<Scalar, 0> fromTensorData(TensorData &tensorData)
    {
      if (!tensorData.isValid())
      {
        // then return an empty FC
        return Camellia::FieldContainer<Scalar, 0>();
      }
      
      // TensorData can be rank 0, 1, 2 or 3
      const int rank = Intrepid2::getFunctorRank(tensorData);
      if (rank == 0)
      {
        return Camellia::FieldContainer<Scalar, 0>();
      }
      else if (rank == 1)
      {
        return fromDeviceContainerWithRank<TensorData,1>(tensorData);
      }
      else if (rank == 2)
      {
        return fromDeviceContainerWithRank<TensorData,2>(tensorData);
      }
      else if (rank == 3)
      {
        return fromDeviceContainerWithRank<TensorData,3>(tensorData);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
    }
    
    static Camellia::FieldContainer<Scalar, 0> fromBasisValues(BasisValues &basisValues)
    {
      // BasisValues can be rank 2, 3, or 4 (though 4 does not actually come up in Camellia's usage)
      const int rank = Intrepid2::getFunctorRank(basisValues);
      if (rank == 2)
      {
        return fromDeviceContainerWithRank<BasisValues,2>(basisValues);
      }
      else if (rank == 3)
      {
        return fromDeviceContainerWithRank<BasisValues,3>(basisValues);
      }
      else if (rank == 4)
      {
        return fromDeviceContainerWithRank<BasisValues,4>(basisValues);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
    }
    
    static Camellia::FieldContainer<Scalar, 0> fromTensorPoints(const TensorPoints &tensorPoints)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!tensorPoints.isValid(), std::invalid_argument, "Invalid TensorPoints container");
      
      // TensorPoints is always rank 2
      const int rank = 2;
      return fromDeviceContainerWithRank<const TensorPoints,rank>(tensorPoints);
    }
    
    static Camellia::FieldContainer<Scalar, 0> fromDynRankView(DynRankView &view)
    {
      // DynRankView can be any rank, up to 7
      return fromDeviceContainer(view);
    }
    
    static TensorData toData(Camellia::FieldContainer<Scalar, 0> &dataFC)
    {
      // we allow empty/invalid containers here; for FCs, this is indicated by having size 0
      if (dataFC.size() == 0)
      {
        return Data();
      }
      
      // dataFC has shape (P)
      auto deviceView = dataFC.checkoutDeviceView(true); // true: deep copy from host
      auto dataView = Allocation<Scalar>::allocateDeviceViewMatchingView(deviceView);
      Kokkos::deep_copy(dataView,deviceView);
      // the following deallocation record is a cheat: since dataView is wrapped in TensorData (which is Intrepid2), we don't have a great hook for determining when it goes out of scope and the deallocation is about to occur.  Ultimately, usage of this conversion method will be replaced by native Intrepid2 methods, which probably will do their own allocations, not necessarily recorded by Camellia.
      Allocation<Scalar>::recordDeviceViewDeallocation(dataView);
      
      // Under CUDA, layouts may differ between Intrepid2 ScalarView and Camellia Device View; the following allocates a Scalar View and copies the data.
      // This serves to underscore the point that all these conversion methods are conveniences, and necessarily represent inefficiencies.  We should use
      // Intrepid2 (or compatible) containers in all performance-critical portions of the code.
      auto dataViewIntrepid2 = Allocation<Scalar>::allocateIntrepid2DeviceViewMatchingView(dataView);
      deepCopy(dataViewIntrepid2, dataView);
      
      auto data = Data(dataViewIntrepid2);
      return data;
    }
    
    static TensorData toTensorData(Camellia::FieldContainer<Scalar, 0> &dataFC)
    {
      // we allow empty/invalid containers here; for FCs, this is indicated by having size 0
      if (dataFC.size() == 0)
      {
        return TensorData();
      }
      
      // dataFC has shape (P)
      auto deviceView = dataFC.checkoutDeviceView(true); // true: deep copy from host
      auto dataView = Allocation<Scalar>::allocateDeviceViewMatchingView(deviceView);
      Kokkos::deep_copy(dataView,deviceView);
      // the following deallocation record is a cheat: since dataView is wrapped in TensorData (which is Intrepid2), we don't have a great hook for determining when it goes out of scope and the deallocation is about to occur.  Ultimately, usage of this conversion method will be replaced by native Intrepid2 methods, which probably will do their own allocations, not necessarily recorded by Camellia.
      Allocation<Scalar>::recordDeviceViewDeallocation(dataView);
      
      // Under CUDA, layouts may differ between Intrepid2 ScalarView and Camellia Device View; the following allocates a Scalar View and copies the data.
      // This serves to underscore the point that all these conversion methods are conveniences, and necessarily represent inefficiencies.  We should use
      // Intrepid2 (or compatible) containers in all performance-critical portions of the code.
      auto dataViewIntrepid2 = Allocation<Scalar>::allocateIntrepid2DeviceViewMatchingView(dataView);
      deepCopy(dataViewIntrepid2, dataView);
      
      auto data = Data(dataViewIntrepid2);
      return TensorData(data); // will have trivial tensor product structure
    }
    
    static TensorPoints toTensorPoints(Camellia::FieldContainer<Scalar, 0> &pointsFC)
    {
      // pointsFC is shape (P,D)
      auto deviceView = pointsFC.checkoutDeviceView(true); // true: deep copy from host
      auto pointsView = Allocation<Scalar>::allocateDeviceViewMatchingView(deviceView);
      Kokkos::deep_copy(pointsView,deviceView);
      pointsFC.checkinDeviceView(false); // false: don't deep copy back to host (no change)
      // the following deallocation record is a cheat: since pointsView is wrapped in TensorPoints (which is Intrepid2), we don't have a great hook for determining when it goes out of scope and the deallocation is about to occur.  Ultimately, usage of this conversion method will be replaced by native Intrepid2 methods, which probably will do their own allocations, not necessarily recorded by Camellia.
      Allocation<Scalar>::recordDeviceViewDeallocation(pointsView);
      
      // Under CUDA, layouts may differ between Intrepid2 ScalarView and Camellia Device View; the following allocates a Scalar View and copies the data.
      // This serves to underscore the point that all these conversion methods are conveniences, and necessarily represent inefficiencies.  We should use
      // Intrepid2 (or compatible) containers in all performance-critical portions of the code.
      auto pointsViewIntrepid2 = Allocation<Scalar>::allocateIntrepid2DeviceViewMatchingView(pointsView);
      deepCopy(pointsViewIntrepid2, pointsView);
      return TensorPoints(pointsViewIntrepid2); // will have trivial tensor product structure
    }
    
    static Intrepid2::ScalarView<double, DeviceType> toScalarView(Camellia::FieldContainer<Scalar, 0> &fc)
    {
      auto deviceView = fc.checkoutDeviceView(true); // true: deep copy from host
      auto scalarView = Allocation<Scalar>::allocateDeviceViewMatchingView(deviceView);
      Kokkos::deep_copy(scalarView,deviceView);
      // the following deallocation record is a cheat: since pointsView is wrapped in TensorPoints (which is Intrepid2), we don't have a great hook for determining when it goes out of scope and the deallocation is about to occur.  Ultimately, usage of this conversion method will be replaced by native Intrepid2 methods, which probably will do their own allocations, not necessarily recorded by Camellia.
      Allocation<Scalar>::recordDeviceViewDeallocation(scalarView);
      return scalarView;
    }
  };
}

#endif /* Camellia_FieldContainerConversion_h */
