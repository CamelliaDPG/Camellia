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
    using ExecSpaceType = Camellia::DeviceMemorySpace::execution_space;
    using BasisValues   = Intrepid2::BasisValues<Scalar,ExecSpaceType>;
    using TensorData    = Intrepid2::TensorData<Scalar,ExecSpaceType>;
    using TensorPoints  = Intrepid2::TensorPoints<Scalar,ExecSpaceType>;
    using DynRankView   = Kokkos::DynRankView<Scalar,ExecSpaceType>;
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
      const int numEntries = fc.size();
      Kokkos::RangePolicy < ExecSpaceType > policy(0,numEntries);
      Kokkos::parallel_for( "copy to fc device view", policy,
        KOKKOS_LAMBDA (const int &enumerationIndex )
        {
          ViewIterator    fcViewIterator(deviceView);
          FunctorIterator viewLikeContainerIterator(viewLikeContainer);

          fcViewIterator.setEnumerationIndex(enumerationIndex);
          viewLikeContainerIterator.setEnumerationIndex(enumerationIndex);
          
          fcViewIterator.set(viewLikeContainerIterator.get());
        }
      );
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
    static Camellia::FieldContainer<Scalar, 0> fromTensorData(TensorData &tensorData)
    {
      // TensorData can be rank 1, 2 or 3
      const int rank = Intrepid2::getFunctorRank(tensorData);
      if (rank == 1)
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
    
    static Camellia::FieldContainer<Scalar, 0> fromTensorPoints(TensorPoints &tensorPoints)
    {
      // TensorPoints is always rank 2
      const int rank = 2;
      return fromDeviceContainerWithRank<TensorPoints,rank>(tensorPoints);
    }
    
    static Camellia::FieldContainer<Scalar, 0> fromDynRankView(DynRankView &view)
    {
      // DynRankView can be any rank, up to 7
      return fromDynRankView(view);
    }
    
    static TensorPoints toTensorPoints(Camellia::FieldContainer<Scalar, 0> &pointsFC)
    {
      // pointsFC is shape (P,D)
      auto deviceView = pointsFC.checkoutDeviceView(true); // true: deep copy from host
      auto pointsView = Allocation<Scalar>::allocateDeviceViewMatchingView(deviceView);
      Kokkos::deep_copy(pointsView,deviceView);
      // the following deallocation record is a cheat: since pointsView is wrapped in TensorPoints (which is Intrepid2), we don't have a great hook for determining when it goes out of scope and the deallocation is about to occur.  Ultimately, usage of this conversion method will be replaced by native Intrepid2 methods, which probably will do their own allocations, not necessarily recorded by Camellia.
      Allocation<Scalar>::recordDeviceViewDeallocation(pointsView);
      return TensorPoints(pointsView); // will have trivial tensor product structure
    }
  };
}

#endif /* Camellia_FieldContainerConversion_h */
