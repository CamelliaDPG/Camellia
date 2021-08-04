// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER
//
//  Camellia_KokkosHelpers.h
//  Camellia
//
//  Created by Roberts, Nathan V on 4/15/21.
//

#ifndef Camellia_KokkosHelpers_h
#define Camellia_KokkosHelpers_h

#include "Intrepid2_FunctorIterator.hpp"
#include "Intrepid2_ViewIterator.hpp"

namespace Camellia {

  //! helper method to perform a deep copy; this supports cases that Kokkos::deep_copy does not, including Views with incompatible layouts.  Note that such copies may have negative performance implications.
  template<int rank, class ViewType, class ViewLikeContainer>
  static void deepCopy(ViewType view, ViewLikeContainer viewLikeContainer)
  {
    using ExecSpaceType = typename ViewType::execution_space;
    using Scalar        = typename ViewType::value_type;
    
    using FunctorIterator = Intrepid2::FunctorIterator<ViewLikeContainer,Scalar,rank>;
    using ViewIterator    = Intrepid2::ViewIterator<ViewType,Scalar>;

    int entryCount = 1;
    for (int r=0; r<rank; r++)
    {
      entryCount *= view.extent_int(r);
    }
    Kokkos::RangePolicy < ExecSpaceType > policy(0,entryCount);
    Kokkos::parallel_for( "copy data to view", policy,
      KOKKOS_LAMBDA (const int &enumerationIndex )
      {
        ViewIterator    viewIterator(view);
        FunctorIterator viewLikeContainerIterator(viewLikeContainer);

        viewIterator.setEnumerationIndex(enumerationIndex);
        viewLikeContainerIterator.setEnumerationIndex(enumerationIndex);
        
        viewIterator.set(viewLikeContainerIterator.get());
      }
    );
  }
    
  //! helper method to perform a deep copy; this supports cases that Kokkos::deep_copy does not, including Views with incompatible layouts.  Note that such copies may have negative performance implications.
  template<class ViewType, class ViewLikeContainer>
  static void deepCopy(ViewType view, ViewLikeContainer viewLikeContainer)
  {
    const auto rank = view.rank();
    switch (rank) {
      case 1: deepCopy<1, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
      case 2: deepCopy<2, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
      case 3: deepCopy<3, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
      case 4: deepCopy<4, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
      case 5: deepCopy<5, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
      case 6: deepCopy<6, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
      case 7: deepCopy<7, ViewType, ViewLikeContainer>(view, viewLikeContainer); break;
        
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported view rank");
    }
  }
} // namespace Camellia

#endif /* Camellia_KokkosHelpers_h */
