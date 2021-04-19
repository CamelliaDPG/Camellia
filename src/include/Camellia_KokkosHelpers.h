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
  template<class ViewType, class ViewLikeContainer, int rank>
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
      case 1: deepCopy<ViewType, ViewLikeContainer, 1>(view, viewLikeContainer); break;
      case 2: deepCopy<ViewType, ViewLikeContainer, 2>(view, viewLikeContainer); break;
      case 3: deepCopy<ViewType, ViewLikeContainer, 3>(view, viewLikeContainer); break;
      case 4: deepCopy<ViewType, ViewLikeContainer, 4>(view, viewLikeContainer); break;
      case 5: deepCopy<ViewType, ViewLikeContainer, 5>(view, viewLikeContainer); break;
      case 6: deepCopy<ViewType, ViewLikeContainer, 6>(view, viewLikeContainer); break;
      case 7: deepCopy<ViewType, ViewLikeContainer, 7>(view, viewLikeContainer); break;
        
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported view rank");
    }
  }
} // namespace Camellia

#endif /* Camellia_KokkosHelpers_h */
