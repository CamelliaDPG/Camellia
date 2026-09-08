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

  namespace Detail {
    template<int rank, class ViewType, class ViewLikeContainer>
    struct DeepCopyImpl
    {
      static void run(ViewType view, ViewLikeContainer viewLikeContainer)
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
    };

    template<class ViewType, class ViewLikeContainer>
    struct DeepCopyImpl<1,ViewType,ViewLikeContainer>
    {
      static void run(ViewType view, ViewLikeContainer viewLikeContainer)
      {
        using ExecSpaceType = typename ViewType::execution_space;
        const int dim0 = view.extent_int(0);
        Kokkos::RangePolicy < ExecSpaceType > policy(0,dim0);
        Kokkos::parallel_for( "copy rank-1 data to view", policy,
          KOKKOS_LAMBDA (const int &i0 )
          {
            view(i0) = viewLikeContainer(i0);
          }
        );
      }
    };

    template<class ViewType, class ViewLikeContainer>
    struct DeepCopyImpl<2,ViewType,ViewLikeContainer>
    {
      static void run(ViewType view, ViewLikeContainer viewLikeContainer)
      {
        using ExecSpaceType = typename ViewType::execution_space;
        using PolicyType = Kokkos::MDRangePolicy<ExecSpaceType,Kokkos::Rank<2>,Kokkos::IndexType<int> >;
        const PolicyType policy({0,0},{view.extent_int(0),view.extent_int(1)});
        Kokkos::parallel_for( "copy rank-2 data to view", policy,
          KOKKOS_LAMBDA (const int &i0, const int &i1 )
          {
            view(i0,i1) = viewLikeContainer(i0,i1);
          }
        );
      }
    };

    template<class ViewType, class ViewLikeContainer>
    struct DeepCopyImpl<3,ViewType,ViewLikeContainer>
    {
      static void run(ViewType view, ViewLikeContainer viewLikeContainer)
      {
        using ExecSpaceType = typename ViewType::execution_space;
        using PolicyType = Kokkos::MDRangePolicy<ExecSpaceType,Kokkos::Rank<3>,Kokkos::IndexType<int> >;
        const PolicyType policy({0,0,0},{view.extent_int(0),view.extent_int(1),view.extent_int(2)});
        Kokkos::parallel_for( "copy rank-3 data to view", policy,
          KOKKOS_LAMBDA (const int &i0, const int &i1, const int &i2 )
          {
            view(i0,i1,i2) = viewLikeContainer(i0,i1,i2);
          }
        );
      }
    };

    template<class ViewType, class ViewLikeContainer>
    struct DeepCopyImpl<4,ViewType,ViewLikeContainer>
    {
      static void run(ViewType view, ViewLikeContainer viewLikeContainer)
      {
        using ExecSpaceType = typename ViewType::execution_space;
        using PolicyType = Kokkos::MDRangePolicy<ExecSpaceType,Kokkos::Rank<4>,Kokkos::IndexType<int> >;
        const PolicyType policy({0,0,0,0},{view.extent_int(0),view.extent_int(1),view.extent_int(2),view.extent_int(3)});
        Kokkos::parallel_for( "copy rank-4 data to view", policy,
          KOKKOS_LAMBDA (const int &i0, const int &i1, const int &i2, const int &i3 )
          {
            view(i0,i1,i2,i3) = viewLikeContainer(i0,i1,i2,i3);
          }
        );
      }
    };
  }

  //! helper method to perform a deep copy; this supports cases that Kokkos::deep_copy does not, including Views with incompatible layouts.  Note that such copies may have negative performance implications.
  template<int rank, class ViewType, class ViewLikeContainer>
  static void deepCopy(ViewType view, ViewLikeContainer viewLikeContainer)
  {
    Detail::DeepCopyImpl<rank,ViewType,ViewLikeContainer>::run(view, viewLikeContainer);
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
