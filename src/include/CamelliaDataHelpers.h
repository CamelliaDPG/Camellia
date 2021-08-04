// @HEADER
// ***********************************************************************
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-Teuchos in the licenses directory.
//
// Some variations on Teuchos_TestingHelpers.hpp and Teuchos_LocalTestingHelpers.hpp, which are included with Trilinos.
//
// Questions? Contact Nathan V. Roberts (nate@nateroberts.com)
//
// ***********************************************************************
// @HEADER

#ifndef CAMELLIA_DATA_HELPERS_HPP
#define CAMELLIA_DATA_HELPERS_HPP

/*! \file CamelliaDataHelpers.h
    \brief Utilities for working with Intrepid2::Data containers.
*/

#include <Intrepid2_Data.hpp>
#include <Intrepid2_TestUtils.hpp>

#include "Function.h"
#include "TypeDefs.h"

namespace Camellia
{
  using ordinal_type = Intrepid2::ordinal_type;

  //! helper method for copying host data to device data with possibly-incompatible layout (such that deep_copy is not available).
  template<class Scalar, int rank>
  void copyHostDataToDevice(Intrepid2::Data<Scalar, Camellia::DeviceType> deviceData,
                            Intrepid2::Data<Scalar,Kokkos::HostSpace::execution_space> hostData)
  {
    using MemorySpace = typename Camellia::DeviceType::memory_space;
    
    auto hostDataMirror = Kokkos::create_mirror_view_and_copy(MemorySpace(), hostData.template getUnderlyingView<rank>());
    deviceData.copyContainer(deviceData.template getUnderlyingView<rank>(), hostDataMirror);
  }

  template<class Scalar>
  void copyHostDataToDevice(Intrepid2::Data<Scalar, Camellia::DeviceType> deviceData,
                            Intrepid2::Data<Scalar,Kokkos::HostSpace::execution_space> hostData)
  {
    using DeviceData = decltype(deviceData);
    using HostData   = decltype(hostData);
    switch (deviceData.getUnderlyingViewRank())
    {
      case 1: copyHostDataToDevice<Scalar, 1>(deviceData, hostData); break;
      case 2: copyHostDataToDevice<Scalar, 2>(deviceData, hostData); break;
      case 3: copyHostDataToDevice<Scalar, 3>(deviceData, hostData); break;
      case 4: copyHostDataToDevice<Scalar, 4>(deviceData, hostData); break;
      case 5: copyHostDataToDevice<Scalar, 5>(deviceData, hostData); break;
      case 6: copyHostDataToDevice<Scalar, 6>(deviceData, hostData); break;
      case 7: copyHostDataToDevice<Scalar, 7>(deviceData, hostData); break;
      default: INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid data rank");
    }
  }

  //! Matches the argument expansion used by Intrepid2::TensorData.
  template<ordinal_type numTensorComponents>
  KOKKOS_INLINE_FUNCTION Intrepid2::ordinal_type tensorEnumerationIndex(const Kokkos::Array<ordinal_type,numTensorComponents> &argComponents,
                                                                        const Kokkos::Array<ordinal_type,numTensorComponents> &dims)
  {
    // TensorData's numbering has the leftmost argument as the fastest-moving
    // We return that numbering here.
    if (numTensorComponents == 1)
    {
      return argComponents[0];
    }
    else
    {
      ordinal_type i = 0;
      for (ordinal_type r=numTensorComponents-1; r>0; r--)
      {
        i += argComponents[r];
        i *= dims[r-1];
      }
      i += argComponents[0];
      return i;
    }
  }

  KOKKOS_INLINE_FUNCTION ordinal_type tensorEnumerationIndex(const ordinal_type &i0, const ordinal_type &i1,
                                                             const ordinal_type &dim0)
  {
    return i0 + dim0 * i1;
  }

  KOKKOS_INLINE_FUNCTION ordinal_type tensorEnumerationIndex(const ordinal_type &i0, const ordinal_type &i1, const ordinal_type &i2,
                                                             const ordinal_type &dim0, const ordinal_type &dim1)
  {
    return i0 + dim0 * (i1 + dim1 * i2);
  }

  KOKKOS_INLINE_FUNCTION ordinal_type tensorEnumerationIndex(const ordinal_type &i0, const ordinal_type &i1, const ordinal_type &i2, const ordinal_type &i3,
                                                             const ordinal_type &dim0, const ordinal_type &dim1, const ordinal_type &dim2)
  {
    return i0 + dim0 * (i1 + dim1 * (i2 + dim2 * i3));
  }

  KOKKOS_INLINE_FUNCTION ordinal_type tensorEnumerationIndex(const ordinal_type &i0, const ordinal_type &i1, const ordinal_type &i2,
                                                             const ordinal_type &i3, const ordinal_type &i4,
                                                             const ordinal_type &dim0, const ordinal_type &dim1,
                                                             const ordinal_type &dim2, const ordinal_type &dim3)
  {
    return i0 + dim0 * (i1 + dim1 * (i2 + dim2 * (i3 + dim3 * i4)));
  }

  //! Wraps a View(-like) container with e.g. (C,P) dimensions, and presents as a container with (C,P0,P1,…,Pn) where the Pi dimensions are tensor components in the P dimension.
  template<class Scalar, class ViewType, ordinal_type expandedRank, ordinal_type startCollapseDim, ordinal_type lastCollapseDim>
  class ViewCollapseDimensions
  {
    static constexpr ordinal_type numCollapsedDims = lastCollapseDim - startCollapseDim + 1;
    static constexpr ordinal_type collapsedRank    = expandedRank - numCollapsedDims;
    
    template< bool B, class T = void >
    using enable_if_t = typename std::enable_if<B,T>::type;
    
    using return_type = typename ViewType::reference_type;
    
    using array_index_type = int;
    
    ViewType _view;
    
    Kokkos::Array<int,numCollapsedDims> _componentDims;
  public:
    static constexpr int rank = expandedRank;
    using value_type          = typename ViewType::value_type;
    using execution_space     = typename ViewType::execution_space;
    
    ViewCollapseDimensions(ViewType view, Kokkos::Array<int,numCollapsedDims> componentDims)
    :
    _view(view),
    _componentDims(componentDims)
    {}
    
    template <int M = expandedRank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==1, return_type>
    operator()(const array_index_type & i0) const
    {
      // M == 1 means that the "collapse" is trivial; pass argument to data
      return _view(i0);
    }
    
    template <int M = expandedRank, int P0 = startCollapseDim, int Pn = lastCollapseDim>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<(M==3) && (P0 == 1) && (Pn == 2), return_type>
    operator()(const array_index_type & i0, const array_index_type & i1, const array_index_type & i2) const
    {
      // the (C,P0,P1) --> (C,P) case
      const ordinal_type pointOrdinal = tensorEnumerationIndex(i1,i2,_componentDims[0]);
      return _view(i0,pointOrdinal);
    }
    
    template <int M = expandedRank, int P0 = startCollapseDim, int Pn = lastCollapseDim>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<(M==5) && (P0 == 1) && (Pn == 4), return_type>
    operator()(const array_index_type & i0, const array_index_type & i1, const array_index_type & i2,
               const array_index_type & i3, const array_index_type & i4) const
    {
      // the (C,P0,P1,P2,P3) --> (C,P) case
      const ordinal_type pointOrdinal = tensorEnumerationIndex(i1,i2,i3,i4,_componentDims[0],_componentDims[1],_componentDims[2]);
      return _view(i0,pointOrdinal);
    }
    
    KOKKOS_INLINE_FUNCTION
    int extent_int(const ordinal_type &d) const
    {
      if (d < startCollapseDim)
      {
        return _view.extent_int(d);
      }
      else if (d <= lastCollapseDim)
      {
        return _componentDims[d - startCollapseDim];
      }
      else
      {
        return _view.extent_int(d - numCollapsedDims + 1);
      }
    }
  };
  
  void testFunctionDataPathMatchesViewPath1V(FunctionPtr f, BasisCachePtr basisCache1V, const double &tol, Teuchos::FancyOStream &out, bool &success);
  void testFunctionDataPathMatchesViewPath3V(FunctionPtr f, BasisCachePtr basisCache3V, const double &tol, Teuchos::FancyOStream &out, bool &success);
} // namespace Camellia
#endif  // CAMELLIA_DATA_HELPERS_HPP
