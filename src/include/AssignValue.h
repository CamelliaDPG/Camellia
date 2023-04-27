//
//  AssignValue.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/18/20.
//

#ifndef AssignValue_h
#define AssignValue_h

#include "Camellia_FieldContainer.hpp"

#include "Kokkos_Core.hpp"

namespace Camellia {
  enum ValueModificationType
  {
    ASSIGN,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    MINIMUM,
    MAXIMUM
  };
  
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  void assignValue(Scalar &existingValue, const ValueModificationType modType, const Scalar valueToAssign)
  {
    switch (modType) {
      case ASSIGN:
        existingValue = valueToAssign;
        break;
      case ADD:
        existingValue += valueToAssign;
        break;
      case MULTIPLY:
        existingValue *= valueToAssign;
        break;
      case DIVIDE:
        existingValue /= valueToAssign;
        break;
      case SUBTRACT:
        existingValue -= valueToAssign;
        break;
      case MINIMUM:
        existingValue = std::fmin(existingValue, valueToAssign);
        break;
      case MAXIMUM:
        existingValue = std::fmax(existingValue, valueToAssign);
    }
  }
  
  template<typename Scalar, ValueModificationType modType, int rank=7>
  struct AssignValuesFunctor
  {
    using DeviceView = typename Camellia::FieldContainer<Scalar>::DeviceView;
    using array_index_type = typename Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type; // hard-coding the rank because MDRangePolicy not defined for rank 1 
    
    template< bool B, class T = void >
    using enable_if_t = typename std::enable_if<B,T>::type;
    
    DeviceView _toValues;
    DeviceView _fromValues;
    const int  _dim6;
    
    AssignValuesFunctor(DeviceView toValues, DeviceView fromValues)
    :
    _toValues(toValues),
    _fromValues(fromValues),
    _dim6(_toValues.extent_int(6))
    {}

    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==1>
    operator()(const array_index_type i0) const
    {
      assignValue(_toValues.access(i0),modType,_fromValues.access(i0));
    }
    
    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==2>
    operator()(const array_index_type i0, const array_index_type i1) const
    {
      assignValue(_toValues.access(i0,i1),modType,_fromValues.access(i0,i1));
    }
    
    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==3>
    operator()(const array_index_type i0, const array_index_type i1, const array_index_type i2) const
    {
      assignValue(_toValues.access(i0,i1,i2),modType,_fromValues.access(i0,i1,i2));
    }
    
    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==4>
    operator()(const array_index_type i0, const array_index_type i1, const array_index_type i2,
               const array_index_type i3) const
    {
      assignValue(_toValues.access(i0,i1,i2,i3),modType,_fromValues.access(i0,i1,i2,i3));
    }
    
    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==5>
    operator()(const array_index_type i0, const array_index_type i1, const array_index_type i2,
               const array_index_type i3, const array_index_type i4) const
    {
      assignValue(_toValues.access(i0,i1,i2,i3,i4),modType,_fromValues.access(i0,i1,i2,i3,i4));
    }
    
    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==6>
    operator()(const array_index_type i0, const array_index_type i1, const array_index_type i2,
               const array_index_type i3, const array_index_type i4, const array_index_type i5) const
    {
      assignValue(_toValues.access(i0,i1,i2,i3,i4,i5),modType,_fromValues.access(i0,i1,i2,i3,i4,i5));
    }
    
    template <int M = rank>
    KOKKOS_INLINE_FUNCTION
    enable_if_t<M==7>
    operator()(const array_index_type i0, const array_index_type i1, const array_index_type i2,
               const array_index_type i3, const array_index_type i4, const array_index_type i5) const
    {
      for (int i6 = 0; i6<_dim6; i6++)
      {
        assignValue(_toValues.access(i0,i1,i2,i3,i4,i5,i6),modType,_fromValues.access(i0,i1,i2,i3,i4,i5,i6));
      }
    }
  };
  
  template<typename Scalar, ValueModificationType modType>
  void assignValues(typename Camellia::FieldContainer<Scalar>::DeviceView &toValues, typename Camellia::FieldContainer<Scalar>::DeviceView &fromValues)
  {
    const int rank = toValues.rank();
    switch (rank)
    {
      case 1:
      {
        AssignValuesFunctor<Scalar,modType,1> functor(toValues, fromValues);
        Kokkos::parallel_for("AssignValuesFunctor", toValues.extent(0), functor);
      }
        break;
      case 2:
      {
        auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy2(toValues);
        AssignValuesFunctor<Scalar,modType,2> functor(toValues, fromValues);
        Kokkos::parallel_for("AssignValuesFunctor", rangePolicy, functor);
      }
        break;
      case 3:
      {
        auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy3(toValues);
        AssignValuesFunctor<Scalar,modType,3> functor(toValues, fromValues);
        Kokkos::parallel_for("AssignValuesFunctor", rangePolicy, functor);
      }
        break;
      case 4:
      {
        auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy4(toValues);
        AssignValuesFunctor<Scalar,modType,4> functor(toValues, fromValues);
        Kokkos::parallel_for("AssignValuesFunctor", rangePolicy, functor);
      }
        break;
      case 5:
      {
        auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy5(toValues);
        AssignValuesFunctor<Scalar,modType,5> functor(toValues, fromValues);
        Kokkos::parallel_for("AssignValuesFunctor", rangePolicy, functor);
      }
        break;
      case 6:
      {
        auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy6(toValues);
        AssignValuesFunctor<Scalar,modType,6> functor(toValues, fromValues);
        Kokkos::parallel_for( "AssignValuesFunctor", rangePolicy, functor);
      }
        break;
      case 7:
      {
        auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy6(toValues); // range policy only goes up to rank 6
        AssignValuesFunctor<Scalar,modType,7> functor(toValues, fromValues);
        Kokkos::parallel_for( "AssignValuesFunctor", rangePolicy, functor);
      }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank");
    }
  }
  
  template<typename Scalar>
  void assignValues(typename Camellia::FieldContainer<Scalar>::DeviceView &toValues, ValueModificationType modType,
                    typename Camellia::FieldContainer<Scalar>::DeviceView &fromValues)
  {
    switch (modType) {
      case ASSIGN:
        assignValues<Scalar,ASSIGN>(toValues, fromValues);
        break;
      case ADD:
        assignValues<Scalar,ADD>(toValues, fromValues);
        break;
      case MULTIPLY:
        assignValues<Scalar,MULTIPLY>(toValues, fromValues);
        break;
      case DIVIDE:
        assignValues<Scalar,DIVIDE>(toValues, fromValues);
        break;
      case SUBTRACT:
        assignValues<Scalar,SUBTRACT>(toValues, fromValues);
        break;
      case MINIMUM:
        assignValues<Scalar,MINIMUM>(toValues, fromValues);
        break;
      case MAXIMUM:
        assignValues<Scalar,MAXIMUM>(toValues, fromValues);
    }
  }
}

#endif /* AssignValue_h */
