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
  
  template<typename Scalar, ValueModificationType modType>
  struct AssignValuesFunctor
  {
    using DeviceView = typename Camellia::FieldContainer<Scalar>::DeviceView;
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    
    DeviceView _toValues;
    DeviceView _fromValues;
    const int  _dim6;
    
    AssignValuesFunctor(DeviceView toValues, DeviceView fromValues)
    :
    _toValues(toValues),
    _fromValues(fromValues),
    _dim6(_toValues.extent_int(6))
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const array_index_type i0, const array_index_type i1, const array_index_type i2,
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
    auto rangePolicy = Camellia::FieldContainer<Scalar>::getMDRangePolicy6(toValues);
    AssignValuesFunctor<Scalar,modType> functor(toValues, fromValues);
    Kokkos::parallel_for( rangePolicy , functor, "AssignValuesFunctor");
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
