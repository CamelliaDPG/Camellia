//
//  AssignValue.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/18/20.
//

#ifndef AssignValue_h
#define AssignValue_h

#include "Kokkos_Core.hpp"

namespace Camellia {
  enum ValueModificationType
  {
    ASSIGN,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
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
    }
  }
}

#endif /* AssignValue_h */
