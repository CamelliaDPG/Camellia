//
//  TensorUtilities.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/15/21.
//

#ifndef TensorUtilities_h
#define TensorUtilities_h

namespace Camellia {
  KOKKOS_INLINE_FUNCTION int getTensorPointOrdinal(const int spatialPointOrdinal, const int velocityPointOrdinal, const int numSpatialPoints)
  {
    return spatialPointOrdinal + numSpatialPoints * velocityPointOrdinal;
  }
}

#endif /* TensorUtilities_h */
