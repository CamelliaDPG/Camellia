//
//  Camellia_TensorIndices.h
//  Camellia
//
//  Created by Roberts, Nathan V on 9/29/21.
//

#ifndef Camellia_TensorIndices_h
#define Camellia_TensorIndices_h

#include <Intrepid2_Types.hpp>
#include <Kokkos_Core.hpp>

namespace Camellia
{
  namespace TensorIndices
  {
    // TODO: name/place this method somewhere more appropriate; defined here because used in SolutionMomentFunction, and outside SolutionMomentFunction to allow external testing
    KOKKOS_INLINE_FUNCTION void extractNextComponentOrdinal(int &ordinalInComponent, int &remainingTensorOrdinal, const int &componentCardinality)
    {
      ordinalInComponent = remainingTensorOrdinal % componentCardinality;
      remainingTensorOrdinal /= componentCardinality;
    }

    // TODO: name/place this method somewhere more appropriate; defined here because used in SolutionMomentFunction, and outside SolutionMomentFunction to allow external testing
    KOKKOS_INLINE_FUNCTION void setTensorOrdinal(int &tensorOrdinal, const int &spatialOrdinal, const int &spatialCardinality, const int &extrusionFieldOrdinal)
    {
      tensorOrdinal = spatialOrdinal + spatialCardinality * extrusionFieldOrdinal;
    }
  
    KOKKOS_INLINE_FUNCTION void setTensorOrdinal(int &tensorOrdinal, const int &spatialOrdinal, const int &spatialCardinality, const int &extrusionFieldOrdinal1, const int &extrusionCardinality1, const int &extrusionFieldOrdinal2)
    {
      tensorOrdinal = spatialOrdinal + spatialCardinality * (extrusionFieldOrdinal1 + extrusionCardinality1 * extrusionFieldOrdinal2);
    }

    KOKKOS_INLINE_FUNCTION void setTensorOrdinal(int &tensorOrdinal, const Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> &componentOrdinals,
                                                 const Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> &componentCardinalities,
                                                 const int &numComponents)
    {
      tensorOrdinal = 0;
      for (int d=numComponents-1; d>0; d--)
      {
        tensorOrdinal += componentOrdinals[d];
        tensorOrdinal *= componentCardinalities[d-1];
      }
      tensorOrdinal += componentOrdinals[0];
    }
  }
};

#endif /* Camellia_TensorIndices_h */
