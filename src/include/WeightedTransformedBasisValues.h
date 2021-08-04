//
//  WeightedTransformedBasisValues.h
//  Camellia
//
//  Created by Roberts, Nathan V on 5/3/21.
//

#ifndef WeightedTransformedBasisValues_h
#define WeightedTransformedBasisValues_h

#include "TypeDefs.h"

#include <Intrepid2_TransformedVectorData.hpp>

namespace Camellia {
/**
 \class WeightedTransformedBasisValues
 
 This probably will be contributed to Intrepid2; it is essentially a generalization of TransformedVectorData to apply to more BasisValues types.
 */
  template<class Scalar, class DeviceType = Camellia::DeviceType>
  class WeightedTransformedBasisValues
  {
    using TransformedVectorData = Intrepid2::TransformedVectorData<Scalar,Camellia::DeviceType>;
    static constexpr auto MaxTensorComponents = Intrepid2::Parameters::MaxTensorComponents;
    
    BasisValues _basisValues; // (F,P[,…])
    Data _transform; // (C,P); can also be (C,P,D,D) for vector-valued basisValues.  If left unspecified, the identity is assumed.
    Data _scalarFunctionWeight; // (C,P1,P2,…) -- points expressed in terms of tensorial components.  If left unspecified, the identity is assumed.
    
    int numPointComponents_;
    
    Kokkos::Array<ordinal_type, MaxTensorComponents> pointModulus_;
    Kokkos::Array<ordinal_type, MaxTensorComponents> pointDivisor_;
    
    int cellCount_ = 1; // 1 is the default cell count; in this case, the weight and transform are unit and identity, respectively, and any cell ordinal is allowed.
    
    void initialize()
    {
      numPointComponents_ = _scalarFunctionWeight.rank() - 1;
      
      const int numPointsTotal = _basisValues.extent_int(1);
      INTREPID2_TEST_FOR_EXCEPTION(numPointComponents > MaxTensorComponents, std::invalid_argument, "Can't have more than MaxTensorComponents point components");
      
      if (_transform.isValid())
      {
        cellCount_ = _transform.extent_int(0);
      }
      else if (_scalarFunctionWeight.isValid())
      {
        cellCount_ = _scalarFunctionWeight.extent_int(0);
      }
      
      int totalPointCount = 1;
      ordinal_type pointDivisor = 1;
      
      ordinal_type d=0;
      ordinal_type dimsSoFar = 0;
      for (ordinal_type r=0; r<numPointComponents; r++)
      {
        const int componentPointCount = _scalarFunctionWeight.extent_int(r+1); // first dimension is cell dimension; subsequent are point dimensions
        
        totalPointCount *= componentPointCount;
        dimsSoFar       += componentDim;
        
        pointModulus_[r] = componentPointCount;
        pointDivisor_[r] = pointDivisor;
        pointDivisor *= componentPointCount;
      }
      extents_[0] = cellCount;
      extents_[1] = totalPointCount;
      extents_[2] = dimsSoFar;
    }
  public:
    WeightedTransformedBasisValues(Data functionWeight, Data transform, BasisValues basisValues)
    :
    _basisValues(basisValues),
    _transform(transform),
    _scalarFunctionWeight(scalarFunctionWeight)
    {}
    
    //! Accessor, with arguments (C,F,P,D), for vector-valued BasisValues.
    KOKKOS_INLINE_FUNCTION Scalar operator()(const int &cellOrdinal, const int &fieldOrdinal, const int &pointOrdinal, const int &dim) const
    {
      const Scalar basisValue = _basisValues(fieldOrdinal,pointOrdinal,dim);
      const Scalar transformedValue;
      
      if (!transform_.isValid())
      {
        // null transform is understood as the identity
        transformedValue = vectorData_(fieldOrdinal,pointOrdinal,dim);
      }
      else if (transform_.isDiagonal())
      {
        transformedValue = transform_(cellOrdinal,pointOrdinal,dim,dim) * vectorData_(fieldOrdinal,pointOrdinal,dim);
      }
      else
      {
        transformedValue = 0.0;
        for (int d2=0; d2<transform_.extent_int(2); d2++)
        {
          transformedValue += transform_(cellOrdinal,pointOrdinal,dim,d2) * vectorData_(fieldOrdinal,pointOrdinal,d2);
        }
      }
      
      Scalar functionWeight;
      if (! _scalarFunctionWeight.isValid())
      {
        // null weight is understood as unit weight
        functionWeight = 1.0;
      }
      else
      {
        Kokkos::Array<int,MaxTensorComponents> pointComponents;
        for (int r=0; r<numPointComponents_; r++)
        {
          pointComponents[r] = (point_ordinal / pointDivisor_[r]) % pointModulus_[r];
        }
        switch (numPointComponents_)
        {
          case 1:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0]);
            break;
          case 2:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0], pointComponents[1]);
            break;
          case 3:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0], pointComponents[1], pointComponents[2]);
            break;
          case 4:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0], pointComponents[1], pointComponents[2], pointComponents[3]);
            break;
          case 5:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0], pointComponents[1], pointComponents[2], pointComponents[3], pointComponents[4]);
            break;
          case 6:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0], pointComponents[1], pointComponents[2], pointComponents[3], pointComponents[4], pointComponents[5]);
            break;
          case 7:
            functionWeight = _scalarFunctionWeight(cell_ordinal, pointComponents[0], pointComponents[1], pointComponents[2], pointComponents[3], pointComponents[4], pointComponents[5], pointComponents[6]);
            break;
            
          default:
            INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(true, std::logic_error, "Unhandled numPointComponents_ value");
            break;
        }
      }
      
      return functionWeight * transformedValue;
    }
    
    KOKKOS_INLINE_FUNCTION
    int extent_int(const int& r) const {
      // shape is (C,F,P[,D])
      switch (r)
      {
        case 0: return _cellCount;
        case 1: return _basisValues.extent_int(0); // _basisValues shape: (F,P[,D])
        case 2: return _basisValues.extent_int(1);
        case 3: return _basisValues.extent_int(2);
        default:
          INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(true, std::logic_error, "invalid dimension");
      }
    }
    
    //! Return the rank of the container, which is 3 or 4, depending on the rank of basis values.
    KOKKOS_INLINE_FUNCTION
    const ordinal_type rank() const
    {
      return _basisValues.rank() + 1; // _basisValues shape: (F,P[,D]); our shape is (C,F,P[,D])
    }
  };
}

#endif /* WeightedTransformedBasisValues_h */
