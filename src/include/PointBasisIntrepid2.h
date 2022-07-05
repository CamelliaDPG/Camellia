// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PointBasisIntrepid2.hpp
//
//  Created by Nathan Roberts on 4/12/22.
//
//

#ifndef Camellia_PointBasisIntrepid2_hpp
#define Camellia_PointBasisIntrepid2_hpp

/** \file
 \brief  Header file for the Camellia PointBasisIntrepid2 class.
 \author Created by N. Roberts
 */

#include "Basis.h"
#include "Camellia_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace std;

namespace Camellia
{
/** \class  PointBasisIntrepid2
    \brief  Implementation of the default HVOL-compatible FEM contstant basis on node topology.
*/
template<typename DeviceType = void,
         typename outputValueType = double,
         typename pointValueType = double>
class PointBasisIntrepid2 : public Intrepid2::Basis<DeviceType,outputValueType,pointValueType> {
public:
 using OrdinalTypeArray1DHost = typename Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::OrdinalTypeArray1DHost;
 using OrdinalTypeArray2DHost = typename Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::OrdinalTypeArray2DHost;
 using OrdinalTypeArray3DHost = typename Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::OrdinalTypeArray3DHost;

 /** \brief Constructor.
  */
  PointBasisIntrepid2();

 using OutputViewType = typename Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::OutputViewType;
 using PointViewType  = typename Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::PointViewType;
 using ScalarViewType = typename Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::ScalarViewType;

 using Intrepid2::Basis<DeviceType,outputValueType,pointValueType>::getValues;

 virtual
 void
 getValues(       OutputViewType outputValues,
            const PointViewType  inputPoints,
            const Intrepid2::EOperator operatorType = Intrepid2::OPERATOR_VALUE ) const override
{
  Kokkos::deep_copy(outputValues,1.0);
}

 virtual
 const char* getName() const override {
   return "PointBasis_Intrepid2";
 }

 virtual Intrepid2::HostBasisPtr<outputValueType,pointValueType>
 getHostBasis() const override{
   return Teuchos::rcp(new PointBasisIntrepid2<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>());
 }
};
} // namespace Camellia

#include "PointBasisIntrepid2Def.h"

#endif
