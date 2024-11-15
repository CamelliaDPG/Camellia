// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PointBasisDef.h
//  Camellia
//
//  Created by Nathan Roberts on 10/24/13.
//
//

#ifndef Camellia_PointBasisDef_h
#define Camellia_PointBasisDef_h

#include "CamelliaIntrepidExtendedTypes.h"

#include "CellTopology.h"

#include "PointBasisIntrepid2.h"

namespace Camellia
{
  template<class Scalar, class ArrayScalar>
  PointBasis<Scalar,ArrayScalar>::PointBasis()
  :
  Intrepid2BasisWrapper<Scalar, ArrayScalar>(Teuchos::rcp(new PointBasisIntrepid2<Camellia::DeviceType,Scalar,Scalar>),
                                             0,0,::Camellia::FUNCTION_SPACE_HGRAD)
  {}

  template<class Scalar, class ArrayScalar>
  Teuchos::RCP< PointBasis<Scalar,ArrayScalar> >
  PointBasis<Scalar,ArrayScalar>::pointBasis()
  {
    using PointBasisPtr = Teuchos::RCP< PointBasis<Scalar,ArrayScalar> >;
    static PointBasisPtr pointBasis = Teuchos::rcp(new PointBasis<Scalar,ArrayScalar>());
    
    Kokkos::push_finalize_hook( [=] {
      pointBasis = Teuchos::null;
    });
    
    return pointBasis;
  }
}

#endif
