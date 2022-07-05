// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PointBasis.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 10/24/13.
//
//

#ifndef Camellia_PointBasis_hpp
#define Camellia_PointBasis_hpp

/** \file
 \brief  Header file for the Camellia PointBasis class.
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
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class PointBasis;
template<class Scalar, class ArrayScalar> class PointBasis : public Camellia::Intrepid2BasisWrapper<>
{
public:
  PointBasis();
  
  static Teuchos::RCP< PointBasis<Scalar,ArrayScalar> > pointBasis();
};

typedef Teuchos::RCP< PointBasis<> > PointBasisPtr;
}

#include "PointBasisDef.h"

#endif
