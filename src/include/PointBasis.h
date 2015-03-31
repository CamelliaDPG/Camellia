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
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace std;

template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class PointBasis;
template<class Scalar, class ArrayScalar> class PointBasis : public Camellia::Basis<Scalar,ArrayScalar> {
  void initializeTags() const;
public:
  PointBasis();
  
  void getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                 const Intrepid::EOperator operatorType) const;
};

typedef Teuchos::RCP< PointBasis<> > PointBasisPtr;

#include "PointBasisDef.h"

#endif
