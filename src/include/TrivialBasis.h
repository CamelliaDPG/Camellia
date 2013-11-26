//
//  TrivialBasis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_TrivialBasis_h
#define Camellia_debug_TrivialBasis_h

#include "Basis.h"
#include "Intrepid_FieldContainer.hpp"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class TrivialBasis;
  template<class Scalar, class ArrayScalar> class TrivialBasis : public Basis<Scalar,ArrayScalar> {
  protected:
    void initializeTags() const;
    int _degree;
    bool _conforming;

    Intrepid::FieldContainer<double> _lobattoL2norms;
    void initializeL2normValues();
  public:
    TrivialBasis(int degree, bool conforming); // conforming means vertex dofs defined...

    void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;

    bool isConforming() const;
  };
}

#include "TrivialBasisDef.h"

#endif
