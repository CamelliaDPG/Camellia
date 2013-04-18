//
//  LobattoHGRAD_LineBasis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LobattoHGRAD_LineBasis_h
#define Camellia_debug_LobattoHGRAD_LineBasis_h

#include "Basis.h"
#include "Intrepid_FieldContainer.hpp"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class LobattoHGRAD_LineBasis;
  template<class Scalar, class ArrayScalar> class LobattoHGRAD_LineBasis : public Basis<Scalar,ArrayScalar> {
  protected:
    void initializeTags() const;
    int _degree;
    bool _conforming;
    
    Intrepid::FieldContainer<double> _lobattoL2norms;
    void initializeL2normValues();
  public:
    LobattoHGRAD_LineBasis(int degree, bool conforming); // conforming means vertex dofs defined...
    
    void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
    
    bool isConforming() const;
  };
}

#include "LobattoHGRAD_LineBasisDef.h"

#endif
