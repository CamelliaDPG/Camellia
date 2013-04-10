//
//  LobattoHGRAD_Quad.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LobattoHGRAD_Quad_h
#define Camellia_debug_LobattoHGRAD_Quad_h

#include "Basis.h"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class LobattoHGRAD_Quad;
  template<class Scalar, class ArrayScalar> class LobattoHGRAD_Quad : public Basis<Scalar,ArrayScalar> {
  protected:
    void initializeTags() const;
    int _degree_x, _degree_y;
    bool _conforming;
    
    FieldContainer<double> _legendreL2normsSquared, _lobattoL2normsSquared;
    void initializeL2normValues();
    int dofOrdinalMap(int xDofOrdinal, int yDofOrdinal) const;
  public:
    LobattoHGRAD_Quad(int degree, bool conforming = false); // conforming means not strictly hierarchical, but has e.g. vertex dofs defined...
    LobattoHGRAD_Quad(int degree_x, int degree_y, bool conforming = false);
    
    void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
  };
}

#include "LobattoHGRAD_QuadDef.h"

#endif
