//
//  LobattoHGRAD_Quad.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LobattoHGRAD_Quad_h
#define Camellia_debug_LobattoHGRAD_Quad_h

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class LobattoHGRAD_Quad;
  template<class Scalar, class ArrayScalar> class LobattoHGRAD_Quad : public Basis<Scalar,ArrayScalar> {
  protected:
    void initializeTags() const;
    int _degree_x, _degree_y;
  public:
    LobattoHGRAD_Quad(int degree);
    LobattoHGRAD_Quad(int degree_x, int degree_y);
    
    // domain info on which the basis is defined:
    shards::CellTopology domainTopology() const;
    
    // range info for basis values:
    int rangeDimension() const;
    int rangeRank() const;
    
    void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
  };
}

#include "LobattoHGRAD_QuadDef.h"

#endif
