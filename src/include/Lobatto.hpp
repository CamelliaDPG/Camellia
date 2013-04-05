//
//  Lobatto.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_Lobatto_hpp
#define Camellia_debug_Lobatto_hpp

#include "Intrepid_FieldContainer.hpp"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<Scalar> > class Lobatto;
  
  template<class Scalar, class ArrayScalar> class Lobatto {
  public:
    // n: poly order; valuesArray should have n+2 entries...
    static void values(ArrayScalar &valuesArray, ArrayScalar &derivativeValuesArray, Scalar x, int n);
    static void l2norms(ArrayScalar &normValues, int n);  // should normValues be ArrayScalar, or hard-coded FieldContainer<double>?  I think the latter, actually...
  };
  
  class LobattoFunction : public SimpleFunction {
    int _polyOrder;
    FieldContainer<double> _values;
    FieldContainer<double> _derivatives;
  public:
    LobattoFunction(int polyOrder) {
      _polyOrder = polyOrder;
      _values.resize(_polyOrder+1);
      _derivatives.resize(_polyOrder+1);
    }
    double value(double x) {
      Lobatto<double>::values(_values,_derivatives,x,_polyOrder);
//      cout << "Lobatto values:\n" << _values;
      return _values[_polyOrder];
    }
  };
}

#include "LobattoDef.hpp"

#endif
