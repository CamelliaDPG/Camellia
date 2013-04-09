//
//  Legendre.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_Legendre_hpp
#define Camellia_debug_Legendre_hpp

#include "Intrepid_FieldContainer.hpp"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<Scalar> > class Legendre;
  
  template<class Scalar, class ArrayScalar> class Legendre {
  public:
    // n: poly order; valuesArray should have n+2 entries...
    static void values(ArrayScalar &valuesArray, ArrayScalar &derivativeValuesArray, Scalar x, int n); 
  };
  
  class LegendreFunction : public SimpleFunction {
    int _polyOrder;
    FieldContainer<double> _values;
    FieldContainer<double> _derivatives;
    
    bool _derivative; // whether this is the derivative of the Lobatto function
  public:
    LegendreFunction(int polyOrder, bool derivative = false) {
      _polyOrder = polyOrder;
      _values.resize(_polyOrder+1);
      _derivatives.resize(_polyOrder+1);
      _derivative = derivative;
    }
    double value(double x) {
      Legendre<double>::values(_values,_derivatives,x,_polyOrder);
      //      cout << "Lobatto values:\n" << _values;
      if (! _derivative) {
        return _values[_polyOrder];
      } else {
        return _derivatives[_polyOrder];
      }
    }
    
    FunctionPtr dx() {
      if (_derivative) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "LegendreFunction only supports first derivatives...");
      }
      return Teuchos::rcp( new LegendreFunction(_polyOrder,true) );
    }
  };
}

#include "LegendreDef.hpp"

#endif
