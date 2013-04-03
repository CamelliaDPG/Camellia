//
//  LobattoDef.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "Legendre.hpp"

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  void Lobatto<Scalar,ArrayScalar>::values(ArrayScalar &valuesArray, ArrayScalar &derivativeValuesArray, Scalar x, int n) {
    valuesArray(0) = 1;
    valuesArray(1) = x;
    
    derivativeValuesArray(0) = 0;
    derivativeValuesArray(1) = 1;
    
    ArrayScalar legendreValues(n+1), legendreDerivatives(n+1);
    Legendre<>::values(legendreValues,legendreDerivatives, x,n);

    double factor = 1 - x*x;
    for (int i=2; i<=n; i++) {
      double i_factor = (i-1)*i;
      valuesArray(i) = -factor * legendreDerivatives(i-1) / i_factor;
      derivativeValuesArray(i) = legendreValues(i-1);
    }
  }
}
