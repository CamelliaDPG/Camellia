//
//  LegendreDef.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  void Legendre<Scalar,ArrayScalar>::values(ArrayScalar &valuesArray, ArrayScalar &derivativeValuesArray, Scalar x, int n) {
    if (n==0) {
      valuesArray(0) = 1;
      return;
    }
    
    valuesArray(0) = 1;
    valuesArray(1) = x;
    
    derivativeValuesArray(0) = 0;
    derivativeValuesArray(1) = 1;

    for (int i=1; i<n; i++) {
      valuesArray(i+1) = ((2*i+1)*x*valuesArray(i) - i * valuesArray(i-1) ) / (i+1);
      derivativeValuesArray(i+1) = derivativeValuesArray(i-1) + (2*i+1)*valuesArray(i);
    }
  }  
}
