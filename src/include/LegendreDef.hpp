//
//  LegendreDef.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  void Legendre<Scalar,ArrayScalar>::values(ArrayScalar &valuesArray, ArrayScalar &derivativeValuesArray, Scalar x, int n, bool conforming) {
    if (n==0) {
      valuesArray(0) = conforming ? (1 - x) / 2 : 1;
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
    
    if (conforming) {
      // overwrite the first two functions in this case:
      valuesArray(0) = (1 - x) / 2;
      valuesArray(1) = (1 + x) / 2;
      derivativeValuesArray(0) = -0.5;
      derivativeValuesArray(1) =  0.5;
    }
  }  
}
