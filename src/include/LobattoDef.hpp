//
//  LobattoDef.hpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "Legendre.hpp"
#include "Function.h"

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  void Lobatto<Scalar,ArrayScalar>::values(ArrayScalar &valuesArray, ArrayScalar &derivativeValuesArray, Scalar x, int n) {
    valuesArray(0) = 1;
    derivativeValuesArray(0) = 0;
    
    if (n==0) return;
    
    valuesArray(1) = x;
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
  
  template<class Scalar, class ArrayScalar>
  void Lobatto<Scalar, ArrayScalar >::l2norms(ArrayScalar &valuesArray, int n) {
    int maxOrder = n + 1; // n + 1 because Lobatto functions have 1 higher degree than the corresponding Legendre polynomial
    
    // TODO: eliminate the assumption that Scalar == double here...
    
    int cubDegree = maxOrder * maxOrder;
    BasisCachePtr basisCache = BasisCache::basisCache1D(-1,1,cubDegree);
    
    for (int i=0; i<=n; i++) {
      FunctionPtr lobatto = Teuchos::rcp( new LobattoFunction(i) ); // this assumes Scalar == double...
      valuesArray(i) = sqrt((lobatto*lobatto)->integrate(basisCache));
    }
  }
}