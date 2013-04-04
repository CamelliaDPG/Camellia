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
  };
}

#include "LobattoDef.hpp"

#endif
