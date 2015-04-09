//
//  SideParityFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SideParityFunction_h
#define Camellia_SideParityFunction_h

#include "Function.h"

namespace Camellia {
  class SideParityFunction : public Function {
  public:
    SideParityFunction();
    bool boundaryValueOnly();
    string displayString();
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };
}

#endif
