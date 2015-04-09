//
//  UnitNormalFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_UnitNormalFunction_h
#define Camellia_UnitNormalFunction_h

#include "Function.h"

namespace Camellia {
  class UnitNormalFunction : public Function {
    int _comp;
    bool _spaceTime;
  public:
    UnitNormalFunction(int comp=-1, bool spaceTime = false); // -1: the vector normal.  Otherwise, picks out the comp component
    
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    FunctionPtr t();
    
    bool boundaryValueOnly();
    string displayString();
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };
}

#endif
