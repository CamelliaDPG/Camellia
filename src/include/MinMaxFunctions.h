//
//  MinMaxFunctions.h
//  Camellia
//
//  Created by Nate Roberts on 4/9/15.
//
//

#ifndef Camellia_MinMaxFunctions_h
#define Camellia_MinMaxFunctions_h

#include "Function.h"

namespace Camellia {
  class MinFunction : public Function {
    FunctionPtr _f1, _f2;
  public:
    MinFunction(FunctionPtr f1, FunctionPtr f2);
    
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();
    
    string displayString();
  };
  
  class MaxFunction : public Function {
    FunctionPtr _f1, _f2;
  public:
    MaxFunction(FunctionPtr f1, FunctionPtr f2);
    
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();
    
    string displayString();
  };
}

#endif
