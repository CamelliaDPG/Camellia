//
//  SumFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/8/15.
//  Copyright (c) 2015 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SumFunction_h
#define Camellia_SumFunction_h

#include "Function.h"

namespace Camellia {
  class SumFunction : public Function {
    FunctionPtr _f1, _f2;
  public:
    SumFunction(FunctionPtr f1, FunctionPtr f2);
    
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    FunctionPtr t();
    
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    
    FunctionPtr grad(int numComponents=-1); // gradient of sum is the sum of gradients
    FunctionPtr div();  // divergence of sum is sum of divergences
    
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();
    
    string displayString();
  };
}

#endif
