//
//  ConstantVectorFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ConstantVectorFunction_h
#define Camellia_ConstantVectorFunction_h

#include "Function.h"

namespace Camellia {
  class ConstantVectorFunction : public Function {
    std::vector<double> _value;
  public:
    ConstantVectorFunction(std::vector<double> value);
    bool isZero();
    
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    std::vector<double> value();
  };
}

#endif
