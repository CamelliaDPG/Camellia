//
//  QuotientFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_QuotientFunction_h
#define Camellia_QuotientFunction_h

#include "Function.h"

namespace Camellia {
  class QuotientFunction : public Function {
    FunctionPtr _f, _scalarDivisor;
  public:
    QuotientFunction(FunctionPtr f, FunctionPtr scalarDivisor);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    std::string displayString();
  };
}

#endif
