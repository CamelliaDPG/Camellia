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
  class MinFunction : public Function<double> {
    FunctionPtr<double> _f1, _f2;
  public:
    MinFunction(FunctionPtr<double> f1, FunctionPtr<double> f2);

    FunctionPtr<double> x();
    FunctionPtr<double> y();
    FunctionPtr<double> z();

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();
  };

  class MaxFunction : public Function<double> {
    FunctionPtr<double> _f1, _f2;
  public:
    MaxFunction(FunctionPtr<double> f1, FunctionPtr<double> f2);

    FunctionPtr<double> x();
    FunctionPtr<double> y();
    FunctionPtr<double> z();

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();
  };
}

#endif
