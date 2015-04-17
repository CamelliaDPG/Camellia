#ifndef Camellia_ParameterFunction_h
#define Camellia_ParameterFunction_h

//
//  ParameterFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/1/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "Function.h"

namespace Camellia {
  class ParameterFunction : public Function<double> {
    FunctionPtr<double> _fxn;
  public:
    ParameterFunction(double value);
    ParameterFunction(FunctionPtr<double> value);

    FunctionPtr<double> getValue() const;

    void setValue(FunctionPtr<double> fxn);
    void setValue(double value);

    // overridden from Function:
    FunctionPtr<double> x();
    FunctionPtr<double> y();
    FunctionPtr<double> z();

    FunctionPtr<double> dx();
    FunctionPtr<double> dy();
    FunctionPtr<double> dz();

    FunctionPtr<double> grad(int numComponents=-1); // gradient of sum is the sum of gradients
    FunctionPtr<double> div();  // divergence of sum is sum of divergences

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();

    static Teuchos::RCP<ParameterFunction> parameterFunction(double value);
    static Teuchos::RCP<ParameterFunction> parameterFunction(FunctionPtr<double> fxn);
  };
}

#endif
