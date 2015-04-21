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
  class ParameterFunction : public TFunction<double> {
    TFunctionPtr<double> _fxn;
  public:
    ParameterFunction(double value);
    ParameterFunction(TFunctionPtr<double> value);

    TFunctionPtr<double> getValue() const;

    void setValue(TFunctionPtr<double> fxn);
    void setValue(double value);

    // overridden from Function:
    TFunctionPtr<double> x();
    TFunctionPtr<double> y();
    TFunctionPtr<double> z();

    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();

    TFunctionPtr<double> grad(int numComponents=-1); // gradient of sum is the sum of gradients
    TFunctionPtr<double> div();  // divergence of sum is sum of divergences

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();

    static Teuchos::RCP<ParameterFunction> parameterFunction(double value);
    static Teuchos::RCP<ParameterFunction> parameterFunction(TFunctionPtr<double> fxn);
  };
}

#endif
