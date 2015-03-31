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

class ParameterFunction : public Function {
  FunctionPtr _fxn;
  typedef Teuchos::RCP<ParameterFunction> ParameterFunctionPtr;
public:
  ParameterFunction(double value);
  ParameterFunction(FunctionPtr value);
  
  FunctionPtr getValue() const;
  
  void setValue(FunctionPtr fxn);
  void setValue(double value);
  
  // overridden from Function:
  FunctionPtr x();
  FunctionPtr y();
  FunctionPtr z();
  
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  
  FunctionPtr grad(int numComponents=-1); // gradient of sum is the sum of gradients
  FunctionPtr div();  // divergence of sum is sum of divergences
  
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();
  
  string displayString();
  
  static ParameterFunctionPtr parameterFunction(double value);
  static ParameterFunctionPtr parameterFunction(FunctionPtr fxn);
};

typedef Teuchos::RCP<ParameterFunction> ParameterFunctionPtr;

#endif