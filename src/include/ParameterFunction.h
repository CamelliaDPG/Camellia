#ifndef Camellia_ParameterFunction_h
#define Camellia_ParameterFunction_h

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  ParameterFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/1/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "Function.h"

namespace Camellia
{
class ParameterFunction : public TFunction<double>
{
  TFunctionPtr<double> _fxn;
  std::string _name = ""; // if name is specified, this will be used by displayString() instead of _fxn->displayString()...
public:
  ParameterFunction(double value);
  ParameterFunction(TFunctionPtr<double> value);

  TFunctionPtr<double> getValue() const;
  
  // ! For scalar constant-valued "value" types only; returns the current value as a double
  // ! Throws an exception if _fxn.get() cannot be cast to ConstantScalarFunction
  double getConstantValue() const;

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

  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();

  std::vector< TFunctionPtr<double> > memberFunctions();
  
  string displayString();
  
  void setName(const std::string &name);

  static Teuchos::RCP<ParameterFunction> parameterFunction(double value);
  static Teuchos::RCP<ParameterFunction> parameterFunction(TFunctionPtr<double> fxn);
};
}

#endif
