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
  TFunctionPtr<double> x() override;
  TFunctionPtr<double> y() override;
  TFunctionPtr<double> z() override;
  
  // TODO: implement numVectorComponents().  Need to be careful about setValue() changing the rank, numVectorComponents of the function.  We can reasonably throw an exception if the rank of the Function changes.  I'm not as sure that it's reasonable to insist that _fxn->numVectorComponents() remains constant.  We could have a construction argument that specifies the (maximum) number of vector components.
  // int numVectorComponents() const override;

  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;

  TFunctionPtr<double> grad(int numComponents=-1) override; // gradient of sum is the sum of gradients
  TFunctionPtr<double> div() override;  // divergence of sum is sum of divergences

  std::vector<Intrepid2::DimensionInfo> valuesDataDims(BasisCachePtr basisCache) override;
  
  using TFunction<double>::values;
  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
  
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  bool boundaryValueOnly() override;

  std::vector< TFunctionPtr<double> > memberFunctions() override;
  
  string displayString() override;
  
  void setName(const std::string &name);

  static Teuchos::RCP<ParameterFunction> parameterFunction(double value);
  static Teuchos::RCP<ParameterFunction> parameterFunction(TFunctionPtr<double> fxn);
};
}

#endif
