// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

namespace Camellia
{
template <typename Scalar>
class SumFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f1, _f2;
public:
  SumFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);

  TFunctionPtr<Scalar> x() override;
  TFunctionPtr<Scalar> y() override;
  TFunctionPtr<Scalar> z() override;
  TFunctionPtr<Scalar> t() override;
  
  int numVectorComponents() const override;

  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;
  TFunctionPtr<Scalar> dt() override;
  
  TFunctionPtr<Scalar> di(int i) override; // 1 for dx(), 2 for dy(), 3 for dz()

  TFunctionPtr<Scalar> grad(int numComponents=-1) override; // gradient of sum is the sum of gradients
  TFunctionPtr<Scalar> div() override;  // divergence of sum is sum of divergences

  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  bool boundaryValueOnly() override;
  
  std::vector<Intrepid2::DimensionInfo> valuesDataDims(BasisCachePtr basisCache) override;
  void values(Intrepid2::Data<Scalar, Camellia::DeviceType> &values, BasisCachePtr basisCache) override;

  string displayString() override;
  
  TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  
  std::vector< TFunctionPtr<Scalar> > memberFunctions() override;
};
}

#endif
