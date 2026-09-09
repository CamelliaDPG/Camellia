// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

namespace Camellia
{
template <typename Scalar>
class QuotientFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f, _scalarDivisor;
public:
  QuotientFunction(TFunctionPtr<Scalar> f, TFunctionPtr<Scalar> scalarDivisor);
  
  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  std::vector<Intrepid2::DimensionInfo> valuesDataDims(BasisCachePtr basisCache) override;
  void values(Intrepid2::Data<Scalar, Camellia::DeviceType> &values, BasisCachePtr basisCache) override;
  
  TFunctionPtr<Scalar> x() override;
  TFunctionPtr<Scalar> y() override;
  TFunctionPtr<Scalar> z() override;
  TFunctionPtr<Scalar> t() override;
  int numVectorComponents() const override;
  
  virtual bool boundaryValueOnly() override;
  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;
  TFunctionPtr<Scalar> dt() override;
  
  TFunctionPtr<Scalar> di(int i) override; // 1 for dx(), 2 for dy(), 3 for dz()
  
  std::string displayString() override;
  
  TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  
  std::vector<TFunctionPtr<Scalar>> memberFunctions() override;
};
}

#endif
