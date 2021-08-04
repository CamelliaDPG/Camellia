// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  ProductFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ProductFunction_h
#define Camellia_ProductFunction_h

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class ProductFunction : public TFunction<Scalar>
{
private:
  int productRank(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  TFunctionPtr<Scalar> _f1, _f2;
public:
  ProductFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  
  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  std::vector<Intrepid2::DimensionInfo> valuesDataDims(BasisCachePtr basisCache) override;
  void values(Intrepid2::Data<Scalar, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
  
  virtual bool boundaryValueOnly() override;

  TFunctionPtr<Scalar> f1();
  TFunctionPtr<Scalar> f2();
  
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

  bool isZero() override;
  bool isZero(BasisCachePtr basisCache) override;
  
  string displayString() override; // _f1->displayString() << " " << _f2->displayString();
  
  TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  
  std::vector<TFunctionPtr<Scalar>> memberFunctions() override;
};
}

#endif
