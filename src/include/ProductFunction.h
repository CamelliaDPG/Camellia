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
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  virtual bool boundaryValueOnly();

  TFunctionPtr<Scalar> f1();
  TFunctionPtr<Scalar> f2();
  
  TFunctionPtr<Scalar> x();
  TFunctionPtr<Scalar> y();
  TFunctionPtr<Scalar> z();
  TFunctionPtr<Scalar> t();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  TFunctionPtr<Scalar> dt();

  bool isZero(BasisCachePtr basisCache);
  
  string displayString(); // _f1->displayString() << " " << _f2->displayString();
  
  TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap);
  TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap);
  
  std::vector<TFunctionPtr<Scalar>> memberFunctions();
};
}

#endif
