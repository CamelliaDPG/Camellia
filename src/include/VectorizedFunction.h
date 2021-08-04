// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  VectorizedFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_VectorizedFunction_h
#define Camellia_VectorizedFunction_h

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class VectorizedFunction : public TFunction<Scalar>
{
private:
  std::vector< TFunctionPtr<Scalar> > _fxns;
  TFunctionPtr<Scalar> di(int i) override; // derivative in the ith coordinate direction
public:
  virtual TFunctionPtr<Scalar> x() override;
  virtual TFunctionPtr<Scalar> y() override;
  virtual TFunctionPtr<Scalar> z() override;
  virtual TFunctionPtr<Scalar> t() override;
  int numVectorComponents() const override;

  virtual TFunctionPtr<Scalar> dx() override;
  virtual TFunctionPtr<Scalar> dy() override;
  virtual TFunctionPtr<Scalar> dz() override;
  virtual TFunctionPtr<Scalar> dt() override;

  VectorizedFunction(const std::vector< TFunctionPtr<Scalar> > &fxns);
  VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3);
  VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3, TFunctionPtr<Scalar> f4);
  
  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;

  std::vector< TFunctionPtr<Scalar> > memberFunctions() override;
  
  virtual string displayString() override;
  int dim();

  using TFunction<Scalar>::isZero;
  bool isZero() override;

  TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap) override;
  
  virtual ~VectorizedFunction() { }
};
}

#endif
