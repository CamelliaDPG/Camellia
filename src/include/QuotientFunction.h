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
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  TFunctionPtr<Scalar> x();
  TFunctionPtr<Scalar> y();
  TFunctionPtr<Scalar> z();
  TFunctionPtr<Scalar> t();
  
  virtual bool boundaryValueOnly();
  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  TFunctionPtr<Scalar> dt();
  std::string displayString();
  
  TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap);
  TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap);
  
  std::vector<TFunctionPtr<Scalar>> memberFunctions();
};
}

#endif
