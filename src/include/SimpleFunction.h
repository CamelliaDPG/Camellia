// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  SimpleFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SimpleFunction_h
#define Camellia_SimpleFunction_h

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class SimpleFunction : public TFunction<Scalar>
{
public:
  SimpleFunction(const std::string &className, int variationDimensions);
  virtual ~SimpleFunction() {}
  virtual Scalar value();
  virtual Scalar value(double x);
  virtual Scalar value(double x, double y);
  virtual Scalar value(double x, double y, double z);
  virtual Scalar value(double x, double y, double z, double t);
  
  using TFunction<Scalar>::values;
  virtual void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
};
}
#endif
