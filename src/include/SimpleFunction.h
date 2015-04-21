//
//  SimpleFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SimpleFunction_h
#define Camellia_SimpleFunction_h

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class SimpleFunction : public TFunction<Scalar> {
  public:
    virtual ~SimpleFunction() {}
    virtual Scalar value(double x);
    virtual Scalar value(double x, double y);
    virtual Scalar value(double x, double y, double z);
    virtual Scalar value(double x, double y, double z, double t);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  };
}
#endif
