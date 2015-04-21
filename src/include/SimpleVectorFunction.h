//
//  SimpleVectorFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SimpleVectorFunction_h
#define Camellia_SimpleVectorFunction_h

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class SimpleVectorFunction : public TFunction<Scalar> {
  public:
    SimpleVectorFunction();
    virtual ~SimpleVectorFunction() {}
    virtual vector<Scalar> value(double x);
    virtual vector<Scalar> value(double x, double y);
    virtual vector<Scalar> value(double x, double y, double z);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  };
}
#endif
