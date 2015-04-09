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
  class SimpleFunction : public Function {
  public:
    virtual ~SimpleFunction() {}
    virtual double value(double x);
    virtual double value(double x, double y);
    virtual double value(double x, double y, double z);
    virtual double value(double x, double y, double z, double t);
    virtual void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };
  typedef Teuchos::RCP<SimpleFunction> SimpleFunctionPtr;
}
#endif
