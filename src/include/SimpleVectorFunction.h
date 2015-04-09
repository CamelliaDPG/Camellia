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
  class SimpleVectorFunction : public Function {
  public:
    SimpleVectorFunction();
    virtual ~SimpleVectorFunction() {}
    virtual vector<double> value(double x);
    virtual vector<double> value(double x, double y);
    virtual vector<double> value(double x, double y, double z);
    virtual void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };
  typedef Teuchos::RCP<SimpleVectorFunction> SimpleVectorFunctionPtr;
}
#endif
