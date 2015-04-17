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

namespace Camellia {
  template <typename Scalar>
  class ProductFunction : public Function<Scalar> {
  private:
    int productRank(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
    FunctionPtr<Scalar> _f1, _f2;
  public:
    ProductFunction(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();

    FunctionPtr<Scalar> x();
    FunctionPtr<Scalar> y();
    FunctionPtr<Scalar> z();
    FunctionPtr<Scalar> t();

    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();
    FunctionPtr<Scalar> dt();

    string displayString(); // _f1->displayString() << " " << _f2->displayString();
  };
}

#endif
