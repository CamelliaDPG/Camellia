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
  class ProductFunction : public TFunction<Scalar> {
  private:
    int productRank(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
    TFunctionPtr<Scalar> _f1, _f2;
  public:
    ProductFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();

    TFunctionPtr<Scalar> x();
    TFunctionPtr<Scalar> y();
    TFunctionPtr<Scalar> z();
    TFunctionPtr<Scalar> t();

    TFunctionPtr<Scalar> dx();
    TFunctionPtr<Scalar> dy();
    TFunctionPtr<Scalar> dz();
    TFunctionPtr<Scalar> dt();

    string displayString(); // _f1->displayString() << " " << _f2->displayString();
  };
}

#endif
