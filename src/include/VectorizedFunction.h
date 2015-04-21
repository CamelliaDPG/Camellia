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

namespace Camellia {
  template <typename Scalar>
  class VectorizedFunction : public TFunction<Scalar> {
  private:
    vector< TFunctionPtr<Scalar> > _fxns;
    TFunctionPtr<Scalar> di(int i); // derivative in the ith coordinate direction
  public:
    virtual TFunctionPtr<Scalar> x();
    virtual TFunctionPtr<Scalar> y();
    virtual TFunctionPtr<Scalar> z();
    virtual TFunctionPtr<Scalar> t();

    virtual TFunctionPtr<Scalar> dx();
    virtual TFunctionPtr<Scalar> dy();
    virtual TFunctionPtr<Scalar> dz();
    virtual TFunctionPtr<Scalar> dt();

    VectorizedFunction(const vector< TFunctionPtr<Scalar> > &fxns);
    VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
    VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3);
    VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3, TFunctionPtr<Scalar> f4);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);

    virtual string displayString();
    int dim();

    bool isZero();

    virtual ~VectorizedFunction() { }
  };
}

#endif
