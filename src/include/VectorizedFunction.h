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
  class VectorizedFunction : public Function<Scalar> {
  private:
    vector< FunctionPtr<Scalar> > _fxns;
    FunctionPtr<Scalar> di(int i); // derivative in the ith coordinate direction
  public:
    virtual FunctionPtr<Scalar> x();
    virtual FunctionPtr<Scalar> y();
    virtual FunctionPtr<Scalar> z();
    virtual FunctionPtr<Scalar> t();

    virtual FunctionPtr<Scalar> dx();
    virtual FunctionPtr<Scalar> dy();
    virtual FunctionPtr<Scalar> dz();
    virtual FunctionPtr<Scalar> dt();

    VectorizedFunction(const vector< FunctionPtr<Scalar> > &fxns);
    VectorizedFunction(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
    VectorizedFunction(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2, FunctionPtr<Scalar> f3);
    VectorizedFunction(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2, FunctionPtr<Scalar> f3, FunctionPtr<Scalar> f4);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);

    virtual string displayString();
    int dim();

    bool isZero();

    virtual ~VectorizedFunction() { }
  };
}

#endif
