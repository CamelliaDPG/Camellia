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
  class VectorizedFunction : public Function {
  private:
    vector< FunctionPtr > _fxns;
    FunctionPtr di(int i); // derivative in the ith coordinate direction
  public:
    virtual FunctionPtr x();
    virtual FunctionPtr y();
    virtual FunctionPtr z();
    virtual FunctionPtr t();
    
    virtual FunctionPtr dx();
    virtual FunctionPtr dy();
    virtual FunctionPtr dz();
    virtual FunctionPtr dt();
    
    VectorizedFunction(const vector< FunctionPtr > &fxns);
    VectorizedFunction(FunctionPtr f1, FunctionPtr f2);
    VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3);
    VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3, FunctionPtr f4);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    
    virtual string displayString();
    int dim();
    
    bool isZero();
    
    virtual ~VectorizedFunction() { }
  };
}

#endif
