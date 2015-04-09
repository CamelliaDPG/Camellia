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
  class ProductFunction : public Function {
  private:
    int productRank(FunctionPtr f1, FunctionPtr f2);
    FunctionPtr _f1, _f2;
  public:
    ProductFunction(FunctionPtr f1, FunctionPtr f2);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();
    
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    FunctionPtr t();
    
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    
    string displayString(); // _f1->displayString() << " " << _f2->displayString();
  };
}

#endif
