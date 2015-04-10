//
//  ConstantScalarFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ConstantScalarFunction_h
#define Camellia_ConstantScalarFunction_h

#include "SimpleFunction.h"

namespace Camellia {
  class ConstantScalarFunction : public SimpleFunction {
    double _value;
    string _stringDisplay;
  public:
    ConstantScalarFunction(double value);
    ConstantScalarFunction(double value, string stringDisplay);
    string displayString();
    bool isZero();
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    void scalarMultiplyFunctionValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    void scalarDivideFunctionValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    void scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);
    void scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);
    
    virtual double value(double x);
    virtual double value(double x, double y);
    virtual double value(double x, double y, double z);
    
    using SimpleFunction::value; // avoid compiler warnings about the value() method below.
    double value();
    
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();  // Hmm... a design issue: if we implement dz() then grad() will return a 3D function, not what we want...  It may be that grad() should require a spaceDim argument.  I'm not sure.
    FunctionPtr dt();
  };
}

#endif
