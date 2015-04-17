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
  template <typename Scalar>
  class ConstantScalarFunction : public SimpleFunction<Scalar> {
    double _value;
    string _stringDisplay;
  public:
    ConstantScalarFunction(Scalar value);
    ConstantScalarFunction(Scalar value, string stringDisplay);
    string displayString();
    bool isZero();
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    void scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    void scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    void scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);
    void scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

    virtual Scalar value(double x);
    virtual Scalar value(double x, double y);
    virtual Scalar value(double x, double y, double z);

    using SimpleFunction<Scalar>::value; // avoid compiler warnings about the value() method below.
    Scalar value();

    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();  // Hmm... a design issue: if we implement dz() then grad() will return a 3D function, not what we want...  It may be that grad() should require a spaceDim argument.  I'm not sure.
    FunctionPtr<Scalar> dt();
  };
}

#endif
