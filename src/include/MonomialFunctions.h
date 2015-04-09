//
//  MonomialFunctions.h
//  Camellia
//
//  Created by Nate Roberts on 4/9/15.
//
//

#ifndef Camellia_MonomialFunctions_h
#define Camellia_MonomialFunctions_h

#include "SimpleFunction.h"

namespace Camellia {
  class Xn : public SimpleFunction {
    int _n;
  public:
    Xn(int n);
    double value(double x);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    string displayString();
  };
  
  class Yn : public SimpleFunction {
    int _n;
  public:
    Yn(int n);
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    string displayString();
  };
  
  class Zn : public SimpleFunction {
    int _n;
  public:
    Zn(int n);
    double value(double x, double y, double z);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    string displayString();
  };
  
  class Tn : public SimpleFunction {
    int _n;
  public:
    Tn(int n);
    double value(double x, double t);
    double value(double x, double y, double t);
    double value(double x, double y, double z, double t);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    FunctionPtr dt();
    string displayString();
  };
}
#endif
