//
//  ExpFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ExpFunction_h
#define Camellia_ExpFunction_h

#include "SimpleFunction.h"
#include "TypeDefs.h"

namespace Camellia {
  class Exp_x : public SimpleFunction {
  public:
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Exp_y : public SimpleFunction {
  public:
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Exp_z : public SimpleFunction {
  public:
    double value(double x, double y, double z);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Exp_ax : public SimpleFunction {
    double _a;
  public:
    Exp_ax(double a);
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    std::string displayString();
  };
  
  class Exp_ay : public SimpleFunction {
    double _a;
  public:
    Exp_ay(double a);
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    string displayString();
  };
}
#endif
