//
//  TrigFunctions.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_TrigFunctions_h
#define Camellia_TrigFunctions_h

#include "SimpleFunction.h"

namespace Camellia {
  class Cos_y : public SimpleFunction {
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Sin_y : public SimpleFunction {
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Cos_x : public SimpleFunction {
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Sin_x : public SimpleFunction {
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    std::string displayString();
  };
  
  class Cos_ax : public SimpleFunction {
    double _a,_b;
  public:
    Cos_ax(double a, double b=0);
    double value(double x);
    FunctionPtr dx();
    FunctionPtr dy();
    
    std::string displayString();
  };
  
  class Sin_ax : public SimpleFunction {
    double _a, _b;
  public:
    Sin_ax(double a, double b=0);
    double value(double x);
    FunctionPtr dx();
    FunctionPtr dy();
    std::string displayString();
  };
  
  class Cos_ay : public SimpleFunction {
    double _a;
  public:
    Cos_ay(double a);
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    
    std::string displayString();
  };
  
  class Sin_ay : public SimpleFunction {
    double _a;
  public:
    Sin_ay(double a);
    double value(double x, double y);
    FunctionPtr dx();
    FunctionPtr dy();
    std::string displayString();
  };

}
#endif