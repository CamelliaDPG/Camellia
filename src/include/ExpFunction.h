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
  class Exp_x : public SimpleFunction<double> {
  public:
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Exp_y : public SimpleFunction<double> {
  public:
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Exp_z : public SimpleFunction<double> {
  public:
    double value(double x, double y, double z);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Exp_ax : public SimpleFunction<double> {
    double _a;
  public:
    Exp_ax(double a);
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    std::string displayString();
  };

  class Exp_ay : public SimpleFunction<double> {
    double _a;
  public:
    Exp_ay(double a);
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    string displayString();
  };
}
#endif
