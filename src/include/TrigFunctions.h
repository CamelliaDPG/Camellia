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
  class Cos_y : public SimpleFunction<double> {
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Sin_y : public SimpleFunction<double> {
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Cos_x : public SimpleFunction<double> {
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Sin_x : public SimpleFunction<double> {
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    std::string displayString();
  };

  class Cos_ax : public SimpleFunction<double> {
    double _a,_b;
  public:
    Cos_ax(double a, double b=0);
    double value(double x);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();

    std::string displayString();
  };

  class Sin_ax : public SimpleFunction<double> {
    double _a, _b;
  public:
    Sin_ax(double a, double b=0);
    double value(double x);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    std::string displayString();
  };

  class Cos_ay : public SimpleFunction<double> {
    double _a;
  public:
    Cos_ay(double a);
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();

    std::string displayString();
  };

  class Sin_ay : public SimpleFunction<double> {
    double _a;
  public:
    Sin_ay(double a);
    double value(double x, double y);
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    std::string displayString();
  };

}
#endif
