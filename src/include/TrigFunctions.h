// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

namespace Camellia
{
class Cos_y : public SimpleFunction<double>
{
public:
  Cos_y() : SimpleFunction<double>("Cos_y") {}
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Sin_y : public SimpleFunction<double>
{
public:
  Sin_y() : SimpleFunction<double>("Sin_y") {}
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Cos_x : public SimpleFunction<double>
{
public:
  Cos_x() : SimpleFunction<double>("Cos_x") {}
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Sin_x : public SimpleFunction<double>
{
public:
  Sin_x() : SimpleFunction<double>("Sin_x") {}
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Cos_ax : public SimpleFunction<double>
{
  double _a,_b;
public:
  Cos_ax(double a, double b=0);
  double value(double x);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();

  std::string displayString();
};

class Sin_ax : public SimpleFunction<double>
{
  double _a, _b;
public:
  Sin_ax(double a, double b=0);
  double value(double x);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  std::string displayString();
};

class Cos_ay : public SimpleFunction<double>
{
  double _a;
public:
  Cos_ay(double a);
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();

  std::string displayString();
};

class Sin_ay : public SimpleFunction<double>
{
  double _a;
public:
  Sin_ay(double a);
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  std::string displayString();
};

class ArcTan_ax : public SimpleFunction<double>
{
  double _a, _b;
public:
  ArcTan_ax(double a, double b=0);
  double value(double x);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  std::string displayString();
};

class ArcTan_ay : public SimpleFunction<double>
{
  double _a, _b;
public:
  ArcTan_ay(double a, double b=0);
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  std::string displayString();
};

  template<class Scalar>
  class TrigFunctions
  {
  public:
    static TFunctionPtr<Scalar> sin(TFunctionPtr<Scalar> argument);
    static TFunctionPtr<Scalar> cos(TFunctionPtr<Scalar> argument);
  };
}
#endif
