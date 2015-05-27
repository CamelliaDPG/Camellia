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

namespace Camellia
{
class Xn : public SimpleFunction<double>
{
  int _n;
public:
  Xn(int n);
  double value(double x);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  TFunctionPtr<double> dt();
  string displayString();
};

class Yn : public SimpleFunction<double>
{
  int _n;
public:
  Yn(int n);
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  TFunctionPtr<double> dt();
  string displayString();
};

class Zn : public SimpleFunction<double>
{
  int _n;
public:
  Zn(int n);
  double value(double x, double y, double z);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  TFunctionPtr<double> dt();
  string displayString();
};

class Tn : public SimpleFunction<double>
{
  int _n;
public:
  Tn(int n);
  double value(double x, double t);
  double value(double x, double y, double t);
  double value(double x, double y, double z, double t);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  TFunctionPtr<double> dt();
  string displayString();
};
}
#endif
