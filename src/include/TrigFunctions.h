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
  Cos_y() : SimpleFunction<double>("Cos_y",VARIES_IN_DIMENSION_1) {}
  using SimpleFunction<double>::value;
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Sin_y : public SimpleFunction<double>
{
public:
  Sin_y() : SimpleFunction<double>("Sin_y",VARIES_IN_DIMENSION_1) {}
  using SimpleFunction<double>::value;
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Cos_x : public SimpleFunction<double>
{
public:
  Cos_x() : SimpleFunction<double>("Cos_x",VARIES_IN_DIMENSION_0) {}
  using SimpleFunction<double>::value;
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Sin_x : public SimpleFunction<double>
{
public:
  Sin_x() : SimpleFunction<double>("Sin_x",VARIES_IN_DIMENSION_0) {}
  using SimpleFunction<double>::value;
  double value(double x, double y);
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Cos_ax : public TFunction<double>
{
  double _a,_b;
public:
  Cos_ax(double a, double b=0);
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;

  using TFunction<double>::values;
  void values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override;
  
  std::string displayString() override;
};

class Sin_ax : public TFunction<double>
{
  double _a, _b;
public:
  Sin_ax(double a, double b=0);
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  
  using TFunction<double>::values;
  void values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  std::string displayString() override;
};

class Cos_ay : public SimpleFunction<double>
{
  double _a;
public:
  Cos_ay(double a);
  using SimpleFunction<double>::value;
  double value(double x, double y) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;

  std::string displayString() override;
};

class Sin_ay : public SimpleFunction<double>
{
  double _a;
public:
  Sin_ay(double a);
  using SimpleFunction<double>::value;
  double value(double x, double y) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  std::string displayString() override;
};

class ArcTan_ax : public SimpleFunction<double>
{
  double _a, _b;
public:
  ArcTan_ax(double a, double b=0);
  using SimpleFunction<double>::value;
  double value(double x) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  std::string displayString() override;
};

class ArcTan_ay : public SimpleFunction<double>
{
  double _a, _b;
public:
  ArcTan_ay(double a, double b=0);
  using SimpleFunction<double>::value;
  double value(double x, double y) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  std::string displayString() override;
};

  template<class Scalar>
  class TrigFunctions
  {
  public:
    static TFunctionPtr<Scalar> sin(TFunctionPtr<Scalar> argument);
    static TFunctionPtr<Scalar> cos(TFunctionPtr<Scalar> argument);
  };

inline FunctionPtr sin(FunctionPtr arg)
{
  return TrigFunctions<double>::sin(arg);
}

inline FunctionPtr cos(FunctionPtr arg)
{
  return TrigFunctions<double>::cos(arg);
}
}
#endif
