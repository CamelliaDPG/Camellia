// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  ExpFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_ExpFunction_h
#define Camellia_ExpFunction_h

#include "Function.h"
#include "SimpleFunction.h"
#include "TypeDefs.h"

namespace Camellia
{
  template<class Scalar>
  class Exp : public TFunction<Scalar>
  {
  private:
    TFunctionPtr<Scalar> _power;
  public:
    Exp(TFunctionPtr<Scalar> power);
    
    using TFunction<Scalar>::values;
    void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    
    TFunctionPtr<Scalar> dx() override;
    TFunctionPtr<Scalar> dy() override;
    TFunctionPtr<Scalar> dz() override;
    
    std::vector< TFunctionPtr<Scalar> > memberFunctions() override;
    
    std::string displayString() override;
    
    static TFunctionPtr<Scalar> exp(TFunctionPtr<Scalar> power);
  };
  
  template<class Scalar>
  class Ln : public TFunction<Scalar>
  {
  private:
    TFunctionPtr<Scalar> _argument;
  public:
    Ln(TFunctionPtr<Scalar> argument);
    
    using TFunction<Scalar>::values;
    void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    
    TFunctionPtr<Scalar> dx() override;
    TFunctionPtr<Scalar> dy() override;
    TFunctionPtr<Scalar> dz() override;
    
    std::vector< TFunctionPtr<Scalar> > memberFunctions() override;
    
    std::string displayString() override;
  };
  
//! tanh(f), for function f
template<class Scalar>
class Tanh : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f;
public:
  Tanh(TFunctionPtr<Scalar> f);
  
  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;
  
  std::vector< TFunctionPtr<Scalar> > memberFunctions() override;
  
  std::string displayString() override;
};

//! sech^2(f), for function f
template<class Scalar>
class Sech2 : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f;
public:
  Sech2(TFunctionPtr<Scalar> argument);
  
  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;
  
  std::vector< TFunctionPtr<Scalar> > memberFunctions() override;
  
  std::string displayString() override;
};

class Exp_x : public SimpleFunction<double>
{
public:
  Exp_x();
  
  using SimpleFunction<double>::value;
  double value(double x, double y);
  
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Exp_y : public SimpleFunction<double>
{
public:
  Exp_y();
  
  using SimpleFunction<double>::value;
  double value(double x, double y);
  
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Exp_z : public SimpleFunction<double>
{
public:
  Exp_z();
  
  using SimpleFunction<double>::value;
  double value(double x, double y, double z);
  
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();
  std::string displayString();
};

class Exp_ax : public SimpleFunction<double>
{
  double _a;
public:
  Exp_ax(double a);
  
  using SimpleFunction<double>::value;
  double value(double x, double y);
  
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  std::string displayString();
};

class Exp_ay : public SimpleFunction<double>
{
  double _a;
public:
  Exp_ay(double a);
  
  using SimpleFunction<double>::value;
  double value(double x, double y);
  
  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  string displayString();
};

class Exp_at : public SimpleFunction<double>
{
  double _a;
public:
  Exp_at(double a);
  
  using SimpleFunction<double>::value;
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
