// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MonomialFunctions.h
//  Camellia
//
//  Created by Nate Roberts on 4/9/15.
//
//

#ifndef Camellia_MonomialFunctions_h
#define Camellia_MonomialFunctions_h

#include "Function.h"

namespace Camellia
{
class Xn : public TFunction<double>
{
  int _n;
public:
  Xn(int n);
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  TFunctionPtr<double> di(int i) override;
  string displayString() override;
};

class Zn : public TFunction<double>
{
  int _n;
public:
  Zn(int n);
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  TFunctionPtr<double> di(int i) override;
  string displayString() override;
};

class Tn : public TFunction<double>
{
  int _n;
public:
  Tn(int n);
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  string displayString() override;
};

class MonomialFunction : public TFunction<double>
{
  int _comp;
  int _n;
public:
  MonomialFunction(int componentDimension, int power);
  
  using TFunction<double>::values;
  void values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  void values(Intrepid2::Data<double, DeviceType> values, BasisCachePtr basisCache) override;
  
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> di(int i) override;
  virtual string displayString() override;
  
  //! the integer power to which the monomial is raised
  int n() const;
  
  static TFunctionPtr<double> xi_n(int i, int n);
};

class Yn : public MonomialFunction
{
public:
  Yn(int n);
  TFunctionPtr<double> dt() override;
  string displayString() override;
};

}
#endif
