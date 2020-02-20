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
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  string displayString() override;
};

class Yn : public TFunction<double>
{
  int _n;
public:
  Yn(int n);
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  string displayString() override;
};

class Zn : public TFunction<double>
{
  int _n;
public:
  Zn(int n);
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  string displayString() override;
};

class Tn : public TFunction<double>
{
  int _n;
public:
  Tn(int n);
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;
  TFunctionPtr<double> dt() override;
  string displayString() override;
};
}
#endif
