// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  MinMaxFunctions.h
//  Camellia
//
//  Created by Nate Roberts on 4/9/15.
//
//

#ifndef Camellia_MinMaxFunctions_h
#define Camellia_MinMaxFunctions_h

#include "Function.h"

namespace Camellia
{
class MinFunction : public TFunction<double>
{
  TFunctionPtr<double> _f1, _f2;
public:
  MinFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2);

  TFunctionPtr<double> x() override;
  TFunctionPtr<double> y() override;
  TFunctionPtr<double> z() override;

  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  bool boundaryValueOnly() override;

  std::vector< TFunctionPtr<double> > memberFunctions() override;
  
  string displayString() override;
};

class MaxFunction : public TFunction<double>
{
  TFunctionPtr<double> _f1, _f2;
public:
  MaxFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2);

  TFunctionPtr<double> x() override;
  TFunctionPtr<double> y() override;
  TFunctionPtr<double> z() override;

  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  bool boundaryValueOnly() override;

  std::vector< TFunctionPtr<double> > memberFunctions() override;
  
  string displayString() override;
};
}

#endif
