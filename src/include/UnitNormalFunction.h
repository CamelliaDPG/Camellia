// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  UnitNormalFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_UnitNormalFunction_h
#define Camellia_UnitNormalFunction_h

#include "Function.h"

namespace Camellia
{
class UnitNormalFunction : public TFunction<double>
{
  int _comp;
  bool _spaceTime;
public:
  UnitNormalFunction(int comp=-1, bool spaceTime = false); // -1: the vector normal.  Otherwise, picks out the comp component

  TFunctionPtr<double> x() override;
  TFunctionPtr<double> y() override;
  TFunctionPtr<double> z() override;
  TFunctionPtr<double> t() override;

  bool boundaryValueOnly() override;
  string displayString() override;
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  int numVectorComponents() const override;
};
}

#endif
