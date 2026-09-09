// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  SideParityFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SideParityFunction_h
#define Camellia_SideParityFunction_h

#include "Function.h"

namespace Camellia
{
class SideParityFunction : public TFunction<double>
{
public:
  SideParityFunction();
  bool boundaryValueOnly() override;
  string displayString() override;
  
  using TFunction<double>::values;
  void values(Intrepid2::Data<double, DeviceType> &values, BasisCachePtr basisCache) override;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
};
}

#endif
