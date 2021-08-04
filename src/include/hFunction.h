// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  hFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_hFunction_h
#define Camellia_hFunction_h

#include "Function.h"

namespace Camellia
{
class hFunction : public TFunction<double>
{
public:
  hFunction() : TFunction<double>("hFunction") {}
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  string displayString() override;
};
}

#endif
