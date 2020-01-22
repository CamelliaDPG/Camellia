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
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache);
  string displayString();
};
}

#endif
