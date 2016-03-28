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
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  string displayString();
};
}

#endif
