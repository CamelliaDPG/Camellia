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

  TFunctionPtr<double> x();
  TFunctionPtr<double> y();
  TFunctionPtr<double> z();

  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();

  string displayString();
};

class MaxFunction : public TFunction<double>
{
  TFunctionPtr<double> _f1, _f2;
public:
  MaxFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2);

  TFunctionPtr<double> x();
  TFunctionPtr<double> y();
  TFunctionPtr<double> z();

  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();

  string displayString();
};
}

#endif
