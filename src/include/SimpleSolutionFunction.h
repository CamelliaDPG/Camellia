//
//  SimpleSolutionFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SimpleSolutionFunction_h
#define Camellia_SimpleSolutionFunction_h

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class SimpleSolutionFunction : public TFunction<Scalar>
{
  TSolutionPtr<Scalar> _soln;
  VarPtr _var;
public:
  SimpleSolutionFunction(VarPtr var, TSolutionPtr<Scalar> soln);
  void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  TFunctionPtr<Scalar> x();
  TFunctionPtr<Scalar> y();
  TFunctionPtr<Scalar> z();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  // for reasons of efficiency, may want to implement div() and grad() as well

  void importCellData(std::vector<GlobalIndexType> cellIDs);

  std::string displayString();
  bool boundaryValueOnly();
};
}

#endif
