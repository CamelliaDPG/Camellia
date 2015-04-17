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

namespace Camellia {
  template <typename Scalar>
  class SimpleSolutionFunction : public Function<Scalar> {
    SolutionPtr<Scalar> _soln;
    VarPtr _var;
  public:
    SimpleSolutionFunction(VarPtr var, SolutionPtr<Scalar> soln);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    FunctionPtr<Scalar> x();
    FunctionPtr<Scalar> y();
    FunctionPtr<Scalar> z();

    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();
    // for reasons of efficiency, may want to implement div() and grad() as well

    void importCellData(std::vector<GlobalIndexType> cellIDs);

    std::string displayString();
    bool boundaryValueOnly();
  };
}

#endif
