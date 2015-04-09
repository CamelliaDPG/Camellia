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
  class SimpleSolutionFunction : public Function {
    SolutionPtr _soln;
    VarPtr _var;
  public:
    SimpleSolutionFunction(VarPtr var, SolutionPtr soln);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();
    
    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();
    // for reasons of efficiency, may want to implement div() and grad() as well
    
    void importCellData(std::vector<GlobalIndexType> cellIDs);
    
    std::string displayString();
    bool boundaryValueOnly();
  };
}

#endif
