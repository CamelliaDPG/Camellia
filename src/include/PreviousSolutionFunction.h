//
//  PreviousSolutionFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_PreviousSolutionFunction_h
#define Camellia_PreviousSolutionFunction_h

#include "TypeDefs.h"

#include "Function.h"
#include "Element.h"
#include "Solution.h"
#include "InnerProductScratchPad.h"

namespace Camellia {
  template <typename Scalar>
  class PreviousSolutionFunction : public TFunction<Scalar> {
    SolutionPtr<Scalar> _soln;
    LinearTermPtr _solnExpression;
    bool _overrideMeshCheck;
  public:
    PreviousSolutionFunction(SolutionPtr<Scalar> soln, LinearTermPtr solnExpression, bool multiplyFluxesByCellParity = true);
    PreviousSolutionFunction(SolutionPtr<Scalar> soln, VarPtr var, bool multiplyFluxesByCellParity = true);
    bool boundaryValueOnly();
    void setOverrideMeshCheck(bool value, bool dontWarn=false);
    void importCellData(std::vector<GlobalIndexType> cells);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    static map<int, TFunctionPtr<Scalar> > functionMap( vector< VarPtr > varPtrs, SolutionPtr<Scalar> soln);
    string displayString();
  };
}


#endif
