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

class PreviousSolutionFunction : public Function {
  SolutionPtr _soln;
  LinearTermPtr _solnExpression;
  bool _overrideMeshCheck;
public:
  PreviousSolutionFunction(SolutionPtr soln, LinearTermPtr solnExpression, bool multiplyFluxesByCellParity = true);
  PreviousSolutionFunction(SolutionPtr soln, VarPtr var, bool multiplyFluxesByCellParity = true);
  bool boundaryValueOnly();
  void setOverrideMeshCheck(bool value, bool dontWarn=false);
  void importCellData(std::vector<GlobalIndexType> cells);
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  static map<int, FunctionPtr > functionMap( vector< VarPtr > varPtrs, SolutionPtr soln);
  string displayString();
};

#endif
