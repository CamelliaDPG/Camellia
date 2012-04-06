//
//  PreviousSolutionFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_PreviousSolutionFunction_h
#define Camellia_PreviousSolutionFunction_h

#include "Function.h"

public class PreviousSolutionFunction : public Function {
  SolutionPtr _soln;
  LinearTermPtr _solnExpression;
public:
  PreviousSolutionFunction(SolutionPtr soln, LinearTermPtr solnExpression) : Function(solnExpression->rank()) { 
    _soln = soln;
    _solnExpression = solnExpression;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // values are stored in (C,P,D) order
    _solnExpression->evaluate(values, _soln, basisCache);
  }
};

#endif
