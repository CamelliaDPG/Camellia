//
//  Constraint.h
//  Camellia
//
//  Created by Nathan Roberts on 4/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Constraint_h
#define Camellia_Constraint_h

#include "SpatialFilter.h"
#include "Function.h"
#include "LinearTerm.h"
#include "SpatiallyFilteredFunction.h"

typedef pair< FunctionPtr, VarPtr > LinearSummand;

class Constraint {
  LinearTermPtr _linearTerm;
  FunctionPtr _f;
public:
  Constraint(LinearTermPtr linearTerm, FunctionPtr f) {
    _linearTerm = linearTerm;
    _f = f;
  }
  LinearTermPtr linearTerm() const {
    return _linearTerm;
  }
  FunctionPtr f() const {
    return _f;
  }
  static Constraint spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf) {
    LinearTermPtr lt = c.linearTerm();
    FunctionPtr f = c.f();
    LinearTermPtr flt = Teuchos::rcp( new LinearTerm ); // filtered linear term
    FunctionPtr ff = Teuchos::rcp( new SpatiallyFilteredFunction(f,sf) );
    
    for (vector< LinearSummand >::const_iterator lsIt = lt->summands().begin(); lsIt != lt->summands().end(); lsIt++) {
      LinearSummand ls = *lsIt;
      FunctionPtr lsWeight = ls.first;
      FunctionPtr filteredWeight = Teuchos::rcp( new SpatiallyFilteredFunction(lsWeight,sf) );
      VarPtr var = ls.second;
      *flt += *(filteredWeight * var);
    }
    
    return Constraint(flt,ff);
  }
};

Constraint operator==(VarPtr v, FunctionPtr f) {
  return Constraint(1.0*v,f);
}

Constraint operator==(FunctionPtr f, VarPtr v) {
  return Constraint(1.0*v,f);
}

Constraint operator==(LinearTermPtr a, FunctionPtr f) {
  return Constraint(a,f);
}

Constraint operator==(FunctionPtr f, LinearTermPtr a) {
  return Constraint(a,f);
}

#endif
