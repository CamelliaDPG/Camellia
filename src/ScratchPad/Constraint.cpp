//
//  Constraint.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Constraint.h"

Constraint::Constraint(LinearTermPtr linearTerm, FunctionPtr f) {
  _linearTerm = linearTerm;
  _f = f;
}
LinearTermPtr Constraint::linearTerm() const {
  return _linearTerm;
}
FunctionPtr Constraint::f() const {
  return _f;
}
Constraint Constraint::spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf) {
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
