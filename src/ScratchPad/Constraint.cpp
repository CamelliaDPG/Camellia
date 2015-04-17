//
//  Constraint.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Constraint.h"
#include "Function.h"

using namespace Camellia;

typedef pair< FunctionPtr<double>, VarPtr > LinearSummand;

Constraint::Constraint(LinearTermPtr linearTerm, FunctionPtr<double> f) {
  _linearTerm = linearTerm;
  _f = f;
}
LinearTermPtr Constraint::linearTerm() const {
  return _linearTerm;
}
FunctionPtr<double> Constraint::f() const {
  return _f;
}
Constraint Constraint::spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf) {
  LinearTermPtr lt = c.linearTerm();
  FunctionPtr<double> f = c.f();
  LinearTermPtr flt = Teuchos::rcp( new LinearTerm ); // filtered linear term
  FunctionPtr<double> ff = Teuchos::rcp( new SpatiallyFilteredFunction<double>(f,sf) );

  for (vector< LinearSummand >::const_iterator lsIt = lt->summands().begin(); lsIt != lt->summands().end(); lsIt++) {
    LinearSummand ls = *lsIt;
    FunctionPtr<double> lsWeight = ls.first;
    FunctionPtr<double> filteredWeight = Teuchos::rcp( new SpatiallyFilteredFunction<double>(lsWeight,sf) );
    VarPtr var = ls.second;
    flt->addTerm(filteredWeight * var, true); //bypass type check...
  }

  return Constraint(flt,ff);
}

namespace Camellia {
  Constraint operator==(VarPtr v, FunctionPtr<double> f) {
    return Constraint(1.0*v,f);
  }

  Constraint operator==(FunctionPtr<double> f, VarPtr v) {
    return Constraint(1.0*v,f);
  }

  Constraint operator==(LinearTermPtr a, FunctionPtr<double> f) {
    return Constraint(a,f);
  }

  Constraint operator==(FunctionPtr<double> f, LinearTermPtr a) {
    return Constraint(a,f);
  }
} // namespace Camellia
