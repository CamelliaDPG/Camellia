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

Constraint::Constraint(LinearTermPtr linearTerm, TFunctionPtr<double> f)
{
  _linearTerm = linearTerm;
  _f = f;
}
LinearTermPtr Constraint::linearTerm() const
{
  return _linearTerm;
}
TFunctionPtr<double> Constraint::f() const
{
  return _f;
}
Constraint Constraint::spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf)
{
  LinearTermPtr lt = c.linearTerm();
  TFunctionPtr<double> f = c.f();
  LinearTermPtr flt = Teuchos::rcp( new LinearTerm ); // filtered linear term
  TFunctionPtr<double> ff = Teuchos::rcp( new SpatiallyFilteredFunction<double>(f,sf) );

  for (vector< LinearSummand >::const_iterator lsIt = lt->summands().begin(); lsIt != lt->summands().end(); lsIt++)
  {
    LinearSummand ls = *lsIt;
    TFunctionPtr<double> lsWeight = ls.first;
    TFunctionPtr<double> filteredWeight = Teuchos::rcp( new SpatiallyFilteredFunction<double>(lsWeight,sf) );
    VarPtr var = ls.second;
    flt->addTerm(filteredWeight * var, true); //bypass type check...
  }

  return Constraint(flt,ff);
}

namespace Camellia
{
Constraint operator==(VarPtr v, TFunctionPtr<double> f)
{
  return Constraint(1.0*v,f);
}

Constraint operator==(TFunctionPtr<double> f, VarPtr v)
{
  return Constraint(1.0*v,f);
}

Constraint operator==(LinearTermPtr a, TFunctionPtr<double> f)
{
  return Constraint(a,f);
}

Constraint operator==(TFunctionPtr<double> f, LinearTermPtr a)
{
  return Constraint(a,f);
}
} // namespace Camellia
