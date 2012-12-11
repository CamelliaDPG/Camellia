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
#include "LinearTerm.h"
#include "SpatiallyFilteredFunction.h"

class Function;
typedef Teuchos::RCP<Function> FunctionPtr;

class Constraint {
  LinearTermPtr _linearTerm;
  FunctionPtr _f;
public:
  Constraint(LinearTermPtr linearTerm, FunctionPtr f);
  LinearTermPtr linearTerm() const;
  FunctionPtr f() const;
  static Constraint spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf);
};

Constraint operator==(VarPtr v, FunctionPtr f);
Constraint operator==(FunctionPtr f, VarPtr v);
Constraint operator==(LinearTermPtr a, FunctionPtr f);
Constraint operator==(FunctionPtr f, LinearTermPtr a);

#endif
