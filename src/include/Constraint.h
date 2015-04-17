//
//  Constraint.h
//  Camellia
//
//  Created by Nathan Roberts on 4/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Constraint_h
#define Camellia_Constraint_h

#include "TypeDefs.h"

#include "SpatialFilter.h"
#include "LinearTerm.h"
#include "SpatiallyFilteredFunction.h"

namespace Camellia {
	class Constraint {
	  LinearTermPtr _linearTerm;
	  FunctionPtr<double> _f;
	public:
	  Constraint(LinearTermPtr linearTerm, FunctionPtr<double> f);
	  LinearTermPtr linearTerm() const;
	  FunctionPtr<double> f() const;
	  static Constraint spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf);
	};

	Constraint operator==(VarPtr v, FunctionPtr<double> f);
	Constraint operator==(FunctionPtr<double> f, VarPtr v);
	Constraint operator==(LinearTermPtr a, FunctionPtr<double> f);
	Constraint operator==(FunctionPtr<double> f, LinearTermPtr a);
}

#endif
