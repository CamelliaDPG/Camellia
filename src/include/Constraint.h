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
	  TFunctionPtr<double> _f;
	public:
	  Constraint(LinearTermPtr linearTerm, TFunctionPtr<double> f);
	  LinearTermPtr linearTerm() const;
	  TFunctionPtr<double> f() const;
	  static Constraint spatiallyFilteredConstraint(const Constraint &c, SpatialFilterPtr sf);
	};

	Constraint operator==(VarPtr v, TFunctionPtr<double> f);
	Constraint operator==(TFunctionPtr<double> f, VarPtr v);
	Constraint operator==(LinearTermPtr a, TFunctionPtr<double> f);
	Constraint operator==(TFunctionPtr<double> f, LinearTermPtr a);
}

#endif
