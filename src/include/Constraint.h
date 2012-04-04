//
//  Constraint.h
//  Camellia
//
//  Created by Nathan Roberts on 4/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Constraint_h
#define Camellia_Constraint_h

class Constraint {
  LinearTermPtr _linearTerm;
  FunctionPtr _f;
public:
  Constraint(LinearTermPtr linearTerm, FunctionPtr f) {
    _linearTerm = linearTerm;
    _f = f;
  }
};

Constraint operator==(LinearTermPtr a, FunctionPtr f) {
  return Constraint(a,f);
}


Constraint operator==(FunctionPtr f, LinearTermPtr a) {
  return Constraint(a,f);
}

#endif
