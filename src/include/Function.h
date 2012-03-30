//
//  Function.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Function_h
#define Camellia_Function_h

#include "BasisCache.h"

class Function;
typedef Teuchos::RCP<Function> FunctionPtr;

class Function {
  // TODO: consider adding a double weight (will allow us to do things like "- 3.0 * mu * f" as a function)
  //       subclasses would have to multiply by this weight inside values()...
protected:
  int _rank;
public:
  Function() {
    _rank = 0;
  }
  Function(int rank) { 
    _rank = rank; 
  }
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) = 0;
  int rank() { return _rank; }
};

class ConstantScalarFunction : public Function {
  double _value;
public:
  ConstantScalarFunction(double value) : Function(0) { _value = value; }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    for (int i=0; i < values.size(); i++) {
      values[i] = _value;
    }
  }
};

class ConstantVectorFunction : public Function {
  vector<double> _value;
public:
  ConstantVectorFunction(vector<double> value) : Function(1) { _value = value; }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // values are stored in (C,P,D) order, the important thing here being that we can do this:
    for (int i=0; i < values.size(); ) {
      for (int d=0; d < _value.size(); d++) {
        values[i++] = _value[d];
      }
    }
  }
};

#endif
