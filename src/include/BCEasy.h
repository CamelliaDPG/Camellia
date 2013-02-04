//
//  BCEasy.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_BCEasy_h
#define Camellia_BCEasy_h

#include "BC.h"
#include "SpatialFilter.h"

class Function;
class Var;
typedef Teuchos::RCP<Var> VarPtr;

class BCEasy : public BC {
  typedef pair< SpatialFilterPtr, FunctionPtr > DirichletBC;
  set< int > _zeroMeanConstraints; // variables on which ZMCs imposed
  //  set< int > _singlePointBCs; // variables on which single-point conditions imposed
  map< int, DirichletBC > _dirichletBCs; // key: trialID
protected:
  map< int, DirichletBC > &dirichletBCs();
public:
  void addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, FunctionPtr valueFunction );
  void addZeroMeanConstraint( VarPtr field );
  bool bcsImposed(int varID);
  void imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache);
  
  bool singlePointBC(int varID);
  
  bool imposeZeroMeanConstraint(int varID);
  
  Teuchos::RCP<BCEasy> copyImposingZero();//returns a copy of this BC object, except with all zero Functions
};

#endif
