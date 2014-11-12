//
//  Var.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Var_h
#define Camellia_Var_h

#include "BilinearForm.h"

class Var;
typedef Teuchos::RCP<Var> VarPtr;

class LinearTerm;
typedef Teuchos::RCP<LinearTerm> LinearTermPtr;

namespace VarFunctionSpaces {
  enum Space { HGRAD, HCURL, HDIV, HGRAD_DISC, HCURL_DISC, HDIV_DISC, HDIV_FREE, L2, CONSTANT_SCALAR, VECTOR_HGRAD, VECTOR_HGRAD_DISC, VECTOR_L2, UNKNOWN_FS };
  enum VarType { TEST, FIELD, TRACE, FLUX, UNKNOWN_TYPE, MIXED_TYPE };
  
  IntrepidExtendedTypes::EFunctionSpaceExtended efsForSpace(Space space);
  Space spaceForEFS(IntrepidExtendedTypes::EFunctionSpaceExtended efs);
  int rankForSpace(Space space);
}

using namespace VarFunctionSpaces;

class Var { // really Var x Operator
  int _rank;
  int _id;
  string _name;
  Space _fs;
  IntrepidExtendedTypes::EOperatorExtended _op; // default is OP_VALUE
  VarType _varType;
  LinearTermPtr _termTraced; // for trace variables, optionally allows identification with fields
  //  map< IntrepidExtendedTypes::EOperatorExtended, VarPtr > _relatedVars; // grad, div, etc. could be cached here
public:
  Var(int ID, int rank, string name, IntrepidExtendedTypes::EOperatorExtended op =  IntrepidExtendedTypes::OP_VALUE,
      Space fs = UNKNOWN_FS, VarType varType = UNKNOWN_TYPE, LinearTermPtr termTraced = Teuchos::rcp((LinearTerm*) NULL));
  
  int ID();
  const string & name();
  string displayString();
  IntrepidExtendedTypes::EOperatorExtended op();
  int rank();  // 0 for scalar, 1 for vector, etc.
  Space space();
  VarType varType();
  
  VarPtr grad();
  VarPtr div();
  VarPtr curl(int spaceDim); // 3D curl differs from 2D
  VarPtr dx();
  VarPtr dy();
  VarPtr dz();
  VarPtr x();
  VarPtr y();
  VarPtr z();
  
  VarPtr cross_normal();
  VarPtr dot_normal();
  VarPtr times_normal();
  VarPtr times_normal_x();
  VarPtr times_normal_y();
  VarPtr times_normal_z();
  
  LinearTermPtr termTraced();
  
  static VarPtr varForTrialID(int trialID, Teuchos::RCP<BilinearForm> bf);
};

#endif
