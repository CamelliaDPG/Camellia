//
//  Var.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Var.h"

IntrepidExtendedTypes::EFunctionSpaceExtended VarFunctionSpaces::efsForSpace(Space space) {
  if (space == HGRAD)
    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
  if (space == HCURL)
    return IntrepidExtendedTypes::FUNCTION_SPACE_HCURL;
  if (space == HDIV)
    return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
  if (space == L2)
    return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
  TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
}

Var::Var(int ID, int rank, string name, EOperatorExtended op, Space fs, VarType varType) {
  _id = ID;
  _rank = rank;
  _name = name;
  _op = op;
  _fs = fs;
  _varType = varType;
}

int Var::ID() { 
  return _id;
}

const string & Var::name() { 
  return _name; 
}

EOperatorExtended Var::op() { 
  return _op; 
}

int Var::rank() {  // 0 for scalar, 1 for vector, etc. 
  return _rank; 
}

Space Var::space() { 
  return _fs; 
}

VarType Var::varType() { 
  return _varType; 
}

VarPtr Var::grad() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "grad() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank + 1, _name, IntrepidExtendedTypes::OPERATOR_GRAD, UNKNOWN_FS, _varType ) );
}

VarPtr Var::div() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "div() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank - 1, _name, IntrepidExtendedTypes::OPERATOR_DIV, UNKNOWN_FS, _varType ) );
}

VarPtr Var::curl() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( (_rank != 0) && (_rank != 1), std::invalid_argument, "curl() can only be applied to vars of ranks 0 or 1.");
  int newRank = (_rank == 0) ? 1 : 0;
  return Teuchos::rcp( new Var(_id, newRank, _name, IntrepidExtendedTypes::OPERATOR_CURL, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dx() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dx() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OPERATOR_DX, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dy() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dy() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OPERATOR_DY, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dz() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dz() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OPERATOR_DZ, UNKNOWN_FS, _varType ) );
}

VarPtr Var::x() { 
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "x() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_X, UNKNOWN_FS, _varType ) );
}

VarPtr Var::y() { 
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "y() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_Y, UNKNOWN_FS, _varType ) );
}

VarPtr Var::z() { 
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "z() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_Z, UNKNOWN_FS, _varType ) );
}

VarPtr Var::cross_normal() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "cross_normal() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_CROSS_NORMAL, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dot_normal() {
  TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "dot_normal() only supported for vars of rank 1.");
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_DOT_NORMAL, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank + 1, _name, OPERATOR_TIMES_NORMAL, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal_x() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank, _name, OPERATOR_TIMES_NORMAL_X, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal_y() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank, _name, OPERATOR_TIMES_NORMAL_Y, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal_z() {
  TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank, _name, OPERATOR_TIMES_NORMAL_Z, UNKNOWN_FS, _varType ) );
}
