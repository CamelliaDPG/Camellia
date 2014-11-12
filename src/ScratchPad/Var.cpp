//
//  Var.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Var.h"
#include "LinearTerm.h"

static const string & S_OP_VALUE = "";
static const string & S_OP_GRAD = "\\nabla ";
static const string & S_OP_CURL = "\\nabla \\times ";
static const string & S_OP_DIV = "\\nabla \\cdot ";
static const string & S_OP_D1 = "D1 ";
static const string & S_OP_D2 = "D2 ";
static const string & S_OP_D3 = "D3 ";
static const string & S_OP_D4 = "D4 ";
static const string & S_OP_D5 = "D5 ";
static const string & S_OP_D6 = "D6 ";
static const string & S_OP_D7 = "D7 ";
static const string & S_OP_D8 = "D8 ";
static const string & S_OP_D9 = "D9 ";
static const string & S_OP_D10 = "D10 ";
static const string & S_OP_X = "{1 \\choose 0} \\cdot ";
static const string & S_OP_Y = "{0 \\choose 1} \\cdot ";
static const string & S_OP_Z = "\\bf{k} \\cdot ";
static const string & S_OP_DX = "\\frac{\\partial}{\\partial x} ";
static const string & S_OP_DY = "\\frac{\\partial}{\\partial y} ";
static const string & S_OP_DZ = "\\frac{\\partial}{\\partial z} ";
static const string & S_OP_CROSS_NORMAL = "\\times \\widehat{n} ";
static const string & S_OP_DOT_NORMAL = "\\cdot \\widehat{n} ";
static const string & S_OP_TIMES_NORMAL = " \\widehat{n} \\cdot ";
static const string & S_OP_TIMES_NORMAL_X = " \\widehat{n}_x ";
static const string & S_OP_TIMES_NORMAL_Y = " \\widehat{n}_y ";
static const string & S_OP_TIMES_NORMAL_Z = " \\widehat{n}_z ";
static const string & S_OP_VECTORIZE_VALUE = ""; // handle this one separately...
static const string & S_OP_UNKNOWN = "[UNKNOWN OPERATOR] ";

IntrepidExtendedTypes::EFunctionSpaceExtended VarFunctionSpaces::efsForSpace(Space space) {
  switch (space) {
    case HDIV:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
    case HGRAD:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    case HCURL:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HCURL;
    case L2:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
    case CONSTANT_SCALAR:
      return IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR;
      
    case HDIV_DISC:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_DISC;
    case HGRAD_DISC:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD_DISC;
    case HCURL_DISC:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HCURL_DISC;

    case VECTOR_HGRAD:
      return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD;
    case VECTOR_L2:
      return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL;
    
    case VECTOR_HGRAD_DISC:
      return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD_DISC;
      
//    case TENSOR_HGRAD:
      
    case HDIV_FREE:
      return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_FREE;
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
      return IntrepidExtendedTypes::FUNCTION_SPACE_UNKNOWN;
  }
  
//  if (space == HGRAD)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
//  if (space == HCURL)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HCURL;
//  if (space == HDIV)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
//  if (space == HGRAD_DISC)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD_DISC;
//  if (space == HCURL_DISC)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HCURL_DISC;
//  if (space == HDIV_DISC)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_DISC;
//  if (space == HDIV_FREE)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_FREE;
//  if (space == L2)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
//  if (space == CONSTANT_SCALAR)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR;
//  if (space == VECTOR_HGRAD)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD;
//  if (space == VECTOR_HGRAD_DISC)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD_DISC;
//  if (space == VECTOR_L2)
//    return IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL;
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
//  return IntrepidExtendedTypes::FUNCTION_SPACE_UNKNOWN;
}

int VarFunctionSpaces::rankForSpace(Space space) {
  switch (space) {
    case HDIV:
      return 1;
    case HGRAD:
      return 0;
    case HCURL:
      return 1;
    case L2:
      return 0;
    case CONSTANT_SCALAR:
      return 0;
      
    case HDIV_DISC:
      return 1;
    case HGRAD_DISC:
      return 0;
    case HCURL_DISC:
      return 1;
      
    case VECTOR_HGRAD:
      return 1;
    case VECTOR_L2:
      return 1;
      
    case VECTOR_HGRAD_DISC:
      return 1;
      
      //    case TENSOR_HGRAD:
      
    case HDIV_FREE:
      return 1;
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
      return IntrepidExtendedTypes::FUNCTION_SPACE_UNKNOWN;
  }
}

Space VarFunctionSpaces::spaceForEFS(IntrepidExtendedTypes::EFunctionSpaceExtended efs) {
  switch (efs) {
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
      return HDIV;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
      return HGRAD;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
      return HCURL;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
      return L2;
    case IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR:
      return CONSTANT_SCALAR;
      
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_DISC:
      return HDIV_DISC;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD_DISC:
      return HGRAD_DISC;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL_DISC:
      return HCURL_DISC;
      
    case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD:
      return VECTOR_HGRAD;
    case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL:
      return VECTOR_L2;
      
    case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD_DISC:
      return VECTOR_HGRAD_DISC;
      
      //    case TENSOR_HGRAD:
      
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_FREE:
      return HDIV_FREE;
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
      return UNKNOWN_FS;
  }
  
//  
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) {
//    return HGRAD;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HCURL) {
//    return HCURL;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HDIV) {
//    return HDIV;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) {
//    return L2;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR) {
//    return CONSTANT_SCALAR;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) {
//    return VECTOR_HGRAD;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL) {
//    return VECTOR_L2;
//  }
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
}

VarPtr Var::varForTrialID(int trialID, Teuchos::RCP<BilinearForm> bf) {
  IntrepidExtendedTypes::EFunctionSpaceExtended efs = bf->functionSpaceForTrial(trialID);
  Space space = spaceForEFS(efs);
  int rank = rankForSpace(space);
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) {
//    space = HGRAD;
//    rank = 0;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HCURL) {
//    space = HCURL;
//    rank = 1;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HDIV) {
//    space = HDIV;
//    rank = 1;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_FREE) {
//    space = HDIV_FREE;
//    rank = 1;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) {
//    space = L2;
//    rank = 0;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR) {
//    space = CONSTANT_SCALAR;
//    rank = 0;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) {
//    space = VECTOR_HGRAD;
//    rank = 1;
//  }
//  if (efs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL) {
//    space = VECTOR_L2;
//    rank = 1;
//  }
  
  VarType varType;
  if (bf->isFluxOrTrace(trialID)) {
    if ((space==L2) || (space==VECTOR_L2)) {
      varType = FLUX;
    } else if ((space==HGRAD) || (space==VECTOR_HGRAD)) {
      varType = TRACE;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unclassifiable variable");
    }
  } else {
    varType = FIELD;
  }
  
  return Teuchos::rcp(new Var(trialID, rank, "trial", OP_VALUE, space, varType));
}

Var::Var(int ID, int rank, string name, IntrepidExtendedTypes::EOperatorExtended op, Space fs, VarType varType, LinearTermPtr termTraced) {
  _id = ID;
  _rank = rank;
  _name = name;
  _op = op;
  _fs = fs;
  _varType = varType;
  _termTraced = termTraced;
}

int Var::ID() { 
  return _id;
}

const string & Var::name() { 
  return _name; 
}

const string & operatorName(IntrepidExtendedTypes::EOperatorExtended op) {
  switch (op) {
    case  IntrepidExtendedTypes::OP_VALUE:
      return S_OP_VALUE; 
      break;
    case IntrepidExtendedTypes::OP_GRAD:
      return S_OP_GRAD; 
      break;
    case IntrepidExtendedTypes::OP_CURL:
      return S_OP_CURL; 
      break;
    case IntrepidExtendedTypes::OP_DIV:
      return S_OP_DIV; 
      break;
    case IntrepidExtendedTypes::OP_D1:
      return S_OP_D1; 
      break;
    case IntrepidExtendedTypes::OP_D2:
      return S_OP_D2; 
      break;
    case IntrepidExtendedTypes::OP_D3:
      return S_OP_D3; 
      break;
    case IntrepidExtendedTypes::OP_D4:
      return S_OP_D4; 
      break;
    case IntrepidExtendedTypes::OP_D5:
      return S_OP_D5; 
      break;
    case IntrepidExtendedTypes::OP_D6:
      return S_OP_D6; 
      break;
    case IntrepidExtendedTypes::OP_D7:
      return S_OP_D7; 
      break;
    case IntrepidExtendedTypes::OP_D8:
      return S_OP_D8; 
      break;
    case IntrepidExtendedTypes::OP_D9:
      return S_OP_D9; 
      break;
    case IntrepidExtendedTypes::OP_D10:
      return S_OP_D10; 
      break;
    case IntrepidExtendedTypes::OP_X:
      return S_OP_X; 
      break;
    case IntrepidExtendedTypes::OP_Y:
      return S_OP_Y; 
      break;
    case IntrepidExtendedTypes::OP_Z:
      return S_OP_Z; 
      break;
    case IntrepidExtendedTypes::OP_DX:
      return S_OP_DX; 
      break;
    case IntrepidExtendedTypes::OP_DY:
      return S_OP_DY; 
      break;
    case IntrepidExtendedTypes::OP_DZ:
      return S_OP_DZ; 
      break;
    case IntrepidExtendedTypes::OP_CROSS_NORMAL:
      return S_OP_CROSS_NORMAL; 
      break;
    case IntrepidExtendedTypes::OP_DOT_NORMAL:
      return S_OP_DOT_NORMAL; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL:
      return S_OP_TIMES_NORMAL; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_X:
      return S_OP_TIMES_NORMAL_X; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_Y:
      return S_OP_TIMES_NORMAL_Y; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_Z:
      return S_OP_TIMES_NORMAL_Z; 
      break;
    case IntrepidExtendedTypes::OP_VECTORIZE_VALUE:
      return S_OP_VECTORIZE_VALUE; 
      break;
    default:
      return S_OP_UNKNOWN;
      break;
  }
}

bool isRightOperator(IntrepidExtendedTypes::EOperatorExtended op) { // as opposed to left
  set<int> _normalOperators;
  _normalOperators.insert(OP_CROSS_NORMAL);
  _normalOperators.insert(OP_DOT_NORMAL);
  _normalOperators.insert(OP_TIMES_NORMAL);
  _normalOperators.insert(OP_TIMES_NORMAL_X);
  _normalOperators.insert(OP_TIMES_NORMAL_Y);
  _normalOperators.insert(OP_TIMES_NORMAL_Z);
  return _normalOperators.find(op) != _normalOperators.end();
}

string Var::displayString() {
  ostringstream varStream;
  if ( isRightOperator(_op) ) {
    varStream << _name << operatorName(_op);
  } else {
    varStream << operatorName(_op) << _name;
  }
  return varStream.str();
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

LinearTermPtr Var::termTraced() {
  return _termTraced;
}

VarPtr Var::grad() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  if (space() != VECTOR_HGRAD)
    TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "grad() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank + 1, _name, IntrepidExtendedTypes::OP_GRAD, UNKNOWN_FS, _varType ) );
}

VarPtr Var::div() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "div() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank - 1, _name, IntrepidExtendedTypes::OP_DIV, UNKNOWN_FS, _varType ) );
}

VarPtr Var::curl(int spaceDim) {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( (_rank != 0) && (_rank != 1), std::invalid_argument, "curl() can only be applied to vars of ranks 0 or 1.");
  if (spaceDim == 2) {
    int newRank = (_rank == 0) ? 1 : 0;
    return Teuchos::rcp( new Var(_id, newRank, _name, IntrepidExtendedTypes::OP_CURL, UNKNOWN_FS, _varType ) );
  } else {
    // in 3D, curl is rank-preserving.  (Vectors map to vectors.)
    return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OP_CURL, UNKNOWN_FS, _varType ) );
  }
}

VarPtr Var::dx() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dx() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OP_DX, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dy() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dy() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OP_DY, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dz() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dz() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OP_DZ, UNKNOWN_FS, _varType ) );
}

VarPtr Var::x() { 
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "x() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OP_X, UNKNOWN_FS, _varType ) );
}

VarPtr Var::y() { 
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "y() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OP_Y, UNKNOWN_FS, _varType ) );
}

VarPtr Var::z() { 
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "z() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OP_Z, UNKNOWN_FS, _varType ) );
}

VarPtr Var::cross_normal() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "cross_normal() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OP_CROSS_NORMAL, UNKNOWN_FS, _varType ) );
}

VarPtr Var::dot_normal() {
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "dot_normal() only supported for vars of rank 1.");
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OP_DOT_NORMAL, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarType newType;
  if (_varType == TRACE)
    newType = FLUX;
  else if (_varType == FLUX) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fluxes can't be multiplied by normal (they already implicitly contain a normal).");
  } else { // tests and field variables, restricted to the boundary, can be multiplied by normal--and are considered of the same type as before.
    newType = _varType;
  }
  return Teuchos::rcp( new Var(_id, _rank + 1, _name, OP_TIMES_NORMAL, UNKNOWN_FS, newType ) );
}

VarPtr Var::times_normal_x() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank, _name, OP_TIMES_NORMAL_X, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal_y() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank, _name, OP_TIMES_NORMAL_Y, UNKNOWN_FS, _varType ) );
}

VarPtr Var::times_normal_z() {
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  IntrepidExtendedTypes::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  return Teuchos::rcp( new Var(_id, _rank, _name, OP_TIMES_NORMAL_Z, UNKNOWN_FS, _varType ) );
}
