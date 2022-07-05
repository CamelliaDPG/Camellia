//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Var.h"
#include "LinearTerm.h"
#include "BF.h"

using namespace Camellia;

Camellia::EFunctionSpace Camellia::efsForSpace(Space space)
{
  switch (space)
  {
  case HDIV:
    return Camellia::FUNCTION_SPACE_HDIV;
  case HGRAD:
    return Camellia::FUNCTION_SPACE_HGRAD;
  case HCURL:
    return Camellia::FUNCTION_SPACE_HCURL;
  case L2:
    return Camellia::FUNCTION_SPACE_HVOL;
  case CONSTANT_SCALAR:
    return Camellia::FUNCTION_SPACE_REAL_SCALAR;

  case HDIV_DISC:
    return Camellia::FUNCTION_SPACE_HDIV_DISC;
  case HGRAD_DISC:
    return Camellia::FUNCTION_SPACE_HGRAD_DISC;
  case HCURL_DISC:
    return Camellia::FUNCTION_SPACE_HCURL_DISC;

  case VECTOR_HGRAD:
    return Camellia::FUNCTION_SPACE_VECTOR_HGRAD;
  case VECTOR_L2:
    return Camellia::FUNCTION_SPACE_VECTOR_HVOL;

  case VECTOR_HGRAD_DISC:
    return Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC;

//    case TENSOR_HGRAD:

  case HDIV_FREE:
    return Camellia::FUNCTION_SPACE_HDIV_FREE;

  case UNKNOWN_FS:
    return Camellia::FUNCTION_SPACE_UNKNOWN;
      
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
//    return Camellia::FUNCTION_SPACE_UNKNOWN;
  }
}

int Camellia::rankForSpace(Space space)
{
  switch (space)
  {
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
//      return Camellia::FUNCTION_SPACE_UNKNOWN;
  }
}

Space Camellia::spaceForEFS(Camellia::EFunctionSpace efs)
{
  switch (efs)
  {
  case Camellia::FUNCTION_SPACE_HDIV:
    return HDIV;
  case Camellia::FUNCTION_SPACE_HGRAD:
    return HGRAD;
  case Camellia::FUNCTION_SPACE_HCURL:
    return HCURL;
  case Camellia::FUNCTION_SPACE_HVOL:
    return L2;
  case Camellia::FUNCTION_SPACE_REAL_SCALAR:
    return CONSTANT_SCALAR;

  case Camellia::FUNCTION_SPACE_HDIV_DISC:
    return HDIV_DISC;
  case Camellia::FUNCTION_SPACE_HGRAD_DISC:
    return HGRAD_DISC;
  case Camellia::FUNCTION_SPACE_HCURL_DISC:
    return HCURL_DISC;

  case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
    return VECTOR_HGRAD;
  case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
    return VECTOR_L2;

  case Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC:
    return VECTOR_HGRAD_DISC;

  case Camellia::FUNCTION_SPACE_HDIV_FREE:
    return HDIV_FREE;

  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
//    return UNKNOWN_FS;
  }

//
//  if (efs == Camellia::FUNCTION_SPACE_HGRAD) {
//    return HGRAD;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HCURL) {
//    return HCURL;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HDIV) {
//    return HDIV;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HVOL) {
//    return L2;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_REAL_SCALAR) {
//    return CONSTANT_SCALAR;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD) {
//    return VECTOR_HGRAD;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_VECTOR_HVOL) {
//    return VECTOR_L2;
//  }
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown function space.");
}

template <typename Scalar>
VarPtr Var::varForTrialID(int trialID, TBFPtr<Scalar> bf)
{
  Camellia::EFunctionSpace efs = bf->functionSpaceForTrial(trialID);
  Space space = spaceForEFS(efs);
  int rank = rankForSpace(space);
//  if (efs == Camellia::FUNCTION_SPACE_HGRAD) {
//    space = HGRAD;
//    rank = 0;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HCURL) {
//    space = HCURL;
//    rank = 1;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HDIV) {
//    space = HDIV;
//    rank = 1;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HDIV_FREE) {
//    space = HDIV_FREE;
//    rank = 1;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_HVOL) {
//    space = L2;
//    rank = 0;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_REAL_SCALAR) {
//    space = CONSTANT_SCALAR;
//    rank = 0;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD) {
//    space = VECTOR_HGRAD;
//    rank = 1;
//  }
//  if (efs == Camellia::FUNCTION_SPACE_VECTOR_HVOL) {
//    space = VECTOR_L2;
//    rank = 1;
//  }

  VarType varType;
  if (bf->isFluxOrTrace(trialID))
  {
    if ((space==L2) || (space==VECTOR_L2))
    {
      varType = FLUX;
    }
    else if ((space==HGRAD) || (space==VECTOR_HGRAD))
    {
      varType = TRACE;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unclassifiable variable");
    }
  }
  else
  {
    varType = FIELD;
  }

  return Teuchos::rcp(new Var(trialID, rank, "trial", OP_VALUE, space, varType));
}

template VarPtr Var::varForTrialID(int trialID, TBFPtr<double> bf);

Var::Var(int ID, int rank, string name, Camellia::EOperator op, Space fs, VarType varType, LinearTermPtr termTraced,
         bool definedOnTemporalInterfaces, Space temporalFS)
{
  _id = ID;
  _rank = rank;
  _name = name;
  _op = op;
  _fs = fs;
  _varType = varType;
  _termTraced = termTraced;
  _definedOnTemporalInterfaces = definedOnTemporalInterfaces;
  _temporalFS = temporalFS;
}

VarPtr Var::applyOp(Camellia::EOperator op, int spaceDim) const
{
  using namespace Teuchos;
  switch (op)
  {
    case OP_VALUE:
      return rcp( new Var(*this) );
    case OP_GRAD:
      return this->grad();
    case OP_CURL:
      return this->curl(spaceDim);
    case OP_DIV:
      return this->div();
    case OP_X:
      return this->x();
    case OP_Y:
      return this->y();
    case OP_Z:
      return this->z();
    case OP_T:
      return this->t();
    case OP_DX:
      return this->dx();
    case OP_DY:
      return this->dy();
    case OP_DZ:
      return this->dz();
    case OP_DT:
      return this->dt();
    case OP_LAPLACIAN:
      return this->laplacian();
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled op type");
  }
}

BasisPtr Var::getBasis(CellTopoPtr cellTopo, const std::vector<int> &p_H1, const int sideOrdinal) const
{
  EFunctionSpace efsBase = efsForSpace(_fs);
  std::vector<EFunctionSpace> fs(1,efsBase);
  
  if (sideOrdinal == -1)
  {
    if (isSpaceTime())
    {
      fs.push_back(efsForSpace(_temporalFS));
    }
    
    return _basisFamily->getBasis(cellTopo, fs, p_H1);
  }
  else
  {
    std::vector<int> p_H1_forSide = p_H1;
    if (isSpaceTime())
    {
      if (cellTopo->sideIsSpatial(sideOrdinal))
      {
        // spatial sides *are* extruded in the final tensorial dimension (time, for space-time variables)
        // therefore we set the final fs as the temporal fs:
        if (fs.size() == 1)
        {
          fs.push_back(efsForSpace(_temporalFS));
        }
      }
      else
      {
        // non-spatial sides have the same fs as that defined on the base (spatial) topology,
        // but the p_H1 can be expected to include the temporal degree; we drop that here:
        p_H1_forSide.pop_back();
      }
      
    }
    return _basisFamily->getBasis(cellTopo->getSide(sideOrdinal), fs, p_H1_forSide);
  }
}

BasisFamilyPtr Var::getBasisFamily() const
{
  return _basisFamily;
}

int Var::ID() const
{
  return _id;
}

bool Var::isDefinedOnTemporalInterface() const
{
  return _definedOnTemporalInterfaces;
}

const string & Var::name() const
{
  return _name;
}

bool isRightOperator(Camellia::EOperator op)   // as opposed to left
{
  set<int> _normalOperators;
  _normalOperators.insert(OP_CROSS_NORMAL);
  _normalOperators.insert(OP_DOT_NORMAL);
  _normalOperators.insert(OP_TIMES_NORMAL);
  _normalOperators.insert(OP_TIMES_NORMAL_X);
  _normalOperators.insert(OP_TIMES_NORMAL_Y);
  _normalOperators.insert(OP_TIMES_NORMAL_Z);
  return _normalOperators.find(op) != _normalOperators.end();
}

string Var::displayString() const
{
  ostringstream varStream;
  if ( isRightOperator(_op) )
  {
    varStream << _name << operatorName(_op);
  }
  else
  {
    varStream << operatorName(_op) << _name;
  }
  return varStream.str();
}

Camellia::EOperator Var::op() const
{
  return _op;
}

int Var::rank() const    // 0 for scalar, 1 for vector, etc.
{
  return _rank;
}

Space Var::space() const
{
  return _fs;
}

VarType Var::varType() const
{
  return _varType;
}

LinearTermPtr Var::termTraced() const
{
  return _termTraced;
}

VarPtr Var::grad() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  if (space() != VECTOR_HGRAD)
    TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "grad() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank + 1, _name, Camellia::OP_GRAD, UNKNOWN_FS, _varType ) );
}

VarPtr Var::div() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "div() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank - 1, _name, Camellia::OP_DIV, UNKNOWN_FS, _varType ) );
}

VarPtr Var::curl(int spaceDim) const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( (_rank != 0) && (_rank != 1), std::invalid_argument, "curl() can only be applied to vars of ranks 0 or 1.");
  if (spaceDim == 2)
  {
    int newRank = (_rank == 0) ? 1 : 0;
    return Teuchos::rcp( new Var(_id, newRank, _name, Camellia::OP_CURL, UNKNOWN_FS, _varType ) );
  }
  else
  {
    // in 3D, curl is rank-preserving.  (Vectors map to vectors.)
    return Teuchos::rcp( new Var(_id, _rank, _name, Camellia::OP_CURL, UNKNOWN_FS, _varType ) );
  }
}

bool Var::isDefinedOnVolume() const
{
  return (_varType != TRACE) && (_varType != FLUX);
}

VarPtr Var::laplacian() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "laplacian() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, Camellia::OP_LAPLACIAN, UNKNOWN_FS, _varType ) );
}

// ! i is 1-based ordinal; 1 for dx, 2 for dy, 3 for dz
VarPtr Var::di(int i) const
{
  switch (i)
  {
    case 1:
//      return this->dx();
    case 2:
//      return this->dy();
    case 3:
//      return this->dz();
    case 4:
    case 5:
    case 6:
    case 7:
    {
      Camellia::EOperator di_op = static_cast<Camellia::EOperator>(Camellia::OP_DI1 + (i-1));
      Camellia::EOperator op = combineOperators(_op, di_op);
      TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "di() only supported for vars of rank 0.");
      return Teuchos::rcp( new Var(_id, _rank, _name, op, UNKNOWN_FS, _varType, _termTraced, _definedOnTemporalInterfaces, _temporalFS) );
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported value of i");
  }
}

VarPtr Var::dx() const
{
  Camellia::EOperator op = combineOperators(_op, Camellia::OP_DX);
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dx() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, op, UNKNOWN_FS, _varType, _termTraced, _definedOnTemporalInterfaces, _temporalFS ) );
}

VarPtr Var::dy() const
{
  Camellia::EOperator op = combineOperators(_op, Camellia::OP_DY);
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dy() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, op, UNKNOWN_FS, _varType, _termTraced, _definedOnTemporalInterfaces, _temporalFS ) );
}

VarPtr Var::dz() const
{
  Camellia::EOperator op = combineOperators(_op, Camellia::OP_DZ);
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dz() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, op, UNKNOWN_FS, _varType, _termTraced, _definedOnTemporalInterfaces, _temporalFS ) );
}

VarPtr Var::dt() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "dt() operator can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dt() only supported for vars of rank 0.");
  return Teuchos::rcp( new Var(_id, _rank, _name, Camellia::OP_DT, UNKNOWN_FS, _varType ) );
}

void Var::setBasisFamily(BasisFamilyPtr basisFamily)
{
  _basisFamily = basisFamily;
}

VarPtr Var::x() const
{
  Camellia::EOperator op = combineOperators(_op, Camellia::OP_X);
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "x() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, op, UNKNOWN_FS, _varType ) );
}

VarPtr Var::y() const
{
  Camellia::EOperator op = combineOperators(_op, Camellia::OP_Y);
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "y() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, op, UNKNOWN_FS, _varType ) );
}

VarPtr Var::z() const
{
  Camellia::EOperator op = combineOperators(_op, Camellia::OP_Z);
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "z() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, op, UNKNOWN_FS, _varType ) );
}

VarPtr Var::t() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "z() only supported for vars of rank 1.");
  return Teuchos::rcp( new Var(_id, _rank-1, _name, OP_T, UNKNOWN_FS, _varType ) );
}

LinearTermPtr Var::cross_normal(int spaceDim) const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "cross_normal() only supported for vars of rank 1.");
  
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, _varType ) );
  
  FunctionPtr n = Function::normal();
  if (spaceDim == 2)
  {
    return var->x() * n->y() - var->y() * n->x();
  }
  else if (spaceDim == 3)
  {
    FunctionPtr x_comp = TFunction<double>::constant({1,0,0});
    FunctionPtr y_comp = TFunction<double>::constant({0,1,0});
    FunctionPtr z_comp = TFunction<double>::constant({0,0,1});
    return x_comp * (var->y() * n->z() - var->z() * n->y()) + y_comp * (var->z() * n->x() - var->x() * n->z())
         + z_comp * (var->x() * n->y() - var->y() * n->x());
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid space dimension for cross_normal()");
}

LinearTermPtr Var::dot_normal() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "dot_normal() only supported for vars of rank 1.");
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, _varType ) );
  FunctionPtr n = Function::normal();
  return var * n;
}

bool Var::isSpaceTime() const
{
  return _temporalFS != UNKNOWN_FS;
}

void Var::setTemporalFunctionSpace(const Space &fs)
{
  bool validOption = (fs == UNKNOWN_FS) || (fs == L2) || (fs == HGRAD) || (fs == HGRAD_DISC);
  TEUCHOS_TEST_FOR_EXCEPTION(!validOption, std::invalid_argument, "Invalid temporal function space choice.");
  _temporalFS = fs;
}

Space Var::temporalFunctionSpace() const
{
  return _temporalFS;
}

// ! i is 1-based ordinal; 1 for x, 2 for y, 3 for z
VarPtr Var::spatialComponent(int i) const
{
  switch (i)
  {
    case 1:
      return this->x();
    case 2:
      return this->y();
    case 3:
      return this->z();
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported value of i");
  }
}

LinearTermPtr Var::times_normal() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarType newType;
  if (_varType == TRACE)
    newType = FLUX;
  else if (_varType == FLUX)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fluxes can't be multiplied by normal (they already implicitly contain a normal).");
  }
  else     // tests and field variables, restricted to the boundary, can be multiplied by normal--and are considered of the same type as before.
  {
    newType = _varType;
  }
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, newType ) );
  FunctionPtr n = Function::normal();
  return var * n;
}

LinearTermPtr Var::times_normal_x() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, _varType ) );
  FunctionPtr n = Function::normal();
  return var * n->x();
}

LinearTermPtr Var::times_normal_y() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, _varType ) );
  FunctionPtr n = Function::normal();
  return var * n->y();
}

LinearTermPtr Var::times_normal_z() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, _varType ) );
  FunctionPtr n = Function::normal();
  return var * n->z();
}

LinearTermPtr Var::times_normal_t() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( _op !=  Camellia::OP_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
  VarPtr var = Teuchos::rcp( new Var(_id, _rank, _name, _op, _fs, _varType ) );
  FunctionPtr n = Function::normalSpaceTime();
  return var * n->t();
}
