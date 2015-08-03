#include "CamelliaIntrepidExtendedTypes.h"

#include "Teuchos_TestForException.hpp"

static const std::string & S_OP_VALUE = "";
static const std::string & S_OP_GRAD = "\\nabla ";
static const std::string & S_OP_CURL = "\\nabla \\times ";
static const std::string & S_OP_DIV = "\\nabla \\cdot ";
static const std::string & S_OP_D1 = "D1 ";
static const std::string & S_OP_D2 = "D2 ";
static const std::string & S_OP_D3 = "D3 ";
static const std::string & S_OP_D4 = "D4 ";
static const std::string & S_OP_D5 = "D5 ";
static const std::string & S_OP_D6 = "D6 ";
static const std::string & S_OP_D7 = "D7 ";
static const std::string & S_OP_D8 = "D8 ";
static const std::string & S_OP_D9 = "D9 ";
static const std::string & S_OP_D10 = "D10 ";
static const std::string & S_OP_X = "{1 \\choose 0} \\cdot ";
static const std::string & S_OP_Y = "{0 \\choose 1} \\cdot ";
static const std::string & S_OP_Z = "\\bf{k} \\cdot ";
static const std::string & S_OP_DX = "\\frac{\\partial}{\\partial x} ";
static const std::string & S_OP_DY = "\\frac{\\partial}{\\partial y} ";
static const std::string & S_OP_DZ = "\\frac{\\partial}{\\partial z} ";
static const std::string & S_OP_DT = "\\frac{\\partial}{\\partial t} ";
static const std::string & S_OP_CROSS_NORMAL = "\\times \\widehat{n} ";
static const std::string & S_OP_DOT_NORMAL = "\\cdot \\widehat{n} ";
static const std::string & S_OP_TIMES_NORMAL = " \\widehat{n} \\cdot ";
static const std::string & S_OP_TIMES_NORMAL_X = " \\widehat{n}_x ";
static const std::string & S_OP_TIMES_NORMAL_Y = " \\widehat{n}_y ";
static const std::string & S_OP_TIMES_NORMAL_Z = " \\widehat{n}_z ";
static const std::string & S_OP_TIMES_NORMAL_T = " \\widehat{n}_t ";
static const std::string & S_OP_VECTORIZE_VALUE = ""; // handle this one separately...
static const std::string & S_OP_UNKNOWN = "[UNKNOWN OPERATOR] ";

using namespace Camellia;

bool Camellia::functionSpaceIsVectorized(EFunctionSpace fs)
{
  return (FUNCTION_SPACE_VECTOR_HGRAD == fs)
         ||     (FUNCTION_SPACE_TENSOR_HGRAD == fs)
         ||     (FUNCTION_SPACE_VECTOR_HVOL  == fs)
         ||     (FUNCTION_SPACE_TENSOR_HVOL == fs)
         ||     (FUNCTION_SPACE_VECTOR_HGRAD_DISC == fs)
         ||     (FUNCTION_SPACE_TENSOR_HGRAD_DISC == fs);
}

bool Camellia::functionSpaceIsDiscontinuous(Camellia::EFunctionSpace fs)
{
  switch (fs)
  {
  case FUNCTION_SPACE_HVOL:
  case FUNCTION_SPACE_VECTOR_HVOL:
  case FUNCTION_SPACE_TENSOR_HVOL:
  case FUNCTION_SPACE_HGRAD_DISC:
  case FUNCTION_SPACE_HCURL_DISC:
  case FUNCTION_SPACE_HDIV_DISC:
  case FUNCTION_SPACE_VECTOR_HGRAD_DISC:
  case FUNCTION_SPACE_TENSOR_HGRAD_DISC:
    return true;
    break;
  default:
    break;
  }
  return false;
}

EFunctionSpace Camellia::discontinuousSpaceForContinuous(Camellia::EFunctionSpace fs_continuous)
{
  switch (fs_continuous)
  {
  case FUNCTION_SPACE_HVOL:
  case FUNCTION_SPACE_VECTOR_HVOL:
  case FUNCTION_SPACE_TENSOR_HVOL:
    return fs_continuous;
  case FUNCTION_SPACE_HGRAD:
    return FUNCTION_SPACE_HGRAD_DISC;
  case FUNCTION_SPACE_HCURL:
    return FUNCTION_SPACE_HCURL_DISC;
  case FUNCTION_SPACE_HDIV:
    return FUNCTION_SPACE_HDIV_DISC;
  case FUNCTION_SPACE_VECTOR_HGRAD:
    return FUNCTION_SPACE_VECTOR_HGRAD_DISC;
  case FUNCTION_SPACE_TENSOR_HGRAD:
    return FUNCTION_SPACE_TENSOR_HGRAD_DISC;
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No known discontinuous version for fs_continuous.");
    return fs_continuous;
    break;
  }
}

EFunctionSpace Camellia::continuousSpaceForDiscontinuous(Camellia::EFunctionSpace fs_disc)
{
  switch (fs_disc)
  {
  case FUNCTION_SPACE_HVOL:
  case FUNCTION_SPACE_VECTOR_HVOL:
  case FUNCTION_SPACE_TENSOR_HVOL:
    return fs_disc; // in a sense, these are both continuous and discontinuous: they conform to the fs, but the fs has no continuity.
  case FUNCTION_SPACE_HGRAD_DISC:
    return FUNCTION_SPACE_HGRAD;
  case FUNCTION_SPACE_HCURL_DISC:
    return FUNCTION_SPACE_HCURL;
  case FUNCTION_SPACE_HDIV_DISC:
    return FUNCTION_SPACE_HDIV;
  case FUNCTION_SPACE_VECTOR_HGRAD_DISC:
    return FUNCTION_SPACE_VECTOR_HGRAD;
  case FUNCTION_SPACE_TENSOR_HGRAD_DISC:
    return FUNCTION_SPACE_TENSOR_HGRAD;
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No known continuous version for fs_disc.");
    return fs_disc;
    break;
  }
}

const std::set<EOperator> & Camellia::normalOperators()
{
  static std::set<EOperator> _normalOperators;
  if (_normalOperators.size() == 0)
  {
    _normalOperators.insert(OP_CROSS_NORMAL);
    _normalOperators.insert(OP_DOT_NORMAL);
    _normalOperators.insert(OP_TIMES_NORMAL);
    _normalOperators.insert(OP_TIMES_NORMAL_X);
    _normalOperators.insert(OP_TIMES_NORMAL_Y);
    _normalOperators.insert(OP_TIMES_NORMAL_Z);
  }
  return _normalOperators;
}

int Camellia::operatorRank(EOperator op, EFunctionSpace fs)
{
  // returns the rank of basis functions in the function space fs when op is applied
  // 0 scalar, 1 vector
  int SCALAR = 0, VECTOR = 1;
  switch (op)
  {
  case  Camellia::OP_VALUE:
    if (   (fs == Camellia::FUNCTION_SPACE_HGRAD)
           || (fs == Camellia::FUNCTION_SPACE_HVOL)
           || (fs == Camellia::FUNCTION_SPACE_REAL_SCALAR) )
      return SCALAR;
    else
      return VECTOR;
  case Camellia::OP_GRAD:
  case Camellia::OP_CURL:
    return VECTOR;
  case Camellia::OP_DIV:
  case Camellia::OP_X:
  case Camellia::OP_Y:
  case Camellia::OP_Z:
  case Camellia::OP_DX:
  case Camellia::OP_DY:
  case Camellia::OP_DZ:
    return SCALAR;
  case Camellia::OP_CROSS_NORMAL:
    return VECTOR;
  case Camellia::OP_DOT_NORMAL:
    return SCALAR;
  case Camellia::OP_TIMES_NORMAL:
    return VECTOR;
  case Camellia::OP_TIMES_NORMAL_X:
    return SCALAR;
  case Camellia::OP_TIMES_NORMAL_Y:
    return SCALAR;
  case Camellia::OP_TIMES_NORMAL_Z:
    return SCALAR;
  case Camellia::OP_VECTORIZE_VALUE:
    return VECTOR;
  default:
    return -1;
  }
}

const std::string & Camellia::operatorName(Camellia::EOperator op)
{
  switch (op)
  {
  case  Camellia::OP_VALUE:
    return S_OP_VALUE;
    break;
  case Camellia::OP_GRAD:
    return S_OP_GRAD;
    break;
  case Camellia::OP_CURL:
    return S_OP_CURL;
    break;
  case Camellia::OP_DIV:
    return S_OP_DIV;
    break;
  case Camellia::OP_D1:
    return S_OP_D1;
    break;
  case Camellia::OP_D2:
    return S_OP_D2;
    break;
  case Camellia::OP_D3:
    return S_OP_D3;
    break;
  case Camellia::OP_D4:
    return S_OP_D4;
    break;
  case Camellia::OP_D5:
    return S_OP_D5;
    break;
  case Camellia::OP_D6:
    return S_OP_D6;
    break;
  case Camellia::OP_D7:
    return S_OP_D7;
    break;
  case Camellia::OP_D8:
    return S_OP_D8;
    break;
  case Camellia::OP_D9:
    return S_OP_D9;
    break;
  case Camellia::OP_D10:
    return S_OP_D10;
    break;
  case Camellia::OP_X:
    return S_OP_X;
    break;
  case Camellia::OP_Y:
    return S_OP_Y;
    break;
  case Camellia::OP_Z:
    return S_OP_Z;
    break;
  case Camellia::OP_DX:
    return S_OP_DX;
    break;
  case Camellia::OP_DY:
    return S_OP_DY;
    break;
  case Camellia::OP_DZ:
    return S_OP_DZ;
    break;
  case Camellia::OP_DT:
    return S_OP_DT;
    break;
  case Camellia::OP_CROSS_NORMAL:
    return S_OP_CROSS_NORMAL;
    break;
  case Camellia::OP_DOT_NORMAL:
    return S_OP_DOT_NORMAL;
    break;
  case Camellia::OP_TIMES_NORMAL:
    return S_OP_TIMES_NORMAL;
    break;
  case Camellia::OP_TIMES_NORMAL_X:
    return S_OP_TIMES_NORMAL_X;
    break;
  case Camellia::OP_TIMES_NORMAL_Y:
    return S_OP_TIMES_NORMAL_Y;
    break;
  case Camellia::OP_TIMES_NORMAL_Z:
    return S_OP_TIMES_NORMAL_Z;
    break;
  case Camellia::OP_VECTORIZE_VALUE:
    return S_OP_VECTORIZE_VALUE;
    break;
  default:
    return S_OP_UNKNOWN;
    break;
  }
}
