//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

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

void Camellia::setOrdinalTagData(std::vector<std::vector<std::vector<int> > > &tagToOrdinal,
                                 std::vector<std::vector<int> > &ordinalToTag,
                                 const int *tags,
                                 const int basisCard,
                                 const int tagSize,
                                 const int posScDim,
                                 const int posScOrd,
                                 const int posDfOrd)
{
  ordinalToTag.resize(basisCard);
  int maxScDim = 0;
  int maxScOrd = 0;
  int maxDfOrd = 0;
  for (int dofOrd=0; dofOrd<basisCard; dofOrd++)
  {
    const int *tag = tags + dofOrd * tagSize;
    ordinalToTag[dofOrd].assign(tag, tag + tagSize);
    maxScDim = std::max(maxScDim, tag[posScDim]);
    maxScOrd = std::max(maxScOrd, tag[posScOrd]);
    maxDfOrd = std::max(maxDfOrd, tag[posDfOrd]);
  }

  tagToOrdinal.assign(maxScDim + 1, std::vector<std::vector<int> >());
  for (int dofOrd=0; dofOrd<basisCard; dofOrd++)
  {
    const int *tag = tags + dofOrd * tagSize;
    const int scDim = tag[posScDim];
    const int scOrd = tag[posScOrd];
    const int dfOrd = tag[posDfOrd];
    if (static_cast<int>(tagToOrdinal[scDim].size()) <= scOrd)
    {
      tagToOrdinal[scDim].resize(scOrd + 1);
    }
    if (static_cast<int>(tagToOrdinal[scDim][scOrd].size()) <= dfOrd)
    {
      tagToOrdinal[scDim][scOrd].resize(dfOrd + 1, -1);
    }
    tagToOrdinal[scDim][scOrd][dfOrd] = dofOrd;
  }
}

bool Camellia::functionSpaceIsVectorized(EFunctionSpace fs)
{
  return (FUNCTION_SPACE_VECTOR_HGRAD == fs)
         ||     (FUNCTION_SPACE_TENSOR_HGRAD == fs)
         ||     (FUNCTION_SPACE_VECTOR_HVOL  == fs)
         ||     (FUNCTION_SPACE_TENSOR_HVOL == fs)
         ||     (FUNCTION_SPACE_VECTOR_HGRAD_DISC == fs)
         ||     (FUNCTION_SPACE_TENSOR_HGRAD_DISC == fs);
}

bool Camellia::functionSpaceIsBroken(Camellia::EFunctionSpace fs)
{
  switch (fs)
  {
  case FUNCTION_SPACE_HGRAD_DISC:
  case FUNCTION_SPACE_HCURL_DISC:
  case FUNCTION_SPACE_HDIV_DISC:
  case FUNCTION_SPACE_VECTOR_HGRAD_DISC:
  case FUNCTION_SPACE_TENSOR_HGRAD_DISC:
    return true;
  default:
    return false;
  }
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
  default:
    return false;
  }
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
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No known discontinuous version for fs_continuous.");
//    return fs_continuous;
  }
}

EFunctionSpace Camellia::continuousSpaceForDiscontinuous(Camellia::EFunctionSpace fs_disc, bool upgradeHVOL)
{
  // in a sense, HVOL spaces are both continuous and discontinuous: they conform to the fs, but the fs has no continuity.
  // if upgradeHVOL is false, we return the HVOL space; otherwise, we return the corresponding H^1 space.
  switch (fs_disc)
  {
  case FUNCTION_SPACE_HVOL:
      return upgradeHVOL ? FUNCTION_SPACE_HGRAD : FUNCTION_SPACE_HVOL;
  case FUNCTION_SPACE_VECTOR_HVOL:
      return upgradeHVOL ? FUNCTION_SPACE_VECTOR_HGRAD : FUNCTION_SPACE_VECTOR_HVOL;
  case FUNCTION_SPACE_TENSOR_HVOL:
      return upgradeHVOL ? FUNCTION_SPACE_TENSOR_HGRAD : FUNCTION_SPACE_TENSOR_HVOL;
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
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No known continuous version for fs_disc.");
//    return fs_disc;
  }
}

EOperator Camellia::combineOperators(EOperator op_first, EOperator op_second)
{
  switch (op_first) {
    case OP_VALUE:
      return op_second;
    case OP_GRAD:
      switch (op_second) {
        case OP_X:
          return OP_DX;
        case OP_Y:
          return OP_DY;
        case OP_Z:
          return OP_DZ;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported op combination");
      }
    case OP_DX:
      switch (op_second) {
        case OP_DX:
          return OP_DXDX;
        case OP_DY:
          return OP_DXDY;
        case OP_DZ:
          return OP_DXDZ;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported op combination");
      }
    case OP_DY:
      switch (op_second) {
        case OP_DX:
          return OP_DXDY;
        case OP_DY:
          return OP_DYDY;
        case OP_DZ:
          return OP_DYDZ;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported op combination");
      }
    case OP_DZ:
      switch (op_second) {
        case OP_DX:
          return OP_DXDZ;
        case OP_DY:
          return OP_DYDZ;
        case OP_DZ:
          return OP_DZDZ;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported op combination");
      }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported op combination");
  }
}

EFunctionSpace Camellia::fromIntrepid2FunctionSpace(const ::Intrepid2::EFunctionSpace &fs)
{
  switch (fs)
  {
    case Intrepid2::FUNCTION_SPACE_HVOL:         return FUNCTION_SPACE_HVOL;
    case Intrepid2::FUNCTION_SPACE_HGRAD:        return FUNCTION_SPACE_HGRAD;
    case Intrepid2::FUNCTION_SPACE_HDIV:         return FUNCTION_SPACE_HDIV;
    case Intrepid2::FUNCTION_SPACE_HCURL:        return FUNCTION_SPACE_HCURL;
    case Intrepid2::FUNCTION_SPACE_VECTOR_HGRAD: return FUNCTION_SPACE_VECTOR_HGRAD;
    case Intrepid2::FUNCTION_SPACE_TENSOR_HGRAD: return FUNCTION_SPACE_TENSOR_HGRAD;
    case Intrepid2::FUNCTION_SPACE_MAX:          return FUNCTION_SPACE_UNKNOWN;
  }
}

::Intrepid2::EFunctionSpace Camellia::toIntrepid2FunctionSpace(const EFunctionSpace &fs)
{
  switch (fs)
  {
    case FUNCTION_SPACE_HVOL:
    case FUNCTION_SPACE_VECTOR_HVOL:
    case FUNCTION_SPACE_TENSOR_HVOL:
    case FUNCTION_SPACE_HVOL_DISC:
    case FUNCTION_SPACE_VECTOR_HVOL_DISC:
    case FUNCTION_SPACE_TENSOR_HVOL_DISC:
      return Intrepid2::FUNCTION_SPACE_HVOL;
    case FUNCTION_SPACE_HGRAD:
    case FUNCTION_SPACE_HGRAD_DISC:
      return Intrepid2::FUNCTION_SPACE_HGRAD;
    case FUNCTION_SPACE_HDIV:
    case FUNCTION_SPACE_HDIV_DISC:
    case FUNCTION_SPACE_HDIV_FREE:
      return Intrepid2::FUNCTION_SPACE_HDIV;
    case FUNCTION_SPACE_HCURL:
    case FUNCTION_SPACE_HCURL_DISC:
      return Intrepid2::FUNCTION_SPACE_HCURL;
    case FUNCTION_SPACE_VECTOR_HGRAD:
    case FUNCTION_SPACE_VECTOR_HGRAD_DISC:
      return Intrepid2::FUNCTION_SPACE_VECTOR_HGRAD;
    case FUNCTION_SPACE_TENSOR_HGRAD:
    case FUNCTION_SPACE_TENSOR_HGRAD_DISC:
      return Intrepid2::FUNCTION_SPACE_TENSOR_HGRAD;
    case FUNCTION_SPACE_UNKNOWN:
    case FUNCTION_SPACE_REAL_SCALAR: // no corresponding FS in Intrepid2
      return Intrepid2::FUNCTION_SPACE_MAX;
  }
}

int Camellia::getH1Order(int basisDegree, EFunctionSpace fs)
{
  switch (fs)
  {
    case FUNCTION_SPACE_HVOL:
    case FUNCTION_SPACE_HVOL_DISC:
    case FUNCTION_SPACE_TENSOR_HVOL:
    case FUNCTION_SPACE_TENSOR_HVOL_DISC:
    case FUNCTION_SPACE_VECTOR_HVOL:
    case FUNCTION_SPACE_VECTOR_HVOL_DISC:
      return basisDegree + 1;
    default:
      return basisDegree;
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
  case Camellia::OP_GRAD:
    return S_OP_GRAD;
  case Camellia::OP_CURL:
    return S_OP_CURL;
  case Camellia::OP_DIV:
    return S_OP_DIV;
  case Camellia::OP_D1:
    return S_OP_D1;
  case Camellia::OP_D2:
    return S_OP_D2;
  case Camellia::OP_D3:
    return S_OP_D3;
  case Camellia::OP_D4:
    return S_OP_D4;
  case Camellia::OP_D5:
    return S_OP_D5;
  case Camellia::OP_D6:
    return S_OP_D6;
  case Camellia::OP_D7:
    return S_OP_D7;
  case Camellia::OP_D8:
    return S_OP_D8;
  case Camellia::OP_D9:
    return S_OP_D9;
  case Camellia::OP_D10:
    return S_OP_D10;
  case Camellia::OP_X:
    return S_OP_X;
  case Camellia::OP_Y:
    return S_OP_Y;
  case Camellia::OP_Z:
    return S_OP_Z;
  case Camellia::OP_DX:
  case Camellia::OP_DI1:
    return S_OP_DX;
  case Camellia::OP_DY:
  case Camellia::OP_DI2:
    return S_OP_DY;
  case Camellia::OP_DZ:
  case Camellia::OP_DI3:
    return S_OP_DZ;
  case Camellia::OP_DT:
    return S_OP_DT;
  case Camellia::OP_CROSS_NORMAL:
    return S_OP_CROSS_NORMAL;
  case Camellia::OP_DOT_NORMAL:
    return S_OP_DOT_NORMAL;
  case Camellia::OP_TIMES_NORMAL:
    return S_OP_TIMES_NORMAL;
  case Camellia::OP_TIMES_NORMAL_X:
    return S_OP_TIMES_NORMAL_X;
  case Camellia::OP_TIMES_NORMAL_Y:
    return S_OP_TIMES_NORMAL_Y;
  case Camellia::OP_TIMES_NORMAL_Z:
    return S_OP_TIMES_NORMAL_Z;
  case Camellia::OP_VECTORIZE_VALUE:
    return S_OP_VECTORIZE_VALUE;
  default:
    return S_OP_UNKNOWN;
  }
}
