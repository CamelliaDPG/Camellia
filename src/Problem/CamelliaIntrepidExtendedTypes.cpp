#include "CamelliaIntrepidExtendedTypes.h"

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

bool Camellia::functionSpaceIsVectorized(EFunctionSpace fs) {
  return (FUNCTION_SPACE_VECTOR_HGRAD == fs)
  ||     (FUNCTION_SPACE_TENSOR_HGRAD == fs)
  ||     (FUNCTION_SPACE_VECTOR_HVOL  == fs)
  ||     (FUNCTION_SPACE_TENSOR_HVOL == fs)
  ||     (FUNCTION_SPACE_VECTOR_HGRAD_DISC == fs)
  ||     (FUNCTION_SPACE_TENSOR_HGRAD_DISC == fs);
}

bool Camellia::functionSpaceIsDiscontinuous(Camellia::EFunctionSpace fs) {
  switch (fs) {
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

EFunctionSpace Camellia::discontinuousSpaceForContinuous(Camellia::EFunctionSpace fs_continuous) {
  switch (fs_continuous) {
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

EFunctionSpace Camellia::continuousSpaceForDiscontinuous(Camellia::EFunctionSpace fs_disc) {
  switch (fs_disc) {
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

const std::set<EOperator> & Camellia::normalOperators() {
  static std::set<EOperator> _normalOperators;
  if (_normalOperators.size() == 0) {
    _normalOperators.insert(OP_CROSS_NORMAL);
    _normalOperators.insert(OP_DOT_NORMAL);
    _normalOperators.insert(OP_TIMES_NORMAL);
    _normalOperators.insert(OP_TIMES_NORMAL_X);
    _normalOperators.insert(OP_TIMES_NORMAL_Y);
    _normalOperators.insert(OP_TIMES_NORMAL_Z);
  }
  return _normalOperators;
}