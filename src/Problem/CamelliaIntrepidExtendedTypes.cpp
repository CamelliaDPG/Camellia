#include "CamelliaIntrepidExtendedTypes.h"

using namespace IntrepidExtendedTypes;

bool IntrepidExtendedTypes::functionSpaceIsVectorized(EFunctionSpaceExtended fs) {
  return (FUNCTION_SPACE_VECTOR_HGRAD == fs)
  ||     (FUNCTION_SPACE_TENSOR_HGRAD == fs)
  ||     (FUNCTION_SPACE_VECTOR_HVOL  == fs)
  ||     (FUNCTION_SPACE_TENSOR_HVOL == fs)
  ||     (FUNCTION_SPACE_VECTOR_HGRAD_DISC == fs)
  ||     (FUNCTION_SPACE_TENSOR_HGRAD_DISC == fs);
}

bool IntrepidExtendedTypes::functionSpaceIsDiscontinuous(IntrepidExtendedTypes::EFunctionSpaceExtended fs) {
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