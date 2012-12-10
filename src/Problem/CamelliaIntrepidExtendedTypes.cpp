#include "CamelliaIntrepidExtendedTypes.h"

using namespace IntrepidExtendedTypes;

bool IntrepidExtendedTypes::functionSpaceIsVectorized(EFunctionSpaceExtended fs) {
  return (FUNCTION_SPACE_VECTOR_HGRAD == fs)
  ||     (FUNCTION_SPACE_TENSOR_HGRAD == fs)
  ||     (FUNCTION_SPACE_VECTOR_HVOL  == fs)
  ||     (FUNCTION_SPACE_TENSOR_HVOL == fs);
}