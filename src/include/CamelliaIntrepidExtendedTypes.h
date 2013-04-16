#ifndef CAMELLIA_INTREPID_EXTENDED_TYPES
#define CAMELLIA_INTREPID_EXTENDED_TYPES

namespace IntrepidExtendedTypes {
  enum EOperatorExtended { // first 13 simply copied from EOperator
    OP_VALUE = 0,
    OP_GRAD,      // 1
    OP_CURL,      // 2
    OP_DIV,       // 3
    OP_D1,        // 4
    OP_D2,        // 5
    OP_D3,        // 6
    OP_D4,        // 7
    OP_D5,        // 8
    OP_D6,        // 9
    OP_D7,        // 10
    OP_D8,        // 11
    OP_D9,        // 12
    OP_D10,       // 13
    OP_X,         // 14 (pick up where EOperator left off...)
    OP_Y,         // 15
    OP_Z,         // 16
    OP_DX,        // 17
    OP_DY,        // 18
    OP_DZ,        // 19
    OP_CROSS_NORMAL,    // 20
    OP_DOT_NORMAL,      // 21
    OP_TIMES_NORMAL,    // 22
    OP_TIMES_NORMAL_X,  // 23
    OP_TIMES_NORMAL_Y,  // 24
    OP_TIMES_NORMAL_Z,  // 25
    OP_VECTORIZE_VALUE  // 26
  };
  
  enum EFunctionSpaceExtended { // all but the last three copied from EFunctionSpace
    FUNCTION_SPACE_HGRAD = 0,
    FUNCTION_SPACE_HCURL,
    FUNCTION_SPACE_HDIV,
    FUNCTION_SPACE_HVOL,
    FUNCTION_SPACE_VECTOR_HGRAD,
    FUNCTION_SPACE_TENSOR_HGRAD,
    FUNCTION_SPACE_VECTOR_HVOL,
    FUNCTION_SPACE_TENSOR_HVOL,
    FUNCTION_SPACE_ONE,
    FUNCTION_SPACE_HDIV_FREE,
    FUNCTION_SPACE_UNKNOWN
  };
  
  bool functionSpaceIsVectorized(EFunctionSpaceExtended fs);
}

#endif