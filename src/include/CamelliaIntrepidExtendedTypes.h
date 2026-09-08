#ifndef CAMELLIA_INTREPID_EXTENDED_TYPES
#define CAMELLIA_INTREPID_EXTENDED_TYPES

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#include <set>
#include <string>
#include <vector>


#include <Intrepid2_Types.hpp>

namespace Camellia
{
  enum EOperator   // first 13 simply copied from Intrepid::EOperator
  {
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
    OP_T,         // 17
    OP_I1,        // 18, synonymous with OP_X
    OP_I2,        // 19, synonymous with OP_Y
    OP_I3,        // 20, synonymous with OP_Z
    OP_I4,        // 21
    OP_I5,        // 22
    OP_I6,        // 23
    OP_I7,        // 24
    OP_DX,        // 25
    OP_DY,        // 26
    OP_DZ,        // 27
    OP_DT,        // 28
    OP_DI1,       // 29, synonymous with OP_DX
    OP_DI2,       // 30, synonymous with OP_DY
    OP_DI3,       // 31, synonymous with OP_DZ
    OP_DI4,       // 32
    OP_DI5,       // 33
    OP_DI6,       // 34
    OP_DI7,       // 35
    OP_CROSS_NORMAL,    // 36
    OP_DOT_NORMAL,      // 37
    OP_TIMES_NORMAL,    // 38
    OP_TIMES_NORMAL_X,  // 39
    OP_TIMES_NORMAL_Y,  // 40
    OP_TIMES_NORMAL_Z,  // 41
    OP_TIMES_NORMAL_T,  // 42
    OP_VECTORIZE_VALUE, // 43
    OP_LAPLACIAN,       // 44
    OP_DXDX,            // 45
    OP_DXDY,            // 46
    OP_DXDZ,            // 47
    OP_DYDY,            // 48
    OP_DYDZ,            // 49
    OP_DZDZ             // 50
  };

  enum EFunctionSpace   // the first four copied from Intrepid::EFunctionSpace
  {
    FUNCTION_SPACE_HGRAD = 0,
    FUNCTION_SPACE_HCURL,
    FUNCTION_SPACE_HDIV,
    FUNCTION_SPACE_HVOL,
    FUNCTION_SPACE_VECTOR_HGRAD,
    FUNCTION_SPACE_TENSOR_HGRAD,
    FUNCTION_SPACE_VECTOR_HVOL,
    FUNCTION_SPACE_TENSOR_HVOL,
    FUNCTION_SPACE_HGRAD_DISC,
    FUNCTION_SPACE_HCURL_DISC,
    FUNCTION_SPACE_HDIV_DISC,
    FUNCTION_SPACE_HVOL_DISC,
    FUNCTION_SPACE_VECTOR_HGRAD_DISC,
    FUNCTION_SPACE_TENSOR_HGRAD_DISC,
    FUNCTION_SPACE_VECTOR_HVOL_DISC,
    FUNCTION_SPACE_TENSOR_HVOL_DISC,
    FUNCTION_SPACE_REAL_SCALAR,
    FUNCTION_SPACE_HDIV_FREE,
    FUNCTION_SPACE_UNKNOWN
  };

  //! returns the corresponding H^1 order for a basis of the specified degree and function space.  (HVOL returns basisDegree +1; all others return basisDegree.)
  int getH1Order(int basisDegree, EFunctionSpace fs);

  bool functionSpaceIsVectorized(EFunctionSpace fs);
  bool functionSpaceIsBroken(EFunctionSpace fs);
  bool functionSpaceIsDiscontinuous(EFunctionSpace fs);
  EFunctionSpace continuousSpaceForDiscontinuous(EFunctionSpace fs_disc, bool upgradeHVOL = true);
  EFunctionSpace discontinuousSpaceForContinuous(EFunctionSpace fs_continuous);
  
  EFunctionSpace fromIntrepid2FunctionSpace(const ::Intrepid2::EFunctionSpace &fs);
  ::Intrepid2::EFunctionSpace toIntrepid2FunctionSpace(const EFunctionSpace &fs);

  //! not all legal combinations yet supported -- for now, just first-order single component derivative (e.g. OP_DX + OP_DY = OP_DXDY)
  EOperator combineOperators(EOperator op_first, EOperator op_second);
  
  const std::set<EOperator> & normalOperators();

  const std::string & operatorName(EOperator op);
  int operatorRank(EOperator op, EFunctionSpace fs);

  void setOrdinalTagData(std::vector<std::vector<std::vector<int> > >   &tagToOrdinal,
                         std::vector<std::vector<int> >                 &ordinalToTag,
                         const int                                      *tags,
                         const int                                      basisCard,
                         const int                                      tagSize,
                         const int                                      posScDim,
                         const int                                      posScOrd,
                         const int                                      posDfOrd);
}

#endif
