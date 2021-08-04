#ifndef CAMELLIA_BC_FUNCTION
#define CAMELLIA_BC_FUNCTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER


#include "TypeDefs.h"

#include "Camellia_FieldContainer.hpp"

#include "Function.h"

namespace Camellia
{
template <typename Scalar>
class BCFunction : public TFunction<Scalar>
{
  Camellia::FieldContainer<bool> _imposeHere;
  int _varID;
  BCPtr _bc;
  TFunctionPtr<Scalar> _spatiallyFilteredFunction;
public:
  BCFunction(BCPtr bc, int varID, TFunctionPtr<Scalar> spatiallyFilteredFunction, int rank);
  
  using TFunction<double>::values;
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  void values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache);
  
  //! cellOrdinal is the ordinal in the last basisCache seen by a call to values
  bool imposeOnCell(int cellOrdinal);
  int varID();

  TFunctionPtr<Scalar> curl();
  TFunctionPtr<Scalar> div();
  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();

  std::vector< TFunctionPtr<Scalar> > memberFunctions();
  
  static Teuchos::RCP<BCFunction<Scalar>> bcFunction(BCPtr bc, int varID);
};
}

#endif
