#ifndef CAMELLIA_BC_FUNCTION
#define CAMELLIA_BC_FUNCTION

#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"
#include "Projector.h"

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class BCFunction : public Function<Scalar> {
    Intrepid::FieldContainer<bool> _imposeHere;
    int _varID;
    BCPtr _bc;
    bool _isTrace; // if false, it's a flux...
    FunctionPtr<Scalar> _spatiallyFilteredFunction;
  public:
    BCFunction(BCPtr bc, int varID, bool isTrace, FunctionPtr<Scalar> spatiallyFilteredFunction, int rank);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    bool imposeOnCell(int cellIndex);
    int varID();
    bool isTrace();

    FunctionPtr<Scalar> curl();
    FunctionPtr<Scalar> div();
    FunctionPtr<Scalar> dx();
    FunctionPtr<Scalar> dy();
    FunctionPtr<Scalar> dz();

    static Teuchos::RCP<BCFunction<Scalar>> bcFunction(BCPtr bc, int varID, bool isTrace);
  };
}

#endif
