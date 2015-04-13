#ifndef BASIS_SUM_FUNCTION
#define BASIS_SUM_FUNCTION

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Intrepid includes
#include "Intrepid_FieldContainer.hpp"

#include "Function.h"
#include "BasisCache.h"

#include "Basis.h"

namespace Camellia {
  class BasisSumFunction : public Function<double> {
   private:
    BasisPtr _basis;
    Intrepid::FieldContainer<double> _coefficients;
    Camellia::EOperator _op;
    bool _boundaryValueOnly;
    BasisCachePtr _overridingBasisCache;
   public:
    BasisSumFunction(BasisPtr basis, const Intrepid::FieldContainer<double> &basisCoefficients,
                        BasisCachePtr overridingBasisCache = Teuchos::null,
                        Camellia::EOperator op = OP_VALUE, bool boundaryValueOnly = false);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);

    FunctionPtr x();
    FunctionPtr y();
    FunctionPtr z();

    FunctionPtr dx();
    FunctionPtr dy();
    FunctionPtr dz();

    bool boundaryValueOnly();

    static FunctionPtr basisSumFunction(BasisPtr basis, const Intrepid::FieldContainer<double> &basisCoefficients);
  };
}

#endif
