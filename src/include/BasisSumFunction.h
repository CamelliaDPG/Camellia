#ifndef BASIS_SUM_FUNCTION
#define BASIS_SUM_FUNCTION

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Intrepid includes
#include "Intrepid_FieldContainer.hpp"

#include "Function.h"
#include "BasisCache.h"

#include "Basis.h"

namespace Camellia
{
class BasisSumFunction : public TFunction<double>
{
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

  TFunctionPtr<double> x();
  TFunctionPtr<double> y();
  TFunctionPtr<double> z();

  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();

  bool boundaryValueOnly();

  static TFunctionPtr<double> basisSumFunction(BasisPtr basis, const Intrepid::FieldContainer<double> &basisCoefficients, Camellia::EOperator op = OP_VALUE);
};
}

#endif
