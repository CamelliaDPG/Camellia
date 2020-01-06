#ifndef BASIS_SUM_FUNCTION
#define BASIS_SUM_FUNCTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Intrepid includes
#include "Camellia_FieldContainer.hpp"

#include "Function.h"
#include "BasisCache.h"

#include "Basis.h"

namespace Camellia
{
class BasisSumFunction : public TFunction<double>
{
private:
  BasisPtr _basis;
  Camellia::FieldContainer<double> _coefficients;
  Camellia::EOperator _op;
  bool _boundaryValueOnly;
  BasisCachePtr _overridingBasisCache;
public:
  BasisSumFunction(BasisPtr basis, const Camellia::FieldContainer<double> &basisCoefficients,
                   BasisCachePtr overridingBasisCache = Teuchos::null,
                   Camellia::EOperator op = OP_VALUE, bool boundaryValueOnly = false);
  void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache);

  TFunctionPtr<double> x();
  TFunctionPtr<double> y();
  TFunctionPtr<double> z();

  TFunctionPtr<double> dx();
  TFunctionPtr<double> dy();
  TFunctionPtr<double> dz();

  bool boundaryValueOnly();

  static TFunctionPtr<double> basisSumFunction(BasisPtr basis, const Camellia::FieldContainer<double> &basisCoefficients, Camellia::EOperator op = OP_VALUE);
};
}

#endif
