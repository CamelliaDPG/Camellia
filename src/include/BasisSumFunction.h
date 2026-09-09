#ifndef BASIS_SUM_FUNCTION
#define BASIS_SUM_FUNCTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

// Teuchos includes
#include "Teuchos_RCP.hpp"

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
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;

  TFunctionPtr<double> x() override;
  TFunctionPtr<double> y() override;
  TFunctionPtr<double> z() override;
  int numVectorComponents() const override;

  TFunctionPtr<double> dx() override;
  TFunctionPtr<double> dy() override;
  TFunctionPtr<double> dz() override;

  bool boundaryValueOnly() override;

  static TFunctionPtr<double> basisSumFunction(BasisPtr basis, const Camellia::FieldContainer<double> &basisCoefficients, Camellia::EOperator op = OP_VALUE);
};
}

#endif
