#ifndef Camellia_TensorCoupledCoefficientEvaluator_h
#define Camellia_TensorCoupledCoefficientEvaluator_h

#include <stdexcept>

#include "TensorFunction.h"

namespace Camellia
{
  /**
   * \brief Point-data evaluation path for genuinely coupled scalar TensorFunction coefficients.
   *
   * This adapter is intentionally separate from TensorCoefficientExpansion: it samples a coupled
   * coefficient at quadrature points and returns ordinary point data, rather than manufacturing
   * fake one-dimensional tensor factors for nonseparable functions such as relativistic gamma or
   * u_i/gamma.
   */
  class TensorCoupledCoefficientEvaluator
  {
  public:
    static bool requiresNonseparablePointData(TensorFunctionPtr coefficient)
    {
      if (coefficient == Teuchos::null) return false;
      return coefficient->isPartiallyCoupled() || coefficient->isFullyCoupled();
    }

    static void values(TensorFunctionPtr coefficient,
                       TensorFunction::PointViewHost pointsIn,
                       TensorFunction::ValueViewHost valuesOut)
    {
      if (coefficient == Teuchos::null) throw std::invalid_argument("TensorCoupledCoefficientEvaluator cannot evaluate a null coefficient.");
      if (!requiresNonseparablePointData(coefficient))
      {
        throw std::invalid_argument("TensorCoupledCoefficientEvaluator is only for coupled/nonseparable coefficients.");
      }
      if (coefficient->hasSolutionDependencies())
      {
        throw std::invalid_argument("TensorCoupledCoefficientEvaluator does not evaluate solution-dependent coupled coefficients without an explicit solution evaluation context.");
      }
      coefficient->values(pointsIn, valuesOut);
    }
  };
}

#endif
