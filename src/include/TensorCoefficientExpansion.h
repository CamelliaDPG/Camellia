#ifndef Camellia_TensorCoefficientExpansion_h
#define Camellia_TensorCoefficientExpansion_h

#include <vector>

#include "TensorFunction.h"

namespace Camellia
{
  using TensorCoefficientFactorTuple = std::vector<TensorFunctionPtr>;
  using TensorCoefficientFactorExpansion = std::vector<TensorCoefficientFactorTuple>;

  /**
   * \brief Expand a tensor coefficient into additive tuples of per-component factors.
   *
   * Returns false when the coefficient is null, solution-dependent, or contains a coupled
   * expression that cannot be represented as a sum of separable per-component products.
   * Constants are attached to component 0 so each returned tuple has at least one non-null
   * factor and can be consumed directly by the existing sum-factorized 1D contraction code.
   */
  bool appendTensorSeparableCoefficientFactorTerms(const TensorFunctionPtr &coefficient,
                                                   int componentCount,
                                                   TensorCoefficientFactorExpansion &coefficientFactorTerms);
}

#endif
