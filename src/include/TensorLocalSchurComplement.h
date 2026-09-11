#ifndef Camellia_TensorLocalSchurComplement_h
#define Camellia_TensorLocalSchurComplement_h

#include <vector>

#include "TensorBasisCache.h"
#include "TensorBilinearForm.h"
#include "TensorProductLocalOperator.h"

namespace Camellia
{
  /**
   * \brief Host-side Phase 0 helper for explicit local B/G/K block construction.
   *
   * TensorProductLocalOperator keeps a separable local B action in factorized form.
   * These utilities materialize the corresponding dense local blocks only at the
   * correctness/assembly boundary: local B as a row-major test-by-trial matrix,
   * diagonal or mass-like G entries as a tensor product of one-dimensional diagonal
   * factors, and the trial-space Schur complement K = B^T G^{-1} B for the current
   * local-coefficient convention.
   */
  class TensorLocalSchurComplementBuilder
  {
  public:
    static TensorProductLocalOperator buildVolumeBOperator(const TensorBilinearTerm &term,
                                                           const TensorFactorSpaceShape &trialShape,
                                                           const TensorFactorSpaceShape &testShape,
                                                           const TensorBasisCacheND &trialBasisCache,
                                                           const TensorBasisCacheND &testBasisCache,
                                                           const std::vector<TensorFunctionPtr> &coefficientFactors);

    static std::vector<double> denseBMatrix(const TensorProductLocalOperator &bOperator);

    static std::vector<std::vector<double>> l2TestMassDiagonalFactors(const TensorBasisCacheND &testBasisCache);

    static std::vector<double> l2TestMassDiagonal(const TensorFactorSpaceShape &testShape,
                                                  const TensorBasisCacheND &testBasisCache);

    static std::vector<double> tensorProductDiagonal(const TensorFactorSpaceShape &shape,
                                                     const std::vector<std::vector<double>> &componentDiagonals);

    static std::vector<double> formDiagonalTestNormSchurComplement(const TensorProductLocalOperator &bOperator,
                                                                   const std::vector<double> &gDiagonal);
  };
}

#endif
