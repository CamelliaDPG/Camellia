// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  LinearTermSumFactorized.h
//  Camellia
//

#ifndef Camellia_LinearTermSumFactorized_h
#define Camellia_LinearTermSumFactorized_h

#include "Camellia_FieldContainer.hpp"
#include "TypeDefs.h"

namespace Camellia
{
  /**
   * \brief Sum-factorized integration of a pair of LinearTerms, using Intrepid2::PAMatrix::assemble().
   *
   * Computes values(c,i,j) += \int (u-term i-th basis contribution) * (v-term j-th basis contribution), the same quantity
   * as TLinearTerm::integrate(values, u, uOrdering, v, vOrdering, basisCache), over the cell (volume cache) or one of its
   * sides (side cache), when that integral is supported here:
   *  - the cell is a hypercube whose bases are Intrepid2 tensor bases with one-dimensional tensor components matching the
   *    cubature's (as for meshes built from 1D component meshes, e.g. ExtrudedMeshTopology);
   *  - for each variable, all of its summands use OP_VALUE, or (volume only) all use single first derivatives (OP_DX,
   *    OP_DY, OP_DZ, OP_DI1, ..., OP_DI7), with scalar Function weights;
   *  - volume variables are HGRAD or HVOL (HGRAD for derivatives; HGRAD on sides); side (trace/flux) variables are
   *    HGRAD or HVOL bases on the side.
   * Coefficients may vary arbitrarily in space; they enter as pointwise weights.  Flux parities and side HVOL scalings
   * are applied as in the standard path.
   *
   * \return true if the integral was performed (and summed into values); false, without modifying values, if any part
   *         of it is unsupported (the caller should then use the standard path).
   *
   * Called from TLinearTerm::integrate(), so bilinear forms, inner products, and other LinearTerm integrals use it
   * automatically.  Disable by setting the environment variable CAMELLIA_DISABLE_SUM_FACTORIZED_LINEAR_TERMS, or by
   * calling setSumFactorizedLinearTermIntegrationEnabled(false).
   *
   * Requires a Trilinos whose Intrepid2 provides PAMatrix (CMake option CAMELLIA_ENABLE_PAMATRIX, defining
   * CAMELLIA_HAVE_PAMATRIX).  Without it, this always returns false, and integration is by quadrature.
   */
  bool integrateLinearTermsSumFactorized(Camellia::FieldContainer<double> &values,
                                               LinearTermPtr u, DofOrderingPtr uOrdering,
                                               LinearTermPtr v, DofOrderingPtr vOrdering,
                                               BasisCachePtr basisCache);

  //! initially false if CAMELLIA_DISABLE_SUM_FACTORIZED_LINEAR_TERMS is set in the environment, true otherwise; always
  //! false (and not settable) when Camellia is built without Intrepid2::PAMatrix
  bool sumFactorizedLinearTermIntegrationEnabled();
  void setSumFactorizedLinearTermIntegrationEnabled(bool enabled);
}

#endif
