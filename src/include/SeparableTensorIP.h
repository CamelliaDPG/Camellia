#ifndef Camellia_SeparableTensorIP_h
#define Camellia_SeparableTensorIP_h

//
//  SeparableTensorIP.h
//  Camellia
//

#include "IP.h"
#include "SeparableGramInverse.h"

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace Camellia
{
  /**
   \brief An inner product whose Gram matrix, on axis-aligned box cells with tensor-product test bases, is assembled
          from one-dimensional factors instead of by quadrature on the cell.

   Applies when every term is c_t(K) d_{k_t} v or c_t(K) v for a single test variable v (k_t a coordinate direction),
   with c_t constant on each cell: a direction-separable norm, such as the Vlasov graph norm's direction-separable
   variant.  On a box cell with widths h_k and a test basis that is a tensor product of one-dimensional bases -- or a
   serendipity subset of one -- each entry is then
     G(a,b) = sum_t c_t(K)^2 prod_k F_{t,k}(a_k, b_k),
   F_{t,k} the one-dimensional mass matrix (scaled by h_k/2) or, in direction k_t, the derivative matrix (scaled by
   2/h_k), a_k and b_k the one-dimensional ordinals of the basis functions.  That is exact (up to roundoff) and costs
   O(N^2) per cell for N test functions, where quadrature costs O(N^2 Q) for Q cubature points.

   Anything else -- other terms, boundary or zero-mean terms, homogeneous BCs, cells that are not axis-aligned boxes,
   bases that are not tensor products, coefficients not constant on a cell -- is computed by TIP<double>.

   On a full tensor-product test basis (not serendipity) with a positive value term, the Gram matrix is a Kronecker sum,
   and applyInverseInnerProductMatrix() applies its inverse exactly by fast diagonalization (SeparableGramInverse),
   without forming or factoring it: about 8 (k+1)^5 multiplications per vector in four dimensions, against (k+1)^12/3 to
   factor the Gram matrix and (k+1)^8 per vector to apply the factor.  BF's local solves use it when it applies.
   */
  class SeparableTensorIP : public TIP<double>
  {
  public:
    //! (The environment variable CAMELLIA_SEPARABLE_IP_QUADRATURE, if set, makes every Gram matrix use quadrature, for
    //! comparison.)
    SeparableTensorIP() : _useTensorAssembly(getenv("CAMELLIA_SEPARABLE_IP_QUADRATURE") == NULL) {}

    void computeInnerProductMatrix(Camellia::FieldContainer<double> &innerProduct,
                                   Teuchos::RCP<DofOrdering> dofOrdering,
                                   Teuchos::RCP<BasisCache> basisCache) override;
    using TIP<double>::computeInnerProductMatrix;

    //! values(c, r, :) <- G_c^{-1} values(c, r, :) by fast diagonalization, where that is exact: the conditions of the
    //! one-dimensional assembly, a full tensor-product test basis, and a positive value term.  Otherwise returns false.
    bool applyInverseInnerProductMatrix(Camellia::FieldContainer<double> &values,
                                        Teuchos::RCP<DofOrdering> dofOrdering,
                                        Teuchos::RCP<BasisCache> basisCache) override;

    //! the number of cells whose Gram matrices were assembled from one-dimensional factors, and by quadrature
    long long tensorCellCount() const { return _tensorCellCount; }
    long long fallbackCellCount() const { return _fallbackCellCount; }
    //! the number of cells whose Gram matrices were inverted by fast diagonalization
    long long inverseCellCount() const { return _inverseCellCount; }

    //! false: always use quadrature (for comparison)
    void setUseTensorAssembly(bool value) { _useTensorAssembly = value; }

    /**
     \brief Weaken the L^2 term on the constant mode of each cell: alpha ||v||^2 becomes, on each cell K,
            alpha ||v - mean_K v||^2 + ratio alpha |K| (mean_K v)^2   (0 < ratio <= 1; 1: unchanged).

     A^* 1 = 0 for the graph norm, so a cell's constant has test norm sqrt(ratio alpha |K|): DPG then weighs the residual
     against it -- for a conservation law, the cell's conservation error -- about 1/ratio times as heavily, while every
     other test function keeps its weight.  The Gram matrix changes by the rank-one term -c m m^T per cell, m_i = int_K
     phi_i for the basis of var, c = (1 - ratio) / int_K (1/alpha) (which keeps it positive definite when alpha varies
     on the cell; for constant alpha, c = (1 - ratio) alpha / |K|).  alpha must be the weight of the IP's L^2 term on var.
     Applied after either assembly path (1D factors or quadrature).
     */
    void setConstantModeWeight(VarPtr var, FunctionPtr alpha, double ratio);

    //! false: applyInverseInnerProductMatrix() always returns false, so that callers form and factor the Gram matrix (for
    //! comparison).  (The environment variable CAMELLIA_SEPARABLE_IP_NO_FAST_INVERSE, if set, makes false the default.)
    void setUseFastInverse(bool value) { _useFastInverse = value; }

    static Teuchos::RCP<SeparableTensorIP> ip() { return Teuchos::rcp(new SeparableTensorIP); }
  private:
    //! a test basis decomposed into one-dimensional components
    struct TensorDecomposition
    {
      bool valid = false;
      BasisPtr basis; // kept alive, so that its address (the key) is not reused
      std::vector<std::vector<int>> componentOrdinals;   // [basisOrdinal][direction]
      std::vector<std::vector<double>> mass, derivative; // [direction]: row-major n_k x n_k, on the reference [-1,1]
      std::vector<int> componentCardinality;
      bool fullTensor = false; // every basis function of the full tensor-product basis is present
      std::vector<int> tensorIndex; // [basisOrdinal]: sum_k componentOrdinals[k] * stride_k, first direction fastest
      std::shared_ptr<SeparableGramInverse> inverse; // when fullTensor
    };
    const TensorDecomposition &decomposition(BasisPtr basis, int spaceDim);

    //! the per-cell data of a Gram matrix sum_t c_t(K)^2 (d_{k_t} v, d_{k_t} v), when the one-dimensional path applies
    struct SeparableCellData
    {
      int varID = -1;
      const TensorDecomposition *decomposition = nullptr;
      std::vector<int> directions;                       // [term]: k_t, or -1 for the value
      std::vector<std::vector<double>> halfWidths;       // [cell][direction]
      std::vector<std::vector<double>> coefficientSquared; // [term][cell]
      int numCells = 0;
    };
    bool separableCellData(DofOrderingPtr dofOrdering, BasisCachePtr basisCache, SeparableCellData &data);
    bool tryTensorAssembly(Camellia::FieldContainer<double> &innerProduct, DofOrderingPtr dofOrdering, BasisCachePtr basisCache);

    std::map<const void*, TensorDecomposition> _decompositions;
    std::mutex _decompositionMutex;
    bool _useTensorAssembly;
    bool _useFastInverse = (getenv("CAMELLIA_SEPARABLE_IP_NO_FAST_INVERSE") == NULL);
    VarPtr _constantModeVar;
    FunctionPtr _constantModeAlpha;
    double _constantModeRatio = 1.0;
    void applyConstantModeWeight(Camellia::FieldContainer<double> &innerProduct, DofOrderingPtr dofOrdering, BasisCachePtr basisCache);
    long long _tensorCellCount = 0, _fallbackCellCount = 0, _inverseCellCount = 0;
  };
}

#endif
