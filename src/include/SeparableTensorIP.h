#ifndef Camellia_SeparableTensorIP_h
#define Camellia_SeparableTensorIP_h

//
//  SeparableTensorIP.h
//  Camellia
//

#include "IP.h"

#include <cstdlib>
#include <map>
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

    //! the number of cells whose Gram matrices were assembled from one-dimensional factors, and by quadrature
    long long tensorCellCount() const { return _tensorCellCount; }
    long long fallbackCellCount() const { return _fallbackCellCount; }

    //! false: always use quadrature (for comparison)
    void setUseTensorAssembly(bool value) { _useTensorAssembly = value; }

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
    };
    const TensorDecomposition &decomposition(BasisPtr basis, int spaceDim);
    bool tryTensorAssembly(Camellia::FieldContainer<double> &innerProduct, DofOrderingPtr dofOrdering, BasisCachePtr basisCache);

    std::map<const void*, TensorDecomposition> _decompositions;
    std::mutex _decompositionMutex;
    bool _useTensorAssembly;
    long long _tensorCellCount = 0, _fallbackCellCount = 0;
  };
}

#endif
