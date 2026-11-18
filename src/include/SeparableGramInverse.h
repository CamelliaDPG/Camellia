#ifndef Camellia_SeparableGramInverse_h
#define Camellia_SeparableGramInverse_h

//
//  SeparableGramInverse.h
//  Camellia
//
//  Exact inverse, by fast diagonalization, of a Gram matrix that is a Kronecker sum on a full tensor-product space.
//  (Lynch, Rice and Thomas, Numer. Math. 6 (1964); for the Vlasov separable graph norm, see the separable-graph-norm
//  notes, Theorem 6.1.)  Free of Trilinos dependencies (it needs only BLAS dgemm), so that matrix-free operators can use it
//  as well as the assembled local solves.
//

#include <vector>

namespace Camellia
{
  /**
   \brief Applies G^{-1} (and G) for Gram matrices of the form

       G = J ( a M_1 (x) ... (x) M_d  +  sum_k b_k M_1 (x) ... (x) K_k (x) ... (x) M_d ),

   on the full tensor product of one-dimensional spaces of dimension n_1, ..., n_d, where M_k and K_k are the reference
   mass and stiffness (derivative) matrices of direction k, J > 0, a > 0 and b_k >= 0.  The per-direction generalized
   eigenproblems K_k V_k = M_k V_k Lambda_k (V_k^T M_k V_k = I) are solved once, at construction; then

       G^{-1} = V diag( 1 / (J (a + sum_k b_k lambda_{k,j_k})) ) V^T,     V = V_1 (x) ... (x) V_d,

   exactly (to roundoff), for any J, a, b: the cell- and norm-dependence enters only through the diagonal.  Applying it
   to one vector costs 2 sum_k n_k N multiplications plus N divisions, N = prod_k n_k (8 (k+1)^5 + (k+1)^4 for four
   directions of n = k+1), against N^2 for a stored Cholesky factor and N^3/3 to compute one.

   Vectors are indexed with the first direction fastest: index = sum_k j_k s_k, s_0 = 1, s_{k+1} = s_k n_k (the ordering
   of Intrepid2::Basis_TensorBasis).  Several vectors are stored one after another, each of length N.
   */
  class SeparableGramInverse
  {
  public:
    //! The cell-dependent coefficients of G: jacobian J, value weight a, derivative weights b_k (one per direction).
    struct Scales
    {
      double jacobian = 1.0;
      double valueWeight = 1.0;
      std::vector<double> derivativeWeights;
    };

    /**
     \param mass       for each direction, the row-major n_k x n_k reference mass matrix (symmetric positive definite)
     \param stiffness  for each direction, the row-major n_k x n_k reference stiffness matrix (symmetric positive semidefinite)
     */
    SeparableGramInverse(const std::vector<std::vector<double>> &mass, const std::vector<std::vector<double>> &stiffness);

    int directionCount() const { return int(_n.size()); }
    int cardinality(int direction) const { return _n[direction]; }
    //! N, the dimension of the tensor-product space
    int size() const { return _size; }

    //! the generalized eigenvalues of (K_k, M_k), in increasing order (the smallest is zero, up to roundoff, when the constants are in the space)
    const std::vector<double> &eigenvalues(int direction) const { return _lambda[direction]; }
    //! row-major n_k x n_k: column j holds the coefficients of the j-th eigenfunction (V_k^T M_k V_k = I)
    const std::vector<double> &eigenvectors(int direction) const { return _V[direction]; }

    //! x <- G^{-1} x, for each of the vectorCount vectors stored consecutively in x
    void applyInverse(const Scales &scales, double *x, int vectorCount = 1) const;

    //! x <- V^T x, V = V_1 (x) ... (x) V_d: coefficients in the eigenbasis, where G is diagonal (eigenvalueDiagonal());
    //! so x^T G^{-1} x = sum_i (V^T x)_i^2 / D_i
    void applyEigenbasisTranspose(double *x, int vectorCount = 1) const;

    //! x <- G x (from the one-dimensional factors; for verification and for matrix-free residuals)
    void apply(const Scales &scales, double *x, int vectorCount = 1) const;

    //! the diagonal of V^T G V, i.e. J (a + sum_k b_k lambda_{k,j_k}), in the tensor ordering
    std::vector<double> eigenvalueDiagonal(const Scales &scales) const;

    //! max_k max(|V_k^T M_k V_k - I|, |V_k^T K_k V_k - Lambda_k| / max(1, max lambda)) (a check on the factors)
    double factorResidual() const;

  private:
    std::vector<int> _n;       // per direction
    std::vector<int> _stride;  // per direction
    int _size = 1;
    std::vector<std::vector<double>> _M, _K, _V, _lambda;

    void checkScales(const Scales &scales) const;
    //! x <- (I (x) ... (x) A (x) ... (x) I) x in direction k for each of vectorCount consecutive vectors, A row-major
    //! n_k x n_k (transpose: use A^T), by BLAS dgemm; work has size N vectorCount
    void modeProduct(int direction, const std::vector<double> &A, bool transpose, double *x, double *work, int vectorCount) const;
  };
}

#endif
