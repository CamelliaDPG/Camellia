#ifndef Camellia_MatrixFreeDPGOperator_h
#define Camellia_MatrixFreeDPGOperator_h

//
//  MatrixFreeDPGOperator.h
//  Camellia
//
//  The global DPG operator S = sum_K P_K^T (B_K^T G_K^{-1} B_K) P_K as a Tpetra operator, applied cell by cell with
//  BoxCellDPGOperator: no global or element matrix is formed.  P_K is the mesh's dof interpretation (including any
//  constraints, e.g. hanging nodes), so S is the matrix that Solution assembles.
//

#include "TypeDefs.h"

#include "BC.h"
#include "BoxCellDPGOperator.h"
#include "RHS.h"

#include "Tpetra_Export.hpp"
#include "Tpetra_Import.hpp"

#include <set>
#include <vector>

namespace Camellia
{
  /**
   \brief y = S x, matrix-free, on the mesh's global dofs (the map Solution::getPartitionMap() gives, without
   condensation).

   With Dirichlet dofs, the operator is S restricted to the free dofs plus the identity on the Dirichlet dofs (rows and
   columns), the form Solution's assembled system takes after imposeBCs(); dirichletRightHandSide() gives the
   corresponding right-hand side.
   */
  class MatrixFreeDPGOperator : public TOperator<double>
  {
  public:
    using MV = TVector<double>;

    //! map: the owned global dofs; dirichletDofs: rank-owned global indices of Dirichlet dofs (may be empty)
    MatrixFreeDPGOperator(MeshPtr mesh, Teuchos::RCP<BoxCellDPGOperator> local, TMapConstPtr map,
                          const std::set<GlobalIndexType> &dirichletDofs = {});

    //! the Dirichlet dofs and values of bc (as Solution::imposeBCs() determines them), rank-owned
    static void dirichletData(MeshPtr mesh, BCPtr bc, std::vector<GlobalIndexType> &indices, std::vector<double> &values);

    Teuchos::RCP<const TMap> getDomainMap() const override { return _map; }
    Teuchos::RCP<const TMap> getRangeMap() const override { return _map; }
    bool hasTransposeApply() const override { return true; } // S is symmetric

    //! Y = beta Y + alpha Op X
    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS, double alpha = 1.0, double beta = 0.0) const override;

    //! Y = S X, without the Dirichlet modification
    void applyUnconstrained(const MV &X, MV &Y) const;

    /**
     \brief The right-hand side of the system with Dirichlet values: b = M (L - S x_D) + x_D, M the restriction to the
     free dofs, L the load (sum_K P_K^T B_K^T G_K^{-1} l_K; zero if rhs is null), x_D the Dirichlet values (zero
     elsewhere).  The solution of Op x = b is then the DPG solution with those boundary values.
     */
    void dirichletRightHandSide(RHSPtr rhs, const std::vector<GlobalIndexType> &dirichletIndices,
                                const std::vector<double> &dirichletValues, MV &b) const;

    //! the load vector L alone
    void load(RHSPtr rhs, MV &L) const;

    /**
     \brief The diagonal of the operator (one column of d): exactly diag(S) = sum_K (P_K e_g)^T S_K (P_K e_g) for each
     global dof g, constraints included, with 1 on the Dirichlet dofs.

     P_K e_g, the cell's coefficients of global basis function g, comes from the dof interpretation; each quadratic form
     is evaluated in the shared eigenbasis of the separable Gram matrices (BoxCellDPGOperator::schurQuadraticForms()),
     so nothing is factored or stored: per cell, one sum-factorized B and half a G^{-1} per global dof touching the cell.
     */
    void diagonal(MV &d) const;

    /**
     \brief Trace-block Jacobi: the dense diagonal blocks of the operator for a partition of the dofs into each face's
     flux dofs and each element's field dofs (identity on the Dirichlet dofs), Cholesky-factored.

     Entries (P_K e_a)^T S_K (P_K e_b) are inner products of the shared-eigenbasis coordinates
     (BoxCellDPGOperator::schurFactorCoordinates()), summed over cells; nothing global is formed.  A flux dof's face
     is that of the first local dof it is met through (on meshes with hanging faces a constrained face's dofs can land in
     neighboring blocks, which is still a valid block Jacobi).  Rank-owned dofs only, with this rank's cells (exact on
     one rank).
     */
    Teuchos::RCP<TOperator<double>> traceBlockJacobi() const;

    /**
     \brief The Galerkin coarse operator P^T Op P for a prolongation P (rows: this operator's dofs; columns: coarse dofs),
     assembled without forming the fine operator.

     For each cell and each coarse dof c touching it, u_c = P_K P e_c (P_K e_g from the dof interpretation, the
     Dirichlet dofs masked out) and the entries u_c^T S_K u_c' are inner products of the shared-eigenbasis coordinates;
     the Dirichlet rows' identity contributes sum_{g Dirichlet} P_gc P_gc'.  Exact: the coarse matrix that
     P^T A P gives for the assembled, Dirichlet-modified A.  Rank-local (one rank).
     */
    TMatrixPtr<double> galerkinCoarseOperator(TMatrixPtr<double> P) const;

    /**
     \brief Additive Schwarz with one block per element: the element's (non-Dirichlet) dofs, fields and the fluxes on
     all its faces, so that the blocks overlap on faces.  y = sum_K R_K^T A_K^{-1} R_K x, A_K = R_K S R_K^T: the global
     operator's entries among the element's dofs, including the neighbors' contributions on shared faces (as
     GMGOperator's cell-block additive Schwarz takes them from the assembled matrix).  The entries are inner products
     of the shared-eigenbasis coordinates; the cost and storage are those of the element matrices.  One rank.
     */
    Teuchos::RCP<TOperator<double>> elementAdditiveSchwarz() const;

  private:
    MeshPtr _mesh;
    Teuchos::RCP<BoxCellDPGOperator> _local;
    TMapConstPtr _map, _overlapMap;
    Teuchos::RCP<Tpetra::Import<>> _importer;
    Teuchos::RCP<Tpetra::Export<>> _exporter;
    std::vector<GlobalIndexType> _cells;
    std::vector<TMap::local_ordinal_type> _dirichletLocal; // local indices (in _map) of the Dirichlet dofs

    //! accumulate sum_K P_K^T y_K, given the local vectors computed by cellAction, into Y (owned map)
    template <typename CellAction>
    void assemble(const MV *X, MV &Y, CellAction cellAction) const;
  };

  //! the outcome of conjugateGradient(): iterations, relative residual |b - A x| / |b|, and the extreme eigenvalues of
  //! M^{-1} A estimated from the Lanczos tridiagonal of the iteration (Ritz values: lambdaMin from above, lambdaMax from
  //! below), whose ratio estimates the condition number
  struct ConjugateGradientResult
  {
    int iterations = 0;
    double relativeResidual = 0, lambdaMin = 0, lambdaMax = 0;
    double conditionNumber() const { return lambdaMax / lambdaMin; }
  };

  //! preconditioned conjugate gradients from x = 0 (M may be null), to |b - A x| <= tolerance |b|
  ConjugateGradientResult conjugateGradient(const TOperator<double> &A, const TOperator<double> *M, const TVector<double> &b,
                                            TVector<double> &x, double tolerance, int maxIterations);

  /**
   \brief Y = beta Y + alpha D^{-1} X for a diagonal D (e.g. MatrixFreeDPGOperator::diagonal()): Jacobi preconditioning.
   */
  class DiagonalScalingOperator : public TOperator<double>
  {
  public:
    using MV = TVector<double>;
    //! diagonal: one column; its entries must be nonzero
    explicit DiagonalScalingOperator(Teuchos::RCP<const MV> diagonal);
    Teuchos::RCP<const TMap> getDomainMap() const override { return _inverseDiagonal->getMap(); }
    Teuchos::RCP<const TMap> getRangeMap() const override { return _inverseDiagonal->getMap(); }
    bool hasTransposeApply() const override { return true; }
    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS, double alpha = 1.0, double beta = 0.0) const override;
  private:
    Teuchos::RCP<MV> _inverseDiagonal;
  };
}

#endif
