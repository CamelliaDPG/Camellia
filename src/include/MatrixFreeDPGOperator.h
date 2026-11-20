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
  class ElementMultiplicativeSchwarz;

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

    /**
     \brief The same element blocks as elementAdditiveSchwarz(), applied multiplicatively (block Gauss-Seidel with
     overlapping blocks), symmetrized: a forward sweep over the cells in a given order, then a backward sweep.  See
     ElementMultiplicativeSchwarz.  The residual updates use each cell's dense S_K (stored, n_K^2 per cell, from the
     shared-eigenbasis coordinates), so storage is about twice elementAdditiveSchwarz()'s.  The initial order is the
     cell order of cells(); set another with ElementMultiplicativeSchwarz::setOrder().  One rank.
     */
    //! With fluxOwner (from upwindCellOrder()), each cell's block is instead its fields and the fluxes it owns (its
    //! outflow faces): non-overlapping blocks, the block analogue of a transport sweep.
    Teuchos::RCP<ElementMultiplicativeSchwarz> elementMultiplicativeSchwarz(const std::vector<int> *fluxOwner = nullptr) const;

    /**
     \brief An upwind ordering of the cells (indices into cells()), for ultraweak transport forms -(beta u, grad v) + <t, v>.

     Each pair of cells sharing a face is oriented by the sign of beta . n on the face, estimated as the sum of the two
     cells' averages of beta_k (BoxCellDPGOperator::testDerivativeCoefficientAverages()), k the face's normal direction;
     the cells are then topologically sorted (Kahn).  Where the face graph has cycles, the remaining cell with the fewest
     unvisited upwind neighbors is taken next, which lags its remaining in-edges; cyclesBroken (if given) counts those
     choices and laggedFaces the faces lagged.  Neighbors are found through shared flux dofs.
     */
    //! fluxOwner (if given): [map lid] the cell (index into cells()) owning each flux dof, the upwind one of the two
    //! cells sharing its face (the only cell on a boundary face); -1 for field dofs.
    //! characteristicFace (if given): [map lid] a face index for the flux dofs on characteristic faces, where
    //! |beta . n| <= characteristicTolerance * max_k |beta_k| (cell averages; for an interior face, of the two cells'
    //! mean), -1 elsewhere.  Those faces get fluxOwner -2 (owned by neither cell) and give no dependency.
    //! weightedCycleBreaking: on a cycle, take next the remaining cell whose unvisited in-edges have the smallest total
    //! weight |beta . n| / h_n (the face's share of the downstream cell's inflow rate), instead of the fewest in-edges;
    //! laggedWeightFraction (if given) is the lagged edges' share of the total weight.
    //! sweepDirections (if given, with characteristicFace): [direction] false leaves that direction's faces out of the
    //! upwind graph, as characteristic faces (e.g. a t-x-only sweep: the velocity faces are face blocks, not swept).
    //! component (if given): [cell] its strongly connected component of the face graph (Tarjan), numbered in an upwind
    //! (topological) order of the components; cells on no cycle are components of size one.
    std::vector<int> upwindCellOrder(int *cyclesBroken = nullptr, int *laggedFaces = nullptr, std::vector<int> *fluxOwner = nullptr,
                                     std::vector<int> *characteristicFace = nullptr, double characteristicTolerance = 0.0,
                                     std::vector<int> *component = nullptr, bool weightedCycleBreaking = false,
                                     double *laggedWeightFraction = nullptr, const std::vector<bool> *sweepDirections = nullptr) const;

    /**
     \brief The transport-sweep preconditioner P = R^{-1} W^{-1} R^{-T} (notes/static-condensation-trace-structure.tex,
     uncondensed form).

     Each cell K owns its field dofs and the flux dofs fluxOwner assigns it (its outflow faces); its other flux dofs are
     its inflow, owned by upwind cells.  With S_K the cell's local DPG matrix split into owned (u) and inflow (i) parts,
     T_K = -S_uu^{-1} S_ui (the local DPG solve from inflow data) and W = diag(S_uu,K), the global operator is exactly
     S = R^T W R + Q with (R x)_K = x_K - T_K x_{i,K} and Q >= 0; P is applied by a backward sweep (R^T), the cell solves,
     and a forward sweep (R).  order must be upwind (upwindCellOrder()); in-edges against it (lagged faces in a cycle)
     are dropped from R in both sweeps, which keeps P symmetric positive definite.  Stores per cell the Cholesky factor
     of S_uu and T_K.  One rank.
     */
    //! Flux dofs with fluxOwner -2 (characteristic faces, upwindCellOrder()) are left out of every cell's split, so the
    //! sweep acts on the rest (S_ss = R^T W R + Q there); with characteristicFace given they are preconditioned by
    //! their face blocks of S (Cholesky), additively (block diagonal over {sweep dofs, characteristic dofs}), or, with
    //! characteristicMultiplicative, by the symmetric product: face blocks, sweep, face blocks (two more applications
    //! of S per application, through op).
    //! residualOperator (if given): the operator whose residuals the multiplicative face composition uses, instead of
    //! this one (when the sweep is built from an approximate operator, e.g. with frozen coefficients).
    //! blockOf (if given): [cell] a block index, blocks numbered in sweep order (e.g. the strongly connected components
    //! from upwindCellOrder(), so that the sweep over blocks needs no lagging).  A block owns its cells' fields and the
    //! fluxes any of its cells owns (so faces between its cells are interior); the sweep then runs over blocks with
    //! dense block solves, and order is ignored.
    Teuchos::RCP<TOperator<double>> transportSweep(const std::vector<int> &order, const std::vector<int> &fluxOwner,
                                                   const std::vector<int> *characteristicFace = nullptr,
                                                   bool characteristicMultiplicative = false,
                                                   const std::vector<int> *blockOf = nullptr,
                                                   const TOperator<double> *residualOperator = nullptr) const;

    /**
     \brief Blocks for transportSweep(): cells merged (union-find) across every face where beta . n changes sign, judged by
     the two cells' ranges of beta_n over their quadrature points (both a value below -tolerance max|beta| and one above
     +tolerance max|beta|).  Blocks are numbered by the first appearance of their cells in order (an upwind order), so
     that transportSweep() lags exactly the inflow from blocks later in that order.  mergedFaces and largestBlock (cells)
     if given.
     */
    std::vector<int> mixedFaceBlocks(const std::vector<int> &order, double tolerance, int *mergedFaces = nullptr,
                                     int *largestBlock = nullptr) const;

    //! the rank's cells, in the order the operator visits them
    const std::vector<GlobalIndexType> &cells() const { return _cells; }

  private:
    //! the element blocks: for each cell (in cells() order), its non-Dirichlet owned dofs (local indices in the map) and
    //! the lower Cholesky factor of A_K = R_K S R_K^T (column-major); and, if cellDofs is given, for each cell its local
    //! trial dofs' (local trial dof, map lid, coefficient) triples, Dirichlet dofs excluded
    struct CellDof { int local; TMap::local_ordinal_type lid; double coefficient; };
    //! each cell's dense S_K (n x n, column-major, local trial dofs) and its (local, lid, coefficient) triples
    void cellMatrices(std::vector<std::vector<CellDof>> &cellDofs, std::vector<std::vector<double>> &matrices) const;
    void elementBlocks(std::vector<std::vector<TMap::local_ordinal_type>> &lids, std::vector<std::vector<double>> &factors,
                       std::vector<std::vector<CellDof>> *cellDofs, const std::vector<int> *fluxOwner = nullptr) const;

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

  /**
   \brief Symmetric multiplicative Schwarz (overlapping block Gauss-Seidel) with element blocks, as a preconditioner:
   Y = beta Y + alpha M^{-1} X, where M^{-1} X is the result of a forward sweep over the blocks in order() and a backward
   sweep, from a zero initial guess, on S z = X.  Each block correction z_K += A_K^{-1} R_K r is followed by the residual
   update r -= S R_K^T delta, computed as sum over the cells K' sharing a dof with the block of P_K'^T S_K' P_K' R_K^T delta
   with the stored dense S_K'.  With exact block solves the symmetrized sweep is symmetric positive definite.  Dirichlet
   dofs (identity rows and columns) are copied.  setAdditive(true) applies the same blocks additively instead (then
   identical to MatrixFreeDPGOperator::elementAdditiveSchwarz()).
   */
  class ElementMultiplicativeSchwarz : public TOperator<double>
  {
  public:
    using MV = TVector<double>;
    using LO = TMap::local_ordinal_type;
    struct CellMatrix
    {
      std::vector<int> local;          // [entry]: local trial dof
      std::vector<LO> lid;             // [entry]: map lid
      std::vector<double> coefficient; // [entry]
      int n = 0;                       // local trial dofs
      std::vector<double> S;           // n x n, column-major: S_K in the local trial dofs
    };
    ElementMultiplicativeSchwarz(TMapConstPtr map, std::vector<std::vector<LO>> blockLids, std::vector<std::vector<double>> blockFactors,
                                 std::vector<CellMatrix> cells, std::vector<LO> dirichletLids);
    Teuchos::RCP<const TMap> getDomainMap() const override { return _map; }
    Teuchos::RCP<const TMap> getRangeMap() const override { return _map; }
    bool hasTransposeApply() const override { return true; }
    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS, double alpha = 1.0, double beta = 0.0) const override;
    //! the block order of the forward sweep (a permutation of the block indices)
    void setOrder(const std::vector<int> &order) { _order = order; }
    const std::vector<int> &order() const { return _order; }
    void setAdditive(bool additive) { _additive = additive; }
    //! only the forward sweep (not symmetric: for experiments)
    void setForwardOnly(bool forwardOnly) { _forwardOnly = forwardOnly; }
  private:
    TMapConstPtr _map;
    std::vector<std::vector<LO>> _blockLids;
    std::vector<std::vector<double>> _blockFactors;
    std::vector<CellMatrix> _cells;
    std::vector<std::vector<int>> _touching; // [block]: the cells sharing a dof with it
    std::vector<LO> _dirichletLids;
    std::vector<int> _order;
    bool _additive = false, _forwardOnly = false;
    void correct(int block, std::vector<double> &r, std::vector<double> &z, std::vector<double> &scratch) const;
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
