#ifndef Camellia_MatrixFreeDPGMultigrid_h
#define Camellia_MatrixFreeDPGMultigrid_h

//
//  MatrixFreeDPGMultigrid.h
//  Camellia
//
//  A two-level preconditioner for the matrix-free DPG operator (MatrixFreeDPGOperator): Chebyshev smoothing with a
//  (trace-)block or point Jacobi diagonal on the fine level, and an exact solve with the Galerkin coarse operator
//  P^T Op P, assembled without forming the fine operator, on a coarse level given by GMGOperator's prolongation.
//  Run through MatrixFreeGMGPreconditioner's V-cycle.
//

#include "TypeDefs.h"

#include "MatrixFreeDPGOperator.h"

#include <functional>

namespace Camellia
{
  /**
   \brief Residual-based Chebyshev smoothing for A x = b with a diagonal (or block-diagonal) preconditioner D^{-1}:
   x <- x + p_k(D^{-1} A) D^{-1} (b - A x), the Chebyshev polynomial on [lambdaMax / eigenRatio, lambdaMax] (Ifpack2's
   convention).  lambdaMax estimates lambda_max(D^{-1} A), e.g. from estimateLambdaMax().
   */
  class ChebyshevSmoother
  {
  public:
    using MV = TVector<double>;
    ChebyshevSmoother(Teuchos::RCP<const TOperator<double>> A, Teuchos::RCP<const TOperator<double>> Dinv, int degree,
                      double lambdaMax, double eigenRatio);
    //! x <- x + (degree Chebyshev steps on A x = b)
    void smooth(const MV &b, MV &x) const;
    //! power iteration on D^{-1} A from a random vector, times boost (Ifpack2 uses 1.1)
    static double estimateLambdaMax(const TOperator<double> &A, const TOperator<double> &Dinv, int iterations, double boost);
    //! the largest Ritz value of D^{-1} A from iterations of D^{-1}-preconditioned CG on a random right-hand side (Lanczos),
    //! times boost: far more accurate than power iteration for the same number of applications (a Ritz value
    //! underestimates lambda_max, and Chebyshev amplifies whatever lies above its interval)
    static double estimateLambdaMaxLanczos(const TOperator<double> &A, const TOperator<double> &Dinv, int iterations, double boost);
  private:
    Teuchos::RCP<const TOperator<double>> _A, _Dinv;
    int _degree;
    double _lambdaMax, _lambdaMin;
  };

  /**
   \brief A two-level V-cycle for the matrix-free DPG operator.

   coarseMesh: a coarser discretization of the same problem (e.g. the lowest polynomial order on the same cells, from
   GMGSolver::meshesForMultigrid()).  The prolongation comes from GMGOperator (so it is the conventional GMG's), the
   coarse operator from MatrixFreeDPGOperator::galerkinCoarseOperator(), factored by Amesos2 (KLU2).
   */
  class MatrixFreeDPGTwoLevel
  {
  public:
    struct Options
    {
      int chebyshevDegree = 2;
      double eigenRatio = 20.0;
      double lambdaMaxBoost = 1.1;
      int powerIterations = 10;     // for the power-iteration estimate
      bool lanczosLambdaMax = true;  // estimate lambda_max by Lanczos (CG) instead of power iteration
      int lanczosIterations = 20;
      bool traceBlockDiagonal = true; // false: the point (shared-eigenbasis) diagonal
      bool elementSchwarz = false;    // true: element-block additive Schwarz (overrides traceBlockDiagonal)
    };

    //! fineSolution: the fine Solution (mesh, dof interpreter, partition map, BCs, inner product); fineOperator on its map
    MatrixFreeDPGTwoLevel(Teuchos::RCP<MatrixFreeDPGOperator> fineOperator, SolutionPtr fineSolution, MeshPtr coarseMesh,
                          const Options &options);
    MatrixFreeDPGTwoLevel(Teuchos::RCP<MatrixFreeDPGOperator> fineOperator, SolutionPtr fineSolution, MeshPtr coarseMesh);

    //! the V-cycle, as a Tpetra operator (a preconditioner for fineOperator)
    Teuchos::RCP<TOperator<double>> preconditioner() const { return _preconditioner; }

    double lambdaMax() const { return _lambdaMax; }
    GlobalIndexType coarseDofCount() const { return _coarseDofCount; }
    //! setup times: prolongation, coarse operator, coarse factorization, smoother diagonal, lambda_max
    std::vector<std::pair<std::string,double>> setupTimes() const { return _setupTimes; }

  private:
    Teuchos::RCP<TOperator<double>> _preconditioner;
    double _lambdaMax = 0;
    GlobalIndexType _coarseDofCount = 0;
    std::vector<std::pair<std::string,double>> _setupTimes;
  };
}

#endif
