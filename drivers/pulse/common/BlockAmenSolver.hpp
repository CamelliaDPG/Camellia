#pragma once

#include "BobaTypes.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

namespace DpgBoba
{
struct AmenOptions
{
  double tolerance = 1e-8;
  double roundingTolerance = 1e-12;
  int maxSweeps = 100;
  int kickrank = 4;
  int maxRank = 64;
  // Coupled site solver: COD, pivoted QR, LU, Eigen GMRES, matrix-free BoBa GMRES, sparse LU (dense LU if dense),
  // sparse LU (COD if dense).
  int localSolver = 0;
  double localTolerance = 1e-6;
  int maxLocalIterations = 300;
  unsigned seed = 7;
  bool verbose = true;
  // Saddle-point enrichment of block 0 (the Riesz block) with the projected image of the trial blocks.
  bool infSupEnrichment = false;
  // Separate relative SVD tolerance for splitting block 0 (the Riesz block); <= 0 uses roundingTolerance.
  double rieszTolerance = 0.0;
  // Sparse-LU site solves (localSolver 5, 6): sites above this fill use a dense solve; > 1 never.
  double denseFillThreshold = 1.1;
  // Row scaling of the coupled site systems (BoBa scale_coupled_site_system).  Off for systems that are already
  // well scaled (the Jacobi-scaled explicit Schur complement): with the matrix-free site solve the row norms cost
  // (operator rank)^4 x rows x cols per block pair and site, every sweep.
  bool scaleSites = true;
  // Matrix-free site GMRES (localSolver 4): right-precondition by the inverse diagonal of the site operator.
  bool siteJacobi = false;
  // Host threads for the matrix-free coupled site apply (BoBa site_apply_threads); 0: BOBA_AMEN_THREADS, else 1.
  int siteThreads = 0;
  // With a caller residual: stop when it fell by less than this fraction over two sweeps (0: off; < 0: the solver
  // path chooses, 0.1 for the explicit Schur complement).
  double stagnationReduction = -1;
};

template <class Tensor>
void round(Tensor &tensor, double tolerance)
{
  tensor.svd_tolerance_relative = tolerance;
  tensor.svd_tolerance_absolute = 0.0;
  tensor.round();
}

/// Apply with an explicit rounding budget, including structurally zero RHS blocks.
template <size_t D>
boba::BlockVector<host_tt_vector<D>> apply(
    const boba::BlockOperator<host_tt_matrix<D>> &matrix,
    const boba::BlockVector<host_tt_vector<D>> &vector, double tolerance)
{
  boba::BlockVector<host_tt_vector<D>> result(matrix.block_rows);
  for (size_t i = 0; i < matrix.block_rows; ++i)
  {
    result(i) = matrix({i, 0}) * vector(0);
    round(result(i), tolerance);
    for (size_t j = 1; j < matrix.block_cols; ++j)
    {
      result(i) += matrix({i, j}) * vector(j);
      round(result(i), tolerance);
    }
  }
  return result;
}

/// A reproducible rank-one guess in every active block, independent of the RHS.
template <size_t D>
boba::BlockVector<host_tt_vector<D>> initialGuess(
    const boba::BlockVector<host_tt_vector<D>> &rhs, unsigned seed)
{
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(0.5, 1.5);
  boba::BlockVector<host_tt_vector<D>> guess(rhs.block_size);
  for (size_t b = 0; b < rhs.block_size; ++b)
  {
    boba::Array<host_vector, D> factors;
    for (size_t d = 0; d < D; ++d)
    {
      const size_t n = rhs(b).sizes(d);
      factors[d] = host_vector({n});
      auto values = factors[d].view();
      for (size_t i = 0; i < n; ++i)
      {
        values(i) = distribution(generator) / std::sqrt(double(n));
      }
    }
    guess(b) = boba::make_tt_from_vectors<D, host_space, double>(factors);
  }
  return guess;
}

/// Solve all mixed blocks together, including the zero trial diagonal blocks.  globalResidual, when set, returns
/// |A x - b| / |b| for AMEn's convergence certificate in place of the exact TT product A x (rank(A) * rank(x)).
template <size_t D>
boba::BlockVector<host_tt_vector<D>> solveAmen(
    const boba::BlockOperator<host_tt_matrix<D>> &matrix,
    const boba::BlockVector<host_tt_vector<D>> &rhs,
    const AmenOptions &options,
    const std::function<double(const boba::BlockVector<host_tt_vector<D>> &)> &globalResidual = {},
    const std::vector<std::vector<std::vector<boba::CoreSliceSparsity<double>>>> *packedOperator = nullptr,
    const boba::BlockVector<host_tt_vector<D>> *guess = nullptr)   // nullptr: random rank-one blocks
{
  if (!(options.tolerance > 0) || !(options.roundingTolerance > 0) ||
      options.maxSweeps < 1 || options.kickrank < 1 || options.maxRank < 1 ||
      options.localSolver < 0 || options.localSolver > 6 ||
      !(options.localTolerance > 0) || options.maxLocalIterations < 1)
  {
    throw std::invalid_argument("Invalid AMEn tolerances, sweep count, or ranks");
  }
  boba::TensorTrainAMENBlock<double> solver;
  solver.convergence_tolerance = options.tolerance;
  solver.max_sweeps = options.maxSweeps;
  solver.minimum_sweeps = std::min(2, options.maxSweeps);
  solver.kickrank = options.kickrank;
  solver.max_allowed_ranks = options.maxRank;
  solver.local_solution_svd_tolerance_relative = options.roundingTolerance;
  solver.residual_enrichment_svd_tolerance_relative = options.roundingTolerance;
  solver.coupled_site_direct_solver_method = options.localSolver < 3 ? options.localSolver : (options.localSolver == 5 ? 3 : (options.localSolver == 6 ? 4 : 0));
  solver.use_eigen_gmres_for_coupled_site_solve = options.localSolver == 3;
  solver.use_boba_gmres_for_coupled_site_solve = options.localSolver == 4;
  solver.solve3d_2ml_options.tolerance_relative = options.localTolerance;
  solver.solve3d_2ml_options.tolerance_absolute = 0.0;
  solver.solve3d_2ml_options.eigen_gmres_max_iterations = options.maxLocalIterations;
  solver.solve3d_2ml_options.eigen_gmres_sparsity = 0.0;
  solver.verbose = options.verbose;
  solver.inf_sup_enrichment_block = options.infSupEnrichment ? 0 : -1;
  solver.coupled_site_dense_fill_threshold = options.denseFillThreshold;
  solver.scale_coupled_site_system = options.scaleSites;
  solver.site_apply_threads = options.siteThreads;
  solver.site_jacobi_preconditioner = options.siteJacobi;
  solver.packed_operator = packedOperator;   // when set, matrix's cores are placeholders (see TensorTrainAMENBlock)
  if (options.rieszTolerance > 0) { solver.local_solution_svd_tolerance_relative_per_block = {options.rieszTolerance}; }
  // a caller residual is cheap: certify by it after every sweep (the largest local residual can stall above the
  // tolerance while |A x - b| / |b| is already below it)
  solver.certify_every_sweep = bool(globalResidual);
  solver.stagnation_reduction = globalResidual ? std::max(0.0, options.stagnationReduction) : 0.0;
  std::mt19937 generator(options.seed);
  return solver.solve(matrix, rhs, guess ? *guess : initialGuess(rhs, options.seed), generator,
                      [&](const boba::BlockVector<host_tt_vector<D>> &x) { return globalResidual ? globalResidual(x) : -1.0; });
}

/// Also report absolute residuals: the trial RHS blocks of a DPG system vanish.
template <size_t D>
double reportResidual(const boba::BlockOperator<host_tt_matrix<D>> &matrix,
                      const boba::BlockVector<host_tt_vector<D>> &rhs,
                      const boba::BlockVector<host_tt_vector<D>> &solution,
                      double roundingTolerance)
{
  double residualSquared = 0.0, rhsSquared = 0.0;
  constexpr double floor = 1e-30;
  for (size_t i = 0; i < rhs.block_size; ++i)
  {
    auto residual = rhs(i);
    const double rhsNorm = boba::norm_frobenius(rhs(i));
    double scale = rhsNorm;
    for (size_t j = 0; j < rhs.block_size; ++j)
    {
      auto term = matrix({i, j}) * solution(j);
      scale += boba::norm_frobenius(term);
      residual -= term;
      round(residual, roundingTolerance);
    }
    const double norm = boba::norm_frobenius(residual);
    std::cout << "residual block=" << i << " absolute=" << norm
              << " scaled=" << norm / (scale + floor)
              << " ranks=" << solution(i).ranks_string() << '\n';
    residualSquared += norm * norm;
    rhsSquared += rhsNorm * rhsNorm;
  }
  const double relative = std::sqrt(residualSquared) / (std::sqrt(rhsSquared) + floor);
  std::cout << "relative_mixed_residual=" << relative << '\n';
  return relative;
}
} // namespace DpgBoba
