#ifndef DPG_BOBA_SCHUR_SOLVER_HPP
#define DPG_BOBA_SCHUR_SOLVER_HPP

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "boba_types.hpp"

namespace dpg_boba
{

struct vlasov_vars
{
  static constexpr size_t test    = 0;
  static constexpr size_t trace_x = 1;
  static constexpr size_t trace_t = 2;
  static constexpr size_t trace_v = 3;
  static constexpr size_t field   = 4;
  static constexpr size_t size    = 5;
};

using matrix_t = host_tt_matrix<3>;
using vector_t = host_tt_vector<3>;
using block_operator_t = ::boba::BlockOperator<matrix_t>;
using block_vector_t = ::boba::BlockVector<vector_t>;

struct SchurSolveOptions
{
  double relative_tolerance = 1.0e-12;
  double absolute_tolerance = 1.0e-12;
  double linear_tolerance = 1.0e-10;
  size_t amen_max_sweeps = 20;
  size_t amen_minimum_sweeps = 7;
  size_t amen_kickrank = 4;
  bool amen_exact_initial_guess = false;
  size_t gmres_outer_iterations = 20;
  size_t gmres_inner_iterations = 4;
  std::string method = "block_amen";
  std::string gram_inverse = "corewise";
};

inline void set_svd_tolerances(matrix_t &matrix, const double relative_tolerance, const double absolute_tolerance)
{
  matrix.svd_tolerance_relative = relative_tolerance;
  matrix.svd_tolerance_absolute = absolute_tolerance;
}

inline void set_svd_tolerances(vector_t &vector, const double relative_tolerance, const double absolute_tolerance)
{
  vector.svd_tolerance_relative = relative_tolerance;
  vector.svd_tolerance_absolute = absolute_tolerance;
}

inline void set_svd_tolerances(block_operator_t &op, const double relative_tolerance, const double absolute_tolerance)
{
  for(size_t row = 0; row < op.block_rows; row++)
  {
    for(size_t col = 0; col < op.block_cols; col++)
    {
      set_svd_tolerances(op({row, col}), relative_tolerance, absolute_tolerance);
    }
  }
}

inline void set_svd_tolerances(block_vector_t &vector, const double relative_tolerance, const double absolute_tolerance)
{
  for(size_t block = 0; block < vector.block_size; block++)
  {
    set_svd_tolerances(vector(block), relative_tolerance, absolute_tolerance);
  }
}

inline void print_ttm_summary(const std::string &label, const matrix_t &matrix)
{
  std::cout << label
            << " rows=" << matrix.rows()
            << " cols=" << matrix.cols()
            << " stored=" << matrix.get_number_elements()
            << " compression=" << matrix.compression_rate()
            << " svd_rel=" << matrix.svd_tolerance_relative
            << " svd_abs=" << matrix.svd_tolerance_absolute
            << " ranks=" << matrix.ranks_string()
            << "\n" << std::flush;
}

inline void print_tt_summary(const std::string &label, const vector_t &vector)
{
  std::cout << label
            << " full_size=" << vector.get_full_size()
            << " stored=" << vector.get_number_elements()
            << " compression=" << vector.compression_rate()
            << " svd_rel=" << vector.svd_tolerance_relative
            << " svd_abs=" << vector.svd_tolerance_absolute
            << " ranks=" << vector.ranks_string()
            << " frobenius_norm=" << ::boba::norm_frobenius(vector)
            << "\n" << std::flush;
}

inline void print_block_vector_summary(const std::string &label, const block_vector_t &vector)
{
  std::cout << "\n" << label
            << " blocks=" << vector.block_size
            << " full_size=" << vector.get_full_size()
            << " stored=" << vector.get_number_elements()
            << " compression=" << vector.compression_rate()
            << " frobenius_norm=" << ::boba::norm_frobenius(vector)
            << "\n";
  for(size_t block = 0; block < vector.block_size; block++)
  {
    print_tt_summary("  block(" + std::to_string(block) + ")", vector(block));
  }
}

inline void print_block_operator_summary(const std::string &label, const block_operator_t &op)
{
  std::cout << "\n" << label
            << " block_shape=" << op.block_rows << "x" << op.block_cols
            << " rows=" << op.rows()
            << " cols=" << op.cols()
            << " stored=" << op.get_number_elements()
            << " compression=" << op.compression_rate()
            << "\n";
  for(size_t row = 0; row < op.block_rows; row++)
  {
    for(size_t col = 0; col < op.block_cols; col++)
    {
      print_ttm_summary(
        "  block(" + std::to_string(row) + "," + std::to_string(col) + ")",
        op({row, col}));
    }
  }
}

inline matrix_t compute_rank_one_corewise_inverse(const matrix_t &matrix)
{
  boba_always_assert_equal(matrix.core_rows(), matrix.core_cols(), "Corewise inverse requires square TTM cores.");

  matrix_t inverse(matrix.core_rows(), matrix.core_cols());
  inverse.svd_tolerance_relative = matrix.svd_tolerance_relative;
  inverse.svd_tolerance_absolute = matrix.svd_tolerance_absolute;

  for(size_t d = 0; d < matrix_t::get_number_cores(); d++)
  {
    boba_always_assert_equal(matrix.get_ranks_left(d), size_t{1}, "Corewise inverse requires rank-one TTM cores.");
    boba_always_assert_equal(matrix.get_ranks_right(d), size_t{1}, "Corewise inverse requires rank-one TTM cores.");

    const auto rows = matrix.core_rows(d);
    const auto cols = matrix.core_cols(d);
    ::boba::Matrix<host_space, double> core_matrix({rows, cols});
    for(size_t row = 0; row < rows; row++)
    {
      for(size_t col = 0; col < cols; col++)
      {
        core_matrix({row, col}) = matrix.cores[d]({0, row, col, 0});
      }
    }

    const auto inverse_core_matrix = ::boba::compute_inverse(core_matrix);
    inverse.cores[d].resize({1, rows, cols, 1});
    for(size_t row = 0; row < rows; row++)
    {
      for(size_t col = 0; col < cols; col++)
      {
        inverse.cores[d]({0, row, col, 0}) = inverse_core_matrix({row, col});
      }
    }
  }

  inverse.rename("K_inverse_corewise");
  return inverse;
}

inline block_operator_t make_constraint_operator(const block_operator_t &stiffness)
{
  block_operator_t constraint(vlasov_vars::size - 1, 1);
  for(size_t row = 0; row < constraint.block_rows; row++)
  {
    constraint({row, 0}) = stiffness({row + 1, vlasov_vars::test});
  }
  return constraint;
}

inline block_operator_t transpose_block_operator(const block_operator_t &op)
{
  block_operator_t transpose(op.block_cols, op.block_rows);
  for(size_t row = 0; row < op.block_rows; row++)
  {
    for(size_t col = 0; col < op.block_cols; col++)
    {
      transpose({col, row}) = op({row, col}).transpose();
    }
  }
  return transpose;
}

inline int solve_compressed_schur_system(
  block_operator_t &stiffness,
  block_vector_t &exact_solution,
  block_vector_t &rhs,
  const SchurSolveOptions &options,
  const std::string &solution_output_file = "")
{
  if(options.gram_inverse != "corewise")
  {
    boba_error("Only --gram_inverse=corewise is implemented. This requires a rank-one separable Gram TTM.");
  }

  set_svd_tolerances(stiffness, options.relative_tolerance, options.absolute_tolerance);
  set_svd_tolerances(exact_solution, options.relative_tolerance, options.absolute_tolerance);
  set_svd_tolerances(rhs, options.relative_tolerance, options.absolute_tolerance);

  std::cout << std::setprecision(10);
  std::cout << "Solving L2 block system in memory\n" << std::flush;
  std::cout << "svd_relative_tolerance=" << options.relative_tolerance << "\n";
  std::cout << "svd_absolute_tolerance=" << options.absolute_tolerance << "\n";
  std::cout << "linear_relative_tolerance=" << options.linear_tolerance << "\n";
  std::cout << "amen_max_sweeps=" << options.amen_max_sweeps << "\n";
  std::cout << "amen_minimum_sweeps=" << options.amen_minimum_sweeps << "\n";
  std::cout << "amen_kickrank=" << options.amen_kickrank << "\n";
  std::cout << "amen_exact_initial_guess=" << options.amen_exact_initial_guess << "\n";
  std::cout << "gmres_outer_iterations=" << options.gmres_outer_iterations << "\n";
  std::cout << "gmres_inner_iterations=" << options.gmres_inner_iterations << "\n";
  std::cout << "gram_inverse=" << options.gram_inverse << "\n";
  print_ttm_summary("Loaded Gram TTM", stiffness({vlasov_vars::test, vlasov_vars::test}));
  print_tt_summary("Loaded rhs test block", rhs(vlasov_vars::test));

  const auto gram_ttm = stiffness({vlasov_vars::test, vlasov_vars::test});
  std::cout << "Computing rank-one corewise Gram inverse TTM\n" << std::flush;
  auto gram_inverse_ttm = compute_rank_one_corewise_inverse(gram_ttm);
  set_svd_tolerances(gram_inverse_ttm, options.relative_tolerance, options.absolute_tolerance);
  print_ttm_summary("Gram inverse TTM", gram_inverse_ttm);

  auto constraint = make_constraint_operator(stiffness);
  auto constraint_transpose = transpose_block_operator(constraint);
  set_svd_tolerances(constraint, options.relative_tolerance, options.absolute_tolerance);
  set_svd_tolerances(constraint_transpose, options.relative_tolerance, options.absolute_tolerance);

  block_operator_t gram_inverse_block(1, 1);
  gram_inverse_block({0, 0}) = gram_inverse_ttm;
  set_svd_tolerances(gram_inverse_block, options.relative_tolerance, options.absolute_tolerance);

  std::cout << "Forming Schur TTM\n" << std::flush;
  auto schur_ttm = (constraint * gram_inverse_block) * constraint_transpose;
  set_svd_tolerances(schur_ttm, options.relative_tolerance, options.absolute_tolerance);
  std::cout << "Schur TTM stored_entries=" << schur_ttm.get_number_elements()
            << " compression_rate=" << schur_ttm.compression_rate() << "\n" << std::flush;
  print_block_operator_summary("Schur TTM diagnostics", schur_ttm);

  std::cout << "Forming Schur RHS in TT block form\n" << std::flush;
  auto gram_inverse_rhs0 = gram_inverse_ttm * rhs(vlasov_vars::test);
  set_svd_tolerances(gram_inverse_rhs0, options.relative_tolerance, options.absolute_tolerance);

  block_vector_t schur_rhs(vlasov_vars::size - 1);
  for(size_t row = 0; row < schur_rhs.block_size; row++)
  {
    schur_rhs(row) = constraint({row, 0}) * gram_inverse_rhs0;
    set_svd_tolerances(schur_rhs(row), options.relative_tolerance, options.absolute_tolerance);
  }
  for(size_t row = 0; row < schur_rhs.block_size; row++)
  {
    schur_rhs(row) -= rhs(row + 1);
    set_svd_tolerances(schur_rhs(row), options.relative_tolerance, options.absolute_tolerance);
    schur_rhs(row).round();
  }
  print_block_vector_summary("Schur RHS diagnostics", schur_rhs);

  block_vector_t exact_lambda(vlasov_vars::size - 1);
  block_vector_t lambda_initial(vlasov_vars::size - 1);
  for(size_t block = 0; block < exact_lambda.block_size; block++)
  {
    exact_lambda(block) = exact_solution(block + 1);
    lambda_initial(block) = exact_lambda(block);
    if(!options.amen_exact_initial_guess)
    {
      lambda_initial(block) *= 0.0;
    }
    set_svd_tolerances(exact_lambda(block), options.relative_tolerance, options.absolute_tolerance);
    set_svd_tolerances(lambda_initial(block), options.relative_tolerance, options.absolute_tolerance);
  }
  print_block_vector_summary("Initial lambda diagnostics", lambda_initial);

  auto exact_residual = schur_rhs - schur_ttm * exact_lambda;
  exact_residual.round();
  print_block_vector_summary("Exact residual diagnostics", exact_residual);

  const auto exact_schur_residual = schur_ttm * exact_lambda - schur_rhs;
  const auto exact_schur_residual_norm = ::boba::norm_frobenius(exact_schur_residual);
  const auto schur_rhs_norm = ::boba::norm_frobenius(schur_rhs);
  std::cout << "exact_relative_schur_residual=" << exact_schur_residual_norm / schur_rhs_norm << "\n";
  std::cout << "exact_schur_residual_norm=" << exact_schur_residual_norm << "\n";
  std::cout << "schur_rhs_norm=" << schur_rhs_norm << "\n";

  block_vector_t lambda_solution(vlasov_vars::size - 1);
  std::cout << "Solving compressed Schur system with " << options.method << "\n" << std::flush;
  if(options.method == "exact")
  {
    lambda_solution = exact_lambda;
  }
  else if(options.method == "block_amen")
  {
    try
    {
      ::boba::TensorTrainAMENBlock<double> amen_solver;
      amen_solver.convergence_tolerance = options.linear_tolerance;
      amen_solver.max_sweeps = options.amen_max_sweeps;
      amen_solver.minimum_sweeps = options.amen_minimum_sweeps;
      amen_solver.kickrank = options.amen_kickrank;
      amen_solver.normalize_coupled_intermediate_quantities = false;
      amen_solver.verbose = true;
      lambda_solution = amen_solver.solve(schur_ttm, schur_rhs, lambda_initial);
    }
    catch(const std::exception &error)
    {
      std::cout << "compressed_schur_solve_failed=1\n";
      std::cout << "compressed_schur_solve_error=\"" << error.what() << "\"\n";
      return 2;
    }
  }
  else if(options.method == "gmres")
  {
    try
    {
      ::boba::Krylov<block_operator_t, block_vector_t> gmres_solver(
        options.gmres_outer_iterations, options.gmres_inner_iterations, options.linear_tolerance);
      gmres_solver.linear_operator = schur_ttm;
      gmres_solver.method = ::boba::KrylovMethods::gmres;
      gmres_solver.verbose = true;
      gmres_solver.solve(schur_rhs, lambda_initial, lambda_solution);
      std::cout << "gmres_used_outer_iterations=" << gmres_solver.used_outer_iterations << "\n";
      std::cout << "gmres_used_inner_iterations=" << gmres_solver.used_inner_iterations << "\n";
    }
    catch(const std::exception &error)
    {
      std::cout << "compressed_schur_solve_failed=1\n";
      std::cout << "compressed_schur_solve_error=\"" << error.what() << "\"\n";
      return 2;
    }
  }
  else
  {
    boba_error("Unknown Schur solve method.");
  }

  std::cout << "Checking compressed Schur residual and reconstructing full solution\n" << std::flush;
  set_svd_tolerances(lambda_solution, options.relative_tolerance, options.absolute_tolerance);
  print_block_vector_summary("Lambda solution diagnostics", lambda_solution);

  for (size_t block = 0; block < lambda_solution.block_size; block++)
  {
    lambda_solution(block).round();
  }
  print_block_vector_summary("Lambda solution diagnostics after round", lambda_solution);

  const auto schur_residual = schur_ttm * lambda_solution - schur_rhs;
  const auto lambda_error = lambda_solution - exact_lambda;
  const auto schur_residual_norm = ::boba::norm_frobenius(schur_residual);
  print_block_vector_summary("Schur residual diagnostics", schur_residual);
  print_block_vector_summary("Schur solution error diagnostics", lambda_error);
  print_block_vector_summary("exact Schur solution diagnostics", exact_lambda);
  std::cout << "relative_schur_residual=" << schur_residual_norm / schur_rhs_norm << "\n";
  std::cout << "schur_residual_norm=" << schur_residual_norm << "\n";
  std::cout << "lambda_error_norm=" << ::boba::norm_frobenius(lambda_error) << "\n";

  auto constraint_transpose_lambda = constraint_transpose({0, 0}) * lambda_solution(0);
  for(size_t block = 1; block < lambda_solution.block_size; block++)
  {
    constraint_transpose_lambda += constraint_transpose({0, block}) * lambda_solution(block);
  }
  auto test_solution = gram_inverse_ttm * (rhs(vlasov_vars::test) - constraint_transpose_lambda);
  set_svd_tolerances(test_solution, options.relative_tolerance, options.absolute_tolerance);
  test_solution.round();
  print_tt_summary("Reconstructed test block diagnostics", test_solution);
  std::cout << "Reconstructed test block norm=" << ::boba::norm_frobenius(test_solution) << "\n";

  block_vector_t full_solution(vlasov_vars::size);
  full_solution(vlasov_vars::test) = test_solution;
  for(size_t block = 0; block < lambda_solution.block_size; block++)
  {
    full_solution(block + 1) = lambda_solution(block);
  }
  set_svd_tolerances(full_solution, options.relative_tolerance, options.absolute_tolerance);
  print_block_vector_summary("Full solution diagnostics", full_solution);
  if(!solution_output_file.empty())
  {
    ::boba::write_to_hdf5_file(full_solution, solution_output_file, "soln");
    std::cout << "full_solution_output=" << solution_output_file << "\n";
  }

  const auto full_residual = stiffness * full_solution - rhs;
  std::cout << "relative_full_residual="
            << ::boba::norm_frobenius(full_residual) / ::boba::norm_frobenius(rhs)
            << "\n";
  for (size_t block = 0; block < lambda_solution.block_size; block++) {
    std::cout << "relative_solution_error(" << block << ")="
              << ::boba::norm_frobenius(exact_solution(block + 1) - lambda_solution(block)) / ::boba::norm_frobenius(exact_solution(block + 1))
              << "\n";
  }

  return 0;
}

} // namespace dpg_boba

#endif
