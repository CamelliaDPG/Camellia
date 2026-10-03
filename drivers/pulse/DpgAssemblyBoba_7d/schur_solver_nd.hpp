#ifndef DPG_BOBA_SCHUR_SOLVER_ND_HPP
#define DPG_BOBA_SCHUR_SOLVER_ND_HPP

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "boba_types.hpp"

namespace dpg_boba {

struct SchurSolveOptionsND {
  double relative_tolerance = 1.0e-10;
  double absolute_tolerance = 1.0e-10;
  double linear_tolerance = 1.0e-8;
  size_t amen_max_sweeps = 2;
  size_t amen_minimum_sweeps = 1;
  size_t amen_kickrank = 2;
  bool amen_exact_initial_guess = true;
  std::string amen_coupled_solver = "cod";
  bool amen_validate_matrix_free_operator = false;
  double amen_local_gmres_tolerance = 1.0e-5;
  size_t amen_local_gmres_max_iterations = 30;
  double amen_local_gmres_sparsity = 1.0e-3;
};

template <size_t dimension> using nd_matrix_t = host_tt_matrix<dimension>;

template <size_t dimension> using nd_vector_t = host_tt_vector<dimension>;

template <size_t dimension>
using nd_block_operator_t = ::boba::BlockOperator<nd_matrix_t<dimension>>;

template <size_t dimension>
using nd_block_vector_t = ::boba::BlockVector<nd_vector_t<dimension>>;

template <typename object_t>
inline void set_nd_tolerances(object_t &object, double relative_tolerance,
                              double absolute_tolerance) {
  object.svd_tolerance_relative = relative_tolerance;
  object.svd_tolerance_absolute = absolute_tolerance;
}

template <size_t dimension>
inline void set_nd_tolerances(nd_block_operator_t<dimension> &op,
                              double relative_tolerance,
                              double absolute_tolerance) {
  for (size_t row = 0; row < op.block_rows; ++row) {
    for (size_t col = 0; col < op.block_cols; ++col) {
      set_nd_tolerances(op({row, col}), relative_tolerance, absolute_tolerance);
    }
  }
}

template <size_t dimension>
inline void set_nd_tolerances(nd_block_vector_t<dimension> &vector,
                              double relative_tolerance,
                              double absolute_tolerance) {
  for (size_t block = 0; block < vector.block_size; ++block) {
    set_nd_tolerances(vector(block), relative_tolerance, absolute_tolerance);
  }
}

template <size_t dimension>
inline nd_matrix_t<dimension>
compute_rank_one_corewise_inverse_nd(const nd_matrix_t<dimension> &matrix) {
  boba_always_assert_equal(matrix.core_rows(), matrix.core_cols(),
                           "Corewise inverse requires square TTM cores.");

  nd_matrix_t<dimension> inverse(matrix.core_rows(), matrix.core_cols());
  inverse.svd_tolerance_relative = matrix.svd_tolerance_relative;
  inverse.svd_tolerance_absolute = matrix.svd_tolerance_absolute;

  for (size_t d = 0; d < dimension; ++d) {
    boba_always_assert_equal(matrix.get_ranks_left(d), size_t{1},
                             "Corewise inverse requires a rank-one Gram TTM.");
    boba_always_assert_equal(matrix.get_ranks_right(d), size_t{1},
                             "Corewise inverse requires a rank-one Gram TTM.");

    const auto rows = matrix.core_rows(d);
    const auto cols = matrix.core_cols(d);
    host_matrix core_matrix({rows, cols});
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        core_matrix({row, col}) = matrix.cores[d]({0, row, col, 0});
      }
    }

    const auto inverse_core = ::boba::compute_inverse(core_matrix);
    inverse.cores[d].resize({1, rows, cols, 1});
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        inverse.cores[d]({0, row, col, 0}) = inverse_core({row, col});
      }
    }
  }
  inverse.rename("K_inverse_corewise_nd");
  return inverse;
}

template <size_t dimension>
inline void print_nd_vector(const std::string &label,
                            const nd_vector_t<dimension> &vector,
                            double norm) {
  std::cout << label << " full_size=" << vector.get_full_size()
            << " stored=" << vector.get_number_elements()
            << " compression=" << vector.compression_rate()
            << " ranks=" << vector.ranks_string()
            << " norm=" << norm << "\n";
}

template <size_t dimension>
inline void print_nd_block_vector_right_orthogonal(
    const std::string &label, const nd_block_vector_t<dimension> &vector) {
  // Both current call sites round every TT block immediately before printing.
  // Reuse one first-core norm per right-orthogonal block instead of performing
  // a full TT self-contraction once for the aggregate and again for each line.
  std::vector<double> block_norms(vector.block_size);
  double norm_squared = 0.0;
  for (size_t block = 0; block < vector.block_size; ++block) {
    block_norms[block] =
        ::boba::norm_frobenius_right_orthogonal(vector(block));
    norm_squared += block_norms[block] * block_norms[block];
  }
  std::cout << "\n"
            << label << " blocks=" << vector.block_size
            << " full_size=" << vector.get_full_size()
            << " stored=" << vector.get_number_elements()
            << " compression=" << vector.compression_rate()
            << " norm=" << std::sqrt(norm_squared) << "\n";
  for (size_t block = 0; block < vector.block_size; ++block) {
    print_nd_vector("  block(" + std::to_string(block) + ")", vector(block),
                    block_norms[block]);
  }
}

template <size_t dimension>
inline int solve_compressed_schur_system_nd(
    nd_block_operator_t<dimension> &stiffness,
    nd_block_vector_t<dimension> &manufactured_solution,
    nd_block_vector_t<dimension> &rhs, const SchurSolveOptionsND &options,
    const std::string &solution_output_file = "",
    const nd_matrix_t<dimension> *field_mass = nullptr,
    const nd_vector_t<dimension> *exact_field_load = nullptr,
    double exact_field_norm_squared = 0.0) {
  const size_t total_blocks = stiffness.block_rows;
  boba_always_assert_equal(total_blocks, stiffness.block_cols,
                           "Saddle operator must be square.");
  boba_always_assert_equal(rhs.block_size, total_blocks,
                           "RHS block count mismatch.");
  boba_always_assert_equal(manufactured_solution.block_size, total_blocks,
                           "Solution block count mismatch.");
  boba_always_assert_gt(total_blocks, size_t{1},
                        "Schur system needs at least one trial block.");

  set_nd_tolerances(stiffness, options.relative_tolerance,
                    options.absolute_tolerance);
  set_nd_tolerances(rhs, options.relative_tolerance,
                    options.absolute_tolerance);
  set_nd_tolerances(manufactured_solution, options.relative_tolerance,
                    options.absolute_tolerance);

  std::cout << std::setprecision(10) << "Solving " << dimension
            << "D TT Schur system with block AMEN\n"
            << "trial_blocks=" << total_blocks - 1 << "\n"
            << "svd_relative_tolerance=" << options.relative_tolerance << "\n"
            << "svd_absolute_tolerance=" << options.absolute_tolerance << "\n"
            << "linear_relative_tolerance=" << options.linear_tolerance << "\n"
            << "amen_max_sweeps=" << options.amen_max_sweeps << "\n"
            << "amen_minimum_sweeps=" << options.amen_minimum_sweeps << "\n"
            << "amen_kickrank=" << options.amen_kickrank << "\n"
            << "amen_coupled_solver=" << options.amen_coupled_solver << "\n"
            << "amen_validate_matrix_free_operator=" << options.amen_validate_matrix_free_operator << "\n"
            << "amen_local_gmres_tolerance=" << options.amen_local_gmres_tolerance << "\n"
            << "amen_local_gmres_max_iterations=" << options.amen_local_gmres_max_iterations << "\n"
            << "amen_local_gmres_sparsity=" << options.amen_local_gmres_sparsity << "\n"
            << "amen_exact_initial_guess=" << options.amen_exact_initial_guess
            << "\n";

  const auto gram = stiffness({0, 0});
  auto gram_inverse = compute_rank_one_corewise_inverse_nd(gram);
  set_nd_tolerances(gram_inverse, options.relative_tolerance,
                    options.absolute_tolerance);

  nd_block_operator_t<dimension> constraint(total_blocks - 1, 1);
  for (size_t row = 0; row < constraint.block_rows; ++row) {
    constraint({row, 0}) = stiffness({row + 1, 0});
  }
  set_nd_tolerances(constraint, options.relative_tolerance,
                    options.absolute_tolerance);

  nd_block_operator_t<dimension> constraint_transpose(1, total_blocks - 1);
  for (size_t col = 0; col < constraint_transpose.block_cols; ++col) {
    constraint_transpose({0, col}) = constraint({col, 0}).transpose();
  }
  set_nd_tolerances(constraint_transpose, options.relative_tolerance,
                    options.absolute_tolerance);

  nd_block_operator_t<dimension> gram_inverse_block(1, 1);
  gram_inverse_block({0, 0}) = gram_inverse;
  auto schur = (constraint * gram_inverse_block) * constraint_transpose;
  set_nd_tolerances(schur, options.relative_tolerance,
                    options.absolute_tolerance);
  std::cout << "schur_rows=" << schur.rows()
            << " schur_stored=" << schur.get_number_elements()
            << " schur_compression=" << schur.compression_rate() << "\n";

  auto gram_inverse_rhs = gram_inverse * rhs(0);
  nd_block_vector_t<dimension> schur_rhs(total_blocks - 1);
  for (size_t row = 0; row < schur_rhs.block_size; ++row) {
    schur_rhs(row) = constraint({row, 0}) * gram_inverse_rhs;
    schur_rhs(row) -= rhs(row + 1);
    set_nd_tolerances(schur_rhs(row), options.relative_tolerance,
                      options.absolute_tolerance);
    schur_rhs(row).round();
  }
  print_nd_block_vector_right_orthogonal("Schur RHS", schur_rhs);

  nd_block_vector_t<dimension> initial(total_blocks - 1);
  for (size_t block = 0; block < initial.block_size; ++block) {
    initial(block) = manufactured_solution(block + 1);
    if (!options.amen_exact_initial_guess) {
      initial(block) *= 0.0;
    }
    set_nd_tolerances(initial(block), options.relative_tolerance,
                      options.absolute_tolerance);
  }

  nd_block_vector_t<dimension> lambda(total_blocks - 1);
  try {
    ::boba::TensorTrainAMENBlock<double> amen;
    amen.convergence_tolerance = options.linear_tolerance;
    amen.max_sweeps = options.amen_max_sweeps;
    amen.minimum_sweeps = options.amen_minimum_sweeps;
    amen.kickrank = options.amen_kickrank;
    amen.normalize_coupled_intermediate_quantities = false;
    amen.use_eigen_gmres_for_coupled_site_solve =
        options.amen_coupled_solver == "eigen_gmres";
    amen.use_boba_gmres_for_coupled_site_solve =
        options.amen_coupled_solver == "boba_gmres";
    amen.validate_coupled_matrix_free_operator =
        options.amen_validate_matrix_free_operator;
    amen.coupled_site_direct_solver_method =
        options.amen_coupled_solver == "qr" ? 1 :
        options.amen_coupled_solver == "lu" ? 2 : 0;
    amen.solve3d_2ml_options.tolerance_relative = options.amen_local_gmres_tolerance;
    amen.solve3d_2ml_options.eigen_gmres_max_iterations = options.amen_local_gmres_max_iterations;
    amen.solve3d_2ml_options.eigen_gmres_sparsity = options.amen_local_gmres_sparsity;
    amen.verbose = true;
    lambda = amen.solve(schur, schur_rhs, initial);
  } catch (const std::exception &error) {
    std::cerr << "compressed_schur_solve_failed=1\n"
              << "compressed_schur_solve_error=\"" << error.what() << "\"\n";
    return 2;
  }

  set_nd_tolerances(lambda, options.relative_tolerance,
                    options.absolute_tolerance);
  for (size_t block = 0; block < lambda.block_size; ++block) {
    lambda(block).round();
  }
  print_nd_block_vector_right_orthogonal("Lambda solution after round", lambda);

  auto schur_residual = schur * lambda - schur_rhs;
  schur_residual.orthogonalize();
  const double relative_schur_residual =
      ::boba::norm_frobenius_left_orthogonal(schur_residual) /
      ::boba::norm_frobenius_right_orthogonal(schur_rhs);

  auto constraint_transpose_lambda = constraint_transpose({0, 0}) * lambda(0);
  for (size_t block = 1; block < lambda.block_size; ++block) {
    constraint_transpose_lambda +=
        constraint_transpose({0, block}) * lambda(block);
  }
  auto test_solution = gram_inverse * (rhs(0) - constraint_transpose_lambda);
  set_nd_tolerances(test_solution, options.relative_tolerance,
                    options.absolute_tolerance);
  test_solution.round();

  nd_block_vector_t<dimension> full_solution(total_blocks);
  full_solution(0) = test_solution;
  for (size_t block = 0; block < lambda.block_size; ++block) {
    full_solution(block + 1) = lambda(block);
  }
  set_nd_tolerances(full_solution, options.relative_tolerance,
                    options.absolute_tolerance);

  auto full_residual = stiffness * full_solution - rhs;
  full_residual.orthogonalize();
  const double relative_full_residual =
      ::boba::norm_frobenius_left_orthogonal(full_residual) /
      ::boba::norm_frobenius(rhs);
  std::cout << "relative_schur_residual=" << relative_schur_residual << "\n"
            << "relative_full_residual=" << relative_full_residual << "\n";

  if (field_mass != nullptr && exact_field_load != nullptr &&
      exact_field_norm_squared > 0.0) {
    const auto &field = lambda(lambda.block_size - 1);
    const auto &projected_field =
        manufactured_solution(manufactured_solution.block_size - 1);
    const auto mass_times_field = (*field_mass) * field;
    const auto projected_error = field - projected_field;
    const auto mass_times_projected_error = (*field_mass) * projected_error;
    const auto mass_times_projected_field = (*field_mass) * projected_field;
    const double projected_error_squared =
        ::boba::inner_product(projected_error, mass_times_projected_error);
    const double projected_field_norm_squared =
        ::boba::inner_product(projected_field, mass_times_projected_field);
    const double field_norm_squared =
        ::boba::inner_product(field, mass_times_field);
    const double exact_field_inner_product =
        ::boba::inner_product(field, *exact_field_load);
    const double field_error_squared =
        std::max(0.0, exact_field_norm_squared + field_norm_squared -
                          2.0 * exact_field_inner_product);
    std::cout << "projected_relative_field_l2_error="
              << std::sqrt(projected_error_squared /
                           projected_field_norm_squared)
              << "\n"
              << "continuous_relative_field_l2_error="
              << std::sqrt(field_error_squared / exact_field_norm_squared)
              << "\n";
  }

  if (!solution_output_file.empty()) {
    ::boba::write_to_hdf5_file(full_solution, solution_output_file, "soln");
    std::cout << "full_solution_output=" << solution_output_file << "\n";
  }
  return 0;
}

} // namespace dpg_boba

#endif
