#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Basis.h"
#include "BasisFamily.h"
#include "Camellia.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "TypeDefs.h"

#include "boba_types.hpp"
#include "global_assembly.hpp"
#include "reference_integrals_1d.hpp"
#include "schur_solver_nd.hpp"

namespace {

constexpr size_t dimension = 7;
constexpr size_t spatial_dimensions = 3;
constexpr size_t trace_blocks = dimension;
constexpr size_t field_block = trace_blocks + 1;
constexpr size_t total_blocks = field_block + 1;

using tt_matrix = host_tt_matrix<dimension>;
using tt_vector = host_tt_vector<dimension>;
using block_operator = boba::BlockOperator<tt_matrix>;
using block_vector = boba::BlockVector<tt_vector>;
using matrix_factors = boba::Array<host_matrix, dimension>;
using vector_factors = boba::Array<host_vector, dimension>;

enum class CoordinateKind { position, time, velocity };

struct Coordinate {
  double lower;
  double upper;
  size_t elements;
  std::string name;
  CoordinateKind kind;
  size_t axis;
};

size_t coordinate_index(const std::array<Coordinate, dimension> &coordinates,
                        CoordinateKind kind, size_t axis = 0) {
  for (size_t d = 0; d < dimension; ++d) {
    if (coordinates[d].kind == kind && coordinates[d].axis == axis) {
      return d;
    }
  }
  boba_error("Requested coordinate is absent from the 7D ordering.");
}

host_vector
rhs_discretization_1d(Camellia::BasisPtr basis, double lower, double upper,
                      size_t elements,
                      const std::function<double(double)> &function,
                      int cubature_degree_override = -1) {
  const int basis_size = basis->getCardinality();
  const double spacing = (upper - lower) / elements;
  const int cubature_degree = cubature_degree_override >= 0
                                  ? cubature_degree_override
                                  : 2 * basis->getDegree();
  auto cubature = Camellia::CubatureFactory().create(
      Camellia::CellTopology::line(), cubature_degree);
  auto points = cubature->allocateCubaturePoints();
  auto weights = cubature->allocateCubatureWeights();
  cubature->getCubature(points, weights);
  auto basis_values =
      basis->allocateBasisValues(points, Intrepid2::OPERATOR_VALUE);
  basis->getValues(basis_values, points, Intrepid2::OPERATOR_VALUE);

  host_vector result({elements * static_cast<size_t>(basis_size)});
  auto result_view = result.view();
  for (size_t element = 0; element < elements; ++element) {
    const double shift = lower + (element + 0.5) * spacing;
    const double scale = 0.5 * spacing;
    for (int basis_index = 0; basis_index < basis_size; ++basis_index) {
      double value = 0.0;
      for (int point = 0; point < points.extent_int(0); ++point) {
        value += scale * basis_values(basis_index, point) *
                 function(shift + scale * points(point, 0)) * weights(point);
      }
      result_view(element * basis_size + basis_index) = value;
    }
  }
  return result;
}

host_vector
nodal_discretization_1d(Camellia::BasisPtr basis, double lower, double upper,
                        size_t elements,
                        const std::function<double(double)> &function) {
  const int basis_size = basis->getCardinality();
  const double spacing = (upper - lower) / elements;
  Intrepid::FieldContainer<double, 0> nodes(basis_size, 1);
  basis->getDofCoords(nodes);

  host_vector result({elements * static_cast<size_t>(basis_size)});
  auto result_view = result.view();
  for (size_t element = 0; element < elements; ++element) {
    const double shift = lower + (element + 0.5) * spacing;
    const double scale = 0.5 * spacing;
    for (int basis_index = 0; basis_index < basis_size; ++basis_index) {
      result_view(element * basis_size + basis_index) =
          function(shift + scale * nodes(basis_index, 0));
    }
  }
  return result;
}

double relative_field_interpolation_l2_error(
    Camellia::BasisPtr field_basis,
    const std::array<Coordinate, dimension> &coordinates) {
  const double pi = 4.0 * std::atan(1.0);
  const int basis_size = field_basis->getCardinality();
  Intrepid::FieldContainer<double, 0> nodes(basis_size, 1);
  field_basis->getDofCoords(nodes);

  // The exact factors are trigonometric, so use a deliberately high-order
  // rule rather than the polynomial-exact assembly rule.
  auto cubature = Camellia::CubatureFactory().create(
      Camellia::CellTopology::line(), 32);
  auto points = cubature->allocateCubaturePoints();
  auto weights = cubature->allocateCubatureWeights();
  cubature->getCubature(points, weights);
  auto basis_values =
      field_basis->allocateBasisValues(points, Intrepid2::OPERATOR_VALUE);
  field_basis->getValues(basis_values, points, Intrepid2::OPERATOR_VALUE);

  long double exact_norm_squared = 1.0L;
  long double interpolant_norm_squared = 1.0L;
  long double exact_interpolant_inner_product = 1.0L;
  for (size_t d = 0; d < dimension; ++d) {
    const auto &coordinate = coordinates[d];
    const bool is_velocity =
        coordinate.kind == CoordinateKind::velocity;
    const auto exact_factor = [pi, is_velocity](double value) {
      return is_velocity ? std::cos(pi * value) : std::sin(pi * value);
    };
    const double spacing =
        (coordinate.upper - coordinate.lower) / coordinate.elements;
    long double exact_1d = 0.0L;
    long double interpolant_1d = 0.0L;
    long double cross_1d = 0.0L;
    for (size_t element = 0; element < coordinate.elements; ++element) {
      const double shift = coordinate.lower + (element + 0.5) * spacing;
      const double scale = 0.5 * spacing;
      std::vector<double> coefficients(basis_size);
      for (int basis_index = 0; basis_index < basis_size; ++basis_index) {
        coefficients[basis_index] =
            exact_factor(shift + scale * nodes(basis_index, 0));
      }
      for (int point = 0; point < points.extent_int(0); ++point) {
        double interpolant = 0.0;
        for (int basis_index = 0; basis_index < basis_size; ++basis_index) {
          interpolant += coefficients[basis_index] *
                         basis_values(basis_index, point);
        }
        const double exact =
            exact_factor(shift + scale * points(point, 0));
        const long double weight = scale * weights(point);
        exact_1d += weight * exact * exact;
        interpolant_1d += weight * interpolant * interpolant;
        cross_1d += weight * exact * interpolant;
      }
    }
    exact_norm_squared *= exact_1d;
    interpolant_norm_squared *= interpolant_1d;
    exact_interpolant_inner_product *= cross_1d;
  }

  const long double error_squared =
      std::max(0.0L, exact_norm_squared + interpolant_norm_squared -
                         2.0L * exact_interpolant_inner_product);
  return std::sqrt(static_cast<double>(error_squared / exact_norm_squared));
}

struct FieldErrorDiagnostics {
  tt_matrix mass;
  tt_vector exact_load;
  double exact_norm_squared;
};

FieldErrorDiagnostics assemble_field_error_diagnostics(
    Camellia::BasisPtr field_basis,
    const std::array<Coordinate, dimension> &coordinates) {
  const double pi = 4.0 * std::atan(1.0);
  const auto sine = [pi](double value) { return std::sin(pi * value); };
  const auto cosine = [pi](double value) { return std::cos(pi * value); };
  reference_integrals_1d reference;
  const auto reference_mass =
      reference.discretization(field_basis, false, field_basis, false);

  matrix_factors mass_factors;
  vector_factors load_factors;
  double exact_norm_squared = 1.0;
  for (size_t d = 0; d < dimension; ++d) {
    const auto &coordinate = coordinates[d];
    const bool is_velocity =
        coordinate.kind == CoordinateKind::velocity;
    const std::function<double(double)> exact_factor =
        is_velocity ? std::function<double(double)>(cosine)
                    : std::function<double(double)>(sine);
    mass_factors[d] = global_field_assembly_1d(
        reference_mass, coordinate.elements, coordinate.lower,
        coordinate.upper);
    load_factors[d] = rhs_discretization_1d(
        field_basis, coordinate.lower, coordinate.upper, coordinate.elements,
        exact_factor, 32);
    // Every factor is sine on [0,1] or cosine on [-1/2,1/2].
    exact_norm_squared *= 0.5;
  }
  return {boba::make_ttm_from_matrices<dimension, host_space, double>(
              mass_factors),
          boba::make_tt_from_vectors<dimension, host_space, double>(
              load_factors),
          exact_norm_squared};
}

host_vector
interior_face_discretization_1d(double lower, double upper, size_t elements,
                                const std::function<double(double)> &function) {
  host_vector result({elements - 1});
  auto result_view = result.view();
  const double spacing = (upper - lower) / elements;
  for (size_t face = 1; face < elements; ++face) {
    result_view(face - 1) = function(lower + face * spacing);
  }
  return result;
}

matrix_factors copy_factors(const matrix_factors &source) {
  matrix_factors result;
  for (size_t d = 0; d < dimension; ++d) {
    result[d] = source[d];
  }
  return result;
}

vector_factors copy_factors(const vector_factors &source) {
  vector_factors result;
  for (size_t d = 0; d < dimension; ++d) {
    result[d] = source[d];
  }
  return result;
}

block_operator
assemble_operator(Camellia::BasisPtr test_basis, Camellia::BasisPtr trace_basis,
                  Camellia::BasisPtr field_basis,
                  const std::array<Coordinate, dimension> &coordinates) {
  reference_integrals_1d reference;
  const auto ref_w_f =
      reference.discretization(test_basis, false, field_basis, false);
  const auto ref_dw_f =
      reference.discretization(test_basis, true, field_basis, false);
  const auto ref_w_vf =
      reference.discretization(test_basis, false, field_basis, false, 1);
  const auto ref_w_t =
      reference.discretization(test_basis, false, trace_basis, false);
  const auto ref_w_w =
      reference.discretization(test_basis, false, test_basis, false);

  matrix_factors mass_w_f;
  matrix_factors derivative_w_f;
  matrix_factors velocity_w_f;
  matrix_factors mass_w_trace;
  matrix_factors mass_w_w;
  matrix_factors face_evaluation;

  for (size_t d = 0; d < dimension; ++d) {
    const auto &coordinate = coordinates[d];
    mass_w_f[d] = global_field_assembly_1d(ref_w_f, coordinate.elements,
                                           coordinate.lower, coordinate.upper);
    derivative_w_f[d] = global_field_assembly_1d(
        ref_dw_f, coordinate.elements, coordinate.lower, coordinate.upper, 1);
    mass_w_trace[d] = global_field_assembly_1d(
        ref_w_t, coordinate.elements, coordinate.lower, coordinate.upper);
    mass_w_w[d] = global_field_assembly_1d(ref_w_w, coordinate.elements,
                                           coordinate.lower, coordinate.upper);
    face_evaluation[d] = global_trace_assembly_1d(test_basis->getCardinality(),
                                                  coordinate.elements);
    velocity_w_f[d] = mass_w_f[d];
  }
  for (size_t i = 0; i < spatial_dimensions; ++i) {
    const size_t velocity_dimension =
        coordinate_index(coordinates, CoordinateKind::velocity, i);
    const auto &coordinate = coordinates[velocity_dimension];
    velocity_w_f[velocity_dimension] =
        global_field_assembly_1d(ref_w_f, ref_w_vf, coordinate.elements,
                                 coordinate.lower, coordinate.upper);
  }

  const size_t time_dimension =
      coordinate_index(coordinates, CoordinateKind::time);
  auto time_factors = copy_factors(mass_w_f);
  time_factors[time_dimension] = derivative_w_f[time_dimension];
  time_factors[0] = -time_factors[0];
  tt_matrix field_operator =
      boba::make_ttm_from_matrices<dimension, host_space, double>(time_factors);

  for (size_t i = 0; i < spatial_dimensions; ++i) {
    const size_t position_dimension =
        coordinate_index(coordinates, CoordinateKind::position, i);
    const size_t velocity_dimension =
        coordinate_index(coordinates, CoordinateKind::velocity, i);
    auto streaming_factors = copy_factors(mass_w_f);
    streaming_factors[position_dimension] =
        derivative_w_f[position_dimension];
    streaming_factors[velocity_dimension] =
        velocity_w_f[velocity_dimension];
    streaming_factors[0] = -streaming_factors[0];
    field_operator +=
        boba::make_ttm_from_matrices<dimension, host_space, double>(
            streaming_factors);

    auto acceleration_factors = copy_factors(mass_w_f);
    acceleration_factors[velocity_dimension] =
        derivative_w_f[velocity_dimension];
    acceleration_factors[0] = -acceleration_factors[0];
    field_operator +=
        boba::make_ttm_from_matrices<dimension, host_space, double>(
            acceleration_factors);
  }
  field_operator.round();

  boba::Array<tt_matrix, trace_blocks> trace_operators;
  for (size_t normal = 0; normal < dimension; ++normal) {
    auto factors = copy_factors(mass_w_trace);
    factors[normal] = face_evaluation[normal];
    trace_operators[normal] =
        boba::make_ttm_from_matrices<dimension, host_space, double>(factors);
  }

  auto gram =
      boba::make_ttm_from_matrices<dimension, host_space, double>(mass_w_w);
  block_operator stiffness(total_blocks, total_blocks);
  stiffness({0, 0}) = gram;
  for (size_t trace = 0; trace < trace_blocks; ++trace) {
    stiffness({0, trace + 1}) = trace_operators[trace];
    stiffness({trace + 1, 0}) = trace_operators[trace].transpose();
  }
  stiffness({0, field_block}) = field_operator;
  stiffness({field_block, 0}) = field_operator.transpose();

  for (size_t row = 1; row < total_blocks; ++row) {
    const auto row_sizes = stiffness({row, 0}).core_rows();
    for (size_t col = 1; col < total_blocks; ++col) {
      const auto col_sizes = stiffness({0, col}).core_cols();
      stiffness({row, col}) = zero_tt_matrix<dimension>(row_sizes, col_sizes);
    }
  }

  std::cout << "7D operator assembled\n  coordinate_order=";
  for (size_t d = 0; d < dimension; ++d) {
    std::cout << (d == 0 ? "" : ",") << coordinates[d].name;
  }
  std::cout << "\n  saddle_blocks=" << total_blocks << "\n"
            << "  test_unknowns=" << gram.rows() << "\n"
            << "  trial_unknowns=" << stiffness({1, 0}).rows();
  size_t trial_unknowns = 0;
  for (size_t row = 1; row < total_blocks; ++row) {
    trial_unknowns += stiffness({row, 0}).rows();
  }
  std::cout << " (first trace), " << trial_unknowns << " (all trial blocks)\n"
            << "  field_operator_ranks=" << field_operator.ranks_string()
            << "\n"
            << "  gram_ranks=" << gram.ranks_string() << "\n";
  for (size_t trace = 0; trace < trace_blocks; ++trace) {
    std::cout << "  trace_" << coordinates[trace].name
              << "_unknowns=" << trace_operators[trace].cols() << "\n";
  }
  std::cout << "  field_unknowns=" << field_operator.cols() << "\n";
  return stiffness;
}

block_vector
assemble_rhs(Camellia::BasisPtr test_basis,
             const std::array<Coordinate, dimension> &coordinates) {
  const double pi = 4.0 * std::atan(1.0);
  const auto sine = [pi](double value) { return std::sin(pi * value); };
  const auto sine_derivative = [pi](double value) {
    return pi * std::cos(pi * value);
  };
  const auto cosine = [pi](double value) { return std::cos(pi * value); };
  const auto cosine_derivative = [pi](double value) {
    return -pi * std::sin(pi * value);
  };
  const auto velocity_cosine = [pi](double value) {
    return value * std::cos(pi * value);
  };

  vector_factors values;
  vector_factors derivatives;
  vector_factors velocity_values;
  for (size_t d = 0; d < dimension; ++d) {
    const auto &coordinate = coordinates[d];
    const bool is_velocity =
        coordinate.kind == CoordinateKind::velocity;
    const std::function<double(double)> value_function =
        is_velocity ? std::function<double(double)>(cosine)
                    : std::function<double(double)>(sine);
    const std::function<double(double)> derivative_function =
        is_velocity ? std::function<double(double)>(cosine_derivative)
                    : std::function<double(double)>(sine_derivative);
    values[d] =
        rhs_discretization_1d(test_basis, coordinate.lower, coordinate.upper,
                              coordinate.elements, value_function);
    derivatives[d] =
        rhs_discretization_1d(test_basis, coordinate.lower, coordinate.upper,
                              coordinate.elements, derivative_function);
    velocity_values[d] = values[d];
    if (is_velocity) {
      velocity_values[d] =
          rhs_discretization_1d(test_basis, coordinate.lower, coordinate.upper,
                                coordinate.elements, velocity_cosine);
    }
  }

  const size_t time_dimension =
      coordinate_index(coordinates, CoordinateKind::time);
  auto factors = copy_factors(values);
  factors[time_dimension] = derivatives[time_dimension];
  tt_vector source =
      boba::make_tt_from_vectors<dimension, host_space, double>(factors);
  for (size_t i = 0; i < spatial_dimensions; ++i) {
    const size_t position_dimension =
        coordinate_index(coordinates, CoordinateKind::position, i);
    const size_t velocity_dimension =
        coordinate_index(coordinates, CoordinateKind::velocity, i);
    factors = copy_factors(values);
    factors[position_dimension] = derivatives[position_dimension];
    factors[velocity_dimension] = velocity_values[velocity_dimension];
    source +=
        boba::make_tt_from_vectors<dimension, host_space, double>(factors);

    factors = copy_factors(values);
    factors[velocity_dimension] = derivatives[velocity_dimension];
    source +=
        boba::make_tt_from_vectors<dimension, host_space, double>(factors);
  }
  source.round();

  block_vector rhs(total_blocks);
  rhs(0) = source;
  return rhs;
}

block_vector assemble_manufactured_solution(
    Camellia::BasisPtr trace_basis, Camellia::BasisPtr field_basis,
    const std::array<Coordinate, dimension> &coordinates,
    const block_operator &stiffness) {
  const double pi = 4.0 * std::atan(1.0);
  const auto sine = [pi](double value) { return std::sin(pi * value); };
  const auto cosine = [pi](double value) { return std::cos(pi * value); };
  const auto velocity_cosine = [pi](double value) {
    return value * std::cos(pi * value);
  };

  vector_factors trace_values;
  vector_factors field_values;
  vector_factors face_values;
  vector_factors velocity_trace_values;
  for (size_t d = 0; d < dimension; ++d) {
    const auto &coordinate = coordinates[d];
    const bool is_velocity =
        coordinate.kind == CoordinateKind::velocity;
    const std::function<double(double)> function =
        is_velocity ? std::function<double(double)>(cosine)
                    : std::function<double(double)>(sine);
    trace_values[d] =
        nodal_discretization_1d(trace_basis, coordinate.lower, coordinate.upper,
                                coordinate.elements, function);
    field_values[d] =
        nodal_discretization_1d(field_basis, coordinate.lower, coordinate.upper,
                                coordinate.elements, function);
    face_values[d] = interior_face_discretization_1d(
        coordinate.lower, coordinate.upper, coordinate.elements, function);
    velocity_trace_values[d] = trace_values[d];
    if (is_velocity) {
      velocity_trace_values[d] = nodal_discretization_1d(
          trace_basis, coordinate.lower, coordinate.upper, coordinate.elements,
          velocity_cosine);
    }
  }

  block_vector solution(total_blocks);
  solution(0) = zero_tt_vector<dimension>(stiffness({0, 0}).core_rows());
  for (size_t normal = 0; normal < trace_blocks; ++normal) {
    auto factors = copy_factors(trace_values);
    factors[normal] = face_values[normal];
    if (coordinates[normal].kind == CoordinateKind::position) {
      const size_t velocity_dimension = coordinate_index(
          coordinates, CoordinateKind::velocity, coordinates[normal].axis);
      factors[velocity_dimension] =
          velocity_trace_values[velocity_dimension];
    }
    solution(normal + 1) =
        boba::make_tt_from_vectors<dimension, host_space, double>(factors);
  }
  solution(field_block) =
      boba::make_tt_from_vectors<dimension, host_space, double>(field_values);
  return solution;
}

void set_zero_rhs_blocks(block_vector &rhs, const block_operator &stiffness) {
  for (size_t block = 1; block < total_blocks; ++block) {
    rhs(block) = zero_tt_vector<dimension>(stiffness({0, block}).core_cols());
  }
}

int run(size_t nx, size_t nt, size_t nv, size_t p, size_t dp, bool solve_schur,
        bool write_system, bool interleaved_order,
        const dpg_boba::SchurSolveOptionsND &options) {
  if (nx < 2 || nt < 2 || nv < 2) {
    std::cerr << "All mesh counts must be at least two because every "
                 "coordinate has an interior trace block.\n";
    return 1;
  }
  if (p < 1) {
    std::cerr << "Polynomial order p must be positive.\n";
    return 1;
  }

  const std::array<Coordinate, dimension> grouped_coordinates{{
      {0.0, 1.0, nx, "x1", CoordinateKind::position, 0},
      {0.0, 1.0, nx, "x2", CoordinateKind::position, 1},
      {0.0, 1.0, nx, "x3", CoordinateKind::position, 2},
      {0.0, 1.0, nt, "t", CoordinateKind::time, 0},
      {-0.5, 0.5, nv, "v1", CoordinateKind::velocity, 0},
      {-0.5, 0.5, nv, "v2", CoordinateKind::velocity, 1},
      {-0.5, 0.5, nv, "v3", CoordinateKind::velocity, 2}}};
  const std::array<Coordinate, dimension> interleaved_coordinates{{
      {0.0, 1.0, nx, "x1", CoordinateKind::position, 0},
      {-0.5, 0.5, nv, "v1", CoordinateKind::velocity, 0},
      {0.0, 1.0, nx, "x2", CoordinateKind::position, 1},
      {-0.5, 0.5, nv, "v2", CoordinateKind::velocity, 1},
      {0.0, 1.0, nx, "x3", CoordinateKind::position, 2},
      {-0.5, 0.5, nv, "v3", CoordinateKind::velocity, 2},
      {0.0, 1.0, nt, "t", CoordinateKind::time, 0}}};
  const auto &coordinates =
      interleaved_order ? interleaved_coordinates : grouped_coordinates;

  auto line = Camellia::CellTopology::line();
  auto basis_family = Camellia::BasisFamily::nodalBasisFamily();
  auto test_basis = basis_family->getBasis(
      line, Camellia::efsForSpace(Camellia::HGRAD_DISC), p + dp);
  auto field_basis =
      basis_family->getBasis(line, Camellia::efsForSpace(Camellia::L2), p);
  auto trace_basis =
      basis_family->getBasis(line, Camellia::efsForSpace(Camellia::L2), p + 1);

  boba::TicToc<boba::tictoc_units::milliseconds> timer;
  timer.tic();
  auto stiffness =
      assemble_operator(test_basis, trace_basis, field_basis, coordinates);
  auto rhs = assemble_rhs(test_basis, coordinates);
  set_zero_rhs_blocks(rhs, stiffness);
  auto manufactured = assemble_manufactured_solution(trace_basis, field_basis,
                                                     coordinates, stiffness);
  auto field_error =
      assemble_field_error_diagnostics(field_basis, coordinates);
  const auto assembly_time = timer.toc();

  const auto manufactured_residual = rhs - stiffness * manufactured;
  std::cout << "manufactured_relative_full_residual="
            << boba::norm_frobenius(manufactured_residual) /
                   boba::norm_frobenius(rhs)
            << "\nmanufactured_relative_field_l2_error="
            << relative_field_interpolation_l2_error(field_basis, coordinates)
            << "\nassembly_time_ms=" << assembly_time << "\n";

  std::ostringstream prefix;
  prefix << "vlasov_3x3v_nx_" << nx << "_nt_" << nt << "_nv_" << nv << "_p_"
         << p << "_dp_" << dp
         << (interleaved_order ? "_interleaved_" : "_grouped_")
         << "L2_norm_";
  if (write_system) {
    boba::write_to_hdf5_file(stiffness, prefix.str() + "stiffness.h5",
                             "stiffness");
    boba::write_to_hdf5_file(rhs, prefix.str() + "rhs.h5", "rhs");
    boba::write_to_hdf5_file(manufactured,
                             prefix.str() + "manufactured_soln.h5", "soln");
  }

  if (!solve_schur) {
    return 0;
  }
  timer.tic();
  const int status = dpg_boba::solve_compressed_schur_system_nd<dimension>(
      stiffness, manufactured, rhs, options,
      write_system ? prefix.str() + "amen_soln.h5" : "", &field_error.mass,
      &field_error.exact_load, field_error.exact_norm_squared);
  std::cout << "solver_time_ms=" << timer.toc() << "\n";
  return status;
}

} // namespace

int main(int argc, char **argv) {
  Teuchos::GlobalMPISession mpi_session(&argc, &argv, nullptr);
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    int nx = 2;
    int nt = 2;
    int nv = 2;
    int p = 1;
    int dp = 1;
    bool solve_schur = false;
    bool write_system = false;
    bool interleaved_order = false;
    dpg_boba::SchurSolveOptionsND options;
    int max_sweeps = static_cast<int>(options.amen_max_sweeps);
    int minimum_sweeps = static_cast<int>(options.amen_minimum_sweeps);
    int kickrank = static_cast<int>(options.amen_kickrank);
    int local_gmres_max_iterations =
        static_cast<int>(options.amen_local_gmres_max_iterations);

    Teuchos::CommandLineProcessor command_line(false, true);
    command_line.setOption("nx", &nx, "elements in each spatial coordinate");
    command_line.setOption("nt", &nt, "elements in time");
    command_line.setOption("nv", &nv, "elements in each velocity coordinate");
    command_line.setOption("p", &p, "field polynomial order");
    command_line.setOption("dp", &dp, "test-space enrichment order");
    command_line.setOption("solve_schur", "assemble_only", &solve_schur,
                           "run block AMEN after assembling the 7D L2 system");
    command_line.setOption("write_system", "no_write_system", &write_system,
                           "write 7D TT operators and vectors to HDF5");
    command_line.setOption(
        "interleaved_order", "grouped_order", &interleaved_order,
        "use (x1,v1,x2,v2,x3,v3,t) instead of (x1,x2,x3,t,v1,v2,v3)");
    command_line.setOption("schur_tolerance", &options.linear_tolerance,
                           "relative block-AMEN convergence tolerance");
    command_line.setOption("svd_relative_tolerance",
                           &options.relative_tolerance,
                           "relative SVD truncation tolerance");
    command_line.setOption("svd_absolute_tolerance",
                           &options.absolute_tolerance,
                           "absolute SVD truncation tolerance");
    command_line.setOption("amen_max_sweeps", &max_sweeps,
                           "maximum block-AMEN sweeps");
    command_line.setOption("amen_minimum_sweeps", &minimum_sweeps,
                           "minimum block-AMEN sweeps");
    command_line.setOption("amen_kickrank", &kickrank,
                           "block-AMEN residual enrichment rank");
    command_line.setOption(
        "amen_coupled_solver", &options.amen_coupled_solver,
        "coupled AMEN site solver: cod, qr, lu, eigen_gmres, or boba_gmres");
    command_line.setOption(
        "amen_local_gmres_tolerance", &options.amen_local_gmres_tolerance,
        "relative tolerance for a coupled-site GMRES solver");
    command_line.setOption(
        "amen_local_gmres_max_iterations", &local_gmres_max_iterations,
        "maximum iterations for a coupled-site GMRES solver");
    command_line.setOption(
        "amen_local_gmres_sparsity", &options.amen_local_gmres_sparsity,
        "threshold used when converting a coupled site matrix to sparse form");
    command_line.setOption(
        "amen_validate_matrix_free_operator", "amen_no_validate_matrix_free_operator",
        &options.amen_validate_matrix_free_operator,
        "compare matrix-free coupled-site actions with explicitly assembled matrices");
    command_line.setOption(
        "amen_exact_initial_guess", "amen_zero_initial_guess",
        &options.amen_exact_initial_guess,
        "initialize from the separable manufactured multipliers");

    if (command_line.parse(argc, argv) !=
        Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      status = -1;
    } else if (nx < 2 || nt < 2 || nv < 2 || p < 1 || dp < 0 ||
               max_sweeps < 1 || minimum_sweeps < 1 || kickrank < 0 ||
               local_gmres_max_iterations < 1 ||
               options.amen_local_gmres_tolerance <= 0.0 ||
               options.amen_local_gmres_sparsity < 0.0 ||
               (options.amen_coupled_solver != "cod" &&
                options.amen_coupled_solver != "qr" &&
                options.amen_coupled_solver != "lu" &&
                options.amen_coupled_solver != "eigen_gmres" &&
                options.amen_coupled_solver != "boba_gmres")) {
      std::cerr << "Invalid arguments: mesh counts must be >= 2; p, sweep "
                   "counts, GMRES iterations, and GMRES tolerance must be positive; "
                   "dp, kickrank, and GMRES sparsity must be nonnegative.\n";
      status = -1;
    } else {
      options.amen_max_sweeps = static_cast<size_t>(max_sweeps);
      options.amen_minimum_sweeps = static_cast<size_t>(minimum_sweeps);
      options.amen_kickrank = static_cast<size_t>(kickrank);
      options.amen_local_gmres_max_iterations =
          static_cast<size_t>(local_gmres_max_iterations);
      status = run(static_cast<size_t>(nx), static_cast<size_t>(nt),
                   static_cast<size_t>(nv), static_cast<size_t>(p),
                   static_cast<size_t>(dp), solve_schur, write_system,
                   interleaved_order, options);
    }
  }
  Camellia::finalize(false);
  return status;
}
