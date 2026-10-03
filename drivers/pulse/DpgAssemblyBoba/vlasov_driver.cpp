#include <cmath>
#include <functional>
#include <iostream>

#include "Basis.h"
#include "BasisFamily.h"
#include "Camellia.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "TypeDefs.h"
#include "Var.h"

#include "boba_types.hpp"
#include "global_assembly.hpp"
#include "reference_integrals_1d.hpp"
#include "schur_solver.hpp"
#include "vlasov_system.hpp"

//
// Given an 1D function rhs(), compute its discretization over uniform mesh with
// nx elements over [x_min, x_max] and specified basis within each element. The
// output is a vector whose (ib + ix * nb)-th entry corresponds to L2 inner
// product of rhs() and ib-th basis over the ix-th element.
//
host_vector
rhs_discretization_1d(Camellia::BasisPtr basis,
                      const double &x_min, const double &x_max, const size_t &nx,
                      const std::function<double(double)> &rhs_fun) {
  const int nb = basis->getCardinality();
  const double dx = (x_max - x_min) / nx;

  const int cubature_degree = 2 * basis->getDegree();
  auto cubature = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubature_degree);

  auto cubature_points = cubature->allocateCubaturePoints();
  auto cubature_weights = cubature->allocateCubatureWeights();
  cubature->getCubature(cubature_points, cubature_weights);

  const int nc = cubature_points.extent_int(0);

  auto basis_values = basis->allocateBasisValues(cubature_points, Intrepid2::OPERATOR_VALUE);
  basis->getValues(basis_values, cubature_points, Intrepid2::OPERATOR_VALUE);

  host_vector rhs_vec({nx * nb});
  auto rhs_vec_view = rhs_vec.view();
  for (int ix = 0; ix < nx; ++ix) {
    const double shift = x_min + 0.5 * (2 * ix + 1) * dx;
    const double scale = 0.5 * dx;
    for (int ib = 0; ib < nb; ++ib) {
      double value = 0.0;
      for (int ic = 0; ic < nc; ++ic) {
        value += scale * basis_values(ib, ic) * rhs_fun(shift + scale * cubature_points(ic, 0)) * cubature_weights(ic);
      }

      const size_t i = static_cast<size_t>(ib + ix * nb);
      rhs_vec_view(i) = value;
    }
  }

  return rhs_vec;
}

//
// Given an 1D function soln(), compute its discretization over uniform mesh with
// nx elements over [x_min, x_max] and specified basis within each element. The
// output is a vector whose (ib + ix * nb)-th entry corresponds to value of the
// function at the node of the ib-th basis over the ix-th element.
//
host_vector
soln_discretization_1d(Camellia::BasisPtr basis,
                       const double &x_min, const double &x_max, const size_t &nx,
                       const std::function<double(double)> &soln_fun) {
  const int nb = basis->getCardinality();
  const double dx = (x_max - x_min) / nx;

  Intrepid::FieldContainer<double, 0> basis_nodes(nb, 1);
  basis->getDofCoords(basis_nodes);

  host_vector soln_vec({nx * nb});
  auto soln_vec_view = soln_vec.view();
  for (int ix = 0; ix < nx; ++ix) {
    const double shift = x_min + 0.5 * (2 * ix + 1) * dx;
    const double scale = 0.5 * dx;
    for (int ib = 0; ib < nb; ++ib) {
      const double value = soln_fun(shift + scale * basis_nodes(ib, 0));

      const size_t i = static_cast<size_t>(ib + ix * nb);
      soln_vec_view(i) = value;
    }
  }

  return soln_vec;
}

host_vector
soln_discretization_1d(const double &x_min, const double &x_max, const size_t &nx,
                       const std::function<double(double)> &soln_fun) {
  const double dx = (x_max - x_min) / nx;

  host_vector soln_vec({nx - 1});
  auto soln_vec_view = soln_vec.view();
  for (int ix = 1; ix < nx; ++ix) {
    const double shift = x_min + ix * dx;
    const double value = soln_fun(shift);

    const size_t i = static_cast<size_t>(ix - 1);
    soln_vec_view(i) = value;
  }

  return soln_vec;
}

//
// Assemble and solve for the Vlasov system with known E = 1:
//
// f_t + v f_x + f_v = c over 0 <= x <= 1, 0 <= t <= 1, -0.5 <= v <= 0.5
//
// with
//
// c(x, t, v) = pi sin(pi x) cos(pi t) cos(pi v) +
//              pi cos(pi x) sin(pi t) v cos(pi v) -
//              pi sin(pi x) sin(pi t) sin(pi v)
//
// and imposing zero Dirichlet boundary condition, which corresponds to the
// exact solution
//
// f(x, t, v) = sin(pi x) sin(pi t) cos(pi v)
//
void vlasov_test_solve(const size_t &nx, const size_t &nt, const size_t &nv, const size_t &p, const size_t &dp,
                       const std::string &test_space_norm_str,
                       const bool solve_schur,
                       const dpg_boba::SchurSolveOptions &schur_options) {
  boba::TicToc<boba::tictoc_units::milliseconds> timer;

  std::string file_prefix;
  {
    std::ostringstream stream;
    stream << "vlasov_system_nx_" << nx << "_nt_" << nt << "_nv_" << nv << "_p_" << p << "_dp_" << dp << "_"
           << test_space_norm_str << "_norm_";
    file_prefix = stream.str();
  }

  TestSpaceNorm test_space_norm = TestSpaceNorm::GRAPH;
  if (test_space_norm_str == "graph") {
    // do nothing
  } else if (test_space_norm_str == "L2") {
    test_space_norm = TestSpaceNorm::L2;
  } else {
    boba_error("unknown test space norm");
  }

  timer.tic();

  const double x_min =  0.0;
  const double x_max =  1.0;
  const double t_min =  0.0;
  const double t_max =  1.0;
  const double v_min = -0.5;
  const double v_max =  0.5;

  auto line = Camellia::CellTopology::line();

  auto basis_family = Camellia::BasisFamily::nodalBasisFamily();
  auto test_basis   = basis_family->getBasis(line, Camellia::efsForSpace(Camellia::HGRAD_DISC), p + dp);
  auto field_basis  = basis_family->getBasis(line, Camellia::efsForSpace(Camellia::L2),         p     );
  auto trace_basis  = basis_family->getBasis(line, Camellia::efsForSpace(Camellia::L2),         p + 1 );

  auto stiffness = construct_vlasov_matrices(test_basis, trace_basis, field_basis,
                                             x_min, x_max, nx,
                                             t_min, t_max, nt,
                                             v_min, v_max, nv,
                                             p, dp,
                                             test_space_norm);

  // Separable FEM L2 metrics used to study mass-weighted TT compression.
  reference_integrals_1d mass_integrals;
  auto ref_test_mass  = mass_integrals.discretization(test_basis,  false, test_basis,  false);
  auto ref_trace_mass = mass_integrals.discretization(trace_basis, false, trace_basis, false);
  auto ref_field_mass = mass_integrals.discretization(field_basis, false, field_basis, false);

  auto x_test_mass = global_field_assembly_1d(ref_test_mass, nx, x_min, x_max);
  auto t_test_mass = global_field_assembly_1d(ref_test_mass, nt, t_min, t_max);
  auto v_test_mass = global_field_assembly_1d(ref_test_mass, nv, v_min, v_max);
  auto x_trace_mass = global_field_assembly_1d(ref_trace_mass, nx, x_min, x_max);
  auto t_trace_mass = global_field_assembly_1d(ref_trace_mass, nt, t_min, t_max);
  auto v_trace_mass = global_field_assembly_1d(ref_trace_mass, nv, v_min, v_max);
  auto x_field_mass = global_field_assembly_1d(ref_field_mass, nx, x_min, x_max);
  auto t_field_mass = global_field_assembly_1d(ref_field_mass, nt, t_min, t_max);
  auto v_field_mass = global_field_assembly_1d(ref_field_mass, nv, v_min, v_max);

  host_matrix x_face_mass({nx - 1, nx - 1});
  host_matrix t_face_mass({nt - 1, nt - 1});
  host_matrix v_face_mass({nv - 1, nv - 1});
  x_face_mass.set_to_identity_matrix();
  t_face_mass.set_to_identity_matrix();
  v_face_mass.set_to_identity_matrix();

  auto test_mass = boba::make_ttm_from_matrices<3, host_space, double>(
    {x_test_mass, t_test_mass, v_test_mass});
  auto trace_x_mass = boba::make_ttm_from_matrices<3, host_space, double>(
    {x_face_mass, t_trace_mass, v_trace_mass});
  auto trace_t_mass = boba::make_ttm_from_matrices<3, host_space, double>(
    {x_trace_mass, t_face_mass, v_trace_mass});
  auto trace_v_mass = boba::make_ttm_from_matrices<3, host_space, double>(
    {x_trace_mass, t_trace_mass, v_face_mass});
  auto field_mass = boba::make_ttm_from_matrices<3, host_space, double>(
    {x_field_mass, t_field_mass, v_field_mass});

  boba::write_to_hdf5_file(test_mass,    file_prefix + "mass_test.h5",    "mass");
  boba::write_to_hdf5_file(trace_x_mass, file_prefix + "mass_trace_x.h5", "mass");
  boba::write_to_hdf5_file(trace_t_mass, file_prefix + "mass_trace_t.h5", "mass");
  boba::write_to_hdf5_file(trace_v_mass, file_prefix + "mass_trace_v.h5", "mass");
  boba::write_to_hdf5_file(field_mass,   file_prefix + "mass_field.h5",   "mass");

  const double PI = 4.0 * std::atan(1.0);

  auto   fx_fun = [&PI](const double &x) { return       std::sin(PI * x); };
  auto  dfx_fun = [&PI](const double &x) { return  PI * std::cos(PI * x); };
  auto   ft_fun = [&PI](const double &t) { return       std::sin(PI * t); };
  auto  dft_fun = [&PI](const double &t) { return  PI * std::cos(PI * t); };
  auto   fv_fun = [&PI](const double &v) { return       std::cos(PI * v); };
  auto  dfv_fun = [&PI](const double &v) { return -PI * std::sin(PI * v); };
  auto v_fv_fun = [&PI](const double &v) { return   v * std::cos(PI * v); };

  auto   fx = rhs_discretization_1d(test_basis, x_min, x_max, nx,   fx_fun);
  auto  dfx = rhs_discretization_1d(test_basis, x_min, x_max, nx,  dfx_fun);
  auto   ft = rhs_discretization_1d(test_basis, t_min, t_max, nt,   ft_fun);
  auto  dft = rhs_discretization_1d(test_basis, t_min, t_max, nt,  dft_fun);
  auto   fv = rhs_discretization_1d(test_basis, v_min, v_max, nv,   fv_fun);
  auto  dfv = rhs_discretization_1d(test_basis, v_min, v_max, nv,  dfv_fun);
  auto v_fv = rhs_discretization_1d(test_basis, v_min, v_max, nv, v_fv_fun);

  auto rhs_test = boba::make_tt_from_vectors<3, host_space, double>({ fx, dft,   fv}) +
                  boba::make_tt_from_vectors<3, host_space, double>({dfx,  ft, v_fv}) +
                  boba::make_tt_from_vectors<3, host_space, double>({ fx,  ft,  dfv});
  rhs_test.round();

  std::cout << "rhs" << std::endl
            << "  cols  = " << rhs_test.sizes() << std::endl
            << "  ranks = " << rhs_test.ranks_string() << std::endl << std::endl;

  boba::BlockVector<host_tt_vector<3>> rhs(vlasov_vars::size);
  rhs(vlasov_vars::test)    = rhs_test;
  rhs(vlasov_vars::trace_x) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_x}).core_cols());
  rhs(vlasov_vars::trace_t) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_t}).core_cols());
  rhs(vlasov_vars::trace_v) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_v}).core_cols());
  rhs(vlasov_vars::field)   = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::field  }).core_cols());

  auto assembly_timing = timer.toc();

  boba::write_to_hdf5_file(stiffness, file_prefix + "stiffness.h5", "stiffness");
  boba::write_to_hdf5_file(rhs, file_prefix + "rhs.h5", "rhs");

  auto fx_bdy = soln_discretization_1d(x_min, x_max, nx,   fx_fun);
  auto ft_bdy = soln_discretization_1d(t_min, t_max, nt,   ft_fun);
  auto fv_bdy = soln_discretization_1d(v_min, v_max, nv,   fv_fun);

  auto   fx_trace = soln_discretization_1d(trace_basis, x_min, x_max, nx,   fx_fun);
  auto   ft_trace = soln_discretization_1d(trace_basis, t_min, t_max, nt,   ft_fun);
  auto   fv_trace = soln_discretization_1d(trace_basis, v_min, v_max, nv,   fv_fun);
  auto v_fv_trace = soln_discretization_1d(trace_basis, v_min, v_max, nv, v_fv_fun);

  auto fx_field = soln_discretization_1d(field_basis, x_min, x_max, nx, fx_fun);
  auto ft_field = soln_discretization_1d(field_basis, t_min, t_max, nx, ft_fun);
  auto fv_field = soln_discretization_1d(field_basis, v_min, v_max, nx, fv_fun);

  auto epsx_test = soln_discretization_1d(test_basis, x_min, x_max, nx, [](const double &x) { return 0.0; });
  auto epst_test = soln_discretization_1d(test_basis, t_min, t_max, nt, [](const double &t) { return 0.0; });
  auto epsv_test = soln_discretization_1d(test_basis, v_min, v_max, nv, [](const double &v) { return 0.0; });

  boba::BlockVector<host_tt_vector<3>> exact_soln(vlasov_vars::size);
  exact_soln(vlasov_vars::test)    = boba::make_tt_from_vectors<3, host_space, double>({ epsx_test,  epst_test,  epsv_test});
  exact_soln(vlasov_vars::trace_x) = boba::make_tt_from_vectors<3, host_space, double>({    fx_bdy,   ft_trace, v_fv_trace});
  exact_soln(vlasov_vars::trace_t) = boba::make_tt_from_vectors<3, host_space, double>({  fx_trace,     ft_bdy,   fv_trace});
  exact_soln(vlasov_vars::trace_v) = boba::make_tt_from_vectors<3, host_space, double>({  fx_trace,   ft_trace,     fv_bdy});
  exact_soln(vlasov_vars::field)   = boba::make_tt_from_vectors<3, host_space, double>({  fx_field,   ft_field,   fv_field});
  boba::write_to_hdf5_file(exact_soln, file_prefix + "soln.h5", "soln");


  auto exact_residual = rhs - stiffness * exact_soln;
  std::cout << "exact_residual_norm" << std::endl;
  for (size_t i = 0; i < vlasov_vars::size; ++i) {
      std::cout << "  [" << i << "] = " << boba::norm_frobenius(exact_residual(i)) << std::endl;
  }
  std::cout << std::endl;

  timer.tic();

  if (solve_schur) {
    if (test_space_norm != TestSpaceNorm::L2) {
      boba_error("The integrated Schur solve currently supports only --test_space_norm=L2.");
    }
    const int schur_status = dpg_boba::solve_compressed_schur_system(
      stiffness, exact_soln, rhs, schur_options, file_prefix + "amen_soln.h5");
    if (schur_status != 0) {
      boba_error("Integrated Schur solve failed.");
    }
  }

  auto solver_timing = timer.toc();

  std::cout << "timing " << std::endl
            << "  assembly: " << assembly_timing << " " << timer.units_string << std::endl
            << "  solver: " << solver_timing << " " << timer.units_string << std::endl << std::endl;
}

int main(int argc, char **argv) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  Teuchos::CommandLineProcessor cmdp(false,  // don't throw exceptions
                                     true);  // return errors for unrecognized options

  {
    int         nx              = 2;
    int         nt              = 2;
    int         nv              = 2;
    int         p               = 3;
    int         dp              = 2;
    std::string test_space_norm = "graph";
    bool        solve_schur     = false;
    dpg_boba::SchurSolveOptions schur_options;
    int         amen_max_sweeps     = static_cast<int>(schur_options.amen_max_sweeps);
    int         amen_minimum_sweeps = static_cast<int>(schur_options.amen_minimum_sweeps);
    int         amen_kickrank       = static_cast<int>(schur_options.amen_kickrank);
    int         gmres_outer_iterations = static_cast<int>(schur_options.gmres_outer_iterations);
    int         gmres_inner_iterations = static_cast<int>(schur_options.gmres_inner_iterations);

    cmdp.setOption("nx",              &nx,              "number of elements along x dimension");
    cmdp.setOption("nt",              &nt,              "number of elements along t dimension");
    cmdp.setOption("nv",              &nv,              "number of elements along v dimension");
    cmdp.setOption("p",               &p,               "basis order for field variables");
    cmdp.setOption("dp",              &dp,              "basis order enrichment for test variables");
    cmdp.setOption("test_space_norm", &test_space_norm, "test space norm choice (graph or L2)");
    cmdp.setOption("solve_schur",     "skip_schur",    &solve_schur, "solve the in-memory compressed L2 Schur system after assembly");
    cmdp.setOption("schur_method", &schur_options.method, "Schur solve method: block_amen, gmres, or exact");
    cmdp.setOption("schur_tolerance", &schur_options.linear_tolerance, "relative tolerance for the compressed Schur solve");
    cmdp.setOption("svd_relative_tolerance", &schur_options.relative_tolerance, "relative SVD tolerance used during compressed Schur operations");
    cmdp.setOption("svd_absolute_tolerance", &schur_options.absolute_tolerance, "absolute SVD tolerance used during compressed Schur operations");
    cmdp.setOption("amen_max_sweeps", &amen_max_sweeps, "maximum block AMEN sweeps");
    cmdp.setOption("amen_minimum_sweeps", &amen_minimum_sweeps, "minimum block AMEN sweeps before convergence");
    cmdp.setOption("amen_kickrank", &amen_kickrank, "block AMEN enrichment kick rank");
    cmdp.setOption("amen_exact_initial_guess", "amen_zero_initial_guess", &schur_options.amen_exact_initial_guess, "use exact multiplier blocks as the AMEN initial guess");
    cmdp.setOption("gmres_outer_iterations", &gmres_outer_iterations, "maximum restarted GMRES outer iterations");
    cmdp.setOption("gmres_inner_iterations", &gmres_inner_iterations, "GMRES Krylov dimension per restart");
    cmdp.setOption("gram_inverse", &schur_options.gram_inverse, "Gram inverse strategy; current L2 path supports corewise");

    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    schur_options.amen_max_sweeps = static_cast<size_t>(amen_max_sweeps);
    schur_options.amen_minimum_sweeps = static_cast<size_t>(amen_minimum_sweeps);
    schur_options.amen_kickrank = static_cast<size_t>(amen_kickrank);
    schur_options.gmres_outer_iterations = static_cast<size_t>(gmres_outer_iterations);
    schur_options.gmres_inner_iterations = static_cast<size_t>(gmres_inner_iterations);

    vlasov_test_solve(nx, nt, nv, p, dp, test_space_norm, solve_schur, schur_options);
  }

  Camellia::finalize(false);
  return 0;
}
