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
// Assemble and solve for the Vlasov system with known E = 1:
//
// f_t + v f_x + f_v = c over 0 <= x <= 1, 0 <= t <= 1, -0.5 <= v <= 0.5
//
// with
//
// c(x, t, v) = (x - x^2) (1 - 2 t) (1 - 4 v^2) +
//              (1 - 2 x) (t - t^2) (v - 4 v^3) +
//              (x - x^2) (t - t^2) (-8v)
//
// and imposing zero Dirichlet boundary condition, which corresponds to the
// exact solution
//
// f(x, t, v) = (x - x^2) (t - t^2) (1 - 4 v^2)
//
void vlasov_test_solve(const size_t &nx, const size_t &nt, const size_t &nv, const size_t &p, const size_t &dp,
                       const bool &solve,
                       const size_t &krylov_outer, const size_t &krylov_inner, const double &krylov_tol) {
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
                                             p, dp);
  stiffness.savetxt("stiffness", ".dat");

  auto fx  = rhs_discretization_1d(test_basis, x_min, x_max, nx, [](const double &x) { return x - x * x; });
  auto fxx = rhs_discretization_1d(test_basis, x_min, x_max, nx, [](const double &x) { return 1.0 - 2.0 * x; });

  auto ft  = rhs_discretization_1d(test_basis, t_min, t_max, nt, [](const double &t) { return t - t * t; });
  auto ftt = rhs_discretization_1d(test_basis, t_min, t_max, nt, [](const double &t) { return 1.0 - 2.0 * t; });

  auto fv  = rhs_discretization_1d(test_basis, v_min, v_max, nv, [](const double &v) { return 1.0 - 4.0 * v * v; });
  auto vfv = rhs_discretization_1d(test_basis, v_min, v_max, nv, [](const double &v) { return v - 4.0 * v * v * v; });
  auto fvv = rhs_discretization_1d(test_basis, v_min, v_max, nv, [](const double &v) { return -8.0 * v; });

  auto rhs_test = boba::make_tt_from_vectors<3, host_space, double, size_t>({ fx, ftt,  fv}) +
                  boba::make_tt_from_vectors<3, host_space, double, size_t>({fxx,  ft, vfv}) +
                  boba::make_tt_from_vectors<3, host_space, double, size_t>({ fx,  ft, fvv});
  rhs_test.round();

  std::cout << "rhs" << std::endl
            << "  cols  = " << rhs_test.sizes() << std::endl
            << "  ranks = " << rhs_test.ranks_string() << std::endl << std::endl;

  boba::block_vector<host_tt_vector<3>, vlasov_vars::size> rhs;
  rhs(vlasov_vars::test)    = rhs_test;
  rhs(vlasov_vars::trace_x) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_x}).core_cols());
  rhs(vlasov_vars::trace_t) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_t}).core_cols());
  rhs(vlasov_vars::trace_v) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_v}).core_cols());
  rhs(vlasov_vars::field)   = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::field  }).core_cols());

  rhs.savetxt("rhs", ".dat");

  if (solve) {
    boba::krylov<boba::block_operator<host_tt_matrix<3>, vlasov_vars::size, vlasov_vars::size>,
                 boba::block_vector<host_tt_vector<3>, vlasov_vars::size>> solver(krylov_outer, krylov_inner, krylov_tol);
    solver.method = boba::krylov_methods::flexible_gmres;
    solver.verbose = true;

    solver.set_matrix(stiffness);

    auto guess = rhs;
    guess.fill_with_zeros();

    auto soln = rhs;
    soln.fill_with_zeros();

    solver.solve(rhs, guess, soln);

    for (size_t i = 0; i < vlasov_vars::size; ++i) {
      std::cout << "soln[" << i << "]" << std::endl
                << "  cols  = " << soln(i).sizes() << std::endl
                << "  ranks = " << soln(i).ranks_string() << std::endl;
    }

    soln.savetxt("soln", ".dat");
  }
}

int main(int argc, char **argv) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  Teuchos::CommandLineProcessor cmdp(false,  // don't throw exceptions
                                     true);  // return errors for unrecognized options

  {
    int    nx    = 2;
    int    nt    = 2;
    int    nv    = 2;
    int    p     = 2;
    int    dp    = 2;
    bool   solve = false;
    int    outer = 5;
    int    inner = 250;
    double tol   = 1.0e-03;

    cmdp.setOption("nx",                  &nx,    "number of elements along x dimension");
    cmdp.setOption("nt",                  &nt,    "number of elements along t dimension");
    cmdp.setOption("nv",                  &nv,    "number of elements along v dimension");
    cmdp.setOption("p",                   &p,     "basis order for field variables");
    cmdp.setOption("dp",                  &dp,    "basis order enrichment for test variables");
    cmdp.setOption("solve", "dont-solve", &solve, "whether to run TT solver");
    cmdp.setOption("outer",               &outer, "number of Krylov outer iterations");
    cmdp.setOption("inner",               &inner, "number of Krylov inner iterations");
    cmdp.setOption("tol",                 &tol,   "Krylov convergence tolerance");

    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }

    vlasov_test_solve(nx, nt, nv, p, dp, solve, outer, inner, tol);
  }

  Camellia::finalize(false);
  return 0;
}
