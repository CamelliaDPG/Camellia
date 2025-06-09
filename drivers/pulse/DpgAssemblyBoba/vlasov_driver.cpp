#include <functional>
#include <iostream>

#include "Camellia.h"
#include "TypeDefs.h"
#include "Var.h"

#include "boba_types.hpp"
#include "vlasov_system.hpp"

#include "Basis1d.hpp"
#include "BasisNd.hpp"
#include "DiscretizedFunction.hpp"
#include "Function1d.hpp"
#include "FunctionNd.hpp"
#include "Mesh1d.hpp"
#include "MeshNd.hpp"

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

  auto x_mesh = newMesh1d(x_min, x_max, nx);
  auto t_mesh = newMesh1d(t_min, t_max, nt);
  auto v_mesh = newMesh1d(v_min, v_max, nv);

  auto mesh = newMeshNd<3>({x_mesh, t_mesh, v_mesh});

  auto field_basis_1d = newBasis1d(Camellia::L2,         p     );
  auto trace_basis_1d = newBasis1d(Camellia::L2,         p + 1 );
  auto test_basis_1d  = newBasis1d(Camellia::HGRAD_DISC, p + dp);

  auto   test_basis = newBasisNd<3>({ test_basis_1d,  test_basis_1d,  test_basis_1d});
  auto tracex_basis = newBasisNd<3>({ Teuchos::null, trace_basis_1d, trace_basis_1d});
  auto tracet_basis = newBasisNd<3>({trace_basis_1d,  Teuchos::null, trace_basis_1d});
  auto tracev_basis = newBasisNd<3>({trace_basis_1d, trace_basis_1d,  Teuchos::null});
  auto  field_basis = newBasisNd<3>({field_basis_1d, field_basis_1d, field_basis_1d});

  auto   fx_fun = [](const double &x) { return x - x * x; };
  auto  dfx_fun = [](const double &x) { return 1.0 - 2.0 * x; };
  auto   ft_fun = [](const double &t) { return t - t * t; };
  auto  dft_fun = [](const double &t) { return 1.0 - 2.0 * t; };
  auto   fv_fun = [](const double &v) { return 1.0 - 4.0 * v * v; };
  auto  dfv_fun = [](const double &v) { return -8.0 * v; };
  auto v_fv_fun = [](const double &v) { return v - 4.0 * v * v * v; };
  auto zero_fun = [](const double &s) { return 0.0; };

  auto load_fun = newFunctionNd<3>();
  load_fun->addTerm(1.0, {newFunction1d( fx_fun), newFunction1d(dft_fun), newFunction1d(  fv_fun)});
  load_fun->addTerm(1.0, {newFunction1d(dfx_fun), newFunction1d( ft_fun), newFunction1d(v_fv_fun)});
  load_fun->addTerm(1.0, {newFunction1d( fx_fun), newFunction1d( ft_fun), newFunction1d( dfv_fun)});

  auto load = discretizeFunction(load_fun, test_basis, mesh)->getTtRepresentation(/* round = */ true);

  auto eps_fun = newFunctionNd<3>({newFunction1d(zero_fun), newFunction1d(zero_fun), newFunction1d(zero_fun)});
  auto exact_eps = discretizeFunction(eps_fun, test_basis, mesh)->getTtRepresentation();

  auto tnx_fun = newFunctionNd<3>({newFunction1d(fx_fun), newFunction1d(ft_fun), newFunction1d(v_fv_fun)});
  auto exact_tnx = discretizeFunction(tnx_fun, tracex_basis, mesh)->getTtRepresentation();

  auto tnt_fun = newFunctionNd<3>({newFunction1d(fx_fun), newFunction1d(ft_fun), newFunction1d(fv_fun)});
  auto exact_tnt = discretizeFunction(tnt_fun, tracet_basis, mesh)->getTtRepresentation();

  auto tnv_fun = newFunctionNd<3>({newFunction1d(fx_fun), newFunction1d(ft_fun), newFunction1d(fv_fun)});
  auto exact_tnv = discretizeFunction(tnv_fun, tracev_basis, mesh)->getTtRepresentation();

  auto f_fun = newFunctionNd<3>({newFunction1d(fx_fun), newFunction1d(ft_fun), newFunction1d(fv_fun)});
  auto exact_f = discretizeFunction(f_fun, field_basis, mesh)->getTtRepresentation();

  auto stiffness = construct_vlasov_matrices(test_basis_1d, trace_basis_1d, field_basis_1d,
                                             x_min, x_max, nx,
                                             t_min, t_max, nt,
                                             v_min, v_max, nv,
                                             p, dp);
  stiffness.savetxt("stiffness", ".dat");

  boba::block_vector<host_tt_vector<3>, vlasov_vars::size> rhs;
  rhs(vlasov_vars::test)    = load;
  rhs(vlasov_vars::trace_x) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_x}).core_cols());
  rhs(vlasov_vars::trace_t) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_t}).core_cols());
  rhs(vlasov_vars::trace_v) = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::trace_v}).core_cols());
  rhs(vlasov_vars::field)   = zero_tt_vector<3>(stiffness({vlasov_vars::test, vlasov_vars::field  }).core_cols());

  rhs.savetxt("rhs", ".dat");

  boba::block_vector<host_tt_vector<3>, vlasov_vars::size> exact_soln;
  exact_soln(vlasov_vars::test)    = exact_eps;
  exact_soln(vlasov_vars::trace_x) = exact_tnx;
  exact_soln(vlasov_vars::trace_t) = exact_tnt;
  exact_soln(vlasov_vars::trace_v) = exact_tnv;
  exact_soln(vlasov_vars::field)   = exact_f;

  exact_soln.savetxt("exact_soln", ".dat");

  auto exact_residual = rhs - stiffness * exact_soln;

  std::cout << "load" << std::endl
            << "  sizes = " << load.sizes() << std::endl
            << "  ranks = " << load.ranks_string() << std::endl << std::endl;

  std::cout << "exact_eps" << std::endl
            << "  sizes = " << exact_eps.sizes() << std::endl
            << "  ranks = " << exact_eps.ranks_string() << std::endl << std::endl;

  std::cout << "exact_tnx" << std::endl
            << "  sizes = " << exact_tnx.sizes() << std::endl
            << "  ranks = " << exact_tnx.ranks_string() << std::endl << std::endl;

  std::cout << "exact_tnt" << std::endl
            << "  sizes = " << exact_tnt.sizes() << std::endl
            << "  ranks = " << exact_tnt.ranks_string() << std::endl << std::endl;

  std::cout << "exact_tnv" << std::endl
            << "  sizes = " << exact_tnv.sizes() << std::endl
            << "  ranks = " << exact_tnv.ranks_string() << std::endl << std::endl;

  std::cout << "exact_f" << std::endl
            << "  sizes = " << exact_f.sizes() << std::endl
            << "  ranks = " << exact_f.ranks_string() << std::endl << std::endl;

  std::cout << "exact_residual" << std::endl;
  for (size_t i = 0; i < vlasov_vars::size; ++i) {
    std::cout << "  component_" << i << std::endl 
              << "    sizes = " << exact_residual(i).sizes() << std::endl
              << "    ranks = " << exact_residual(i).sizes() << std::endl
              << "    norm  = " << exact_residual(i).norm_frobenius() << std::endl;
  }
  std::cout << std::endl;

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

    std::cout << "solution" << std::endl;
    for (size_t i = 0; i < vlasov_vars::size; ++i) {
      std::cout << "  soln[" << i << "]" << std::endl
                << "    cols  = " << soln(i).sizes() << std::endl
                << "    ranks = " << soln(i).ranks_string() << std::endl;
    }
    std::cout << std::endl;

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
    int    p     = 3;
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
