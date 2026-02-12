#include <iostream>

#include "Camellia.h"
#include "TypeDefs.h"
#include "Var.h"

#include "boba_types.hpp"
#include "vlasov_system.hpp"

#include "Basis1d.hpp"
#include "BasisNd.hpp"
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
  //
  // simulation mesh
  //

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

  //
  // basis
  //

  auto field_basis_1d = newBasis1d(Camellia::L2,         p     );
  auto trace_basis_1d = newBasis1d(Camellia::L2,         p + 1 );
  auto test_basis_1d  = newBasis1d(Camellia::HGRAD_DISC, p + dp);

  auto   test_basis = newBasisNd<3>({ test_basis_1d,  test_basis_1d,  test_basis_1d});
  auto tracex_basis = newBasisNd<3>({ Teuchos::null, trace_basis_1d, trace_basis_1d});
  auto tracet_basis = newBasisNd<3>({trace_basis_1d,  Teuchos::null, trace_basis_1d});
  auto tracev_basis = newBasisNd<3>({trace_basis_1d, trace_basis_1d,  Teuchos::null});
  auto  field_basis = newBasisNd<3>({field_basis_1d, field_basis_1d, field_basis_1d});

  //
  // functions
  //

  auto zero_fun_1d = newConstant1d(0.0);
  auto pow0_fun_1d = newConstant1d(1.0);
  auto pow1_fun_1d = newMonomial1d(1);
  auto pow2_fun_1d = newMonomial1d(2);
  auto pow3_fun_1d = newMonomial1d(3);

  auto  f1_fun_1d = pow1_fun_1d - pow2_fun_1d;
  auto df1_fun_1d = pow0_fun_1d - 2.0 * pow1_fun_1d;
  auto  f2_fun_1d = pow0_fun_1d - 4.0 * pow2_fun_1d;
  auto df2_fun_1d = -8.0 * pow1_fun_1d;

  // v(x, t, v) = v
  auto v_fun = newFunctionNd<3>({pow0_fun_1d, pow0_fun_1d, pow1_fun_1d});

  // f(x, t, v) = f1(x) * f1(t) * f2(v) and its derivatives
  auto f_fun    = newFunctionNd<3>({ f1_fun_1d,  f1_fun_1d,  f2_fun_1d});
  auto dfdx_fun = newFunctionNd<3>({df1_fun_1d,  f1_fun_1d,  f2_fun_1d});
  auto dfdt_fun = newFunctionNd<3>({ f1_fun_1d, df1_fun_1d,  f2_fun_1d});
  auto dfdv_fun = newFunctionNd<3>({ f1_fun_1d,  f1_fun_1d, df2_fun_1d});

  // trace functions
  auto tnx_fun = v_fun * f_fun;
  auto tnt_fun = f_fun;
  auto tnv_fun = f_fun;

  // c(x, t, v) = v * dfdx(x, t, v) + dfdt(x, t, v) + dfdv(x, t, v)
  auto c_fun = v_fun * dfdx_fun + dfdt_fun + dfdv_fun;

  // zero(x, t, v) = 0
  auto zero_fun = newFunctionNd<3>({zero_fun_1d, zero_fun_1d, zero_fun_1d});

  //
  // block rhs and expected exact lhs
  //

  boba::block_vector<host_tt_vector<3>> rhs(vlasov_vars::size);
  rhs(vlasov_vars::test)    = c_fun->getDiscretization(  test_basis, mesh);
  rhs(vlasov_vars::trace_x) = zero_fun->getDiscretization(tracex_basis, mesh);
  rhs(vlasov_vars::trace_t) = zero_fun->getDiscretization(tracet_basis, mesh);
  rhs(vlasov_vars::trace_v) = zero_fun->getDiscretization(tracev_basis, mesh);
  rhs(vlasov_vars::field)   = zero_fun->getDiscretization( field_basis, mesh);
//  rhs.savetxt("rhs", ".dat");

  boba::block_vector<host_tt_vector<3>> exact_soln(vlasov_vars::size);
  exact_soln(vlasov_vars::test)    = zero_fun->getBasisExpansion(  test_basis, mesh);
  exact_soln(vlasov_vars::trace_x) =  tnx_fun->getBasisExpansion(tracex_basis, mesh);
  exact_soln(vlasov_vars::trace_t) =  tnt_fun->getBasisExpansion(tracet_basis, mesh);
  exact_soln(vlasov_vars::trace_v) =  tnv_fun->getBasisExpansion(tracev_basis, mesh);
  exact_soln(vlasov_vars::field)   =    f_fun->getBasisExpansion( field_basis, mesh);
//  exact_soln.savetxt("exact_soln", ".dat");

  //
  // block linear system
  //

  auto stiffness = construct_vlasov_matrices(test_basis_1d, trace_basis_1d, field_basis_1d,
                                             x_min, x_max, nx,
                                             t_min, t_max, nt,
                                             v_min, v_max, nv,
                                             p, dp);
//  stiffness.savetxt("stiffness", ".dat");

  //
  // verification
  //

  auto exact_residual = rhs - stiffness * exact_soln;

  std::cout << "rhs_w" << std::endl
            << "  sizes = " << rhs(vlasov_vars::test).sizes() << std::endl
            << "  ranks = " << rhs(vlasov_vars::test).ranks_string() << std::endl << std::endl;

  std::cout << "exact_eps" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::test).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::test).ranks_string() << std::endl << std::endl;

  std::cout << "exact_tnx" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::trace_x).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::trace_x).ranks_string() << std::endl << std::endl;

  std::cout << "exact_tnt" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::trace_t).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::trace_t).ranks_string() << std::endl << std::endl;

  std::cout << "exact_tnv" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::trace_v).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::trace_v).ranks_string() << std::endl << std::endl;

  std::cout << "exact_f" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::field).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::field).ranks_string() << std::endl << std::endl;

  std::cout << "exact_residual" << std::endl;
  for (size_t i = 0; i < vlasov_vars::size; ++i) {
    std::cout << "  component_" << i << std::endl 
              << "    sizes = " << exact_residual(i).sizes() << std::endl
              << "    ranks = " << exact_residual(i).ranks_string() << std::endl
              << "    norm  = " << exact_residual(i).norm_frobenius() << std::endl;
  }
  std::cout << std::endl;

  //
  // attempt to solve
  //

  if (solve) {
    boba::krylov<boba::block_operator<host_tt_matrix<3>>, boba::block_vector<host_tt_vector<3>>> solver(krylov_outer, krylov_inner, krylov_tol);
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

//    soln.savetxt("soln", ".dat");
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
