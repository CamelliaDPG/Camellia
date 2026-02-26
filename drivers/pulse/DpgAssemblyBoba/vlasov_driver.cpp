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

#include "assembly_verification.hpp"
#include "reference_integrals_1d.hpp"

template<class Basis1D>
host_tensor<2>
assemble_ref_mass_intrepid2(Teuchos::RCP<Basis1D> basisPtr)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using Scalar    = double;
  using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

  auto& basis = *basisPtr->intrepid2Basis();
  auto cellTopo = basis.getBaseCellTopology();
  const int N = basis.getCardinality();

  const int p = basis.getDegree();
  const int cubDegree = 2 * p + 2; // safe margin

  Intrepid2::DefaultCubatureFactory cubFactory;
  auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(cellTopo, cubDegree);

  const int Q = cub->getNumPoints();
  DRV refPts("refPts", Q, 1);
  DRV refWts("refWts", Q);
  cub->getCubature(refPts, refWts);

  DRV phi("phi", N, Q);
  basis.getValues(phi, refPts, Intrepid2::OPERATOR_VALUE);

  auto refPts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), refPts);
  auto refWts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), refWts);
  auto phi_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);

  host_tensor<2> M({(size_t)N, (size_t)N});
  auto Mv = M.view();

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      double sum = 0.0;
      for (int q = 0; q < Q; ++q)
      {
        sum += refWts_h(q) * phi_h(i,q) * phi_h(j,q);
      }
      Mv({(size_t)i,(size_t)j}) = sum;
    }
  }

  return M;
}

inline void
compare_ref_mass_matrices(Camellia::BasisPtr basis1d)
{
  reference_integrals_1d refInts;

  // Camellia integral cache mass matrix on reference cell
  host_tensor<2> M_cam = refInts.discretization(basis1d, false, basis1d, false);

  // Intrepid2-assembled reference mass matrix
  host_tensor<2> M_i2  = assemble_ref_mass_intrepid2(basis1d);

  const size_t N = M_cam.sizes(0);
  boba_assert(M_cam.sizes(1) == N, "Camellia mass not square");
  boba_assert(M_i2.sizes(0)  == N && M_i2.sizes(1) == N, "Intrepid2 mass size mismatch");

  auto Mc = M_cam.view();
  auto Mi = M_i2.view();

  double maxAbsDiff = 0.0;
  double maxAbsCam  = 0.0;
  double maxAbsI2   = 0.0;

  size_t maxi=0, maxj=0;

  for (size_t i = 0; i < N; ++i)
  {
    for (size_t j = 0; j < N; ++j)
    {
      const double a = Mc({i,j});
      const double b = Mi({i,j});
      const double d = std::fabs(a - b);

      if (d > maxAbsDiff) { maxAbsDiff = d; maxi=i; maxj=j; }
      maxAbsCam = std::max(maxAbsCam, std::fabs(a));
      maxAbsI2  = std::max(maxAbsI2,  std::fabs(b));
    }
  }

  const double denom = std::max(1.0, std::max(maxAbsCam, maxAbsI2));
  const double rel = maxAbsDiff / denom;

  std::cout << std::setprecision(16);
  std::cout << "compare_ref_mass_matrices:\n"
            << "  N = " << N << "\n"
            << "  maxAbsCam  = " << maxAbsCam << "\n"
            << "  maxAbsI2   = " << maxAbsI2  << "\n"
            << "  maxAbsDiff = " << maxAbsDiff << " at (" << maxi << "," << maxj << ")\n"
            << "  relDiff    = " << rel << "\n";

  // Optional: print the worst entry
  std::cout << "  Camellia("<<maxi<<","<<maxj<<") = " << Mc({maxi,maxj}) << "\n"
            << "  I2     ("<<maxi<<","<<maxj<<") = " << Mi({maxi,maxj}) << "\n";
}

template<class Basis1D_w, class Basis1D_t>
host_tensor<2>
assemble_ref_mixed_mass_intrepid2(Teuchos::RCP<Basis1D_w> wBasisPtr,
                                  Teuchos::RCP<Basis1D_t> tBasisPtr)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using Scalar    = double;
  using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

  auto& wBasis = *wBasisPtr->intrepid2Basis();
  auto& tBasis = *tBasisPtr->intrepid2Basis();

  const int Nw = wBasis.getCardinality();
  const int Nt = tBasis.getCardinality();

  // They should live on the same reference cell topology (line).
  auto cellTopo = wBasis.getBaseCellTopology();

  // Cubature degree: enough for product of the two bases
  const int pw = wBasis.getDegree();
  const int pt = tBasis.getDegree();
  const int cubDegree = pw + pt + 2; // margin

  Intrepid2::DefaultCubatureFactory cubFactory;
  auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(cellTopo, cubDegree);

  const int Q = cub->getNumPoints();
  DRV refPts("refPts", Q, 1);
  DRV refWts("refWts", Q);
  cub->getCubature(refPts, refWts);

  // Basis values on reference points
  DRV wVals("wVals", Nw, Q);
  DRV tVals("tVals", Nt, Q);
  wBasis.getValues(wVals, refPts, Intrepid2::OPERATOR_VALUE);
  tBasis.getValues(tVals, refPts, Intrepid2::OPERATOR_VALUE);

  auto refWts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), refWts);
  auto wVals_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wVals);
  auto tVals_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tVals);

  host_tensor<2> M({(size_t)Nw, (size_t)Nt});
  auto Mv = M.view();

  for (int i = 0; i < Nw; ++i)
  {
    for (int j = 0; j < Nt; ++j)
    {
      double sum = 0.0;
      for (int q = 0; q < Q; ++q)
      {
        sum += refWts_h(q) * wVals_h(i,q) * tVals_h(j,q);
      }
      Mv({(size_t)i, (size_t)j}) = sum;
    }
  }

  return M;
}

inline void
compare_ref_mixed_mass_matrices(Camellia::BasisPtr w_basis,
                                Camellia::BasisPtr t_basis)
{
  reference_integrals_1d refInts;

  // Camellia mixed mass on reference cell
  host_tensor<2> M_cam = refInts.discretization(w_basis, false, t_basis, false);

  // Intrepid2 assembled mixed mass
  host_tensor<2> M_i2  = assemble_ref_mixed_mass_intrepid2(w_basis, t_basis);

  const size_t Nw = M_cam.sizes(0);
  const size_t Nt = M_cam.sizes(1);
  boba_assert(M_i2.sizes(0) == Nw && M_i2.sizes(1) == Nt, "mixed mass size mismatch");

  auto Mc = M_cam.view();
  auto Mi = M_i2.view();

  double maxAbsDiff = 0.0;
  double maxAbsCam  = 0.0;
  double maxAbsI2   = 0.0;
  size_t maxi=0, maxj=0;

  for (size_t i = 0; i < Nw; ++i)
  {
    for (size_t j = 0; j < Nt; ++j)
    {
      const double a = Mc({i,j});
      const double b = Mi({i,j});
      const double d = std::fabs(a - b);

      if (d > maxAbsDiff) { maxAbsDiff = d; maxi=i; maxj=j; }
      maxAbsCam = std::max(maxAbsCam, std::fabs(a));
      maxAbsI2  = std::max(maxAbsI2,  std::fabs(b));
    }
  }

  const double denom = std::max(1.0, std::max(maxAbsCam, maxAbsI2));
  const double rel = maxAbsDiff / denom;

  std::cout << std::setprecision(16);
  std::cout << "compare_ref_mixed_mass_matrices:\n"
            << "  Nw = " << Nw << ", Nt = " << Nt << "\n"
            << "  maxAbsCam  = " << maxAbsCam << "\n"
            << "  maxAbsI2   = " << maxAbsI2  << "\n"
            << "  maxAbsDiff = " << maxAbsDiff << " at (" << maxi << "," << maxj << ")\n"
            << "  relDiff    = " << rel << "\n"
            << "  Camellia("<<maxi<<","<<maxj<<") = " << Mc({maxi,maxj}) << "\n"
            << "  I2     ("<<maxi<<","<<maxj<<") = " << Mi({maxi,maxj}) << "\n";
}

void print_endpoint_basis_ordinals(Camellia::BasisPtr w_basis)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using Scalar = double;
  using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

  const int N = w_basis->getCardinality();

  // Build TensorPoints on device from a DynRankView
  DRV ptsView("ptsView", 2, 1);
  auto ptsHost = Kokkos::create_mirror_view(ptsView);
  ptsHost(0,0) = -1.0;
  ptsHost(1,0) =  1.0;
  Kokkos::deep_copy(ptsView, ptsHost);

  // Construct TensorPoints wrapper (this is the step you mentioned)
  Intrepid2::TensorPoints<Scalar, ExecSpace> pts(ptsView);

  auto vals = w_basis->allocateBasisValues(pts, Intrepid2::OPERATOR_VALUE);

  // Fill vals
  w_basis->getValues(vals, pts, Intrepid2::OPERATOR_VALUE);

  auto valsHost = vals; // assume we're not on device for now

  std::cout << "Endpoint basis values: N=" << N << "\n";
  int leftMaxI = -1, rightMaxI = -1;
  double leftMax = 0.0, rightMax = 0.0;

  for (int i = 0; i < N; ++i)
  {
    const double vL = valsHost(i,0);
    const double vR = valsHost(i,1);
    std::cout << "  i=" << i << "  phi(-1)=" << vL << "  phi(+1)=" << vR << "\n";

    if (std::fabs(vL) > leftMax)  { leftMax  = std::fabs(vL); leftMaxI  = i; }
    if (std::fabs(vR) > rightMax) { rightMax = std::fabs(vR); rightMaxI = i; }
  }

  std::cout << "  max |phi(-1)| at i=" << leftMaxI  << " val=" << leftMax  << "\n"
            << "  max |phi(+1)| at i=" << rightMaxI << " val=" << rightMax << "\n";
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

  auto field_basis_1d = newBasis1d(Camellia::L2,    p     );
  auto trace_basis_1d = newBasis1d(Camellia::L2,    p + 1 );
  auto test_basis_1d  = newBasis1d(Camellia::HGRAD, p + dp);

  {
    compare_ref_mass_matrices(field_basis_1d);
    compare_ref_mass_matrices(test_basis_1d);
    compare_ref_mass_matrices(trace_basis_1d);
    
    compare_ref_mixed_mass_matrices(test_basis_1d, trace_basis_1d);
    compare_ref_mixed_mass_matrices(test_basis_1d, field_basis_1d);
    
    // pick the 1D bases used in v-direction in your operator:
    auto w_basis_1d = test_basis_1d;   // same as in construct_vlasov_matrices()
    auto f_basis_1d = field_basis_1d;

    test_global_field_assembly_v_weighted(w_basis_1d, f_basis_1d, nv, v_min, v_max);
    
    test_1d_mass_consistency(field_basis_1d, /*ne=*/8, 0.0, 1.0);
    
    print_endpoint_basis_ordinals(w_basis_1d);
    
    test_1d_trace_coupling_consistency(test_basis_1d, trace_basis_1d, /*ne=*/8, 0.0, 1.0);
    
    test_1d_trace_coupling_consistency(test_basis_1d, trace_basis_1d, /*ne=*/8, -0.5, 0.5);
    
    test_boba_tt_vector_ordering();
    test_boba_ttm_mode_action();
    
    test_1d_moments_vs_coeffs(test_basis_1d);
  }
  
  auto   test_basis = newBasisNd<3>({ test_basis_1d,  test_basis_1d,  test_basis_1d});
  auto tracex_basis = newBasisNd<3>({ Teuchos::null, trace_basis_1d, trace_basis_1d});
  auto tracet_basis = newBasisNd<3>({trace_basis_1d,  Teuchos::null, trace_basis_1d});
  auto tracev_basis = newBasisNd<3>({trace_basis_1d, trace_basis_1d,  Teuchos::null});
  auto  field_basis = newBasisNd<3>({field_basis_1d, field_basis_1d, field_basis_1d});

  auto one_1d = newConstant1d(1.0);
  auto one_3d = newFunctionNd<3>({one_1d, one_1d, one_1d});

  auto rhs1 = one_3d->getDiscretization(test_basis, mesh);    // should be moments (nonzero)
  auto exp1 = one_3d->getBasisExpansion(test_basis, mesh);    // coefficients (also nonzero, but different)

  std::cout << "DEBUG one_3d discretization sizes=" << rhs1.sizes()
            << " ranks=" << rhs1.ranks_string()
            << " norm=" << rhs1.norm_frobenius() << "\n";

  std::cout << "DEBUG one_3d expansion      sizes=" << exp1.sizes()
            << " ranks=" << exp1.ranks_string()
            << " norm=" << exp1.norm_frobenius() << "\n";
  
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
            << "  ranks = " << exact_soln(vlasov_vars::test).ranks_string() << std::endl
            << "  norm  = " << exact_soln(vlasov_vars::test).norm_frobenius() << std::endl << std::endl;

  std::cout << "exact_tnx" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::trace_x).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::trace_x).ranks_string() << std::endl
            << "  norm  = " << exact_soln(vlasov_vars::trace_x).norm_frobenius() << std::endl << std::endl;

  std::cout << "exact_tnt" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::trace_t).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::trace_t).ranks_string() << std::endl
            << "  norm  = " << exact_soln(vlasov_vars::trace_t).norm_frobenius() << std::endl << std::endl;

  std::cout << "exact_tnv" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::trace_v).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::trace_v).ranks_string() << std::endl
            << "  norm  = " << exact_soln(vlasov_vars::trace_v).norm_frobenius() << std::endl << std::endl;

  std::cout << "exact_f" << std::endl
            << "  sizes = " << exact_soln(vlasov_vars::field).sizes() << std::endl
            << "  ranks = " << exact_soln(vlasov_vars::field).ranks_string() << std::endl
            << "  norm  = " << exact_soln(vlasov_vars::field).norm_frobenius() << std::endl << std::endl;

  std::cout << "exact_residual" << std::endl;
  for (size_t i = 0; i < vlasov_vars::size; ++i) {
    std::cout << "  component_" << i << std::endl 
              << "    sizes = " << exact_residual(i).sizes() << std::endl
              << "    ranks = " << exact_residual(i).ranks_string() << std::endl
              << "    norm  = " << exact_residual(i).norm_frobenius() << std::endl;
  }
  std::cout << std::endl;

  {
    auto Bf = stiffness({vlasov_vars::test, vlasov_vars::field});
    auto Bx = stiffness({vlasov_vars::test, vlasov_vars::trace_x});
    auto Bt = stiffness({vlasov_vars::test, vlasov_vars::trace_t});
    auto Bv = stiffness({vlasov_vars::test, vlasov_vars::trace_v});

    auto rf = Bf * exact_soln(vlasov_vars::field);
    auto rx = Bx * exact_soln(vlasov_vars::trace_x);
    auto rt = Bt * exact_soln(vlasov_vars::trace_t);
    auto rv = Bv * exact_soln(vlasov_vars::trace_v);

    auto r0 = rhs(vlasov_vars::test);

    auto res = rx + rt + rv + rf - r0;
    res.round();

    std::cout << "DEBUG norms:\n"
              << "  ||rhs_test|| = " << r0.norm_frobenius() << "\n"
              << "  ||Bf f||     = " << rf.norm_frobenius() << "\n"
              << "  ||Bx tx||    = " << rx.norm_frobenius() << "\n"
              << "  ||Bt tt||    = " << rt.norm_frobenius() << "\n"
              << "  ||Bv tv||    = " << rv.norm_frobenius() << "\n"
              << "  ||sum-rhs||  = " << res.norm_frobenius() << "\n";
  }
  
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
    // pick a basis to test
    std::size_t p = 3;
    auto basis1d = newBasis1d(Camellia::L2, p);
    
    // Call the test (choose any element)
    // f(x) = a + b x, element [xL, xL+h]
    double a  = 0.7;
    double b  = -1.3;
    double xL = 0.2;
    double h  = 0.5;
    
    test_reproduce_line(basis1d, a, b, xL, h);
  }
  
  {
    auto basis1d = newBasis1d(Camellia::L2, 4); // degree >= 3

    double a=0.1, b=-0.7, c=2.0, d=-1.5;
    double xL=0.2, h=0.5;

    test_reproduce_cubic(basis1d, a,b,c,d, xL,h);
  }
  
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
