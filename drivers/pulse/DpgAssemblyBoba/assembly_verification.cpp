#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <vector>
#include <stdexcept>

#include "Camellia.h"
#include "CubatureFactory.h"

#include "BOBA/boba.hpp"
#include "reference_integrals_1d.hpp"
#include "global_assembly.hpp"
#include "boba_types.hpp"
#include "Projection1d.hpp"


// brute force assemble b_w(e,i) = \int_{K_e} w_i(x) * g(x) dx
host_vector
assemble_mixed_rhs_bruteforce(Camellia::BasisPtr w_basis,
                              std::size_t ne,
                              double x_min,
                              double x_max,
                              const std::function<double(double)>& g,
                              int cubatureDegree = -1)
{
  const std::size_t Nw = w_basis->getCardinality();
  const double h = (x_max - x_min) / (double)ne;

  if (cubatureDegree < 0) {
    cubatureDegree = 2 * w_basis->getDegree() + 6;
  }

  auto cub = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubatureDegree);
  auto pts = cub->allocateCubaturePoints();
  auto wts = cub->allocateCubatureWeights();
  cub->getCubature(pts, wts);
  const std::size_t Q = pts.extent(0);

  auto wVals = w_basis->allocateBasisValues(pts, Intrepid2::OPERATOR_VALUE);
  w_basis->getValues(wVals, pts, Intrepid2::OPERATOR_VALUE);

  host_vector b({ne * Nw});
  b.fill_with_zeros();
  auto bv = b.view();

  for (std::size_t e = 0; e < ne; ++e) {
    const double xL = x_min + e * h;
    const double J  = 0.5 * h;

    auto x_of_xi = [&](double xi) {
      return xL + 0.5 * (xi + 1.0) * h;
    };

    for (std::size_t i = 0; i < Nw; ++i) {
      double sum = 0.0;
      for (std::size_t q = 0; q < Q; ++q) {
        const double xi = pts(q,0);
        const double x  = x_of_xi(xi);
        sum += (double)wts(q) * (double)wVals(i,q) * g(x) * J;
      }
      bv(e * Nw + i) = sum;
    }
  }

  return b;
}

// dense matvec: y = A x
host_vector
matvec_dense(const host_matrix& A, const host_vector& x)
{
  const std::size_t m = A.sizes(0);
  const std::size_t n = A.sizes(1);

  host_vector y({m});
  y.fill_with_zeros();

  auto Av = A.view();
  auto xv = x.view();
  auto yv = y.view();

  for (std::size_t i = 0; i < m; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      sum += Av({i,j}) * xv(j);
    }
    yv(i) = sum;
  }
  return y;
}

// Main test: does (M_{w,t} * c_t) equal b_w, where c_t comes from projection onto trace basis?
void
test_1d_trace_coupling_consistency(Camellia::BasisPtr w_basis,
                                  Camellia::BasisPtr t_basis,
                                  std::size_t ne,
                                  double x_min,
                                  double x_max)
{
  const std::size_t Nw = w_basis->getCardinality();
  const std::size_t Nt = t_basis->getCardinality();
  const double h = (x_max - x_min) / (double)ne;

  // Pick a polynomial g(x) that should be representable by t_basis on each element
  // Adjust degree if you like; this is cubic-ish.
  auto g = [](double x) { return 0.4 + 0.2*x - 0.7*x*x + 0.1*x*x*x; };

  // Build trace coeff vector c (global ordering: element-major then local)
  host_vector c({ne * Nt});
  c.fill_with_zeros();
  auto cv = c.view();

  for (std::size_t e = 0; e < ne; ++e) {
    const double xL = x_min + e * h;
    auto coeffs = Camellia::Projection1d::project_onto_1d_basis_ref(t_basis, g, xL, h);
    for (std::size_t j = 0; j < Nt; ++j) {
      cv(e * Nt + j) = coeffs[j];
    }
  }

  // Assemble mixed matrix using your Camellia reference integrals + global assembly
  reference_integrals_1d refInts;
  auto Mwt_ref = refInts.discretization(w_basis, false, t_basis, false, 0);
  host_matrix Mwt = global_field_assembly_1d(Mwt_ref, ne, x_min, x_max, 0);

  // Compute b_w by brute force and by Mwt*c
  host_vector b_bf   = assemble_mixed_rhs_bruteforce(w_basis, ne, x_min, x_max, g);
  host_vector b_code = matvec_dense(Mwt, c);

  // Compare
  auto b1 = b_bf.view();
  auto b2 = b_code.view();

  double maxDiff=0.0, maxAbs=0.0;
  std::size_t imax=0;

  for (std::size_t i = 0; i < ne * Nw; ++i) {
    const double d = std::fabs(b1(i) - b2(i));
    if (d > maxDiff) { maxDiff = d; imax = i; }
    maxAbs = std::max(maxAbs, std::fabs(b1(i)));
  }

  std::cout << std::setprecision(16);
  std::cout << "test_1d_trace_coupling_consistency: ne=" << ne
            << " Nw=" << Nw << " Nt=" << Nt << "\n"
            << "  maxDiff=" << maxDiff << " at i=" << imax << "\n"
            << "  rel=" << (maxDiff / std::max(1.0, maxAbs)) << "\n"
            << "  b_bf=" << b1(imax) << "  b_code=" << b2(imax) << "\n";
}

host_matrix assemble_mass_bruteforce(Camellia::BasisPtr basis,
                                     std::size_t ne,
                                     double x_min,
                                     double x_max,
                                     int cubatureDegree)
{
  const std::size_t N = basis->getCardinality();
  const double h = (x_max - x_min) / (double)ne;

  if (cubatureDegree < 0) {
    cubatureDegree = 2 * basis->getDegree() + 4;
  }

  auto cub = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubatureDegree);
  auto pts = cub->allocateCubaturePoints();
  auto wts = cub->allocateCubatureWeights();
  cub->getCubature(pts, wts);
  const std::size_t Q = pts.extent(0);

  auto vals = basis->allocateBasisValues(pts, Intrepid2::OPERATOR_VALUE);
  basis->getValues(vals, pts, Intrepid2::OPERATOR_VALUE);

  host_matrix M({ne * N, ne * N});
  M.fill_with_zeros();
  auto Mv = M.view();

  for (std::size_t e=0; e<ne; ++e) {
    const double xL = x_min + e * h;
    const double J  = 0.5 * h;

    for (std::size_t i=0; i<N; ++i) {
      for (std::size_t j=0; j<N; ++j) {
        double sum = 0.0;
        for (std::size_t q=0; q<Q; ++q) {
          sum += (double)wts(q) * (double)vals(i,q) * (double)vals(j,q) * J;
        }
        Mv({e*N+i, e*N+j}) = sum;
      }
    }
  }
  return M;
}

host_vector assemble_rhs_bruteforce(Camellia::BasisPtr basis,
                                   std::size_t ne,
                                   double x_min,
                                   double x_max,
                                   const std::function<double(double)>& f,
                                   int cubatureDegree = -1)
{
  const std::size_t N = basis->getCardinality();
  const double h = (x_max - x_min) / (double)ne;

  if (cubatureDegree < 0) {
    cubatureDegree = 2 * basis->getDegree() + 6;
  }

  auto cub = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubatureDegree);
  auto pts = cub->allocateCubaturePoints();
  auto wts = cub->allocateCubatureWeights();
  cub->getCubature(pts, wts);
  const std::size_t Q = pts.extent(0);

  auto vals = basis->allocateBasisValues(pts, Intrepid2::OPERATOR_VALUE);
  basis->getValues(vals, pts, Intrepid2::OPERATOR_VALUE);

  host_vector b({ne * N});
  b.fill_with_zeros();
  auto bv = b.view();

  for (std::size_t e=0; e<ne; ++e) {
    const double xL = x_min + e * h;
    const double J  = 0.5 * h;

    auto x_of_xi = [&](double xi) {
      return xL + 0.5*(xi+1.0)*h;
    };

    for (std::size_t i=0; i<N; ++i) {
      double sum = 0.0;
      for (std::size_t q=0; q<Q; ++q) {
        const double xi = pts(q,0);
        const double x  = x_of_xi(xi);
        sum += (double)wts(q) * (double)vals(i,q) * f(x) * J;
      }
      bv(e*N + i) = sum;
    }
  }
  return b;
}

host_vector matvec(const host_matrix& A, const host_vector& x)
{
  auto Av = A.view();
  auto xv = x.view();
  host_vector y({A.sizes(0)});
  y.fill_with_zeros();
  auto yv = y.view();

  for (std::size_t i=0; i<A.sizes(0); ++i) {
    double sum = 0.0;
    for (std::size_t j=0; j<A.sizes(1); ++j) {
      sum += Av({i,j}) * xv(j);
    }
    yv(i) = sum;
  }
  return y;
}

void test_1d_mass_consistency(Camellia::BasisPtr basis,
                              std::size_t ne,
                              double x_min,
                              double x_max)
{
  // Choose f(x) representable in your space; e.g. cubic
  auto f = [](double x){ return 0.3 + 0.1*x - 0.7*x*x + 0.2*x*x*x; };

  // Build c by per-element projection (your routine)
  // We'll assemble c in the same global ordering e*N + i.
  const std::size_t N = basis->getCardinality();
  const double h = (x_max - x_min) / (double)ne;

  host_vector c({ne * N});
  c.fill_with_zeros();
  auto cv = c.view();

  for (std::size_t e=0; e<ne; ++e) {
    const double xL = x_min + e*h;
    auto coeffs = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h); // uses reference projection
    for (std::size_t i=0; i<N; ++i) cv(e*N+i) = coeffs[i];
  }

  // M from your global assembly
  reference_integrals_1d refInts;
  auto M0 = refInts.discretization(basis, false, basis, false, 0);
  host_matrix M_code = global_field_assembly_1d(M0, ne, x_min, x_max, 0);

  // b brute force and b_code = M_code * c
  host_vector b_bf   = assemble_rhs_bruteforce(basis, ne, x_min, x_max, f);
  host_vector b_code = matvec(M_code, c);

  // compare
  auto b1 = b_bf.view();
  auto b2 = b_code.view();
  double maxDiff=0, maxB=0;
  std::size_t imax=0;
  for (std::size_t i=0; i<ne*N; ++i) {
    const double d = std::fabs(b1(i) - b2(i));
    if (d > maxDiff) { maxDiff=d; imax=i; }
    maxB = std::max(maxB, std::fabs(b1(i)));
  }

  std::cout << std::setprecision(16)
            << "test_1d_mass_consistency: ne="<<ne<<" N="<<N<<"\n"
            << "  maxDiff="<<maxDiff<<" at i="<<imax<<"\n"
            << "  rel="<< (maxDiff / std::max(1.0, maxB)) << "\n"
            << "  b_bf="<< b1(imax) << "  b_code="<< b2(imax) << "\n";
}

// Assemble global A = \int w(v) * v * f(v) dv directly by element quadrature.
// Returns a dense host_matrix of size (nv*Nw) x (nv*Nf).
host_matrix
assemble_v_weighted_bruteforce(Camellia::BasisPtr w_basis,
                               Camellia::BasisPtr f_basis,
                               std::size_t nv,
                               double v_min,
                               double v_max,
                               int cubatureDegree = -1)
{
  const std::size_t Nw = w_basis->getCardinality();
  const std::size_t Nf = f_basis->getCardinality();

  const double h = (v_max - v_min) / (double)nv;

  // choose a safe cubature degree if not provided
  if (cubatureDegree < 0) {
    const int pw = w_basis->getDegree();
    const int pf = f_basis->getDegree();
    // integrand is (poly pw) * (linear v) * (poly pf) => degree pw+pf+1 in physical v,
    // still polynomial in reference xi.
    cubatureDegree = pw + pf + 3; // margin
  }

  auto cubature = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubatureDegree);
  auto cubPts = cubature->allocateCubaturePoints();   // (Q,1)
  auto cubWts = cubature->allocateCubatureWeights();  // (Q)
  cubature->getCubature(cubPts, cubWts);
  const std::size_t Q = cubPts.extent(0);

  // basis values on reference points: (N,Q)
  auto wVals = w_basis->allocateBasisValues(cubPts, Intrepid2::OPERATOR_VALUE);
  auto fVals = f_basis->allocateBasisValues(cubPts, Intrepid2::OPERATOR_VALUE);
  w_basis->getValues(wVals, cubPts, Intrepid2::OPERATOR_VALUE);
  f_basis->getValues(fVals, cubPts, Intrepid2::OPERATOR_VALUE);

  host_matrix A({nv * Nw, nv * Nf});
  A.fill_with_zeros();
  auto Av = A.view();

  // For each element, compute local matrix and place on block diagonal
  for (std::size_t e = 0; e < nv; ++e) {
    const double vL = v_min + e * h;
    const double J  = 0.5 * h; // dv/dxi

    for (std::size_t i = 0; i < Nw; ++i) {
      for (std::size_t j = 0; j < Nf; ++j) {

        double sum = 0.0;
        for (std::size_t q = 0; q < Q; ++q) {
          const double xi = cubPts(q,0);
          const double v  = vL + 0.5 * (xi + 1.0) * h;
          sum += (double)cubWts(q) * (double)wVals(i,q) * v * (double)fVals(j,q) * J;
        }

        Av({e * Nw + i, e * Nf + j}) = sum;
      }
    }
  }

  return A;
}

void test_1d_moments_vs_coeffs(Camellia::BasisPtr basis)
{
  const double xL = 0.0;
  const double h  = 0.25;

  auto f = [](double x){ return 0.4 + 0.2*x - 0.7*x*x + 0.1*x*x*x; };

  auto c = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);
  auto b = Camellia::Projection1d::moments_onto_1d_basis_phys(basis, f, xL, h);

  // build M_ref via reference_integrals_1d (consistent with your assembly)
  reference_integrals_1d refInts;
  auto Mref_t = refInts.discretization(basis, false, basis, false, 0);
  auto Mref = Mref_t.view();

  const size_t N = basis->getCardinality();
  const double J = 0.5*h;

  double maxDiff = 0.0;
  for (size_t i=0; i<N; ++i) {
    double sum = 0.0;
    for (size_t j=0; j<N; ++j) {
      sum += (J * Mref({i,j})) * c[j];
    }
    maxDiff = std::max(maxDiff, std::fabs(sum - b[i]));
  }

  std::cout << "test_1d_moments_vs_coeffs maxDiff = " << maxDiff << "\n";
}

void test_boba_tt_vector_ordering()
{
  host_vector ux({2});
  host_vector ut({3});
  host_vector uv({2});

  auto uxv = ux.view(); uxv(0)=1.0;  uxv(1)=2.0;
  auto utv = ut.view(); utv(0)=10.0; utv(1)=20.0; utv(2)=30.0;
  auto uvv = uv.view(); uvv(0)=100.0; uvv(1)=200.0;

  boba::Array<host_vector, 3> factors{ux, ut, uv};
  auto tt = boba::make_tt_from_vectors(factors);

  std::cout << "sizes = " << tt.sizes() << "\n";
  std::cout << "ranks = " << tt.ranks_string() << "\n";

  const double expected = uxv(1) * utv(2) * uvv(1); // 2*30*200 = 12000
  const double got = tt.unroll_value({1,2,1});
  std::cout << "u(1,2,1): expected=" << expected << " got=" << got
            << " diff=" << std::fabs(expected-got) << "\n";
}

void test_boba_ttm_mode_action()
{
  // Same factors as above
  host_vector ux({2}); host_vector ut({3}); host_vector uv({2});
  auto uxv = ux.view(); uxv(0)=1.0;  uxv(1)=2.0;
  auto utv = ut.view(); utv(0)=10.0; utv(1)=20.0; utv(2)=30.0;
  auto uvv = uv.view(); uvv(0)=100.0; uvv(1)=200.0;

  boba::Array<host_vector, 3> factors{ux, ut, uv};
  auto u = boba::make_tt_from_vectors<3>(factors);

  // Build D that scales first component differently than second, etc.
  host_matrix Dx({2,2}); Dx.fill_with_zeros();
  auto Dxv = Dx.view(); Dxv({0,0})=2.0; Dxv({1,1})=5.0;

  host_matrix Dt({3,3}); Dt.fill_with_zeros();
  auto Dtv = Dt.view(); Dtv({0,0})=3.0; Dtv({1,1})=7.0; Dtv({2,2})=11.0;

  host_matrix Dv({2,2}); Dv.fill_with_zeros();
  auto Dvv = Dv.view(); Dvv({0,0})=13.0; Dvv({1,1})=17.0;

  // Identity matrices
  host_matrix Ix({2,2}); Ix.fill_with_zeros();
  auto Ixv=Ix.view(); Ixv({0,0})=1.0; Ixv({1,1})=1.0;

  host_matrix It({3,3}); It.fill_with_zeros();
  auto Itv=It.view(); Itv({0,0})=1.0; Itv({1,1})=1.0; Itv({2,2})=1.0;

  host_matrix Iv({2,2}); Iv.fill_with_zeros();
  auto Ivv=Iv.view(); Ivv({0,0})=1.0; Ivv({1,1})=1.0;

  // Apply three Kronecker operators
  auto Kx = boba::make_ttm_from_matrices<3, host_space, double, size_t>({Dx, It, Iv});
  auto Kt = boba::make_ttm_from_matrices<3, host_space, double, size_t>({Ix, Dt, Iv});
  auto Kv = boba::make_ttm_from_matrices<3, host_space, double, size_t>({Ix, It, Dv});

  auto ux_out = Kx * u;
  auto ut_out = Kt * u;
  auto uv_out = Kv * u;

  // Probe a single entry and compare expected scaling
  // pick idx=(1,2,1): expected u = 2 * 30 * 200 = 12000
  const boba::Array<size_t,3> idx{1,2,1};
  const double u0 = u.unroll_value(idx);

  const double got_x = ux_out.unroll_value(idx);
  const double got_t = ut_out.unroll_value(idx);
  const double got_v = uv_out.unroll_value(idx);

  // Expected scalings if order is (x,t,v):
  // Kx scales by Dx(1,1)=5
  // Kt scales by Dt(2,2)=11
  // Kv scales by Dv(1,1)=17
  std::cout << std::setprecision(16);
  std::cout << "test_boba_ttm_mode_action:\n";
  std::cout << "  u(idx)=" << u0 << "\n";
  std::cout << "  (Dx⊗I⊗I)u(idx) got=" << got_x << " expected=" << (5.0*u0) << "\n";
  std::cout << "  (I⊗Dt⊗I)u(idx) got=" << got_t << " expected=" << (11.0*u0) << "\n";
  std::cout << "  (I⊗I⊗Dv)u(idx) got=" << got_v << " expected=" << (17.0*u0) << "\n";
}

void
test_global_field_assembly_v_weighted(Camellia::BasisPtr w_basis,
                                      Camellia::BasisPtr f_basis,
                                      std::size_t nv,
                                      double v_min,
                                      double v_max)
{
  reference_integrals_1d refInts;

  // Reference integrals:
  // M0_ij = \int_{-1}^1 w_i f_j dxi
  // M1_ij = \int_{-1}^1 xi w_i f_j dxi  (monomial_degree=1)
  auto M0 = refInts.discretization(w_basis, false, f_basis, false, 0);
  auto M1 = refInts.discretization(w_basis, false, f_basis, false, 1);

  // Your assembled global matrix
  host_matrix A_code = global_field_assembly_1d(M0, M1, nv, v_min, v_max, /*num_derivative=*/0);

  // Brute force assembled global matrix
  host_matrix A_bf = assemble_v_weighted_bruteforce(w_basis, f_basis, nv, v_min, v_max);

  // Compare
  auto Ac = A_code.view();
  auto Ab = A_bf.view();

  const std::size_t nrow = A_code.sizes(0);
  const std::size_t ncol = A_code.sizes(1);

  if (A_bf.sizes(0) != nrow || A_bf.sizes(1) != ncol) {
    throw std::runtime_error("Dimension mismatch between A_code and A_bf");
  }

  double maxAbsDiff = 0.0, maxAbsCode = 0.0, maxAbsBf = 0.0;
  std::size_t imax=0, jmax=0;

  for (std::size_t i = 0; i < nrow; ++i) {
    for (std::size_t j = 0; j < ncol; ++j) {
      const double a = Ac({i,j});
      const double b = Ab({i,j});
      const double d = std::fabs(a - b);

      if (d > maxAbsDiff) { maxAbsDiff = d; imax=i; jmax=j; }
      maxAbsCode = std::max(maxAbsCode, std::fabs(a));
      maxAbsBf   = std::max(maxAbsBf,   std::fabs(b));
    }
  }

  const double denom = std::max(1.0, std::max(maxAbsCode, maxAbsBf));
  const double rel = maxAbsDiff / denom;

  std::cout << std::setprecision(16);
  std::cout << "test_global_field_assembly_v_weighted:\n"
            << "  nv = " << nv << "  v_min="<<v_min<<" v_max="<<v_max<<"\n"
            << "  rows="<<nrow<<" cols="<<ncol<<"\n"
            << "  maxAbsCode = " << maxAbsCode << "\n"
            << "  maxAbsBf   = " << maxAbsBf   << "\n"
            << "  maxAbsDiff = " << maxAbsDiff << " at ("<<imax<<","<<jmax<<")\n"
            << "  relDiff    = " << rel << "\n"
            << "  A_code("<<imax<<","<<jmax<<") = " << Ac({imax,jmax}) << "\n"
            << "  A_bf  ("<<imax<<","<<jmax<<") = " << Ab({imax,jmax}) << "\n";
}


