#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "Camellia.h"
#include "CubatureFactory.h"

#include "Basis1d.hpp"
#include "Projection1d.hpp"
#include "reference_integrals_1d.hpp"
#include "global_assembly.hpp"
#include "assembly_verification.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Camellia;

namespace
{
  enum class BasisFamilyChoice
  {
    Nodal,
    Hierarchical
  };

  std::string famLabel(BasisFamilyChoice fam)
  {
    return (fam == BasisFamilyChoice::Nodal) ? "Nodal" : "Hierarchical";
  }

  std::string spaceLabel(Camellia::Space sp)
  {
    switch (sp)
    {
      case Camellia::L2:    return "L2";
      case Camellia::HGRAD: return "HGRAD";
      case Camellia::HDIV:  return "HDIV";
      case Camellia::HCURL: return "HCURL";
      default:              return "Space(" + std::to_string((int)sp) + ")";
    }
  }

  static std::pair<int,int>
  lineEndpointDofOrdinalsFromTags(Camellia::BasisPtr basis)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(basis == Teuchos::null, std::invalid_argument,
                               "lineEndpointDofOrdinalsFromTags: basis is null");
    auto i2Basis = basis->intrepid2Basis();
    TEUCHOS_TEST_FOR_EXCEPTION(i2Basis == Teuchos::null, std::invalid_argument,
                               "lineEndpointDofOrdinalsFromTags: intrepid2Basis() returned null");

    const int N = i2Basis->getCardinality();
    int left  = -1;
    int right = -1;

    for (int bfOrdinal = 0; bfOrdinal < N; ++bfOrdinal)
    {
      auto tag = i2Basis->getDofTag(bfOrdinal); // [subcellDim, subcellOrd, dofOrdOnSubcell, totalDofsOnSubcell]
      const int subcellDim      = tag[0];
      const int subcellOrd      = tag[1];
      const int dofOrdOnSubcell = tag[2];

      if ((subcellDim == 0) && (dofOrdOnSubcell == 0))
      {
        if (subcellOrd == 0) left  = bfOrdinal;
        if (subcellOrd == 1) right = bfOrdinal;
      }
    }
    return {left,right};
  }

  BasisPtr newBasis1d_family(Camellia::Space space, std::size_t h1Order, BasisFamilyChoice fam)
  {
    static auto cellTopo = Camellia::CellTopology::line();
    auto basisFamily = (fam == BasisFamilyChoice::Nodal) ?
      Camellia::BasisFamily::nodalBasisFamily() :
      Camellia::BasisFamily::hierarchicalBasisFamily();

    return basisFamily->getBasis(cellTopo, Camellia::efsForSpace(space), static_cast<int>(h1Order));
  }

// TODO: delete this -- just using for debugging the tests
  static host_tensor<2> assemble_ref_mixed_mass_intrepid2_from_basisPtrs(BasisPtr w, BasisPtr t)
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using Scalar    = double;
    using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

    auto& wB = *w->intrepid2Basis();
    auto& tB = *t->intrepid2Basis();

    const int Nw = wB.getCardinality();
    const int Nt = tB.getCardinality();

    const int pw = wB.getDegree();
    const int pt = tB.getDegree();
    const int cubDegree = pw + pt + 6;

    Intrepid2::DefaultCubatureFactory cubFactory;
    auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(wB.getBaseCellTopology(), cubDegree);

    const int Q = cub->getNumPoints();
    DRV pts("pts", Q, 1);
    DRV wts("wts", Q);
    cub->getCubature(pts, wts);

    DRV wVals("wVals", Nw, Q);
    DRV tVals("tVals", Nt, Q);
    wB.getValues(wVals, pts, Intrepid2::OPERATOR_VALUE);
    tB.getValues(tVals, pts, Intrepid2::OPERATOR_VALUE);

    auto wts_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wts);
    auto wVals_h= Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wVals);
    auto tVals_h= Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tVals);

    host_tensor<2> M({(size_t)Nw,(size_t)Nt});
    auto Mv = M.view();
    for (int i=0;i<Nw;++i)
      for (int j=0;j<Nt;++j)
      {
        double sum=0;
        for (int q=0;q<Q;++q) sum += wts_h(q)*wVals_h(i,q)*tVals_h(j,q);
        Mv({(size_t)i,(size_t)j}) = sum;
      }
    return M;
  }

  // Assemble reference mass matrix using Intrepid2 directly (host-side reduction).
  template<class Basis1D>
  host_tensor<2> assemble_ref_mass_intrepid2(Teuchos::RCP<Basis1D> basisPtr)
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using Scalar    = double;
    using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

    auto& basis = *basisPtr->intrepid2Basis();
    auto cellTopo = basis.getBaseCellTopology();
    const int N = basis.getCardinality();

    const int p = basis.getDegree();
    const int cubDegree = 2 * p + 2;

    Intrepid2::DefaultCubatureFactory cubFactory;
    auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(cellTopo, cubDegree);

    const int Q = cub->getNumPoints();
    DRV refPts("refPts", Q, 1);
    DRV refWts("refWts", Q);
    cub->getCubature(refPts, refWts);

    DRV phi("phi", N, Q);
    basis.getValues(phi, refPts, Intrepid2::OPERATOR_VALUE);

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

  template<class Basis1D_w, class Basis1D_t>
  host_tensor<2> assemble_ref_mixed_mass_intrepid2(Teuchos::RCP<Basis1D_w> wBasisPtr,
                                                  Teuchos::RCP<Basis1D_t> tBasisPtr)
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using Scalar    = double;
    using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

    auto& wBasis = *wBasisPtr->intrepid2Basis();
    auto& tBasis = *tBasisPtr->intrepid2Basis();

    const int Nw = wBasis.getCardinality();
    const int Nt = tBasis.getCardinality();

    auto cellTopo = wBasis.getBaseCellTopology();

    const int pw = wBasis.getDegree();
    const int pt = tBasis.getDegree();
    const int cubDegree = pw + pt + 2;

    Intrepid2::DefaultCubatureFactory cubFactory;
    auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(cellTopo, cubDegree);

    const int Q = cub->getNumPoints();
    DRV refPts("refPts", Q, 1);
    DRV refWts("refWts", Q);
    cub->getCubature(refPts, refWts);

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

  struct MatrixDiff
  {
    double maxAbsDiff = 0.0;
    double maxAbsA = 0.0;
    double maxAbsB = 0.0;
  };

  MatrixDiff diff_2d(const host_tensor<2>& A, const host_tensor<2>& B)
  {
    MatrixDiff d;
    const size_t m = A.sizes(0);
    const size_t n = A.sizes(1);
    if (B.sizes(0) != m || B.sizes(1) != n) throw std::runtime_error("diff_2d: size mismatch");

    auto Av = A.view();
    auto Bv = B.view();
    for (size_t i=0; i<m; ++i)
    {
      for (size_t j=0; j<n; ++j)
      {
        const double a = Av({i,j});
        const double b = Bv({i,j});
        d.maxAbsDiff = std::max(d.maxAbsDiff, std::fabs(a-b));
        d.maxAbsA    = std::max(d.maxAbsA,    std::fabs(a));
        d.maxAbsB    = std::max(d.maxAbsB,    std::fabs(b));
      }
    }
    return d;
  }

  void testGlobalTraceAssemblyUsesEndpointTags(BasisFamilyChoice fam,
                                               Teuchos::FancyOStream &out, bool &success)
  {
    const size_t nx = 3;
    const auto bc = bc_kind_1d::both;

    auto w_basis = newBasis1d_family(Camellia::HGRAD, /*h1Order=*/4, fam);
    const size_t nb = w_basis->getCardinality();

    auto [leftOrd, rightOrd] = lineEndpointDofOrdinalsFromTags(w_basis);

    TEST_ASSERT(leftOrd >= 0);
    TEST_ASSERT(rightOrd >= 0);

    host_matrix E = global_trace_assembly_1d(w_basis, nx, bc);

    const size_t expectedCols = nx - 1;

    TEST_EQUALITY(E.sizes(0), nx * nb);
    TEST_EQUALITY(E.sizes(1), expectedCols);

    auto Ev = E.view();

    const double tol = 1e-15;
    size_t nnz = 0;

    for (size_t row = 0; row < E.sizes(0); ++row)
    {
      for (size_t col = 0; col < E.sizes(1); ++col)
      {
        const double val = Ev({row,col});
        if (std::fabs(val) > tol) nnz++;
      }
    }

    TEST_EQUALITY(nnz, 2 * (nx - 1));

    for (size_t j = 0; j < nx - 1; ++j)
    {
      const size_t rowPlus  = j     * nb + (size_t)rightOrd;
      const size_t rowMinus = (j+1) * nb + (size_t)leftOrd;

      TEST_FLOATING_EQUALITY(Ev({rowPlus,  j}),  1.0, 1e-15);
      TEST_FLOATING_EQUALITY(Ev({rowMinus, j}), -1.0, 1e-15);
    }

    // boundary rows should be uncoupled when bc=both
    const size_t boundaryLeftRow  = 0        * nb + (size_t)leftOrd;
    const size_t boundaryRightRow = (nx - 1) * nb + (size_t)rightOrd;

    for (size_t col = 0; col < E.sizes(1); ++col)
    {
      TEST_FLOATING_EQUALITY(Ev({boundaryLeftRow,  col}), 0.0, 1e-15);
      TEST_FLOATING_EQUALITY(Ev({boundaryRightRow, col}), 0.0, 1e-15);
    }

    out << "GlobalTraceAssemblyUsesEndpointTags passed for fam=" << famLabel(fam) << "\n";
  }

  void testTraceCouplingConsistency(BasisFamilyChoice fam,
                                  Teuchos::FancyOStream &out, bool &success)
  {
    // Choose bases (match the intent of the previous nodal-only test)
    auto w_basis = newBasis1d_family(Camellia::HGRAD, 4, fam);
    auto t_basis = newBasis1d_family(Camellia::L2,    3, fam);

    // Two domains (mirroring the earlier test)
    std::vector<std::pair<double,double>> domains = { {0.0, 1.0}, {-0.5, 0.5} };

    // IMPORTANT: g must lie in span(t_basis) on each element for equality to hold.
    // With degree(t_basis)=2, choose g quadratic.
    auto g = [](double x) { return 0.4 + 0.2*x - 0.7*x*x; };

    for (auto dom : domains)
    {
      const std::size_t ne = 8;
      const double x_min = dom.first;
      const double x_max = dom.second;

      const std::size_t Nw = w_basis->getCardinality();
      const std::size_t Nt = t_basis->getCardinality();
      const double h = (x_max - x_min) / (double)ne;

      // Build trace coefficient vector c (element-major)
      host_vector c({ne * Nt});
      c.fill_with_zeros();
      auto cv = c.view();

      for (std::size_t e=0; e<ne; ++e)
      {
        const double xL = x_min + e*h;
        auto coeffs = Camellia::Projection1d::project_onto_1d_basis_ref(t_basis, g, xL, h);
        for (std::size_t j=0; j<Nt; ++j) cv(e*Nt + j) = coeffs[j];
      }

      // Assemble mixed matrix using reference integrals + global assembly
      reference_integrals_1d refInts;
      auto Mwt_ref = refInts.discretization(w_basis, false, t_basis, false, 0);
      host_matrix Mwt = global_field_assembly_1d(Mwt_ref, ne, x_min, x_max, 0);

      // Compare b_w by brute force and by Mwt*c
      host_vector b_bf   = assemble_mixed_rhs_bruteforce(w_basis, ne, x_min, x_max, g);
      host_vector b_code = matvec_dense(Mwt, c);

      auto b1 = b_bf.view();
      auto b2 = b_code.view();

      double maxDiff=0.0, maxAbs=0.0;
      for (std::size_t i = 0; i < ne * Nw; ++i)
      {
        maxDiff = std::max(maxDiff, std::fabs(b1(i) - b2(i)));
        maxAbs  = std::max(maxAbs,  std::fabs(b1(i)));
      }

      const double rel = maxDiff / std::max(1.0, maxAbs);

      out << "fam=" << famLabel(fam)
          << " domain=[" << x_min << "," << x_max << "]"
          << " maxDiff=" << maxDiff << " rel=" << rel << "\n";

      // tight tolerances; g is representable
      TEST_ASSERT(maxDiff < 1e-11 || rel < 1e-11);
    }
  }

  void assertMatricesClose(const host_tensor<2>& A, const host_tensor<2>& B,
                           double relTol, double absTol,
                           Teuchos::FancyOStream& out, bool& success,
                           const std::string& label)
  {
    auto d = diff_2d(A,B);
    const double denom = std::max(1.0, std::max(d.maxAbsA, d.maxAbsB));
    const double rel = d.maxAbsDiff / denom;

    out << label << "\n"
        << "  maxAbsDiff = " << d.maxAbsDiff << "\n"
        << "  relDiff    = " << rel << "\n";

    TEST_ASSERT(d.maxAbsDiff <= absTol || rel <= relTol);
  }
} // namespace

// ------------------------------
// Projection reproduction tests
// ------------------------------

TEUCHOS_UNIT_TEST(DPGBobaAssembly, Projection_ReproducesLinear_Nodal_L2)
{
  auto basis = newBasis1d_family(Camellia::L2, /*h1Order=*/3, BasisFamilyChoice::Nodal);

  const double a  = 0.7;
  const double b  = -1.3;
  const double xL = 0.2;
  const double h  = 0.5;

  auto f = [&](double x){ return a + b*x; };
  auto c = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);

  auto x_of_xi = [&](double xi) { return xL + 0.5*(xi+1.0)*h; };
  const double xis[] = {-1.0, -0.3, 0.0, 0.7, 1.0};

  double maxErr = 0.0;
  for (double xi : xis)
  {
    const double exact = f(x_of_xi(xi));
    const double approx = Camellia::Projection1d::eval_reconstruction_ref(basis, c, xi);
    maxErr = std::max(maxErr, std::fabs(exact-approx));
  }

  out << "maxErr = " << maxErr << "\n";
  TEST_COMPARE(maxErr, <, 1e-12);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, Projection_ReproducesLinear_Hierarchical_L2)
{
  auto basis = newBasis1d_family(Camellia::L2, /*h1Order=*/3, BasisFamilyChoice::Hierarchical);

  const double a  = 0.7;
  const double b  = -1.3;
  const double xL = 0.2;
  const double h  = 0.5;

  auto f = [&](double x){ return a + b*x; };
  auto c = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);

  auto x_of_xi = [&](double xi) { return xL + 0.5*(xi+1.0)*h; };
  const double xis[] = {-1.0, -0.3, 0.0, 0.7, 1.0};

  double maxErr = 0.0;
  for (double xi : xis)
  {
    const double exact = f(x_of_xi(xi));
    const double approx = Camellia::Projection1d::eval_reconstruction_ref(basis, c, xi);
    maxErr = std::max(maxErr, std::fabs(exact-approx));
  }

  out << "maxErr = " << maxErr << "\n";
  TEST_COMPARE(maxErr, <, 1e-12);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, Projection_ReproducesCubic_Nodal_L2)
{
  auto basis = newBasis1d_family(Camellia::L2, /*h1Order=*/4, BasisFamilyChoice::Nodal); // degree>=3 typical

  const double a=0.1, b=-0.7, c=2.0, d=-1.5;
  const double xL=0.2, h=0.5;
  auto f = [&](double x){ return a + b*x + c*x*x + d*x*x*x; };

  auto coeffs = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);

  auto x_of_xi = [&](double xi) { return xL + 0.5*(xi+1.0)*h; };
  const double xis[] = {-1.0, -0.75, -0.3, 0.0, 0.2, 0.7, 1.0};

  double maxErr = 0.0;
  for (double xi : xis)
  {
    const double exact = f(x_of_xi(xi));
    const double approx = Camellia::Projection1d::eval_reconstruction_ref(basis, coeffs, xi);
    maxErr = std::max(maxErr, std::fabs(exact-approx));
  }

  out << "maxErr = " << maxErr << "\n";
  TEST_COMPARE(maxErr, <, 1e-12);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, Projection_ReproducesCubic_Hierarchical_L2)
{
  auto basis = newBasis1d_family(Camellia::L2, /*h1Order=*/4, BasisFamilyChoice::Hierarchical);

  const double a=0.1, b=-0.7, c=2.0, d=-1.5;
  const double xL=0.2, h=0.5;
  auto f = [&](double x){ return a + b*x + c*x*x + d*x*x*x; };

  auto coeffs = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);

  auto x_of_xi = [&](double xi) { return xL + 0.5*(xi+1.0)*h; };
  const double xis[] = {-1.0, -0.75, -0.3, 0.0, 0.2, 0.7, 1.0};

  double maxErr = 0.0;
  for (double xi : xis)
  {
    const double exact = f(x_of_xi(xi));
    const double approx = Camellia::Projection1d::eval_reconstruction_ref(basis, coeffs, xi);
    maxErr = std::max(maxErr, std::fabs(exact-approx));
  }

  out << "maxErr = " << maxErr << "\n";
  TEST_COMPARE(maxErr, <, 1e-12);
}

// ----------------------------------------
// Reference integral cache vs Intrepid2
// ----------------------------------------

TEUCHOS_UNIT_TEST(DPGBobaAssembly, RefMassMatchesIntrepid2_Nodal)
{
  const double relTol = 1e-12;
  const double absTol = 1e-12;

  // try a couple spaces/orders
  struct Case { Camellia::Space sp; int p; };
  std::vector<Case> cases = {
    {Camellia::L2, 4},
    {Camellia::HGRAD, 4},
  };

  for (auto c : cases)
  {
    auto basis = newBasis1d_family(c.sp, c.p, BasisFamilyChoice::Nodal);

    reference_integrals_1d refInts;
    auto M_cam = refInts.discretization(basis, false, basis, false);
    auto M_i2  = assemble_ref_mass_intrepid2(basis);

    assertMatricesClose(M_cam, M_i2, relTol, absTol, out, success,
                        "Ref mass: " + spaceLabel(c.sp) + " p=" + std::to_string(c.p) + " nodal");
  }
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, RefMassMatchesIntrepid2_Hierarchical)
{
  const double relTol = 1e-12;
  const double absTol = 1e-12;

  struct Case { Camellia::Space sp; int p; };
  std::vector<Case> cases = {
    {Camellia::L2, 4},
    {Camellia::HGRAD, 4},
  };

  for (auto c : cases)
  {
    auto basis = newBasis1d_family(c.sp, c.p, BasisFamilyChoice::Hierarchical);

    reference_integrals_1d refInts;
    auto M_cam = refInts.discretization(basis, false, basis, false);
    auto M_i2  = assemble_ref_mass_intrepid2(basis);

    assertMatricesClose(M_cam, M_i2, relTol, absTol, out, success,
                        "Ref mass: " + spaceLabel(c.sp) + " p=" + std::to_string(c.p) + " hierarchical");
  }
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, RefMixedMassMatchesIntrepid2_Nodal)
{
  const double relTol = 1e-12;
  const double absTol = 1e-12;

  auto w = newBasis1d_family(Camellia::HGRAD, 4, BasisFamilyChoice::Nodal);
  auto t = newBasis1d_family(Camellia::L2,    3, BasisFamilyChoice::Nodal);
  auto f = newBasis1d_family(Camellia::L2,    2, BasisFamilyChoice::Nodal);

  reference_integrals_1d refInts;
  auto Mwt_cam = refInts.discretization(w, false, t, false);
  auto Mwt_i2  = assemble_ref_mixed_mass_intrepid2(w, t);
  assertMatricesClose(Mwt_cam, Mwt_i2, relTol, absTol, out, success, "Ref mixed mass (w,t) nodal");

  auto Mwf_cam = refInts.discretization(w, false, f, false);
  auto Mwf_i2  = assemble_ref_mixed_mass_intrepid2(w, f);
  assertMatricesClose(Mwf_cam, Mwf_i2, relTol, absTol, out, success, "Ref mixed mass (w,f) nodal");
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, RefMixedMassMatchesIntrepid2_Hierarchical)
{
  const double relTol = 1e-12;
  const double absTol = 1e-12;

  auto w = newBasis1d_family(Camellia::HGRAD, 4, BasisFamilyChoice::Hierarchical);
  auto t = newBasis1d_family(Camellia::L2,    3, BasisFamilyChoice::Hierarchical);
  auto f = newBasis1d_family(Camellia::L2,    2, BasisFamilyChoice::Hierarchical);

  reference_integrals_1d refInts;
  auto Mwt_cam = refInts.discretization(w, false, t, false);
  auto Mwt_i2  = assemble_ref_mixed_mass_intrepid2(w, t);
  assertMatricesClose(Mwt_cam, Mwt_i2, relTol, absTol, out, success, "Ref mixed mass (w,t) hierarchical");

  auto Mwf_cam = refInts.discretization(w, false, f, false);
  auto Mwf_i2  = assemble_ref_mixed_mass_intrepid2(w, f);
  assertMatricesClose(Mwf_cam, Mwf_i2, relTol, absTol, out, success, "Ref mixed mass (w,f) hierarchical");
}

// ----------------------------------------
// Global assembly verification tests
// ----------------------------------------

TEUCHOS_UNIT_TEST(DPGBobaAssembly, GlobalFieldAssembly_VWeighted_MatchesBruteforce_NodalAndHierarchical)
{
  // This just invokes your existing check, but in unit-test form by asserting rel/abs thresholds.
  // We reproduce the maxDiff computation here instead of printing-only.

  auto run = [&](BasisFamilyChoice fam)
  {
    auto w_basis = newBasis1d_family(Camellia::HGRAD, 4, fam);
    auto f_basis = newBasis1d_family(Camellia::L2,    3, fam);

    const std::size_t nv = 4;
    const double v_min = -0.5;
    const double v_max =  0.5;

    reference_integrals_1d refInts;
    auto M0 = refInts.discretization(w_basis, false, f_basis, false, 0);
    auto M1 = refInts.discretization(w_basis, false, f_basis, false, 1);

    host_matrix A_code = global_field_assembly_1d(M0, M1, nv, v_min, v_max, 0);
    host_matrix A_bf   = assemble_v_weighted_bruteforce(w_basis, f_basis, nv, v_min, v_max);

    auto Ac = A_code.view();
    auto Ab = A_bf.view();

    double maxAbsDiff = 0.0, maxAbsCode = 0.0, maxAbsBf = 0.0;
    for (std::size_t i=0; i<A_code.sizes(0); ++i)
    {
      for (std::size_t j=0; j<A_code.sizes(1); ++j)
      {
        const double a = Ac({i,j});
        const double b = Ab({i,j});
        maxAbsDiff = std::max(maxAbsDiff, std::fabs(a-b));
        maxAbsCode = std::max(maxAbsCode, std::fabs(a));
        maxAbsBf   = std::max(maxAbsBf,   std::fabs(b));
      }
    }
    const double denom = std::max(1.0, std::max(maxAbsCode, maxAbsBf));
    const double rel = maxAbsDiff / denom;

    out << "fam=" << famLabel(fam) << " maxAbsDiff=" << maxAbsDiff << " rel=" << rel << "\n";
    TEST_ASSERT(maxAbsDiff < 1e-12 || rel < 1e-12);
  };

  run(BasisFamilyChoice::Nodal);
  run(BasisFamilyChoice::Hierarchical);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, MomentsVsCoeffs_Consistent_NodalAndHierarchical)
{
  auto run = [&](BasisFamilyChoice fam)
  {
    auto basis = newBasis1d_family(Camellia::HGRAD, 4, fam);

    const double xL = 0.0;
    const double h  = 0.25;
    auto f = [](double x){ return 0.4 + 0.2*x - 0.7*x*x + 0.1*x*x*x; };

    auto c = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);
    auto b = Camellia::Projection1d::moments_onto_1d_basis_phys(basis, f, xL, h);

    reference_integrals_1d refInts;
    auto Mref_t = refInts.discretization(basis, false, basis, false, 0);
    auto Mref = Mref_t.view();

    const size_t N = basis->getCardinality();
    const double J = 0.5*h;

    double maxDiff = 0.0;
    for (size_t i=0; i<N; ++i)
    {
      double sum = 0.0;
      for (size_t j=0; j<N; ++j)
      {
        sum += (J * Mref({i,j})) * c[j];
      }
      maxDiff = std::max(maxDiff, std::fabs(sum - b[i]));
    }

    out << "fam=" << famLabel(fam) << " maxDiff=" << maxDiff << "\n";
    TEST_COMPARE(maxDiff, <, 1e-11);
  };

  run(BasisFamilyChoice::Nodal);
  run(BasisFamilyChoice::Hierarchical);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, GlobalMassConsistency_MatchesBruteforce_NodalAndHierarchical)
{
  auto run = [&](BasisFamilyChoice fam)
  {
    auto basis = newBasis1d_family(Camellia::L2, 4, fam);
    // uses your existing helper which prints; but we do our own assertion by recomputing quickly:
    const std::size_t ne = 8;
    const double x_min = 0.0;
    const double x_max = 1.0;

    auto f = [](double x){ return 0.3 + 0.1*x - 0.7*x*x + 0.2*x*x*x; };

    const std::size_t N = basis->getCardinality();
    const double h = (x_max - x_min) / (double)ne;

    host_vector cvec({ne * N});
    cvec.fill_with_zeros();
    auto cv = cvec.view();

    for (std::size_t e=0; e<ne; ++e)
    {
      const double xL = x_min + e*h;
      auto coeffs = Camellia::Projection1d::project_onto_1d_basis_ref(basis, f, xL, h);
      for (std::size_t i=0; i<N; ++i) cv(e*N+i) = coeffs[i];
    }

    reference_integrals_1d refInts;
    auto M0 = refInts.discretization(basis, false, basis, false, 0);
    host_matrix M_code = global_field_assembly_1d(M0, ne, x_min, x_max, 0);

    host_vector b_bf   = assemble_rhs_bruteforce(basis, ne, x_min, x_max, f);
    host_vector b_code = matvec_dense(M_code, cvec);

    auto b1 = b_bf.view();
    auto b2 = b_code.view();

    double maxDiff=0, maxB=0;
    for (std::size_t i=0; i<ne*N; ++i)
    {
      maxDiff = std::max(maxDiff, std::fabs(b1(i) - b2(i)));
      maxB = std::max(maxB, std::fabs(b1(i)));
    }
    const double rel = maxDiff / std::max(1.0, maxB);

    out << "fam=" << famLabel(fam) << " maxDiff=" << maxDiff << " rel=" << rel << "\n";
    TEST_ASSERT(maxDiff < 1e-11 || rel < 1e-11);
  };

  run(BasisFamilyChoice::Nodal);
  run(BasisFamilyChoice::Hierarchical);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, GlobalTraceAssembly_UsesEndpointTags_HGRAD_Nodal)
{
  testGlobalTraceAssemblyUsesEndpointTags(BasisFamilyChoice::Nodal, out, success);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, GlobalTraceAssembly_UsesEndpointTags_HGRAD_Hierarchical)
{
  testGlobalTraceAssemblyUsesEndpointTags(BasisFamilyChoice::Hierarchical, out, success);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, TraceCouplingConsistency_Nodal)
{
  testTraceCouplingConsistency(BasisFamilyChoice::Nodal, out, success);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, TraceCouplingConsistency_Hierarchical)
{
  testTraceCouplingConsistency(BasisFamilyChoice::Hierarchical, out, success);
}

// ------------------------------
// BoBa TT sanity checks
// ------------------------------

TEUCHOS_UNIT_TEST(DPGBobaAssembly, Boba_TTVectorOrdering)
{
  // This just reproduces the logic from test_boba_tt_vector_ordering(), but with assertion.
  host_vector ux({2});
  host_vector ut({3});
  host_vector uv({2});

  auto uxv = ux.view(); uxv(0)=1.0;  uxv(1)=2.0;
  auto utv = ut.view(); utv(0)=10.0; utv(1)=20.0; utv(2)=30.0;
  auto uvv = uv.view(); uvv(0)=100.0; uvv(1)=200.0;

  boba::Array<host_vector, 3> factors{ux, ut, uv};
  auto tt = boba::make_tt_from_vectors(factors);

  const double expected = uxv(1) * utv(2) * uvv(1); // 12000
  const double got = tt.unroll_value({1,2,1});

  out << "expected=" << expected << " got=" << got << "\n";
  TEST_FLOATING_EQUALITY(expected, got, 1e-15);
}

TEUCHOS_UNIT_TEST(DPGBobaAssembly, Boba_TTMModeAction)
{
  host_vector ux({2}); host_vector ut({3}); host_vector uv({2});
  auto uxv = ux.view(); uxv(0)=1.0;  uxv(1)=2.0;
  auto utv = ut.view(); utv(0)=10.0; utv(1)=20.0; utv(2)=30.0;
  auto uvv = uv.view(); uvv(0)=100.0; uvv(1)=200.0;

  boba::Array<host_vector, 3> factors{ux, ut, uv};
  auto u = boba::make_tt_from_vectors<3>(factors);

  host_matrix Dx({2,2}); Dx.fill_with_zeros();
  auto Dxv = Dx.view(); Dxv({0,0})=2.0; Dxv({1,1})=5.0;

  host_matrix Dt({3,3}); Dt.fill_with_zeros();
  auto Dtv = Dt.view(); Dtv({0,0})=3.0; Dtv({1,1})=7.0; Dtv({2,2})=11.0;

  host_matrix Dv({2,2}); Dv.fill_with_zeros();
  auto Dvv = Dv.view(); Dvv({0,0})=13.0; Dvv({1,1})=17.0;

  host_matrix Ix({2,2}); Ix.fill_with_zeros();
  auto Ixv2=Ix.view(); Ixv2({0,0})=1.0; Ixv2({1,1})=1.0;

  host_matrix It({3,3}); It.fill_with_zeros();
  auto Itv2=It.view(); Itv2({0,0})=1.0; Itv2({1,1})=1.0; Itv2({2,2})=1.0;

  host_matrix Iv({2,2}); Iv.fill_with_zeros();
  auto Ivv2=Iv.view(); Ivv2({0,0})=1.0; Ivv2({1,1})=1.0;

  auto Kx = boba::make_ttm_from_matrices<3, host_space, double, size_t>({Dx, It, Iv});
  auto Kt = boba::make_ttm_from_matrices<3, host_space, double, size_t>({Ix, Dt, Iv});
  auto Kv = boba::make_ttm_from_matrices<3, host_space, double, size_t>({Ix, It, Dv});

  auto ux_out = Kx * u;
  auto ut_out = Kt * u;
  auto uv_out = Kv * u;

  const boba::Array<size_t,3> idx{1,2,1};
  const double u0 = u.unroll_value(idx);

  TEST_FLOATING_EQUALITY(5.0  * u0, ux_out.unroll_value(idx), 1e-14);
  TEST_FLOATING_EQUALITY(11.0 * u0, ut_out.unroll_value(idx), 1e-14);
  TEST_FLOATING_EQUALITY(17.0 * u0, uv_out.unroll_value(idx), 1e-14);
}
