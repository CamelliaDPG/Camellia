#ifndef PROJECTION_1D_HPP
#define PROJECTION_1D_HPP

#include "Camellia.h"

namespace Camellia
{
  class Projection1d
  {
  public:
    template<class Basis1D>
    static
    std::pair<std::vector<double>, bool> // bool: true if the mass matrix is diagonal (in which case, the return is just the inverse diagonal; otherwise, it's a Cholesky factorization.)
    factorRefMassMatrix(Teuchos::RCP<Basis1D> basisPtr)
    {
      using ExecSpace = Kokkos::DefaultExecutionSpace;
      using Scalar    = double;
      using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;
      
      auto & basis = *basisPtr->intrepid2Basis();
      
      auto cellTopo = basis.getBaseCellTopology(); // line
      const int N   = basis.getCardinality();
      
      // per-cell order p_cell (varies)
      const int p = basis.getDegree();
      const int cubDegree = 2*p;  // typical for mass matrix exactness
      
      Intrepid2::DefaultCubatureFactory cubFactory;
      auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(cellTopo, cubDegree);
      
      const int Q = cub->getNumPoints();
      DRV refPts("refPts", Q, 1);
      DRV refWts("refWts", Q);
      cub->getCubature(refPts, refWts);
      
      // Basis values on reference points: (N,Q)
      DRV phi("phi", N, Q);
      basis.getValues(phi, refPts, Intrepid2::OPERATOR_VALUE);
      
      //  std::vector<double> M(N*N, 0.0);
      Kokkos::View<Scalar**> M_view("M", N, N);
      
      // Fill M
      Kokkos::parallel_for("assemble", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0,0},{N,N}),
                           KOKKOS_LAMBDA (int i, int j) {
        double value = 0;
        for (int q=0; q<Q; ++q) {
          // For a single cell with endpoints x0,x1:
          const Scalar x0 = -1.0;
          const Scalar x1 =  1.0;
          
          const Scalar w = refWts(q);
          value += phi(i,q) * phi(j,q) * w;
        }
        M_view(i,j) = value;
      });
      
      std::vector<double> M(N*N);
      auto M_view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace::memory_space(), M_view);
      for (int i=0; i<N; i++)
      {
        for (int j=0; j<N; j++)
        {
          M[i + j*N] = M_view_host(i,j);
        }
      }
      
      double maxOffDiagonal = (N > 1) ? std::fabs(M[0*N + 1]) : 0.0;
      double maxDiagonal = (N > 0) ? M[0] : 0.0;
      double minDiagonal = (N > 0) ? M[0] : 0.0;
      
      for (int i=0; i<N; i++)
      {
        for (int j=0; j<N; j++)
        {
          if (i==j)
          {
            maxDiagonal = std::max(M[i+j*N], maxDiagonal);
            minDiagonal = std::min(M[i+j*N], minDiagonal);
          }
          else
          {
            maxOffDiagonal = std::max(std::fabs(M[i+j*N]), maxOffDiagonal);
          }
        }
      }
//      std::cout << "minDiagonal: " << minDiagonal << std::endl;
//      std::cout << "maxDiagonal: " << maxDiagonal << std::endl;
//      std::cout << "maxOffDiagonal: " << maxOffDiagonal << std::endl;
      
      double tol = 1e-12;
      if (maxOffDiagonal > tol)
      {
        // Cholesky factorization (SPD): Mref = L * L^T (UPLO='L') in-place in Mref_cm
        Teuchos::LAPACK<int, double> lapack;
        int info = 0;
        const char UPLO = 'L';
        lapack.POTRF(UPLO, N, M.data(), N, &info);
        if (info != 0) {
          throw std::runtime_error("LAPACK POTRF failed in factorRefMassMatrix() with info = "
                                   + std::to_string(info));
        }
        
        return {M, false};
      }
      else
      {
        std::vector<double> diag(N,0.0);
        for (int i=0; i<N; i++)
        {
          diag[i] = 1./M[i + i*N];
        }
        return {diag, true};
      }
    }
    
    template<class Basis1D, class Func>
    static
    std::vector<double>
    moments_onto_1d_basis_phys(Teuchos::RCP<Basis1D> basisPtr,
                               Func&& func,
                               double xL,
                               double h,
                               int cubDegree = -1)
    {
      using ExecSpace = Kokkos::DefaultExecutionSpace;
      using Scalar = double;
      using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;

      auto &basis = *basisPtr->intrepid2Basis();
      const int N = basis.getCardinality();

      if (cubDegree < 0) {
        const int p = basis.getDegree();
        cubDegree = 2*p + 6; // safe default
      }

      auto cellTopo = basis.getBaseCellTopology();
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

      auto x_of_xi = [&](double xi) { return xL + 0.5*(xi + 1.0)*h; };
      const double J = 0.5*h;

      std::vector<double> b((size_t)N, 0.0);
      for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int q = 0; q < Q; ++q) {
          const double xi = refPts_h(q,0);
          const double x  = x_of_xi(xi);
          sum += J * refWts_h(q) * phi_h(i,q) * func(x);
        }
        b[(size_t)i] = sum;
      }
      return b;
    }
    
    // L2 projection coefficients c on a single physical element [xL, xL+h].
    // Returns c such that f(x(xi)) ≈ sum_i c_i * phi_i(xi) on reference [-1,1].
    template<class Basis1D, class Func>
    static
    std::vector<double>
    project_onto_1d_basis_ref(Teuchos::RCP<Basis1D> basisPtr,
                              Func&& func,
                              double xL,
                              double h)
    {
      using ExecSpace = Kokkos::DefaultExecutionSpace;
      using Scalar    = double;
      using DRV       = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;
      
      auto& basis = *basisPtr->intrepid2Basis();
      const int N = basis.getCardinality();
      
      // Factorized reference mass matrix (or inv diagonal)
      auto massEntry = factorRefMassMatrix(basisPtr);
      const bool massIsDiagonal = massEntry.second;
      
      // Cubature (use same scheme family as mass matrix to be consistent)
      const int p = basis.getDegree();
      const int cubDegree = 2 * p + 2; // add a little margin; safe for linears etc.
      
      auto cellTopo = basis.getBaseCellTopology();
      Intrepid2::DefaultCubatureFactory cubFactory;
      auto cub = cubFactory.create<ExecSpace, Scalar, Scalar>(cellTopo, cubDegree);
      
      const int Q = cub->getNumPoints();
      DRV refPts("refPts", Q, 1);
      DRV refWts("refWts", Q);
      cub->getCubature(refPts, refWts);
      
      // Basis values phi(i,q)
      DRV phi("phi", N, Q);
      basis.getValues(phi, refPts, Intrepid2::OPERATOR_VALUE);
      
      // Build b_i = ∫ phi_i(xi) f(x(xi)) dxi  (reference measure)
      std::vector<double> b((size_t)N, 0.0);
      
      auto refPts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), refPts);
      auto refWts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), refWts);
      auto phi_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);
      
      auto x_of_xi = [&](double xi) {
        return xL + 0.5 * (xi + 1.0) * h;
      };
      
      for (int i = 0; i < N; ++i)
      {
        double sum = 0.0;
        for (int q = 0; q < Q; ++q)
        {
          const double xi = refPts_h(q, 0);
          const double x  = x_of_xi(xi);
          sum += refWts_h(q) * phi_h(i, q) * func(x);
        }
        b[(size_t)i] = sum;
      }
      
      // Solve M_ref c = b
      std::vector<double> c = b;
      
      if (massIsDiagonal)
      {
        const auto& invDiag = massEntry.first;
        if ((int)invDiag.size() != N) throw std::runtime_error("invDiag size mismatch");
        for (int i = 0; i < N; ++i) c[(size_t)i] *= invDiag[(size_t)i];
      }
      else
      {
        // massEntry.first is column-major Cholesky factor in-place (from your factorRefMassMatrix)
        std::vector<double> L = massEntry.first; // copy (POTRS overwrites neither L nor b, but safe)
        Teuchos::LAPACK<int, double> lapack;
        int info = 0;
        const char UPLO = 'L';
        const int NRHS = 1;
        lapack.POTRS(UPLO, N, NRHS, L.data(), N, c.data(), N, &info);
        if (info != 0) {
          throw std::runtime_error("POTRS failed with info=" + std::to_string(info));
        }
      }
      
      return c;
    }
    
    // Evaluate reconstructed function on reference [-1,1]:
    template<class Basis1D>
    static
    double
    eval_reconstruction_ref(Teuchos::RCP<Basis1D> basisPtr,
                            const std::vector<double>& c,
                            double xi)
    {
      using ExecSpace = Kokkos::DefaultExecutionSpace;
      using Scalar = double;
      using DRV = Kokkos::DynRankView<Scalar, Kokkos::LayoutRight, ExecSpace>;
      
      auto& basis = *basisPtr->intrepid2Basis();
      const int N = basis.getCardinality();
      
      // point view (1,1)
      DRV pt("pt", 1, 1);
      auto pt_h = Kokkos::create_mirror_view(pt);
      pt_h(0,0) = xi;
      Kokkos::deep_copy(pt, pt_h);
      
      DRV phi("phi", N, 1);
      basis.getValues(phi, pt, Intrepid2::OPERATOR_VALUE);
      
      auto phi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);
      
      double val = 0.0;
      for (int i = 0; i < N; ++i) val += c[(size_t)i] * phi_h(i,0);
      return val;
    }
  };

  // Simple test: reproduce f(x)=a+b*x on a single element
  template<class Basis1D>
  void test_reproduce_line(Teuchos::RCP<Basis1D> basisPtr,
                           double a, double b,
                           double xL, double h)
  {
    auto f = [&](double x) { return a + b*x; };
    
    
    auto c = Projection1d::project_onto_1d_basis_ref(basisPtr, f, xL, h);
    
    auto x_of_xi = [&](double xi) {
      return xL + 0.5 * (xi + 1.0) * h;
    };
    
    const double xis[] = {-1.0, -0.3, 0.0, 0.7, 1.0};
    double max_err = 0.0;
    for (double xi : xis)
    {
      const double x = x_of_xi(xi);
      const double exact = f(x);
      const double approx = Projection1d::eval_reconstruction_ref(basisPtr, c, xi);
      max_err = std::max(max_err, std::fabs(exact - approx));
    }
    
    std::cout << "line reproduction max_err = " << max_err << "\n";
  }

  template<class Basis1D>
  void test_reproduce_cubic(Teuchos::RCP<Basis1D> basisPtr,
                            double a, double b, double c, double d,
                            double xL, double h)
  {
    auto f = [&](double x) { return a + b*x + c*x*x + d*x*x*x; };
    
    auto coeffs = Projection1d::project_onto_1d_basis_ref(basisPtr, f, xL, h);
    
    auto x_of_xi = [&](double xi) {
      return xL + 0.5 * (xi + 1.0) * h;
    };
    
    const double xis[] = {-1.0, -0.75, -0.3, 0.0, 0.2, 0.7, 1.0};
    double max_err = 0.0;
    
    for (double xi : xis)
    {
      const double x = x_of_xi(xi);
      const double exact  = f(x);
      const double approx = Projection1d::eval_reconstruction_ref(basisPtr, coeffs, xi);
      max_err = std::max(max_err, std::fabs(exact - approx));
    }
    
    std::cout << "cubic reproduction max_err = " << max_err << "\n";
  }
}

#endif
