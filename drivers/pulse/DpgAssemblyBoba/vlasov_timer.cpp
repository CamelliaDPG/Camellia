#include <iostream>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>

#include "../SpaceTimeMixedVlasovBoba/SpaceTimeMixedVlasovSetup.hpp"
#include "vlasov_system.hpp"

void vlasov_assembly_camellia(const double &x_min, const double &x_max, const int &nx,
                              const double &t_min, const double &t_max, const int &nt,
                              const double &v_min, const double &v_max, const int &nv,
                              const int &p, const int &dp,
                              Teuchos::GlobalMPISession &mpiSession) {
  boba::TicToc<::boba::tictoc_units::milliseconds> timer;
  timer.tic();

  Camellia::SolutionPtr soln;
  Camellia::MeshPtr mesh;
  Camellia::VarFactoryPtr vf;
  Camellia::VarPtr tn_hat;
  Camellia::VarPtr f;
  Camellia::VarPtr eps_w;

  const std::vector<std::vector<double>> domainDim{{x_min, x_max}, {t_min, t_max}, {v_min, v_max}};
  const double beta = 1.0;
  const bool solveSystem = false;
  std::tie(soln, mesh, vf, tn_hat, f, eps_w) = setupVlasovSystem(nx, nt, nv, p, dp, domainDim, beta, mpiSession, solveSystem );

  auto timing = timer.toc();
  std::cout << "total_timing " << std::endl
            << "  time = " << timing << " " << timer.units_string << std::endl << std::endl;
}

void vlasov_assembly_boba(const double &x_min, const double &x_max, const size_t &nx,
                          const double &t_min, const double &t_max, const size_t &nt,
                          const double &v_min, const double &v_max, const size_t &nv,
                          const size_t &p, const size_t &dp) {
  auto line = Camellia::CellTopology::line();

  auto basis_family = Camellia::BasisFamily::nodalBasisFamily();
  auto test_basis  = basis_family->getBasis(line, Camellia::efsForSpace(Camellia::HGRAD_DISC), p + dp);
  auto trace_basis = basis_family->getBasis(line, Camellia::efsForSpace(Camellia::L2), p + 1);
  auto field_basis = basis_family->getBasis(line, Camellia::efsForSpace(Camellia::L2), p);

  auto stiffness = construct_vlasov_matrices(test_basis, trace_basis, field_basis,
                                             x_min, x_max, nx,
                                             t_min, t_max, nt,
                                             v_min, v_max, nv,
                                             p, dp);
}

int main(int argc, char **argv) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  Teuchos::CommandLineProcessor cmdp(false,  // don't throw exceptions
                                     true);  // return errors for unrecognized options
  {
    double x_min = 0.0;
    double x_max = 1.0;
    int nx = 1;
    double t_min = 0.0;
    double t_max = 1.0;
    int nt = 1;
    double v_min = 0.0;
    double v_max = 1.0;
    int nv = 1;
    int p = 2;
    int dp = 2;
    int factor = 1;

    cmdp.setOption("x_min", &x_min, "lower x domain extent (default: 0.0)");
    cmdp.setOption("x_max", &x_max, "lower x domain extent (default: 1.0)");
    cmdp.setOption("num_element_x", &nx, "number of elements along x dimension (default: 1)");
    cmdp.setOption("t_min", &t_min, "lower t domain extent (default: 0.0)");
    cmdp.setOption("t_max", &t_max, "lower t domain extent (default: 1.0)");
    cmdp.setOption("num_element_t", &nt, "number of elements along t dimension (default: 1)");
    cmdp.setOption("v_min", &v_min, "lower v domain extent (default: 0.0)");
    cmdp.setOption("v_max", &v_max, "lower v domain extent (default: 1.0)");
    cmdp.setOption("num_element_v", &nv, "number of elements along v dimension (default: 1)");
    cmdp.setOption("p", &p, "monomial order for field variables (default: 2)");
    cmdp.setOption("dp", &dp, "monomial order enrichment for test variables (default: 2)");
    cmdp.setOption("elements_factor", &factor, "Multiplicative factor on number of elements per dimension (default: 1)");

    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }

    nx *= factor;
    nt *= factor;
    nv *= factor;

    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "BOBA Assembly" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    vlasov_assembly_boba(x_min, x_max, nx, t_min, t_max, nt, v_min, v_max, nv, p, dp);

    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Camellia Assembly" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    vlasov_assembly_camellia(x_min, x_max, nx, t_min, t_max, nt, v_min, v_max, nv, p, dp, mpiSession);
  }
  Camellia::finalize(false);
  return 0;
}
