#include <vector>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>

#include "Camellia.h"
#include "BOBA/boba.hpp"

#include "EpetraToBoba.hpp"
#include "SpaceTimeMixedVlasovSetup.hpp"
#include "SpaceTimeMixedVlasovCompress.hpp"

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  {
    int numElements_x = 2;
    int numElements_t = 2;
    int numElements_v = 2;
    double beta = 1.0;

    std::vector<std::vector<double>> domainDim(3,std::vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    int p = 2;
    int dp = 2;

    cmdp.setOption("xElements", &numElements_x );
    cmdp.setOption("tElements", &numElements_t );
    cmdp.setOption("vElements", &numElements_v );
    cmdp.setOption("p",  &p );
    cmdp.setOption("dp", &dp);
    cmdp.setOption("x0", &domainDim[0][0] );
    cmdp.setOption("x1", &domainDim[0][1] );
    cmdp.setOption("t0", &domainDim[1][0] );
    cmdp.setOption("t1", &domainDim[1][1] );
    cmdp.setOption("v0", &domainDim[2][0] );
    cmdp.setOption("v1", &domainDim[2][1] );
    cmdp.setOption("beta", &beta, "beta weight for L^2 term in graph norm (beta -> 0 should be L^2 projection, according to usual DPG theory)");

    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }

    Camellia::SolutionPtr soln;
    boba::Array<std::set<Camellia::GlobalIndexType>, subvectors::size> partition;
    {
      Camellia::MeshPtr mesh;
      Camellia::VarFactoryPtr vf;
      Camellia::VarPtr tn_hat;
      Camellia::VarPtr f;
      Camellia::VarPtr eps_w;
      std::tie(soln, mesh, vf, tn_hat, f, eps_w) = setupVlasovSystem(numElements_x, numElements_t, numElements_v,
                                                                     p, dp, domainDim, beta, mpiSession, false); // don't solve

      partition = identifyVlasovBlockPartition(soln, mesh, vf, tn_hat, f, eps_w);
    }

    auto matrix = soln->getStiffnessMatrix();
    const auto compression_from_sparsity = static_cast<double>(matrix->NumGlobalRows()) * static_cast<double>(matrix->NumGlobalCols()) / static_cast<double>(matrix->NumGlobalNonzeros());

    // ---------------------------------------------
    // Block-structured matrices and vectors
    // ---------------------------------------------

    auto block_matrix = setupBlockSystem<host_space, double, size_t, Camellia::GlobalIndexType, subvectors::size, subvectors::size>(matrix, partition, partition);

    // ---------------------------------------------
    // TT compressed matrix/vectors
    // ---------------------------------------------

    auto block_tt_matrix = compressBlockSystem(block_matrix, numElements_x, numElements_t, numElements_v, p, dp);
    const auto compression_from_tt = block_tt_matrix.compression_rate();

    block_tt_matrix = compressBlockSystem(block_matrix, numElements_x, numElements_t, numElements_v, p, dp, true); // permute = true
    const auto compression_from_tt_with_permute = block_tt_matrix.compression_rate();

    std::cout << "sparse_matrix                CR " << std::scientific << compression_from_sparsity << std::endl;
    std::cout << "block_tt_matrix              CR " << std::scientific << compression_from_tt << std::endl;
    std::cout << "block_tt_matrix_with_permute CR " << std::scientific << compression_from_tt_with_permute << std::endl;
  }

  Camellia::finalize(false);

  return 0;
}
