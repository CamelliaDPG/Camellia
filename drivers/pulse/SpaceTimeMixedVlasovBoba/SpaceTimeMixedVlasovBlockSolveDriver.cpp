#include <vector>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Kokkos_Core.hpp>

#include "Camellia.h"
#include "BOBA/boba.hpp"

#include "EpetraToBoba.hpp"
#include "SpaceTimeMixedVlasovSetup.hpp"
#include "SpaceTimeMixedVlasovCompress.hpp"

boba::BlockOperator<host_matrix>
computeBlockPreconditioner(const boba::BlockOperator<host_matrix> &matrix,
                           const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                           const size_t &polyOrder, const size_t &polyOrderEnrich) {
  const size_t numElement = numElementX * numElementT * numElementV;
  const size_t numTestNode1d = polyOrder + polyOrderEnrich + 1;
  const size_t numTestNode = numTestNode1d * numTestNode1d * numTestNode1d;

  boba::BlockOperator<host_matrix> precond(subvectors::size, subvectors::size);
  for (size_t row = 0; row < subvectors::size; ++row) {
    for (size_t col = 0; col < subvectors::size; ++col) {
      precond({row, col}) = host_matrix(matrix({row, col}).sizes());

      if (row == 0 && col == 0) {
        const auto &gram = matrix({row, col});
        const auto &gram_view = gram.view();

        auto &gram_inv = precond({row, col});
        auto gram_inv_view = gram_inv.view();

        gram_inv.fill_with_zeros();

        // block-by-block inverse
        for (size_t i = 0; i < numElement; ++i) {
          const size_t offset = i * numTestNode;

          host_matrix block({numTestNode, numTestNode});
          auto block_view = block.view();

          for (size_t c = 0; c < numTestNode; ++c) {
            for (size_t r = 0; r < numTestNode; ++r) {
              block_view({r, c}) = gram_view({offset + r, offset + c});
            }
          }

          host_matrix block_inv = boba::compute_inverse(block);
          auto block_inv_view = block_inv.view();

          for (size_t c = 0; c < numTestNode; ++c) {
            for (size_t r = 0; r < numTestNode; ++r) {
              gram_inv_view({offset + r, offset + c}) = block_inv_view({r, c});
            }
          }
        }
      } else if (row == col) {
        precond({row, col}).set_to_identity_matrix();
      } else {
        precond({row, col}).fill_with_zeros();
      }
    }
  }

  return precond;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  {
    int numElements_x = 2;
    int numElements_t = 2;
    int numElements_v = 2;
    int p = 2;
    int dp = 2;
    std::vector<std::vector<double>> domainDim(3,std::vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    double beta = 1.0;
    size_t krylov_method = ::boba::KrylovMethods::flexible_gmres;

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
    cmdp.setOption("krylov_method", &krylov_method );

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
                                                                     p, dp, domainDim, beta, mpiSession);

      partition = identifyVlasovBlockPartition(soln, mesh, vf, tn_hat, f, eps_w);
    }

    auto matrix = soln->getStiffnessMatrix();
    auto rhs = soln->getRHSVector();
    auto lhs = soln->getLHSVector();

    // ---------------------------------------------
    // Block-structured matrices and vectors
    // ---------------------------------------------

    boba::BlockOperator<host_matrix> block_matrix;
    boba::BlockVector<host_vector> block_rhs;
    boba::BlockVector<host_vector> block_lhs;
    std::tie(block_matrix, block_rhs, block_lhs) = setupBlockSystem<host_space, double, size_t, Camellia::GlobalIndexType, subvectors::size, subvectors::size>(matrix, rhs, lhs, partition, partition);

    block_matrix.savetxt("block_matrix", ".dat");
    block_rhs.savetxt("block_rhs", ".dat");
    block_lhs.savetxt("block_lhs", ".dat");

    // ---------------------------------------------
    // Sanity check
    // ---------------------------------------------
    
    auto block_residual = block_rhs - block_matrix * block_lhs;
    std::cout << "block_residual_norm = " << std::scientific << boba::norm_frobenius(block_residual) << std::endl;

    // ---------------------------------------------
    // Preconditioner construction
    // ---------------------------------------------

    auto block_precond = computeBlockPreconditioner(block_matrix, numElements_x, numElements_t, numElements_v, p, dp);

    block_precond.savetxt("block_precond", ".dat");

    // ---------------------------------------------
    // Block system solve
    // ---------------------------------------------

    size_t outer_iterations = 10;
    size_t inner_iterations = static_cast<size_t>(matrix->NumGlobalRows());
    double tolerance = 1.0e-08;

    boba::Krylov<boba::BlockOperator<host_matrix>,
                 boba::BlockVector<host_vector>> block_solver(outer_iterations, inner_iterations, tolerance);
    block_solver.verbose = true;
    block_solver.method = krylov_method;

    block_solver.linear_operator = block_matrix;

    auto guess = block_lhs;
    guess.fill_with_zeros();

    auto solution = guess;
    block_solver.solve(block_rhs, guess, solution);
    std::cout << "block_residual_norm_from_gmres = " << std::scientific << boba::norm_frobenius(block_rhs - block_matrix * solution) << std::endl;

    block_solver.set_left_preconditioner(block_precond);

    auto solution_precond = guess;
    block_solver.solve(block_rhs, guess, solution_precond);
    std::cout << "block_residual_norm_from_gmres = " << std::scientific << boba::norm_frobenius(block_rhs - block_matrix * solution_precond) << std::endl;
  }

  Camellia::finalize(false);

  return 0;
}
