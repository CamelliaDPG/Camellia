//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "BF.h"
#include "BC.h"
#include "Camellia.h"
#include "Function.h"
#include "GMGOperator.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "Solution.h"
#include "VarFactory.h"

#include <Kokkos_Core.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Camellia;

namespace
{
  struct StudyOptions
  {
    std::string outputFile = "vlasov-gram-gmg-study.txt";
    int fineH1Order = 7;
    int deltaK = 0;
    int elementsPerDim = 1;
    int maxIterations = 200;
    int smootherApplications = 1;
    double tolerance = 1e-8;
    double massWeight = 1.0;
    bool quick = false;
  };

  struct Strategy
  {
    std::string name;
    GMGOperator::MultigridStrategy multigridStrategy = GMGOperator::V_CYCLE;
    GMGOperator::SmootherChoice smoother = GMGOperator::CHEBYSHEV;
    std::string coarseSolver = "direct"; // direct, chebyshev, or jacobi
    int chebyshevDegree = 8;
    double chebyshevRatio = 30.0;
    int chebyshevMaxEigenIterations = 10;
  };

  std::string multigridStrategyString(GMGOperator::MultigridStrategy strategy)
  {
    switch (strategy)
    {
      case GMGOperator::TWO_LEVEL: return "TWO_LEVEL";
      case GMGOperator::V_CYCLE: return "V_CYCLE";
      case GMGOperator::W_CYCLE: return "W_CYCLE";
      case GMGOperator::FULL_MULTIGRID_V: return "FULL_MULTIGRID_V";
      case GMGOperator::FULL_MULTIGRID_W: return "FULL_MULTIGRID_W";
      case GMGOperator::SMOOTHER_ONLY: return "SMOOTHER_ONLY";
      default: return "UNKNOWN";
    }
  }

  std::vector<int> pCoarsenSchedule(int degree)
  {
    if (degree < 1) throw std::invalid_argument("fineH1Order/fine p degree must be positive");

    std::vector<int> pCoarsenSchedule;
    int pCoarse = std::max(degree / 2, 1);
    pCoarsenSchedule.push_back(pCoarse);
    while (pCoarse > 1)
    {
      pCoarse = std::max(pCoarse / 2, 1);
      pCoarsenSchedule.push_back(pCoarse);
    }
    return pCoarsenSchedule;
  }

  std::string pCoarsenScheduleString(int degree)
  {
    std::string schedule = std::to_string(degree);
    for (int p : pCoarsenSchedule(degree))
    {
      schedule += " -> " + std::to_string(p);
    }
    return schedule;
  }

  std::vector<MeshPtr> meshesForPCoarseningSchedule(SolutionPtr solution, const StudyOptions &options)
  {
    MeshPtr fineMesh = solution->mesh();
    std::vector<MeshPtr> meshesCoarseToFine;
    std::vector<int> schedule = pCoarsenSchedule(options.fineH1Order);

    for (auto it = schedule.rbegin(); it != schedule.rend(); ++it)
    {
      meshesCoarseToFine.push_back(MeshFactory::minRuleMesh(fineMesh->getTopology(), fineMesh->bilinearForm(),
                                                            *it, options.deltaK));
    }
    meshesCoarseToFine.push_back(fineMesh);
    return meshesCoarseToFine;
  }

  SolutionPtr makeVlasovGraphGramSolution(const StudyOptions &options)
  {
    const int spaceTimePhaseDim = 4; // component ordering: x, t, ux, uy
    VarFactoryPtr vf = VarFactory::varFactory();
    vf->setBasisFamily(BasisFamily::hierarchicalBasisFamily());

    VarPtr f = vf->fieldVar("f", HGRAD_DISC);
    VarFactoryPtr bubnovVF = vf->getBubnovFactory(VarFactory::BUBNOV_TRIAL);
    BFPtr bf = BF::bf(bubnovVF);
    VarPtr w = f; // Bubnov-Galerkin: the test variable is the trial variable relabeled by the Bubnov factory.

    FunctionPtr x  = Function::xn(1);
    FunctionPtr t  = Function::yn(1);
    FunctionPtr ux = Function::zn(1);
    FunctionPtr uy = Function::tn(1);

    FunctionPtr ax = Function::constant(0.5) * (x + t);
    FunctionPtr ay = Function::constant(0.75) * t;

    LinearTermPtr Lf = -f->di(2);      // -d/dt f; di() is 1-based and ordering is x,t,ux,uy.
    Lf = Lf - ux * f->di(1);           // -ux d/dx f
    Lf = Lf - ax * f->di(3);           // -ax d/dux f
    Lf = Lf - ay * f->di(4);           // -ay d/duy f

    LinearTermPtr Lw = -w->di(2);
    Lw = Lw - ux * w->di(1);
    Lw = Lw - ax * w->di(3);
    Lw = Lw - ay * w->di(4);

    bf->addTerm(options.massWeight * f, w);
    bf->addTerm(Lf, Lw);

    std::vector<double> dimensions(spaceTimePhaseDim, 1.0);
    std::vector<int> elementCounts(spaceTimePhaseDim, options.elementsPerDim);
    std::vector<double> origin(spaceTimePhaseDim, 0.0);

    MeshTopologyPtr meshTopology = MeshFactory::hypercubeMeshTopology(dimensions, elementCounts, origin);
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopology, bf, options.fineH1Order, options.deltaK);

    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(Function::constant(1.0) * w);
    BCPtr bc = BC::bc();

    SolutionPtr solution = Solution::solution(mesh, bc, rhs, Teuchos::null);
    solution->setUseCondensedSolve(false);
    solution->setZMCsAsGlobalLagrange(false);
    return solution;
  }

  Teuchos::RCP<GMGSolver> makeSmootherOnlySolver(SolutionPtr solution,
                                                 GMGOperator::SmootherChoice smoother,
                                                 const Strategy &strategy,
                                                 const StudyOptions &options)
  {
    std::vector<MeshPtr> meshes {solution->mesh()};
    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp(new GMGSolver(solution, meshes,
                                                                options.maxIterations,
                                                                options.tolerance,
                                                                GMGOperator::SMOOTHER_ONLY,
                                                                smoother,
                                                                Solver::getDirectSolver(true),
                                                                false));
    // Use GMRES rather than CG: even when the assembled Gram matrix is SPD,
    // the right-preconditioned GMG/smoother operator is not guaranteed to be
    // Hermitian positive definite in Belos CG's p^H*A*p test.
    solver->setUseConjugateGradient(false);
    solver->setAztecOutput(0);
    solver->setComputeConditionNumberEstimate(false);
    solver->setSmootherApplicationCount(options.smootherApplications);
    solver->setSmootherType(smoother);

    auto opStack = solver->getOperatorStack(false);
    for (auto op : opStack)
    {
      if (op == Teuchos::null) continue;
      op->setChebyshevDegree(strategy.chebyshevDegree);
      op->setChebyshevRatio(strategy.chebyshevRatio);
      op->setChebyshevMaxEigenIterations(strategy.chebyshevMaxEigenIterations);
      op->setMultigridStrategy(GMGOperator::SMOOTHER_ONLY);
    }
    return solver;
  }

  Teuchos::RCP<GMGSolver> makeStudySolver(SolutionPtr solution,
                                          const Strategy &strategy,
                                          const StudyOptions &options)
  {
    std::vector<MeshPtr> meshes;
    if (strategy.multigridStrategy == GMGOperator::SMOOTHER_ONLY)
    {
      meshes.push_back(solution->mesh());
    }
    else
    {
      meshes = meshesForPCoarseningSchedule(solution, options);
    }

    Teuchos::RCP<GMGSolver> solver = Teuchos::rcp(new GMGSolver(solution, meshes,
                                                                options.maxIterations,
                                                                options.tolerance,
                                                                strategy.multigridStrategy,
                                                                strategy.smoother,
                                                                Solver::getDirectSolver(true),
                                                                false));
    // Use GMRES rather than CG: even when the assembled Gram matrix is SPD,
    // the right-preconditioned GMG/smoother operator is not guaranteed to be
    // Hermitian positive definite in Belos CG's p^H*A*p test.
    solver->setUseConjugateGradient(false);
    solver->setAztecOutput(0);
    solver->setComputeConditionNumberEstimate(false);
    solver->setSmootherApplicationCount(options.smootherApplications);
    solver->setSmootherType(strategy.smoother);

    auto opStack = solver->getOperatorStack(false);
    for (auto op : opStack)
    {
      if (op == Teuchos::null) continue;
      op->setChebyshevDegree(strategy.chebyshevDegree);
      op->setChebyshevRatio(strategy.chebyshevRatio);
      op->setChebyshevMaxEigenIterations(strategy.chebyshevMaxEigenIterations);
      op->setSmootherType(strategy.smoother);
      op->setMultigridStrategy(strategy.multigridStrategy);
    }

    if ((strategy.multigridStrategy != GMGOperator::SMOOTHER_ONLY) && (strategy.coarseSolver != "direct"))
    {
      GMGOperator::SmootherChoice coarseSmoother = (strategy.coarseSolver == "jacobi")
                                                 ? GMGOperator::POINT_JACOBI
                                                 : GMGOperator::CHEBYSHEV;
      for (auto op : opStack)
      {
        if (op == Teuchos::null) continue;
        if (op->getCoarseOperator() == Teuchos::null)
        {
          SolutionPtr coarseSolution = op->getCoarseSolution();
          if (coarseSolution != Teuchos::null)
          {
            auto coarseSolver = makeSmootherOnlySolver(coarseSolution, coarseSmoother, strategy, options);
            op->setCoarseSolver(Teuchos::rcp_implicit_cast<Solver>(coarseSolver));
          }
        }
      }
    }

    return solver;
  }

  std::vector<Strategy> defaultStrategies(bool quick)
  {
    std::vector<Strategy> strategies;
    strategies.push_back({"diagonal-jacobi-smoother-only", GMGOperator::SMOOTHER_ONLY, GMGOperator::POINT_JACOBI, "direct", 1, 30.0, 0});
    strategies.push_back({"chebyshev-smoother-only-degree-4-ratio-20", GMGOperator::SMOOTHER_ONLY, GMGOperator::CHEBYSHEV, "direct", 4, 20.0, 10});
    strategies.push_back({"chebyshev-smoother-only-degree-8-ratio-30", GMGOperator::SMOOTHER_ONLY, GMGOperator::CHEBYSHEV, "direct", 8, 30.0, 10});
    if (!quick)
    {
      strategies.push_back({"chebyshev-smoother-only-degree-16-ratio-60", GMGOperator::SMOOTHER_ONLY, GMGOperator::CHEBYSHEV, "direct", 16, 60.0, 20});
    }
    strategies.push_back({"pmg-chebyshev-smoother-chebyshev-coarse", GMGOperator::V_CYCLE, GMGOperator::CHEBYSHEV, "chebyshev", 8, 30.0, 10});
    strategies.push_back({"pmg-jacobi-smoother-chebyshev-coarse", GMGOperator::V_CYCLE, GMGOperator::POINT_JACOBI, "chebyshev", 8, 30.0, 10});
    strategies.push_back({"pmg-chebyshev-smoother-jacobi-coarse", GMGOperator::V_CYCLE, GMGOperator::CHEBYSHEV, "jacobi", 8, 30.0, 10});
    return strategies;
  }

  void writeHeader(std::ostream &out, const StudyOptions &options)
  {
    out << "Vlasov 1D2V space-time graph-norm Gram GMG study\n";
    out << "=================================================\n\n";
    out << "Operator: assembled Bubnov-Galerkin Gram matrix for\n";
    out << "  (Lf,Lw) + massWeight*(f,w), L = -d_t - ux*d_x - ax*d_ux - ay*d_uy\n";
    out << "  ax = 0.5*(x+t), ay = 0.75*t; coordinate ordering is x,t,ux,uy.\n\n";
    out << "Discretization:\n";
    out << "  space: HGRAD_DISC\n";
    out << "  basis family: hierarchical\n";
    out << "  elements per dimension: " << options.elementsPerDim << "\n";
    out << "  fine p degree range: 1 .. " << options.fineH1Order << "\n";
    out << "  p coarsening schedule: for each fine p, "
        << "p -> max(p/2,1) -> ... -> 1; max-p schedule is "
        << pCoarsenScheduleString(options.fineH1Order) << "\n";
    out << "  delta_k: " << options.deltaK << "\n";
    out << "  massWeight: " << options.massWeight << "\n\n";
    out << "Solver settings:\n";
    out << "  outer Krylov: GMRES through GMGSolver\n";
    out << "  tolerance: " << options.tolerance << "\n";
    out << "  max iterations: " << options.maxIterations << "\n";
    out << "  smoother applications: " << options.smootherApplications << "\n\n";
    out << std::left << std::setw(56) << "strategy"
        << std::setw(16) << "status"
        << std::setw(12) << "iterations"
        << std::setw(14) << "seconds"
        << std::setw(18) << "mg"
        << std::setw(18) << "smoother"
        << std::setw(14) << "coarse"
        << std::setw(12) << "chebDeg"
        << std::setw(12) << "chebRatio"
        << "maxEigIters" << "\n";
    out << std::string(180, '-') << "\n";
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  int returnCode = 0;

  try
  {
    StudyOptions options;
    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("outputFile", &options.outputFile, "Human-readable study output file.");
    cmdp.setOption("fineH1Order", &options.fineH1Order, "Maximum fine p degree/H1Order for the HGRAD_DISC Gram sweep; default studies p=1..7.");
    cmdp.setOption("deltaK", &options.deltaK, "Camellia test-space enrichment parameter used by Mesh/GMGOperator.");
    cmdp.setOption("elementsPerDim", &options.elementsPerDim, "Uniform element count in each of the four dimensions; default is the requested single element.");
    cmdp.setOption("maxIterations", &options.maxIterations, "Maximum outer Krylov iterations.");
    cmdp.setOption("smootherApplications", &options.smootherApplications, "Smoother applications per GMG level.");
    cmdp.setOption("tolerance", &options.tolerance, "Outer Krylov residual tolerance.");
    cmdp.setOption("massWeight", &options.massWeight, "L2 mass weight in the graph-norm Gram operator.");
    cmdp.setOption("quick", "full", &options.quick, "Run a reduced strategy set.");
    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      Camellia::finalize(false);
      Kokkos::finalize();
      return 1;
    }

    const int rank = Teuchos::GlobalMPISession::getRank();
    std::ofstream output;
    if (rank == 0)
    {
      output.open(options.outputFile.c_str());
      if (!output.good()) throw std::runtime_error("Could not open output file: " + options.outputFile);
      writeHeader(output, options);
    }

    for (int fineP = 1; fineP <= options.fineH1Order; fineP++)
    {
      StudyOptions degreeOptions = options;
      degreeOptions.fineH1Order = fineP;

      for (const Strategy &strategy : defaultStrategies(options.quick))
      {
        auto solution = makeVlasovGraphGramSolution(degreeOptions);
        auto gmgSolver = makeStudySolver(solution, strategy, degreeOptions);
        SolverPtr solver = Teuchos::rcp_implicit_cast<Solver>(gmgSolver);

        const auto start = std::chrono::steady_clock::now();
        int solveResult = solution->solve(solver);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const int iterations = (solveResult == 0) ? gmgSolver->iterationCount() : -1;

        // This executable is a study harness: an individual strategy failing to
        // converge is a result to record, not a driver failure.  Exceptions still
        // produce a nonzero process return code via the catch block below.
        if (rank == 0)
        {
          const std::string strategyName = "p=" + std::to_string(fineP) + " " + strategy.name;
          output << std::left << std::setw(56) << strategyName
                 << std::setw(16) << (solveResult == 0 ? "converged" : "failed")
                 << std::setw(12) << iterations
                 << std::setw(14) << std::setprecision(6) << std::fixed << seconds
                 << std::setw(18) << multigridStrategyString(strategy.multigridStrategy)
                 << std::setw(18) << GMGOperator::smootherString(strategy.smoother)
                 << std::setw(14) << strategy.coarseSolver
                 << std::setw(12) << strategy.chebyshevDegree
                 << std::setw(12) << std::setprecision(3) << strategy.chebyshevRatio
                 << strategy.chebyshevMaxEigenIterations << "\n";
        }
      }
    }

    if (rank == 0)
    {
      output << "\nNotes:\n";
      output << "  This driver intentionally uses Camellia assembled operators and GMGOperator/GMGSolver.\n";
      output << "  The study sweeps fine p from 1 through the configured maximum.\n";
      output << "  For each fine p, p-multigrid levels follow p -> max(p/2,1) -> ... -> 1.\n";
      output << "  The coarse Chebyshev/Jacobi variants replace the lowest-level coarse solver with\n";
      output << "  a smoother-only GMGSolver on the coarsest p mesh.\n";
      output << "  Failed strategy rows are recorded as study results; they do not make the driver fail.\n";
      std::cout << "Wrote Vlasov Gram GMG study results to " << options.outputFile << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "VlasovGramGMGStudyDriver failed: " << e.what() << std::endl;
    returnCode = 1;
  }

  Camellia::finalize(false);
  Kokkos::finalize();
  return returnCode;
}
