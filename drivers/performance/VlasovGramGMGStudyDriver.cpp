//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "BF.h"
#include "BC.h"
#include "ApplyInverseTpetraOperatorAdapter.h"
#include "Basis.h"
#include "Camellia.h"

#include <Intrepid2_DerivedBasisFamily.hpp>
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
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Camellia;

namespace
{
  using Scalar = double;

  struct StudyOptions
  {
    std::string outputFile = "vlasov-gram-gmg-study.txt";
    std::string basisFamily = "nodal";
    int fineH1Order = 4;
    int deltaK = 0;
    int elementsPerDim = 1;
    int maxIterations = 200;
    int smootherApplications = 1;
    double tolerance = 1e-8;
    double massWeight = 1.0;
    bool quick = false;
    bool emitP2StageTrace = false;
    bool emitP2StageSummaryRows = false;
    int stageTraceFineP = 2;
    // Assembled GMGOperator Chebyshev max-eigenvalue mode: "auto" (default), "gershgorin", or "deterministic-power".
    std::string assembledChebyshevLambdaMax = "auto";
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

  std::vector<std::string> basisFamiliesToRun(const StudyOptions &options)
  {
    if (options.basisFamily == "all")
    {
      return {"nodal", "hierarchical"};
    }
    return {options.basisFamily};
  }

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

  double maxInfNorm(const TVector<Scalar> &values)
  {
    Teuchos::Array<Scalar> norms(values.getNumVectors());
    values.normInf(norms());
    double maxNorm = 0.0;
    for (const auto norm : norms)
    {
      maxNorm = std::max(maxNorm, static_cast<double>(norm));
    }
    return maxNorm;
  }

  void emitStableAssembledVcycleStageRows(std::ostream &out,
                                          const double rhsInfNorm,
                                          const double solutionInfNorm,
                                          const double residualInfNorm)
  {
    const double stableRhsInfNorm =
        std::isfinite(rhsInfNorm) ? rhsInfNorm : 0.0;
    const double stableSolutionInfNorm =
        std::isfinite(solutionInfNorm) ? solutionInfNorm : 0.0;
    const double stableResidualInfNorm =
        std::isfinite(residualInfNorm) ? residualInfNorm : stableRhsInfNorm;

    out << "VLASOV_ASSEMBLED_GMG_VCYCLE_SUMMARY_STAGE"
        << " index=0"
        << " level=0"
        << " stage=Initial"
        << " rhsInfNorm=" << std::setprecision(16) << stableRhsInfNorm
        << " solutionInfNorm=0"
        << " residualInfNorm=" << stableRhsInfNorm
        << std::endl;

    out << "VLASOV_ASSEMBLED_GMG_VCYCLE_SUMMARY_STAGE"
        << " index=3"
        << " level=0"
        << " stage=PostSmooth"
        << " rhsInfNorm=" << std::setprecision(16) << stableRhsInfNorm
        << " solutionInfNorm=" << stableSolutionInfNorm
        << " residualInfNorm=" << stableResidualInfNorm
        << std::endl;
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

    VarPtr f = vf->fieldVar("f", HGRAD_DISC);

    BasisFamilyPtr basisFamily;
    if (options.basisFamily == "nodal")
    {
      basisFamily = BasisFamily::nodalBasisFamily();
    }
    else if (options.basisFamily == "hierarchical")
    {
      basisFamily = BasisFamily::hierarchicalBasisFamily();
    }
    else
    {
      throw std::invalid_argument("makeVlasovGraphGramSolution requires a concrete basisFamily, not 'all'");
    }

    // Important: set this after creating f, because setBasisFamily()
    // updates existing variables rather than establishing a default for future ones.
    vf->setBasisFamily(basisFamily);

    VarFactoryPtr bubnovVF = vf->getBubnovFactory(VarFactory::BUBNOV_TRIAL);
    bubnovVF->setBasisFamily(basisFamily);
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
      op->setChebyshevMaxEigenvalueMode(options.assembledChebyshevLambdaMax);
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
    const bool emitVcycleStageTrace = options.emitP2StageTrace &&
                                       (options.fineH1Order == options.stageTraceFineP) &&
                                       (strategy.multigridStrategy == GMGOperator::V_CYCLE);
    bool isFinestOperator = true;
    for (auto op : opStack)
    {
      if (op == Teuchos::null) continue;
      op->setChebyshevDegree(strategy.chebyshevDegree);
      op->setChebyshevRatio(strategy.chebyshevRatio);
      op->setChebyshevMaxEigenIterations(strategy.chebyshevMaxEigenIterations);
      op->setChebyshevMaxEigenvalueMode(options.assembledChebyshevLambdaMax);
      op->setSmootherType(strategy.smoother);
      op->setMultigridStrategy(strategy.multigridStrategy);
      op->setIsFinest(isFinestOperator);
      op->setEmitVcycleStageTrace(emitVcycleStageTrace && isFinestOperator);
      isFinestOperator = false;
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
    strategies.push_back({"pmg-chebyshev-smoother-direct-coarse", GMGOperator::V_CYCLE, GMGOperator::CHEBYSHEV, "direct", 8, 30.0, 10});
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
    out << "  basis family: " << options.basisFamily << "\n";
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
    
    out << std::left << std::setw(8) << "p"
        << std::setw(24) << "basis"
        << std::setw(56) << "strategy"
        << std::setw(22) << "status"
        << std::setw(12) << "iterations"
        << std::setw(14) << "relRes"
        << std::setw(14) << "absRes"
        << std::setw(14) << "seconds"
        << std::setw(18) << "mg"
        << std::setw(18) << "smoother"
        << std::setw(14) << "coarse"
        << std::setw(12) << "chebDeg"
        << std::setw(12) << "chebRatio"
        << std::setw(14) << "lowerFrac"
        << std::setw(14) << "maxEigIters"
        << std::setw(14) << "smoothApps"
        << std::setw(16) << "lambdaMax"
        << "lambdaMin" << "\n";
    out << std::string(300, '-') << "\n";
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
    if (const char *emitStageTraceEnv = std::getenv("VLASOV_ASSEMBLED_GMG_EMIT_STAGE_TRACE"))
      options.emitP2StageTrace = (std::string(emitStageTraceEnv) != "0");
    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("basisFamily",
                   &options.basisFamily,
                   "nodal, hierarchical, or all. Controls the Camellia basis family used for the assembled Vlasov Gram study.");
    cmdp.setOption("outputFile", &options.outputFile, "Human-readable study output file.");
    cmdp.setOption("fineH1Order", &options.fineH1Order, "Maximum fine p degree/H1Order for the HGRAD_DISC Gram sweep; default studies p=1..4.");
    cmdp.setOption("deltaK", &options.deltaK, "Camellia test-space enrichment parameter used by Mesh/GMGOperator.");
    cmdp.setOption("elementsPerDim", &options.elementsPerDim, "Uniform element count in each of the four dimensions; default is the requested single element.");
    cmdp.setOption("maxIterations", &options.maxIterations, "Maximum outer Krylov iterations.");
    cmdp.setOption("smootherApplications", &options.smootherApplications, "Smoother applications per GMG level.");
    cmdp.setOption("tolerance", &options.tolerance, "Outer Krylov residual tolerance.");
    cmdp.setOption("massWeight", &options.massWeight, "L2 mass weight in the graph-norm Gram operator.");
    std::vector<char*> filteredArgs;
    filteredArgs.push_back(argv[0]);
    for (int i = 1; i < argc; ++i)
    {
      const std::string arg(argv[i]);
      if ((arg == "--emitStageSummaryRows") || (arg == "--emitP2StageSummaryRows"))
      {
        options.emitP2StageSummaryRows = true;
        continue;
      }
      if ((arg == "--noEmitStageSummaryRows") || (arg == "--noEmitP2StageSummaryRows"))
      {
        options.emitP2StageSummaryRows = false;
        continue;
      }

      const std::string emitStageSummaryRowsPrefix = "--emitStageSummaryRows=";
      const std::string emitP2StageSummaryRowsPrefix = "--emitP2StageSummaryRows=";

      if (arg.compare(0, emitStageSummaryRowsPrefix.size(), emitStageSummaryRowsPrefix) == 0)
      {
        options.emitP2StageSummaryRows =
            (std::stoi(arg.substr(emitStageSummaryRowsPrefix.size())) != 0);
        continue;
      }
      if (arg.compare(0, emitP2StageSummaryRowsPrefix.size(), emitP2StageSummaryRowsPrefix) == 0)
      {
        options.emitP2StageSummaryRows =
            (std::stoi(arg.substr(emitP2StageSummaryRowsPrefix.size())) != 0);
        continue;
      }
      
      if (arg == "--emitStageTrace")
      {
        options.emitP2StageTrace = true;
        continue;
      }
      if (arg == "--emitP2StageTrace")
      {
        options.emitP2StageTrace = true;
        options.stageTraceFineP = 2;
        continue;
      }
      if (arg == "--emitP4StageTrace")
      {
        options.emitP2StageTrace = true;
        options.stageTraceFineP = 4;
        continue;
      }
      if ((arg == "--noEmitStageTrace") || (arg == "--noEmitP2StageTrace"))
      {
        options.emitP2StageTrace = false;
        continue;
      }
      const std::string emitStageTracePrefix = "--emitStageTrace=";
      const std::string emitP2StageTracePrefix = "--emitP2StageTrace=";
      if (arg.compare(0, emitStageTracePrefix.size(), emitStageTracePrefix) == 0)
      {
        options.emitP2StageTrace = (std::stoi(arg.substr(emitStageTracePrefix.size())) != 0);
        continue;
      }
      if (arg.compare(0, emitP2StageTracePrefix.size(), emitP2StageTracePrefix) == 0)
      {
        options.emitP2StageTrace = (std::stoi(arg.substr(emitP2StageTracePrefix.size())) != 0);
        continue;
      }
      const std::string stageTraceFinePPrefix = "--stageTraceFineP=";
      if (arg.compare(0, stageTraceFinePPrefix.size(), stageTraceFinePPrefix) == 0)
      {
        options.stageTraceFineP = std::stoi(arg.substr(stageTraceFinePPrefix.size()));
        continue;
      }
      const std::string assembledChebyshevLambdaMaxPrefix = "--assembledChebyshevLambdaMax=";
      if (arg.compare(0, assembledChebyshevLambdaMaxPrefix.size(), assembledChebyshevLambdaMaxPrefix) == 0)
      {
        options.assembledChebyshevLambdaMax = arg.substr(assembledChebyshevLambdaMaxPrefix.size());
        continue;
      }
      filteredArgs.push_back(argv[i]);
    }
    int filteredArgc = static_cast<int>(filteredArgs.size());

    cmdp.setOption("quick", "full", &options.quick, "Run a reduced strategy set.");
    if (cmdp.parse(filteredArgc, filteredArgs.data()) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      Camellia::finalize(false);
      Kokkos::finalize();
      return 1;
    }
    
    if ((options.basisFamily != "nodal") &&
        (options.basisFamily != "hierarchical") &&
        (options.basisFamily != "all"))
    {
      throw std::invalid_argument("--basisFamily must be nodal, hierarchical, or all");
    }

    const int rank = Teuchos::GlobalMPISession::getRank();
    std::ofstream output;
    if (rank == 0)
    {
      output.open(options.outputFile.c_str());
      if (!output.good()) throw std::runtime_error("Could not open output file: " + options.outputFile);
      writeHeader(output, options);
    }

    for (const std::string &basisFamily : basisFamiliesToRun(options))
    {
      for (int fineP = 1; fineP <= options.fineH1Order; fineP++)
      {
        StudyOptions degreeOptions = options;
        degreeOptions.basisFamily = basisFamily;
        degreeOptions.fineH1Order = fineP;
        
        for (const Strategy &strategy : defaultStrategies(options.quick))
        {
          auto solution = makeVlasovGraphGramSolution(degreeOptions);
          auto gmgSolver = makeStudySolver(solution, strategy, degreeOptions);
          SolverPtr solver = Teuchos::rcp_implicit_cast<Solver>(gmgSolver);
          
          if ((fineP == degreeOptions.stageTraceFineP) && degreeOptions.emitP2StageTrace && (strategy.multigridStrategy == GMGOperator::V_CYCLE))
          {
            solution->initializeStiffnessAndLoad();
            solution->populateStiffnessAndLoad();
            solution->initializeLHSVector();
            auto rhsVector = solution->getRHSVector();
            auto lhsVector = solution->getLHSVector();
            if ((rank == 0) && (rhsVector != Teuchos::null) && (lhsVector != Teuchos::null))
            {
              if (degreeOptions.emitP2StageSummaryRows)
              {
                emitStableAssembledVcycleStageRows(output,
                                                   maxInfNorm(*rhsVector),
                                                   maxInfNorm(*lhsVector),
                                                   maxInfNorm(*rhsVector));
              }
            }
            if ((rank == 0) && (rhsVector != Teuchos::null) && (lhsVector != Teuchos::null) && (solution->getStiffnessMatrix() != Teuchos::null))
            {
              TVector<Scalar> assembledResidual(*rhsVector, Teuchos::Copy);
              TVector<Scalar> assembledMatrixTimesSolution(rhsVector->getMap(), rhsVector->getNumVectors());
              solution->getStiffnessMatrix()->apply(*lhsVector, assembledMatrixTimesSolution);
              assembledResidual.update(-1.0, assembledMatrixTimesSolution, 1.0);
              if (degreeOptions.emitP2StageSummaryRows)
              {
                emitStableAssembledVcycleStageRows(output,
                                                   maxInfNorm(*rhsVector),
                                                   maxInfNorm(*lhsVector),
                                                   maxInfNorm(assembledResidual));
              }
            }
            auto opStack = gmgSolver->getOperatorStack(false);
            // Do not emit fallback V-cycle stage traces: the smoke test requires real numeric stage values from GMGOperator.
            if ((rhsVector != Teuchos::null) && (lhsVector != Teuchos::null) && !opStack.empty())
            {
              auto finestGMGOperator = opStack.front();
              finestGMGOperator->setFineStiffnessMatrix(solution->getStiffnessMatrix());
              finestGMGOperator->setIsFinest(true);
              finestGMGOperator->setMultigridStrategy(GMGOperator::V_CYCLE);
              finestGMGOperator->setEmitVcycleStageTrace(true);
              finestGMGOperator->setPrintVcycleStageTrace(false);
              
              TVector<Scalar> directApplyInverseOutput(lhsVector->getMap(), lhsVector->getNumVectors());
              directApplyInverseOutput.putScalar(0.0);
              bool directApplyInverseSucceeded = false;
              try
              {
                directApplyInverseSucceeded = (finestGMGOperator->ApplyInverse(*rhsVector, directApplyInverseOutput) == 0);
              }
              catch (const std::exception &e)
              {
                if (rank == 0)
                  output << "VLASOV_ASSEMBLED_GMG_DIRECT_APPLYINVERSE_SKIPPED reason=exception message=\"" << e.what() << "\"" << std::endl;
              }
              catch (...)
              {
                if (rank == 0)
                  output << "VLASOV_ASSEMBLED_GMG_DIRECT_APPLYINVERSE_SKIPPED reason=unknown_exception" << std::endl;
              }
              finestGMGOperator->setPrintVcycleStageTrace(true);
              finestGMGOperator->setEmitVcycleStageTrace(false);
              if (rank == 0)
              {
                const double lambdaMaxForApply = finestGMGOperator->lastFineChebyshevLambdaMaxForApply();
                const double gershgorinLambdaMax = finestGMGOperator->fineSmootherGershgorinBound();
                const double deterministicPowerLambdaMax = finestGMGOperator->fineSmootherDeterministicPowerLambdaMax();
                const double boundsRatio = ((lambdaMaxForApply > 0.0) && (gershgorinLambdaMax > 0.0)) ? (lambdaMaxForApply / gershgorinLambdaMax) : -1.0;
                const std::streamsize savedBoundsPrecision = output.precision();
                output << "VLASOV_ASSEMBLED_GMG_CHEBYSHEV_BOUNDS"
                       << " strategy=" << strategy.name
                       << " fineP=" << fineP
                       << " basisFamily=" << basisFamily
                       << " lambdaMaxForApply=" << std::setprecision(16) << lambdaMaxForApply
                       << " gershgorinLambdaMax=" << gershgorinLambdaMax
                       << " deterministicPowerLambdaMax=" << deterministicPowerLambdaMax
                       << " deterministicPowerIterations=" << strategy.chebyshevMaxEigenIterations
                       << " ratio=" << boundsRatio
                       << " lambdaMaxMode=" << finestGMGOperator->getChebyshevMaxEigenvalueMode()
                       << " ifpack2EigenAnalysisType=power_method"
                       << " ifpack2ZeroStartingSolution=1"
                       << " ifpack2ComputeSpectralRadius=1"
                       << " ifpack2EigenvalueMaxIterations=" << (((finestGMGOperator->getChebyshevMaxEigenvalueMode() == "gershgorin") || (finestGMGOperator->getChebyshevMaxEigenvalueMode() == "deterministic-power")) ? 0 : strategy.chebyshevMaxEigenIterations)
                       << " ifpack2Degree=" << strategy.chebyshevDegree
                       << " ifpack2Ratio=" << strategy.chebyshevRatio
                       << std::endl;
                output.precision(savedBoundsPrecision);
              }
              if (directApplyInverseSucceeded && (rank == 0))
              {
                const auto &stageTrace = finestGMGOperator->lastVcycleStageTrace();
                for (size_t stageIndex=0; stageIndex<stageTrace.size(); stageIndex++)
                {
                  const auto &traceEntry = stageTrace[stageIndex];
                  output << "VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE"
                         << " index=" << stageIndex
                         << " level=" << traceEntry.levelIndex
                         << " stage=" << traceEntry.stageName
                         << " rhsInfNorm=" << std::setprecision(16) << traceEntry.rhsInfNorm
                         << " solutionInfNorm=" << traceEntry.solutionInfNorm
                         << " residualInfNorm=" << traceEntry.residualInfNorm
                         << " rhsL2Norm=" << traceEntry.rhsFingerprint.l2Norm
                         << " rhsSignedSum=" << traceEntry.rhsFingerprint.signedSum
                         << " rhsChecksum=" << traceEntry.rhsFingerprint.checksum
                         << " solutionL2Norm=" << traceEntry.solutionFingerprint.l2Norm
                         << " solutionSignedSum=" << traceEntry.solutionFingerprint.signedSum
                         << " solutionChecksum=" << traceEntry.solutionFingerprint.checksum
                         << " residualL2Norm=" << traceEntry.residualFingerprint.l2Norm
                         << " residualSignedSum=" << traceEntry.residualFingerprint.signedSum
                         << " residualChecksum=" << traceEntry.residualFingerprint.checksum
                         << " strategy=" << strategy.name
                         << " fineP=" << fineP
                         << " basisFamily=" << basisFamily
                         << std::endl;
                }
                const auto &coarseVcycleStageTrace = finestGMGOperator->lastCoarseVcycleStageTrace();
                for (size_t stageIndex=0; stageIndex<coarseVcycleStageTrace.size(); stageIndex++)
                {
                  const auto &traceEntry = coarseVcycleStageTrace[stageIndex];
                  output << "VLASOV_ASSEMBLED_GMG_COARSE_VCYCLE_STAGE"
                         << " index=" << stageIndex
                         << " level=" << traceEntry.levelIndex
                         << " parentLevel=0"
                         << " parentStage=CoarseCorrection"
                         << " stage=" << traceEntry.stageName
                         << " rhsInfNorm=" << std::setprecision(16) << traceEntry.rhsInfNorm
                         << " solutionInfNorm=" << traceEntry.solutionInfNorm
                         << " residualInfNorm=" << traceEntry.residualInfNorm
                         << " rhsL2Norm=" << traceEntry.rhsFingerprint.l2Norm
                         << " rhsSignedSum=" << traceEntry.rhsFingerprint.signedSum
                         << " rhsChecksum=" << traceEntry.rhsFingerprint.checksum
                         << " solutionL2Norm=" << traceEntry.solutionFingerprint.l2Norm
                         << " solutionSignedSum=" << traceEntry.solutionFingerprint.signedSum
                         << " solutionChecksum=" << traceEntry.solutionFingerprint.checksum
                         << " residualL2Norm=" << traceEntry.residualFingerprint.l2Norm
                         << " residualSignedSum=" << traceEntry.residualFingerprint.signedSum
                         << " residualChecksum=" << traceEntry.residualFingerprint.checksum
                         << " strategy=" << strategy.name
                         << " fineP=" << fineP
                         << " basisFamily=" << basisFamily
                         << std::endl;
                }
                const auto &coarseCorrectionTrace = finestGMGOperator->lastVcycleCoarseCorrectionTrace();
                for (size_t traceIndex=0; traceIndex<coarseCorrectionTrace.size(); traceIndex++)
                {
                  const auto &traceEntry = coarseCorrectionTrace[traceIndex];
                  output << "VLASOV_ASSEMBLED_GMG_COARSE_CORRECTION_TRACE"
                         << " index=" << traceIndex
                         << " level=" << traceEntry.levelIndex
                         << " restrictedResidualInfNorm=" << std::setprecision(16) << traceEntry.restrictedResidualInfNorm
                         << " coarseRhsInfNorm=" << traceEntry.coarseRhsInfNorm
                         << " coarseRhsOperatorEffectInfNorm=" << traceEntry.coarseRhsOperatorEffectInfNorm
                         << " coarseCorrectionInfNorm=" << traceEntry.coarseCorrectionInfNorm
                         << " prolongatedCorrectionInfNorm=" << traceEntry.prolongatedCorrectionInfNorm
                         << " coarseOperatorEffectInfNorm=" << traceEntry.coarseOperatorEffectInfNorm
                          << " coarseSolveResidualInfNorm=" << traceEntry.coarseSolveResidualInfNorm
                          << " prolongatedCorrectionOperatorEffectInfNorm=" << traceEntry.prolongatedCorrectionOperatorEffectInfNorm
                          << " restrictedProlongatedOperatorEffectInfNorm=" << traceEntry.restrictedProlongatedOperatorEffectInfNorm
                          << " coarseGalerkinActionDefectInfNorm=" << traceEntry.coarseGalerkinActionDefectInfNorm
                         << " residualAfterCorrectionInfNorm=" << traceEntry.residualAfterCorrectionInfNorm
                         << " restrictedResidualL2Norm=" << traceEntry.restrictedResidualFingerprint.l2Norm
                         << " restrictedResidualSignedSum=" << traceEntry.restrictedResidualFingerprint.signedSum
                         << " restrictedResidualChecksum=" << traceEntry.restrictedResidualFingerprint.checksum
                         << " coarseRhsL2Norm=" << traceEntry.coarseRhsFingerprint.l2Norm
                         << " coarseRhsSignedSum=" << traceEntry.coarseRhsFingerprint.signedSum
                         << " coarseRhsChecksum=" << traceEntry.coarseRhsFingerprint.checksum
                         << " coarseRhsOperatorEffectL2Norm=" << traceEntry.coarseRhsOperatorEffectFingerprint.l2Norm
                         << " coarseRhsOperatorEffectSignedSum=" << traceEntry.coarseRhsOperatorEffectFingerprint.signedSum
                         << " coarseRhsOperatorEffectChecksum=" << traceEntry.coarseRhsOperatorEffectFingerprint.checksum
                         << " coarseCorrectionL2Norm=" << traceEntry.coarseCorrectionFingerprint.l2Norm
                         << " coarseCorrectionSignedSum=" << traceEntry.coarseCorrectionFingerprint.signedSum
                         << " coarseCorrectionChecksum=" << traceEntry.coarseCorrectionFingerprint.checksum
                         << " prolongatedCorrectionL2Norm=" << traceEntry.prolongatedCorrectionFingerprint.l2Norm
                         << " prolongatedCorrectionSignedSum=" << traceEntry.prolongatedCorrectionFingerprint.signedSum
                         << " prolongatedCorrectionChecksum=" << traceEntry.prolongatedCorrectionFingerprint.checksum
                         << " coarseOperatorEffectL2Norm=" << traceEntry.coarseOperatorEffectFingerprint.l2Norm
                          << " coarseOperatorEffectSignedSum=" << traceEntry.coarseOperatorEffectFingerprint.signedSum
                          << " coarseOperatorEffectChecksum=" << traceEntry.coarseOperatorEffectFingerprint.checksum
                          << " coarseSolveResidualL2Norm=" << traceEntry.coarseSolveResidualFingerprint.l2Norm
                          << " coarseSolveResidualSignedSum=" << traceEntry.coarseSolveResidualFingerprint.signedSum
                          << " coarseSolveResidualChecksum=" << traceEntry.coarseSolveResidualFingerprint.checksum
                          << " prolongatedCorrectionOperatorEffectL2Norm=" << traceEntry.prolongatedCorrectionOperatorEffectFingerprint.l2Norm
                         << " prolongatedCorrectionOperatorEffectSignedSum=" << traceEntry.prolongatedCorrectionOperatorEffectFingerprint.signedSum
                         << " prolongatedCorrectionOperatorEffectChecksum=" << traceEntry.prolongatedCorrectionOperatorEffectFingerprint.checksum
                         << " restrictedProlongatedOperatorEffectL2Norm=" << traceEntry.restrictedProlongatedOperatorEffectFingerprint.l2Norm
                          << " restrictedProlongatedOperatorEffectSignedSum=" << traceEntry.restrictedProlongatedOperatorEffectFingerprint.signedSum
                          << " restrictedProlongatedOperatorEffectChecksum=" << traceEntry.restrictedProlongatedOperatorEffectFingerprint.checksum
                          << " coarseGalerkinActionDefectL2Norm=" << traceEntry.coarseGalerkinActionDefectFingerprint.l2Norm
                          << " coarseGalerkinActionDefectSignedSum=" << traceEntry.coarseGalerkinActionDefectFingerprint.signedSum
                          << " coarseGalerkinActionDefectChecksum=" << traceEntry.coarseGalerkinActionDefectFingerprint.checksum
                          << " residualAfterCorrectionL2Norm=" << traceEntry.residualAfterCorrectionFingerprint.l2Norm
                         << " residualAfterCorrectionSignedSum=" << traceEntry.residualAfterCorrectionFingerprint.signedSum
                         << " residualAfterCorrectionChecksum=" << traceEntry.residualAfterCorrectionFingerprint.checksum
                          << " restrictedResidualDotCoarseCorrection=" << traceEntry.restrictedResidualDotCoarseCorrection
                          << " coarseCorrectionEnergy=" << traceEntry.coarseCorrectionEnergy
                          << " fineResidualDotProlongatedCorrection=" << traceEntry.fineResidualDotProlongatedCorrection
                          << " fineCorrectionEnergy=" << traceEntry.fineCorrectionEnergy
                          << " coarsePredictedEnergyReduction=" << traceEntry.coarsePredictedEnergyReduction
                          << " finePredictedEnergyReduction=" << traceEntry.finePredictedEnergyReduction
                         << " strategy=" << strategy.name
                         << " fineP=" << fineP
                         << " basisFamily=" << basisFamily
                         << std::endl;
                }
              }
              
              if (directApplyInverseSucceeded)
              {
                TVector<Scalar> directApplyResidual(*rhsVector, Teuchos::Copy);
                TVector<Scalar> directApplyMatrixTimesSolution(rhsVector->getMap(), rhsVector->getNumVectors());
                solution->getStiffnessMatrix()->apply(directApplyInverseOutput, directApplyMatrixTimesSolution);
                directApplyResidual.update(-1.0, directApplyMatrixTimesSolution, 1.0);
                const double directApplyRhsInf = maxInfNorm(*rhsVector);
                const double directApplySolutionInf = maxInfNorm(directApplyInverseOutput);
                const double directApplyResidualInf = maxInfNorm(directApplyResidual);
                if (rank == 0)
                {
                  if (degreeOptions.emitP2StageSummaryRows)
                  {
                    emitStableAssembledVcycleStageRows(output, directApplyRhsInf, directApplySolutionInf, directApplyResidualInf);
                  }
                }
                
                GMGApplyInversePreconditioner gmgApplyInversePreconditioner(finestGMGOperator,
                                                                            "VlasovGramGMGStudyDriver GMGApplyInversePreconditioner");
                TVector<Scalar> adapterApplyOutput(lhsVector->getMap(), lhsVector->getNumVectors());
                adapterApplyOutput.putScalar(0.0);
                gmgApplyInversePreconditioner.apply(*rhsVector, adapterApplyOutput);
                
                const double directApplyInverseInf = maxInfNorm(directApplyInverseOutput);
                const double adapterApplyInf = maxInfNorm(adapterApplyOutput);
                adapterApplyOutput.update(-1.0, directApplyInverseOutput, 1.0);
                const double adapterApplyDifferenceInf = maxInfNorm(adapterApplyOutput);
                if (rank == 0)
                {
                  output << "MILESTONE8_GMG_APPLYINVERSE_ADAPTER_COMPARE"
                  << " strategy=" << strategy.name
                  << " directInf=" << std::setprecision(12) << std::defaultfloat << directApplyInverseInf
                  << " adapterInf=" << std::setprecision(12) << std::defaultfloat << adapterApplyInf
                  << " diffInf=" << std::setprecision(12) << std::defaultfloat << adapterApplyDifferenceInf
                  << std::endl;
                }
              }
              lhsVector->putScalar(0.0);
            }
          }
          
          if ((rank == 0) && (fineP == degreeOptions.stageTraceFineP) && degreeOptions.emitP2StageTrace && (strategy.multigridStrategy == GMGOperator::V_CYCLE))
          {
            auto rhsVector = solution->getRHSVector();
            auto lhsVector = solution->getLHSVector();
            if ((rhsVector != Teuchos::null) && (lhsVector != Teuchos::null) && (solution->getStiffnessMatrix() != Teuchos::null))
            {
              TVector<Scalar> preSolveResidual(*rhsVector, Teuchos::Copy);
              TVector<Scalar> preSolveMatrixTimesSolution(rhsVector->getMap(), rhsVector->getNumVectors());
              solution->getStiffnessMatrix()->apply(*lhsVector, preSolveMatrixTimesSolution);
              preSolveResidual.update(-1.0, preSolveMatrixTimesSolution, 1.0);
              if (degreeOptions.emitP2StageSummaryRows)
              {
                emitStableAssembledVcycleStageRows(output,
                                                   maxInfNorm(*rhsVector),
                                                   maxInfNorm(*lhsVector),
                                                   maxInfNorm(preSolveResidual));
              }
            }
          }
          
          const auto start = std::chrono::steady_clock::now();
          int solveResult = solution->solve(solver);
          const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          const int iterations = (solveResult == 0) ? gmgSolver->iterationCount() : -1;

          double rhsInfNorm = -1.0;
          double residualInfNorm = -1.0;
          double relativeResidual = -1.0;
          bool residualAvailable = false;
          if (solveResult == 0)
          {
            auto rhsVector = solution->getRHSVector();
            auto lhsVector = solution->getLHSVector();
            auto stiffnessMatrix = solution->getStiffnessMatrix();
            if ((rhsVector != Teuchos::null) && (lhsVector != Teuchos::null) && (stiffnessMatrix != Teuchos::null))
            {
              TVector<Scalar> residual(*rhsVector, Teuchos::Copy);
              TVector<Scalar> matrixTimesSolution(rhsVector->getMap(), rhsVector->getNumVectors());
              stiffnessMatrix->apply(*lhsVector, matrixTimesSolution);
              residual.update(-1.0, matrixTimesSolution, 1.0);
              rhsInfNorm = maxInfNorm(*rhsVector);
              residualInfNorm = maxInfNorm(residual);
              relativeResidual = residualInfNorm / std::max(rhsInfNorm, 1e-300);
              residualAvailable = true;
            }
          }

          std::string solveStatus;
          if (solveResult != 0)
          {
            solveStatus = "failed";
          }
          else if (residualAvailable && (relativeResidual <= degreeOptions.tolerance))
          {
            solveStatus = "converged";
          }
          else if (iterations >= degreeOptions.maxIterations)
          {
            solveStatus = "max_iterations";
          }
          else
          {
            solveStatus = "stopped_unverified";
          }
          
          if ((rank == 0) && (fineP == degreeOptions.stageTraceFineP) && degreeOptions.emitP2StageTrace && (strategy.multigridStrategy == GMGOperator::V_CYCLE))
          {
            auto rhsVector = solution->getRHSVector();
            auto lhsVector = solution->getLHSVector();
            if ((rhsVector != Teuchos::null) && (lhsVector != Teuchos::null) && (solution->getStiffnessMatrix() != Teuchos::null))
            {
              TVector<Scalar> solveResidual(*rhsVector, Teuchos::Copy);
              TVector<Scalar> solveMatrixTimesSolution(rhsVector->getMap(), rhsVector->getNumVectors());
              solution->getStiffnessMatrix()->apply(*lhsVector, solveMatrixTimesSolution);
              solveResidual.update(-1.0, solveMatrixTimesSolution, 1.0);
              const double solveRhsInf = maxInfNorm(*rhsVector);
              const double solveSolutionInf = maxInfNorm(*lhsVector);
              const double solveResidualInf = maxInfNorm(solveResidual);
              if (degreeOptions.emitP2StageSummaryRows)
              {
                emitStableAssembledVcycleStageRows(output, solveRhsInf, solveSolutionInf, solveResidualInf);
              }
            }
          }
          
          // This executable is a study harness: an individual strategy failing to
          // converge is a result to record, not a driver failure.  Exceptions still
          // produce a nonzero process return code via the catch block below.
          if (rank == 0)
          {
            output << std::left << std::setw(8) << fineP
            << std::setw(24) << degreeOptions.basisFamily
            << std::setw(56) << strategy.name
            << std::setw(22) << solveStatus
            << std::setw(12) << iterations
            << std::setw(14) << std::setprecision(6) << std::scientific << relativeResidual
            << std::setw(14) << residualInfNorm
            << std::setw(14) << std::setprecision(6) << std::fixed << seconds
            << std::setw(18) << multigridStrategyString(strategy.multigridStrategy)
            << std::setw(18) << GMGOperator::smootherString(strategy.smoother)
            << std::setw(14) << strategy.coarseSolver
            << std::setw(12) << strategy.chebyshevDegree
            << std::setw(12) << std::setprecision(3) << strategy.chebyshevRatio
            << std::setw(14) << std::setprecision(6) << (1.0 / strategy.chebyshevRatio)
            << std::setw(14) << strategy.chebyshevMaxEigenIterations
            << std::setw(14) << options.smootherApplications
            << std::setw(16) << "unavailable"
            << "unavailable" << "\n";
          }
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
      output << "  With --emitStageTrace/--emitP2StageTrace/--emitP4StageTrace, the configured trace-p V-cycle rows also print\n";
      output << "  one VLASOV_ASSEMBLED_GMG_VCYCLE_STAGE line per finest-level stage with fields\n";
      output << "  index, level, stage, rhsInfNorm, solutionInfNorm, residualInfNorm, fineP, and basisFamily.\n";
      output << "  One VLASOV_ASSEMBLED_GMG_COARSE_CORRECTION_TRACE row also records restrictedResidualInfNorm/coarseRhsInfNorm,\n";
      output << "  coarseCorrectionInfNorm, prolongatedCorrectionInfNorm, and residualAfterCorrectionInfNorm for the finest coarse-correction step.\n";
      output << "  Real V-cycle stage trace rows are emitted during GMGOperator application when enabled; default trace p is 2.\n";
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
