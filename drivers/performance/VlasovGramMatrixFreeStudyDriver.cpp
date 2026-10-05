//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace
{
  struct StudyOptions
  {
    std::string outputFile = "vlasov-gram-matrix-free-study.txt";
    std::string tensorDriverPath;
    std::string basisFamily = "hierarchical";
    int fineH1Order = 4;
    int elementsPerDim = 1;
    int maxIterations = 200;
    int smootherApplications = 1;
    double tolerance = 1e-8;
    double massWeight = 1.0;
    double pMultigridFineSmootherWeight = 1.0;
    long long maxAssembledEntries = 30000000;
    bool quick = false;
    bool keepChildOutput = false;
    bool pMultigridFineWeightSweep = false;
    bool matrixFreeGMGStageTrace = false;
  };

  struct Strategy
  {
    std::string name;
    std::string matrixFreeAlgorithm;
    std::string multigridStrategy;
    std::string smoother;
    std::string coarseSolver;
    int chebyshevDegree = 8;
    double chebyshevRatio = 30.0;
    int chebyshevMaxEigenIterations = 10;
    double fineSmootherWeight = 1.0;
    bool supported = true;
  };

  struct DriverRun
  {
    int status = -1;
    double seconds = -1.0;
    std::string command;
    std::string output;
  };

  struct Diagnostic
  {
    bool found = false;
    int iterations = -1;
    int converged = 0;
    double residualReduction = -1.0;
    double initialResidualNorm = -1.0;
    double finalResidualNorm = -1.0;
    double pMultigridFineSmootherWeight = -1.0;
    int matrixFreeGMGStageTrace = 0;
    int matrixFreeGMGStageTraceEntries = 0;
    double localPMultigridWrapperDifferenceInfNorm = -1.0;
    double localPMultigridWrapperRelativeDifferenceInfNorm = -1.0;
    std::string stoppedReason = "missing";
    std::string inversePath = "missing";
    std::string pMultigridPath = "missing";
  };

  std::vector<int> pCoarsenSchedule(int degree)
  {
    if (degree < 1) throw std::invalid_argument("fineH1Order/fine p degree must be positive");

    std::vector<int> schedule;
    int pCoarse = std::max(degree / 2, 1);
    schedule.push_back(pCoarse);
    while (pCoarse > 1)
    {
      pCoarse = std::max(pCoarse / 2, 1);
      schedule.push_back(pCoarse);
    }
    return schedule;
  }

  std::string pCoarsenScheduleString(int degree)
  {
    std::string schedule = std::to_string(degree);
    for (int p : pCoarsenSchedule(degree))
      schedule += " -> " + std::to_string(p);
    return schedule;
  }

  std::vector<Strategy> defaultStrategies(bool quick)
  {
    std::vector<Strategy> strategies;
    strategies.push_back({"diagonal-jacobi-smoother-only",
                          "gmres-jacobi", "SMOOTHER_ONLY", "POINT_JACOBI", "direct",
                          1, 30.0, 0, 1.0, true});
    strategies.push_back({"chebyshev-smoother-only-degree-4-ratio-20",
                          "gmres-chebyshev", "SMOOTHER_ONLY", "CHEBYSHEV", "direct",
                          4, 20.0, 10, 1.0, true});
    strategies.push_back({"chebyshev-smoother-only-degree-8-ratio-30",
                          "gmres-chebyshev", "SMOOTHER_ONLY", "CHEBYSHEV", "direct",
                          8, 30.0, 10, 1.0, true});
    strategies.push_back({"chebyshev-smoother-only-degree-16-ratio-60",
                          "gmres-chebyshev", "SMOOTHER_ONLY", "CHEBYSHEV", "direct",
                          16, 60.0, 20, 1.0, true});

    // Keep the matrix-free p=2 comparison table aligned with the assembled
    // GMG study rows.  The Tpetra/right-preconditioner and direct-coarse
    // pMG variants are useful diagnostics, but they are not rows in the
    // reference vlasov-gram-gmg-study.txt table.
    strategies.push_back({"pmg-chebyshev-smoother-chebyshev-coarse",
                          "gmres-pmultigrid-chebyshev", "V_CYCLE", "CHEBYSHEV", "chebyshev",
                          8, 30.0, 10, 1.0, true});
    strategies.push_back({"pmg-jacobi-smoother-chebyshev-coarse",
                          "gmres-pmultigrid-jacobi-chebyshev", "V_CYCLE", "POINT_JACOBI", "chebyshev",
                          8, 30.0, 10, 1.0, true});
    strategies.push_back({"pmg-chebyshev-smoother-jacobi-coarse",
                          "pmultigrid-chebyshev-jacobi-coarse", "V_CYCLE", "CHEBYSHEV", "jacobi",
                          8, 30.0, 10, 1.0, false});
    return strategies;
  }

  std::vector<std::string> basisFamiliesToRun(const StudyOptions &options)
  {
    if (options.basisFamily == "all")
      return {"nodal", "hierarchical"};
    return {options.basisFamily};
  }

  std::string tensorPoissonBasisFamilyArgument(const std::string &basisFamily)
  {
    if ((basisFamily == "hierarchical") || (basisFamily == "intrepid2-hierarchical"))
      return "hgrad-hierarchical";
    if ((basisFamily == "nodal") || (basisFamily == "intrepid2-nodal"))
      return "hgrad-nodal";
    throw std::invalid_argument("Unknown Vlasov Gram matrix-free basisFamily: " + basisFamily);
  }

  bool strategyRequiresPMultigrid(const Strategy &strategy)
  {
    return (strategy.multigridStrategy == "V_CYCLE") ||
           (strategy.matrixFreeAlgorithm.find("pmultigrid") != std::string::npos) ||
           (strategy.matrixFreeAlgorithm.find("matrix-free-gmg") != std::string::npos);
  }

  std::string shellQuote(const std::string &value)
  {
    std::string quoted = "'";
    for (const char c : value)
    {
      if (c == '\'') quoted += "'\\''";
      else quoted += c;
    }
    quoted += "'";
    return quoted;
  }

  std::string parentDirectory(const std::string &path)
  {
    const std::string::size_type slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    if (slash == 0) return path.substr(0, 1);
    return path.substr(0, slash);
  }

  std::string defaultTensorDriverPath(const char *argv0)
  {
    const std::string executable = (argv0 == nullptr) ? std::string() : std::string(argv0);
#ifdef _WIN32
    const std::string suffix = ".exe";
#else
    const std::string suffix;
#endif
    if (executable.empty()) return std::string("TensorPoissonPAMatrixDriver") + suffix;
    return parentDirectory(executable) + "/TensorPoissonPAMatrixDriver" + suffix;
  }

  std::string temporaryOutputPath()
  {
    static int counter = 0;
    const char *tmpDir = std::getenv("TMPDIR");
    if (tmpDir == nullptr) tmpDir = "/tmp";
#ifdef _WIN32
    const long processId = 0;
#else
    const long processId = static_cast<long>(getpid());
#endif
    std::ostringstream path;
    path << tmpDir;
    const std::string tmpDirString(tmpDir);
    if (!tmpDirString.empty() && (tmpDirString.back() != '/')) path << "/";
    path << "camellia_vlasov_gram_matrix_free_study_" << processId << "_" << counter++ << ".txt";
    return path.str();
  }

  std::string trimLeadingDashes(std::string key)
  {
    while (!key.empty() && key[0] == '-') key.erase(key.begin());
    return key;
  }

  bool parseBoolValue(const std::string &value)
  {
    return (value == "1") || (value == "true") || (value == "TRUE") || (value == "on") || (value == "ON") || (value == "yes") || (value == "YES");
  }

  void parseArguments(int argc, char *argv[], StudyOptions &options)
  {
    for (int i=1; i<argc; i++)
    {
      std::string argument = argv[i];
      if (argument == "--quick")
      {
        options.quick = true;
        continue;
      }
      if (argument == "--full")
      {
        options.quick = false;
        continue;
      }
      if ((argument == "--help") || (argument == "-h"))
      {
        std::cout << "Usage: VlasovGramMatrixFreeStudyDriver [--outputFile=file] [--tensorDriverPath=path] [--fineH1Order=N] [--quick]\n"
                  << "       [--basisFamily=hierarchical|nodal|all]\n"
                  << "       [--elementsPerDim=N] [--maxIterations=N] [--smootherApplications=N]\n"
                  << "       [--tolerance=tol] [--massWeight=w] [--pMultigridFineSmootherWeight=w]\n"
                  << "       [--maxAssembledEntries=N] [--keepChildOutput=0|1] [--pMultigridFineWeightSweep=0|1]\n"
                  << "       [--matrixFreeGMGStageTrace=0|1]\n";
        std::exit(0);
      }

      const std::string::size_type equals = argument.find('=');
      if (equals == std::string::npos)
        throw std::invalid_argument("Expected --key=value argument, got: " + argument);
      const std::string key = trimLeadingDashes(argument.substr(0, equals));
      const std::string value = argument.substr(equals + 1);
      if (key == "outputFile") options.outputFile = value;
      else if (key == "tensorDriverPath") options.tensorDriverPath = value;
      else if ((key == "basisFamily") || (key == "basisMode")) options.basisFamily = value;
      else if ((key == "fineH1Order") || (key == "maxFineP")) options.fineH1Order = std::stoi(value);
      else if (key == "elementsPerDim") options.elementsPerDim = std::stoi(value);
      else if (key == "maxIterations") options.maxIterations = std::stoi(value);
      else if (key == "smootherApplications") options.smootherApplications = std::stoi(value);
      else if (key == "tolerance") options.tolerance = std::stod(value);
      else if (key == "massWeight") options.massWeight = std::stod(value);
      else if (key == "pMultigridFineSmootherWeight") options.pMultigridFineSmootherWeight = std::stod(value);
      else if (key == "maxAssembledEntries") options.maxAssembledEntries = std::stoll(value);
      else if (key == "quick") options.quick = parseBoolValue(value);
      else if (key == "keepChildOutput") options.keepChildOutput = parseBoolValue(value);
      else if (key == "pMultigridFineWeightSweep") options.pMultigridFineWeightSweep = parseBoolValue(value);
      else if ((key == "matrixFreeGMGStageTrace") || (key == "emitMatrixFreeGMGStageTrace")) options.matrixFreeGMGStageTrace = parseBoolValue(value);
      else throw std::invalid_argument("Unknown option: " + argument);
    }

    if (options.fineH1Order < 1) throw std::invalid_argument("--fineH1Order must be positive");
    if (options.elementsPerDim < 1) throw std::invalid_argument("--elementsPerDim must be positive");
    if (options.maxIterations < 1) throw std::invalid_argument("--maxIterations must be positive");
    if (options.smootherApplications < 1) throw std::invalid_argument("--smootherApplications must be positive");
    if (options.tolerance <= 0.0) throw std::invalid_argument("--tolerance must be positive");
    if (options.massWeight <= 0.0) throw std::invalid_argument("--massWeight must be positive");
    if (options.pMultigridFineSmootherWeight <= 0.0) throw std::invalid_argument("--pMultigridFineSmootherWeight must be positive");
    if (options.maxAssembledEntries < 1) throw std::invalid_argument("--maxAssembledEntries must be positive");
    if (options.basisFamily == "intrepid2-hierarchical") options.basisFamily = "hierarchical";
    if (options.basisFamily == "intrepid2-nodal") options.basisFamily = "nodal";
    if ((options.basisFamily != "hierarchical") &&
        (options.basisFamily != "nodal") &&
        (options.basisFamily != "all"))
    {
      throw std::invalid_argument("--basisFamily must be hierarchical, nodal, or all");
    }
  }

  std::string keyValueAfter(const std::string &text, const std::string &key, std::string::size_type start)
  {
    const std::string token = key + "=";
    const std::string::size_type keyStart = text.find(token, start);
    if (keyStart == std::string::npos) return std::string();
    const std::string::size_type valueStart = keyStart + token.size();
    std::string::size_type valueEnd = text.find_first_of(" \t\r\n", valueStart);
    if (valueEnd == std::string::npos) valueEnd = text.size();
    return text.substr(valueStart, valueEnd - valueStart);
  }

  int parseIntAfter(const std::string &text, const std::string &key, std::string::size_type start, int fallback)
  {
    const std::string value = keyValueAfter(text, key, start);
    return value.empty() ? fallback : std::stoi(value);
  }

  double parseDoubleAfter(const std::string &text, const std::string &key, std::string::size_type start, double fallback)
  {
    const std::string value = keyValueAfter(text, key, start);
    return value.empty() ? fallback : std::stod(value);
  }

  Diagnostic parseDiagnostic(const std::string &output)
  {
    Diagnostic diagnostic;
    const std::string::size_type lineStart = output.find("INVERSE_DIAGNOSTIC ");
    if (lineStart == std::string::npos) return diagnostic;
    diagnostic.found = true;
    diagnostic.iterations = parseIntAfter(output, "iterations", lineStart, -1);
    diagnostic.converged = parseIntAfter(output, "inverseConverged", lineStart, 0);
    diagnostic.residualReduction = parseDoubleAfter(output, "residualReduction", lineStart, -1.0);
    diagnostic.initialResidualNorm = parseDoubleAfter(output, "initialResidualNorm", lineStart, -1.0);
    diagnostic.finalResidualNorm = parseDoubleAfter(output, "finalResidualNorm", lineStart, -1.0);
    diagnostic.pMultigridFineSmootherWeight = parseDoubleAfter(output, "pMultigridFineSmootherWeight", lineStart, -1.0);
    diagnostic.matrixFreeGMGStageTrace = parseIntAfter(output, "matrixFreeGMGStageTrace", lineStart, 0);
    diagnostic.matrixFreeGMGStageTraceEntries = parseIntAfter(output, "matrixFreeGMGStageTraceEntries", lineStart, 0);
    diagnostic.localPMultigridWrapperDifferenceInfNorm = parseDoubleAfter(output, "localPMultigridWrapperDifferenceInfNorm", lineStart, -1.0);
    diagnostic.localPMultigridWrapperRelativeDifferenceInfNorm = parseDoubleAfter(output, "localPMultigridWrapperRelativeDifferenceInfNorm", lineStart, -1.0);
    diagnostic.stoppedReason = keyValueAfter(output, "inverseStoppedReason", lineStart);
    diagnostic.inversePath = keyValueAfter(output, "inversePath", lineStart);
    diagnostic.pMultigridPath = keyValueAfter(output, "pMultigridPath", lineStart);
    if (diagnostic.stoppedReason.empty()) diagnostic.stoppedReason = "missing";
    if (diagnostic.inversePath.empty()) diagnostic.inversePath = "missing";
    if (diagnostic.pMultigridPath.empty()) diagnostic.pMultigridPath = "missing";
    return diagnostic;
  }

  void writeMatrixFreeGMGStageTraceRows(std::ostream &out,
                                        const std::string &childOutput,
                                        const Strategy &strategy,
                                        const int fineP,
                                        const double fineSmootherWeight)
  {
    const std::string marker = "MATRIX_FREE_GMG_STAGE_TRACE";
    std::istringstream input(childOutput);
    std::string line;
    while (std::getline(input, line))
    {
      const std::string boundsMarker = "VLASOV_MATRIX_FREE_CHEBYSHEV_BOUNDS";
      if (line.rfind(boundsMarker, 0) == 0)
      {
        out << boundsMarker
            << " strategy=" << strategy.name
            << " " << ((line.size() > boundsMarker.size()) ? line.substr(boundsMarker.size() + 1) : std::string())
            << "\n";
        continue;
      }
      if (line.rfind(marker, 0) != 0) continue;
      out << "VLASOV_MATRIX_FREE_GMG_STAGE_TRACE"
          << " strategy=" << strategy.name
          << " fineP=" << fineP
          << " fineWt=" << std::setprecision(12) << std::defaultfloat << fineSmootherWeight
          << " " << line.substr(marker.size() + 1)
          << "\n";
    }
  }

  DriverRun runDriver(const std::string &tensorDriverPath, const std::string &arguments, const bool keepChildOutput)
  {
    const std::string outputPath = temporaryOutputPath();
    DriverRun result;
    result.command = shellQuote(tensorDriverPath) + " " + arguments + " > " + shellQuote(outputPath) + " 2>&1";

    const auto start = std::chrono::steady_clock::now();
    result.status = std::system(result.command.c_str());
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::ifstream input(outputPath.c_str());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    result.output = buffer.str();
    if (!keepChildOutput) std::remove(outputPath.c_str());

    std::cout << "VLASOV_GRAM_MATRIX_FREE_CHILD_COMMAND " << result.command << std::endl;
    if (keepChildOutput)
      std::cout << "VLASOV_GRAM_MATRIX_FREE_CHILD_OUTPUT " << outputPath << std::endl;
    if (result.status != 0)
      std::cout << result.output;
    std::cout << "VLASOV_GRAM_MATRIX_FREE_CHILD_STATUS rawStatus=" << result.status << std::endl;
    return result;
  }

  std::string commandArguments(const StudyOptions &options, const Strategy &strategy, const int p, const double fineSmootherWeight, const std::string &basisFamily)
  {
    std::ostringstream arguments;
    arguments << "--dim=4"
              << " --polyOrder=" << p
              << " --cellsPerDim=" << options.elementsPerDim
              << " --operator=vlasov-graph-norm-solution-coeff-1d1t2v"
              << " --basisMode=" << tensorPoissonBasisFamilyArgument(basisFamily)
              << " --mode=inverse-diagnostic"
              << " --inverseDiagnosticAlgorithm=" << strategy.matrixFreeAlgorithm
              << " --numInverseIterations=" << options.maxIterations
              << " --inverseTolerance=" << std::setprecision(16) << options.tolerance
              << " --inverseInitialGuess=zero"
              << " --inverseRhs=constant-test-function"
              << " --massWeight=" << std::setprecision(16) << options.massWeight
              << " --stiffnessWeight=1"
              << " --maxAssembledEntries=" << options.maxAssembledEntries
              << " --printTermWeights=0";

    if (strategy.matrixFreeAlgorithm == "gmres-chebyshev")
    {
      arguments << " --inversePreconditionerDegree=" << strategy.chebyshevDegree
                << " --inverseEigenRatio=" << std::setprecision(16) << strategy.chebyshevRatio
                << " --inverseSpectralEstimateIterations=" << strategy.chebyshevMaxEigenIterations
                << " --inverseDiagnosticRequireAssembledReference=1";
    }
    else if (strategy.matrixFreeAlgorithm == "gmres-jacobi")
    {
      arguments << " --inverseDiagnosticRequireAssembledReference=1";
    }
    else if ((strategy.matrixFreeAlgorithm == "gmres-pmultigrid-chebyshev") ||
             (strategy.matrixFreeAlgorithm == "pmultigrid-chebyshev"))
    {
      arguments << " --pMultigridChebyshevSmootherIterations=" << std::max(1, options.smootherApplications * strategy.chebyshevDegree)
                << " --pMultigridChebyshevSmootherLowerFraction=" << std::setprecision(16) << (1.0 / strategy.chebyshevRatio)
                << " --pMultigridFineSmootherWeight=" << std::setprecision(16) << fineSmootherWeight
                << " --diagnoseLocalPMultigridWrapperEquivalence=1"
                << " --inverseSpectralEstimate=assembled-gershgorin"
                << " --inverseLowerSpectralEstimate=ratio"
                << " --inverseEigenRatio=" << std::setprecision(16) << strategy.chebyshevRatio;
    }
    else if (strategy.matrixFreeAlgorithm == "gmres-pmultigrid-jacobi-chebyshev")
    {
      arguments << " --pMultigridJacobiSmootherIterations=" << options.smootherApplications
                << " --pMultigridCoarseChebyshevIterations=" << strategy.chebyshevDegree
                << " --pMultigridCoarseChebyshevLowerFraction=" << std::setprecision(16) << (1.0 / strategy.chebyshevRatio)
                << " --pMultigridFineSmootherWeight=" << std::setprecision(16) << fineSmootherWeight
                << " --diagnoseLocalPMultigridWrapperEquivalence=1"
                << " --inverseSpectralEstimate=assembled-gershgorin"
                << " --inverseLowerSpectralEstimate=ratio"
                << " --inverseEigenRatio=" << std::setprecision(16) << strategy.chebyshevRatio;
    }
    const bool supportsMatrixFreeGMGStageTrace =
      (strategy.matrixFreeAlgorithm == "gmres-pmultigrid-chebyshev") ||
      (strategy.matrixFreeAlgorithm == "gmres-pmultigrid-jacobi-chebyshev");
    if (options.matrixFreeGMGStageTrace && supportsMatrixFreeGMGStageTrace)
      arguments << " --matrixFreeGMGStageTrace=1";

    return arguments.str();
  }

  void writeHeader(std::ostream &out, const StudyOptions &options, const std::string &tensorDriverPath)
  {
    out << "Vlasov 1D2V space-time graph-norm Gram matrix-free study\n";
    out << "=========================================================\n\n";
    out << "Operator: tensor/PAMatrix graph-norm Gram proxy for\n";
    out << "  (Lf,Lw) + massWeight*(f,w), L = -d_t - ux*d_x - ax*d_ux - ay*d_uy\n";
    out << "  ax = 0.5*(x+t), ay = 0.75*t; coordinate ordering is x,t,ux,uy.\n\n";
    out << "Execution model:\n";
    out << "  This harness drives TensorPoissonPAMatrixDriver inverse diagnostics.\n";
    out << "  Outer operator applications use LocalPAMatrixTpetraOperator / PAMatrix terms.\n";
    out << "  GMRES smoother-only rows use the existing right-preconditioned Belos+Ifpack2 path.\n";
    out << "  p-multigrid rows use TensorPoissonPAMatrixDriver's local two-level p-multigrid preconditioner;\n";
    out << "  hierarchical p-multigrid uses Intrepid2::HierarchicalBasisProlongator for P and R=P^T for restriction.\n";
    out << "  These rows are still TensorPoisson local V-cycle rows and are not yet asserted to match assembled GMGOperator stage-by-stage semantics.\n\n";
    out << "Tensor driver: " << tensorDriverPath << "\n\n";
    out << "Discretization:\n";
    out << "  space: tensor HGRAD\n";
    out << "  basis family option: " << options.basisFamily << "\n";
    out << "  TensorPoisson basis mapping: hierarchical -> --basisMode=hgrad-hierarchical; nodal -> --basisMode=hgrad-nodal\n";
    out << "  intrepid2-hierarchical and intrepid2-nodal are accepted as legacy aliases for hierarchical and nodal\n";
    out << "  nodal p-multigrid rows are skipped until nodal p-prolongation/restriction is implemented\n";
    out << "  elements per dimension: " << options.elementsPerDim << "\n";
    out << "  fine p degree range: 1 .. " << options.fineH1Order << "\n";
    out << "  p coarsening schedule: for each fine p, p -> max(p/2,1) -> ... -> 1; max-p schedule is "
        << pCoarsenScheduleString(options.fineH1Order) << "\n";
    out << "  massWeight: " << options.massWeight << "\n\n";
    out << "Solver settings:\n";
    out << "  outer Krylov: Belos GMRES for smoother-only strategies\n";
    out << "  tolerance: " << options.tolerance << "\n";
    out << "  max iterations / V-cycles: " << options.maxIterations << "\n";
    out << "  smoother applications: " << options.smootherApplications << "\n";
    out << "  pMultigridFineSmootherWeight default: " << options.pMultigridFineSmootherWeight << "\n";
    out << "  matrixFreeGMGStageTrace: " << (options.matrixFreeGMGStageTrace ? "enabled" : "disabled") << "\n";
    out << "  Optional MatrixFreeGMG stage traces are emitted as VLASOV_MATRIX_FREE_GMG_STAGE_TRACE rows.\n";
    out << "  P2_PMG_FINEWT_RESULT includes matrixFreeGMGStageTrace and matrixFreeGMGStageTraceEntries.\n\n";
    out << std::left << std::setw(8) << "p"
        << std::setw(24) << "basis"
        << std::setw(56) << "strategy"
        << std::setw(22) << "status"
        << std::setw(12) << "iterations"
        << std::setw(14) << "seconds"
        << std::setw(18) << "mg"
        << std::setw(18) << "smoother"
        << std::setw(14) << "coarse"
        << std::setw(12) << "chebDeg"
        << std::setw(12) << "chebRatio"
        << std::setw(14) << "maxEigIters"
        << std::setw(14) << "fineWt"
        << std::setw(14) << "relRes"
        << std::setw(14) << "absRes"
        << std::setw(18) << "wrapperDiff"
        << std::setw(18) << "wrapperRelDiff"
        << "path" << "\n";
    out << std::string(316, '-') << "\n";
  }
}

int main(int argc, char *argv[])
{
  int returnCode = 0;
  try
  {
    StudyOptions options;
    parseArguments(argc, argv, options);
    if (options.tensorDriverPath.empty()) options.tensorDriverPath = defaultTensorDriverPath(argv[0]);

    std::ofstream output(options.outputFile.c_str());
    if (!output.good()) throw std::runtime_error("Could not open output file: " + options.outputFile);
    writeHeader(output, options, options.tensorDriverPath);

    for (const std::string &basisFamily : basisFamiliesToRun(options))
    {
      for (int fineP = 1; fineP <= options.fineH1Order; fineP++)
      {
        for (const Strategy &strategy : defaultStrategies(options.quick))
        {
        std::vector<double> fineSmootherWeights(1, strategy.fineSmootherWeight * options.pMultigridFineSmootherWeight);
        if ((fineP == 2) && (strategy.multigridStrategy == "V_CYCLE") && strategy.supported && options.pMultigridFineWeightSweep)
        {
          fineSmootherWeights.push_back(0.9 * options.pMultigridFineSmootherWeight);
          fineSmootherWeights.push_back(0.85 * options.pMultigridFineSmootherWeight);
          fineSmootherWeights.push_back(0.8 * options.pMultigridFineSmootherWeight);
          fineSmootherWeights.push_back(0.75 * options.pMultigridFineSmootherWeight);
          fineSmootherWeights.push_back(0.7 * options.pMultigridFineSmootherWeight);
          fineSmootherWeights.push_back((2.0 / 3.0) * options.pMultigridFineSmootherWeight);
        }
        for (const double fineSmootherWeight : fineSmootherWeights)
        {
        const std::string weightSuffix = (strategy.multigridStrategy == "V_CYCLE")
                                       ? " fineWeight=" + std::to_string(fineSmootherWeight)
                                       : std::string();
        const std::string strategyName = "p=" + std::to_string(fineP) + " " + strategy.name + weightSuffix;
        const bool nodalPMultigridUnsupported =
            (basisFamily == "nodal") && strategyRequiresPMultigrid(strategy);
        if (!strategy.supported || nodalPMultigridUnsupported ||
            ((strategy.multigridStrategy == "V_CYCLE") && (fineP < 2)))
        {
          const std::string unsupportedReason = nodalPMultigridUnsupported
              ? "nodal_pmultigrid_prolongator_not_implemented"
              : "matrix_free_preconditioner_gap";
          output << std::left << std::setw(8) << fineP
                 << std::setw(24) << basisFamily
                 << std::setw(56) << strategy.name
                 << std::setw(22) << "not_supported"
                 << std::setw(12) << -1
                 << std::setw(14) << std::setprecision(6) << std::fixed << 0.0
                 << std::setw(18) << strategy.multigridStrategy
                 << std::setw(18) << strategy.smoother
                 << std::setw(14) << strategy.coarseSolver
                 << std::setw(12) << strategy.chebyshevDegree
                 << std::setw(12) << std::setprecision(3) << strategy.chebyshevRatio
                 << std::setw(14) << strategy.chebyshevMaxEigenIterations
                 << std::setw(14) << ((strategy.multigridStrategy == "V_CYCLE") ? std::to_string(fineSmootherWeight) : std::string("n/a"))
                 << std::setw(14) << "n/a"
                 << std::setw(14) << "n/a"
                 << std::setw(18) << "n/a"
                 << std::setw(18) << "n/a"
                 << unsupportedReason << "\n";
          continue;
        }

        const DriverRun run = runDriver(options.tensorDriverPath, commandArguments(options, strategy, fineP, fineSmootherWeight, basisFamily), options.keepChildOutput);
        const Diagnostic diagnostic = parseDiagnostic(run.output);
        if (options.matrixFreeGMGStageTrace && (diagnostic.matrixFreeGMGStageTraceEntries > 0))
          writeMatrixFreeGMGStageTraceRows(output, run.output, strategy, fineP, fineSmootherWeight);
        const bool childSucceeded = (run.status == 0);
        std::string status;
        if (!childSucceeded)
        {
          status = "failed";
        }
        else if (!diagnostic.found)
        {
          status = "missing_diag";
        }
        else if ((diagnostic.converged != 0) && (diagnostic.residualReduction <= options.tolerance))
        {
          status = "converged";
        }
        else if (diagnostic.iterations >= options.maxIterations)
        {
          status = "max_iterations";
        }
        else if (diagnostic.converged != 0)
        {
          status = "stopped_unverified";
        }
        else
        {
          status = "not_converged";
        }
        const std::string path = (strategy.matrixFreeAlgorithm.find("pmultigrid") == 0)
                               ? diagnostic.pMultigridPath
                               : diagnostic.inversePath;

        output << std::left << std::setw(8) << fineP
               << std::setw(24) << basisFamily
               << std::setw(56) << strategy.name
               << std::setw(22) << status
               << std::setw(12) << diagnostic.iterations
               << std::setw(14) << std::setprecision(6) << std::fixed << run.seconds
               << std::setw(18) << strategy.multigridStrategy
               << std::setw(18) << strategy.smoother
               << std::setw(14) << strategy.coarseSolver
               << std::setw(12) << strategy.chebyshevDegree
               << std::setw(12) << std::setprecision(3) << strategy.chebyshevRatio
               << std::setw(14) << strategy.chebyshevMaxEigenIterations
               << std::setw(14) << std::setprecision(6) << std::fixed
               << ((diagnostic.pMultigridFineSmootherWeight > 0.0) ? diagnostic.pMultigridFineSmootherWeight : fineSmootherWeight)
               << std::setw(14) << std::setprecision(6) << std::scientific << diagnostic.residualReduction
               << std::setw(14) << diagnostic.finalResidualNorm
               << std::setw(18) << diagnostic.localPMultigridWrapperDifferenceInfNorm
               << std::setw(18) << diagnostic.localPMultigridWrapperRelativeDifferenceInfNorm
               << path << "\n";
        if ((fineP == 2) && (strategy.multigridStrategy == "V_CYCLE"))
        {
          const double printedFineSmootherWeight = (diagnostic.pMultigridFineSmootherWeight > 0.0) ? diagnostic.pMultigridFineSmootherWeight : fineSmootherWeight;
          std::ostringstream fineWtRow;
          fineWtRow << "P2_PMG_FINEWT_RESULT"
                    << " basisFamily=" << basisFamily
                    << " strategy=" << strategy.name
                    << " fineWt=" << std::setprecision(12) << std::defaultfloat << printedFineSmootherWeight
                    << " status=" << status
                    << " iterations=" << diagnostic.iterations
                    << " residualReduction=" << std::setprecision(16) << std::scientific << diagnostic.residualReduction
                    << " initialResidualNorm=" << diagnostic.initialResidualNorm
                    << " finalResidualNorm=" << diagnostic.finalResidualNorm
                    << " matrixFreeGMGStageTrace=" << diagnostic.matrixFreeGMGStageTrace
                    << " matrixFreeGMGStageTraceEntries=" << diagnostic.matrixFreeGMGStageTraceEntries
                    << " path=" << path
                    << "\n";
          output << fineWtRow.str();
          if (options.keepChildOutput) std::cout << fineWtRow.str();
        }
        }
      }
    }
    }

    output << "\nNotes:\n";
    output << "  This driver is intentionally matrix-free at the operator-application layer by\n";
    output << "  reusing TensorPoissonPAMatrixDriver's Vlasov graph-norm PAMatrix terms.\n";
    output << "  The smoother-only rows are the closest GMGOperator-study analog currently ready:\n";
    output << "  Belos GMRES with right Ifpack2 Jacobi/Chebyshev preconditioning and the same RHS.\n";
    output << "  The p-multigrid rows are TensorPoisson local two-level V-cycle preconditioner rows;\n";
    output << "  hierarchical transfer uses Intrepid2::HierarchicalBasisProlongator for P and R=P^T.\n";
    output << "  They are not yet asserted to match assembled GMGOperator stage-by-stage semantics.\n";
    output << "  Unsupported lower-degree p-multigrid rows are retained to make remaining gaps explicit.\n";
    output << "  Child command lines are echoed to stdout with VLASOV_GRAM_MATRIX_FREE_CHILD_COMMAND.\n";

    std::cout << "VLASOV_GRAM_MATRIX_FREE_DRIVER_VERSION matrix_free_gmg_stage_trace_diagnostic" << std::endl;
    std::cout << "Wrote Vlasov Gram matrix-free study results to " << options.outputFile << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "VlasovGramMatrixFreeStudyDriver failed: " << e.what() << std::endl;
    returnCode = 1;
  }
  return returnCode;
}
