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
    int fineH1Order = 7;
    int elementsPerDim = 1;
    int maxIterations = 200;
    int smootherApplications = 1;
    double tolerance = 1e-8;
    double massWeight = 1.0;
    long long maxAssembledEntries = 30000000;
    bool quick = false;
    bool keepChildOutput = false;
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
    double finalResidualNorm = -1.0;
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
                          1, 30.0, 0, true});
    strategies.push_back({"chebyshev-smoother-only-degree-4-ratio-20",
                          "gmres-chebyshev", "SMOOTHER_ONLY", "CHEBYSHEV", "direct",
                          4, 20.0, 10, true});
    strategies.push_back({"chebyshev-smoother-only-degree-8-ratio-30",
                          "gmres-chebyshev", "SMOOTHER_ONLY", "CHEBYSHEV", "direct",
                          8, 30.0, 10, true});
    if (!quick)
    {
      strategies.push_back({"chebyshev-smoother-only-degree-16-ratio-60",
                            "gmres-chebyshev", "SMOOTHER_ONLY", "CHEBYSHEV", "direct",
                            16, 60.0, 20, true});
    }

    // These are matrix-free p-multigrid inverse diagnostics implemented in
    // TensorPoissonPAMatrixDriver as Belos GMRES right-preconditioner operators,
    // matching the intended GMGOperator-style V-cycle study rows.
    strategies.push_back({"pmg-chebyshev-smoother-direct-coarse",
                          "pmultigrid-chebyshev", "V_CYCLE", "CHEBYSHEV", "direct",
                          8, 30.0, 10, true});
    strategies.push_back({"pmg-tpetra-chebyshev-smoother-direct-coarse",
                          "gmres-pmultigrid-chebyshev", "V_CYCLE", "CHEBYSHEV", "direct",
                          8, 30.0, 10, true});
    strategies.push_back({"pmg-jacobi-smoother-chebyshev-coarse",
                          "pmultigrid-jacobi-chebyshev", "V_CYCLE", "POINT_JACOBI", "chebyshev",
                          8, 30.0, 10, true});
    strategies.push_back({"pmg-tpetra-jacobi-smoother-chebyshev-coarse",
                          "gmres-pmultigrid-jacobi-chebyshev", "V_CYCLE", "POINT_JACOBI", "chebyshev",
                          8, 30.0, 10, true});
    strategies.push_back({"pmg-chebyshev-smoother-chebyshev-coarse",
                          "pmultigrid-chebyshev", "V_CYCLE", "CHEBYSHEV", "chebyshev",
                          8, 30.0, 10, true});
    strategies.push_back({"pmg-chebyshev-smoother-jacobi-coarse",
                          "pmultigrid-chebyshev-jacobi-coarse", "V_CYCLE", "CHEBYSHEV", "jacobi",
                          8, 30.0, 10, false});
    return strategies;
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
                  << "       [--elementsPerDim=N] [--maxIterations=N] [--smootherApplications=N]\n"
                  << "       [--tolerance=tol] [--massWeight=w] [--maxAssembledEntries=N] [--keepChildOutput=0|1]\n";
        std::exit(0);
      }

      const std::string::size_type equals = argument.find('=');
      if (equals == std::string::npos)
        throw std::invalid_argument("Expected --key=value argument, got: " + argument);
      const std::string key = trimLeadingDashes(argument.substr(0, equals));
      const std::string value = argument.substr(equals + 1);
      if (key == "outputFile") options.outputFile = value;
      else if (key == "tensorDriverPath") options.tensorDriverPath = value;
      else if ((key == "fineH1Order") || (key == "maxFineP")) options.fineH1Order = std::stoi(value);
      else if (key == "elementsPerDim") options.elementsPerDim = std::stoi(value);
      else if (key == "maxIterations") options.maxIterations = std::stoi(value);
      else if (key == "smootherApplications") options.smootherApplications = std::stoi(value);
      else if (key == "tolerance") options.tolerance = std::stod(value);
      else if (key == "massWeight") options.massWeight = std::stod(value);
      else if (key == "maxAssembledEntries") options.maxAssembledEntries = std::stoll(value);
      else if (key == "quick") options.quick = parseBoolValue(value);
      else if (key == "keepChildOutput") options.keepChildOutput = parseBoolValue(value);
      else throw std::invalid_argument("Unknown option: " + argument);
    }

    if (options.fineH1Order < 1) throw std::invalid_argument("--fineH1Order must be positive");
    if (options.elementsPerDim < 1) throw std::invalid_argument("--elementsPerDim must be positive");
    if (options.maxIterations < 1) throw std::invalid_argument("--maxIterations must be positive");
    if (options.smootherApplications < 1) throw std::invalid_argument("--smootherApplications must be positive");
    if (options.tolerance <= 0.0) throw std::invalid_argument("--tolerance must be positive");
    if (options.massWeight <= 0.0) throw std::invalid_argument("--massWeight must be positive");
    if (options.maxAssembledEntries < 1) throw std::invalid_argument("--maxAssembledEntries must be positive");
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
    diagnostic.finalResidualNorm = parseDoubleAfter(output, "finalResidualNorm", lineStart, -1.0);
    diagnostic.stoppedReason = keyValueAfter(output, "inverseStoppedReason", lineStart);
    diagnostic.inversePath = keyValueAfter(output, "inversePath", lineStart);
    diagnostic.pMultigridPath = keyValueAfter(output, "pMultigridPath", lineStart);
    if (diagnostic.stoppedReason.empty()) diagnostic.stoppedReason = "missing";
    if (diagnostic.inversePath.empty()) diagnostic.inversePath = "missing";
    if (diagnostic.pMultigridPath.empty()) diagnostic.pMultigridPath = "missing";
    return diagnostic;
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

  std::string commandArguments(const StudyOptions &options, const Strategy &strategy, const int p)
  {
    std::ostringstream arguments;
    arguments << "--dim=4"
              << " --polyOrder=" << p
              << " --cellsPerDim=" << options.elementsPerDim
              << " --operator=vlasov-graph-norm-solution-coeff-1d1t2v"
              << " --basisMode=hgrad"
              << " --mode=inverse-diagnostic"
              << " --inverseDiagnosticAlgorithm=" << strategy.matrixFreeAlgorithm
              << " --numInverseIterations=" << options.maxIterations
              << " --inverseTolerance=" << std::setprecision(16) << options.tolerance
              << " --inverseInitialGuess=zero"
              << " --inverseRhs=constant-test-function"
              << " --massWeight=" << std::setprecision(16) << options.massWeight
              << " --stiffnessWeight=1"
              << " --maxAssembledEntries=" << options.maxAssembledEntries;

    if ((strategy.matrixFreeAlgorithm == "gmres-chebyshev") || (strategy.matrixFreeAlgorithm == "gmres-jacobi"))
    {
      arguments << " --inversePreconditionerDegree=" << strategy.chebyshevDegree
                << " --inverseEigenRatio=" << std::setprecision(16) << strategy.chebyshevRatio
                << " --inverseSpectralEstimateIterations=" << strategy.chebyshevMaxEigenIterations
                << " --inverseDiagnosticRequireAssembledReference=1";
    }
    else if (strategy.matrixFreeAlgorithm == "gmres-pmultigrid-chebyshev")
    {
      arguments << " --pMultigridChebyshevSmootherIterations=" << std::max(1, options.smootherApplications * strategy.chebyshevDegree)
                << " --pMultigridChebyshevSmootherLowerFraction=" << std::setprecision(16) << (1.0 / strategy.chebyshevRatio)
                << " --inverseSpectralEstimate=assembled-gershgorin"
                << " --inverseLowerSpectralEstimate=ratio"
                << " --inverseEigenRatio=" << std::setprecision(16) << strategy.chebyshevRatio;
    }
    else if (strategy.matrixFreeAlgorithm == "gmres-pmultigrid-jacobi-chebyshev")
    {
      arguments << " --pMultigridJacobiSmootherIterations=" << options.smootherApplications
                << " --pMultigridCoarseChebyshevIterations=" << strategy.chebyshevDegree
                << " --pMultigridCoarseChebyshevLowerFraction=" << std::setprecision(16) << (1.0 / strategy.chebyshevRatio)
                << " --inverseSpectralEstimate=assembled-gershgorin"
                << " --inverseLowerSpectralEstimate=ratio"
                << " --inverseEigenRatio=" << std::setprecision(16) << strategy.chebyshevRatio;
    }
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
    out << "  p-multigrid rows use the currently available local p-multigrid diagnostic recurrences;\n";
    out << "  GMGOperator-equivalent matrix-free V-cycle-as-right-preconditioner rows remain marked\n";
    out << "  by strategy names/paths until that lower-level preconditioner object exists.\n\n";
    out << "Tensor driver: " << tensorDriverPath << "\n\n";
    out << "Discretization:\n";
    out << "  space: tensor HGRAD\n";
    out << "  basis family: Intrepid2 hierarchical tensor HGRAD\n";
    out << "  elements per dimension: " << options.elementsPerDim << "\n";
    out << "  fine p degree range: 1 .. " << options.fineH1Order << "\n";
    out << "  p coarsening schedule: for each fine p, p -> max(p/2,1) -> ... -> 1; max-p schedule is "
        << pCoarsenScheduleString(options.fineH1Order) << "\n";
    out << "  massWeight: " << options.massWeight << "\n\n";
    out << "Solver settings:\n";
    out << "  outer Krylov: Belos GMRES for smoother-only strategies\n";
    out << "  tolerance: " << options.tolerance << "\n";
    out << "  max iterations / V-cycles: " << options.maxIterations << "\n";
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
        << std::setw(14) << "maxEigIters"
        << std::setw(18) << "residualRed."
        << "path" << "\n";
    out << std::string(220, '-') << "\n";
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

    for (int fineP = 1; fineP <= options.fineH1Order; fineP++)
    {
      for (const Strategy &strategy : defaultStrategies(options.quick))
      {
        const std::string strategyName = "p=" + std::to_string(fineP) + " " + strategy.name;
        if (!strategy.supported || ((strategy.multigridStrategy == "V_CYCLE") && (fineP < 2)))
        {
          output << std::left << std::setw(56) << strategyName
                 << std::setw(16) << "not_supported"
                 << std::setw(12) << -1
                 << std::setw(14) << std::setprecision(6) << std::fixed << 0.0
                 << std::setw(18) << strategy.multigridStrategy
                 << std::setw(18) << strategy.smoother
                 << std::setw(14) << strategy.coarseSolver
                 << std::setw(12) << strategy.chebyshevDegree
                 << std::setw(12) << std::setprecision(3) << strategy.chebyshevRatio
                 << std::setw(14) << strategy.chebyshevMaxEigenIterations
                 << std::setw(18) << "n/a"
                 << "matrix_free_preconditioner_gap" << "\n";
          continue;
        }

        const DriverRun run = runDriver(options.tensorDriverPath, commandArguments(options, strategy, fineP), options.keepChildOutput);
        const Diagnostic diagnostic = parseDiagnostic(run.output);
        const bool childSucceeded = (run.status == 0);
        const std::string status = (!childSucceeded) ? "failed"
                                 : (!diagnostic.found) ? "missing_diag"
                                 : (diagnostic.converged != 0) ? "converged"
                                 : "not_converged";
        const std::string path = (strategy.matrixFreeAlgorithm.find("pmultigrid") == 0)
                               ? diagnostic.pMultigridPath
                               : diagnostic.inversePath;

        output << std::left << std::setw(56) << strategyName
               << std::setw(16) << status
               << std::setw(12) << diagnostic.iterations
               << std::setw(14) << std::setprecision(6) << std::fixed << run.seconds
               << std::setw(18) << strategy.multigridStrategy
               << std::setw(18) << strategy.smoother
               << std::setw(14) << strategy.coarseSolver
               << std::setw(12) << strategy.chebyshevDegree
               << std::setw(12) << std::setprecision(3) << strategy.chebyshevRatio
               << std::setw(14) << strategy.chebyshevMaxEigenIterations
               << std::setw(18) << std::setprecision(6) << std::scientific << diagnostic.residualReduction
               << path << "\n";
      }
    }

    output << "\nNotes:\n";
    output << "  This driver is intentionally matrix-free at the operator-application layer by\n";
    output << "  reusing TensorPoissonPAMatrixDriver's Vlasov graph-norm PAMatrix terms.\n";
    output << "  The smoother-only rows are the closest GMGOperator-study analog currently ready:\n";
    output << "  Belos GMRES with right Ifpack2 Jacobi/Chebyshev preconditioning and the same RHS.\n";
    output << "  The p-multigrid rows are diagnostic local V-cycle recurrences, not yet the exact\n";
    output << "  GMGOperator semantics of a matrix-free V-cycle object used as a right preconditioner.\n";
    output << "  Unsupported rows are retained to make that remaining implementation gap explicit.\n";
    output << "  Child command lines are echoed to stdout with VLASOV_GRAM_MATRIX_FREE_CHILD_COMMAND.\n";

    std::cout << "Wrote Vlasov Gram matrix-free study results to " << options.outputFile << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << "VlasovGramMatrixFreeStudyDriver failed: " << e.what() << std::endl;
    returnCode = 1;
  }
  return returnCode;
}
