//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace
{
  bool shouldRunExternalDiagnosticsOnThisRank()
  {
    if (Teuchos::GlobalMPISession::getRank() != 0)
    {
      std::cout << "VLASOV_GRAPH_NORM_DRIVER_SKIP rank=" << Teuchos::GlobalMPISession::getRank()
                << " reason=external_diagnostics_run_on_rank_0_only" << std::endl;
      return false;
    }
    return true;
  }

  struct DriverRun
  {
    int status = -1;
    std::string command;
    std::string output;
  };

  std::string tensorPoissonPAMatrixDriverPath()
  {
#ifdef TENSOR_POISSON_PAMATRIX_DRIVER_PATH
    return TENSOR_POISSON_PAMATRIX_DRIVER_PATH;
#else
    return std::string();
#endif
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

  std::string makeTemporaryPath()
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
    path << "camellia_vlasov_graph_norm_chebyshev_" << processId << "_" << counter++ << ".txt";
    return path.str();
  }

  DriverRun runDriver(const std::string &arguments)
  {
    const std::string driver = tensorPoissonPAMatrixDriverPath();
    if (driver.empty())
      throw std::runtime_error("TENSOR_POISSON_PAMATRIX_DRIVER_PATH is not defined; TensorPoissonPAMatrixDriver target is unavailable");

    const std::string outputPath = makeTemporaryPath();
    DriverRun result;
    result.command = shellQuote(driver) + " " + arguments + " > " + shellQuote(outputPath) + " 2>&1";
    result.status = std::system(result.command.c_str());

    std::ifstream input(outputPath.c_str());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    result.output = buffer.str();
    std::remove(outputPath.c_str());

    std::cout << "VLASOV_GRAPH_NORM_DRIVER_COMMAND " << result.command << std::endl;
    std::cout << result.output;
    if (result.output.empty() || (result.output.back() != '\n')) std::cout << std::endl;
    std::cout << "VLASOV_GRAPH_NORM_DRIVER_STATUS rawStatus=" << result.status << std::endl;
    return result;
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

  double parseDoubleKeyAfter(const std::string &text, const std::string &key, std::string::size_type start)
  {
    const std::string value = keyValueAfter(text, key, start);
    if (value.empty())
      throw std::runtime_error("Could not find key '" + key + "' in TensorPoissonPAMatrixDriver output");
    return std::stod(value);
  }

  std::vector<double> parseAllInverseDiagnosticResidualReductions(const std::string &output)
  {
    std::vector<double> reductions;
    std::string::size_type cursor = 0;
    while (true)
    {
      const std::string::size_type lineStart = output.find("INVERSE_DIAGNOSTIC ", cursor);
      if (lineStart == std::string::npos) break;
      reductions.push_back(parseDoubleKeyAfter(output, "residualReduction", lineStart));
      cursor = lineStart + 1;
    }
    return reductions;
  }

  double minimumResidualReduction(const std::vector<double> &values)
  {
    if (values.empty())
      throw std::runtime_error("No INVERSE_DIAGNOSTIC residualReduction fields were found");
    return *std::min_element(values.begin(), values.end());
  }
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, ReportsL2MassWeight )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=1 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=diagonal "
      "--massWeight=2.0 --stiffnessWeight=0.5 --maxAssembledEntries=10000");

  TEST_EQUALITY(run.status, 0);
  TEST_INEQUALITY(run.output.find("termWeights="), std::string::npos);
  TEST_INEQUALITY(run.output.find("vlasov_graph_norm_l2:2.000000"), std::string::npos);
  std::cout << "VLASOV_GRAPH_NORM_L2_WEIGHT observed=2.000000 source=massWeight" << std::endl;
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, AssembledBoundsReduceResidualForP1P2 )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=1:2 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=chebyshev "
      "--inverseSpectralEstimate=assembled-gershgorin "
      "--inverseLowerSpectralEstimate=assembled-inverse-iteration "
      "--inverseLowerSpectralEstimateIterations=20 "
      "--numInverseIterations=30 --inverseInitialGuess=zero "
      "--massWeight=2.0 --stiffnessWeight=0.5 --maxAssembledEntries=100000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> reductions = parseAllInverseDiagnosticResidualReductions(run.output);
  TEST_EQUALITY(reductions.size(), static_cast<std::size_t>(2));
  if (reductions.size() == 2)
  {
    std::cout << "VLASOV_GRAPH_NORM_CHEBYSHEV_ASSEMBLED_BOUNDS p=1 residualReduction=" << reductions[0] << std::endl;
    std::cout << "VLASOV_GRAPH_NORM_CHEBYSHEV_ASSEMBLED_BOUNDS p=2 residualReduction=" << reductions[1] << std::endl;
    TEST_COMPARE(reductions[0], <, 1.0);
  }
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, MatrixFreeCellPowerSweepFindsImprovingParameters )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const std::vector<double> ratios {20.0, 30.0, 60.0};
  const std::vector<int> iterationCounts {10, 30};
  double bestReduction = std::numeric_limits<double>::infinity();
  double bestRatio = -1.0;
  int bestIterations = -1;

  for (const double ratio : ratios)
  {
    for (const int iterations : iterationCounts)
    {
      std::ostringstream args;
      args << "--dim=4 --polyOrder=2 --cellsPerDim=1 "
           << "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
           << "--inverseDiagnosticAlgorithm=chebyshev "
           << "--inverseSpectralEstimate=cell-power --inverseSpectralEstimateIterations=12 "
           << "--inverseLowerSpectralEstimate=ratio --inverseEigenRatio=" << ratio << " "
           << "--numInverseIterations=" << iterations << " --inverseInitialGuess=zero "
           << "--inverseDiagnosticRequireAssembledReference=0 "
           << "--massWeight=2.0 --stiffnessWeight=0.5 --maxAssembledEntries=1";
      const DriverRun run = runDriver(args.str());
      TEST_EQUALITY(run.status, 0);
      const double reduction = minimumResidualReduction(parseAllInverseDiagnosticResidualReductions(run.output));
      std::cout << "VLASOV_GRAPH_NORM_CHEBYSHEV_CELL_POWER_SWEEP ratio=" << ratio
                << " iterations=" << iterations
                << " residualReduction=" << reduction << std::endl;
      if (reduction < bestReduction)
      {
        bestReduction = reduction;
        bestRatio = ratio;
        bestIterations = iterations;
      }
    }
  }

  std::cout << "VLASOV_GRAPH_NORM_CHEBYSHEV_CELL_POWER_BEST p=2 cellsPerDim=1 ratio=" << bestRatio
            << " iterations=" << bestIterations
            << " residualReduction=" << bestReduction << std::endl;
  TEST_COMPARE(bestReduction, <, 2.0);
}
