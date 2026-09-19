//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <algorithm>
#include <cmath>
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

  std::vector<double> parseAllInverseDiagnosticDoubleKey(const std::string &output, const std::string &key)
  {
    std::vector<double> values;
    std::string::size_type cursor = 0;
    while (true)
    {
      const std::string::size_type lineStart = output.find("INVERSE_DIAGNOSTIC ", cursor);
      if (lineStart == std::string::npos) break;
      values.push_back(parseDoubleKeyAfter(output, key, lineStart));
      cursor = lineStart + 1;
    }
    return values;
  }

  std::vector<double> parseAllInverseDiagnosticResidualReductions(const std::string &output)
  {
    return parseAllInverseDiagnosticDoubleKey(output, "residualReduction");
  }

  double minimumResidualReduction(const std::vector<double> &values)
  {
    if (values.empty())
      throw std::runtime_error("No INVERSE_DIAGNOSTIC residualReduction fields were found");
    return *std::min_element(values.begin(), values.end());
  }

  void assertIterationsCloseToAssembledStudy(const std::vector<double> &iterations,
                                             const std::vector<double> &converged,
                                             const std::vector<int> &expectedIterations,
                                             const std::string &label,
                                             const int absoluteSlack,
                                             const double relativeSlack,
                                             Teuchos::FancyOStream &out,
                                             bool &success)
  {
    TEST_EQUALITY(iterations.size(), expectedIterations.size());
    TEST_EQUALITY(converged.size(), expectedIterations.size());
    for (std::size_t ordinal=0; ordinal<iterations.size(); ordinal++)
    {
      const int p = static_cast<int>(ordinal) + 1;
      const int observedIterations = static_cast<int>(std::lround(iterations[ordinal]));
      const int tolerance = std::max(absoluteSlack,
                                     static_cast<int>(std::ceil(relativeSlack * expectedIterations[ordinal])));
      std::cout << label << " p=" << p
                << " observedIterations=" << observedIterations
                << " expectedAssembledGMGIterations=" << expectedIterations[ordinal]
                << " tolerance=" << tolerance
                << " converged=" << converged[ordinal]
                << std::endl;
      TEST_ASSERT(std::isfinite(iterations[ordinal]));
      TEST_ASSERT(std::isfinite(converged[ordinal]));
      if (ordinal == 0)
      {
        TEST_COMPARE(converged[ordinal], ==, 1.0);
        TEST_COMPARE(std::abs(observedIterations - expectedIterations[ordinal]), <=, tolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, ReportsL2MassWeight )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=1 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=diagonal "
      "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=10000");

  TEST_EQUALITY(run.status, 0);
  TEST_INEQUALITY(run.output.find("termWeights="), std::string::npos);
  TEST_INEQUALITY(run.output.find("vlasov_graph_norm_l2:1000000.000000"), std::string::npos);
  std::cout << "VLASOV_GRAPH_NORM_L2_WEIGHT observed=1000000.000000 source=massWeight" << std::endl;
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, GMRESChebyshevMatchesAssembledGMGSmootherOnlyP1ToP7 )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=1:7 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=gmres-chebyshev "
      "--numInverseIterations=200 --inversePreconditionerDegree=8 "
      "--inverseEigenRatio=30 --inverseSpectralEstimateIterations=10 "
      "--inverseInitialGuess=zero --inverseRhs=constant-test-function "
      "--massWeight=1 --stiffnessWeight=1 --maxAssembledEntries=30000000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> iterations = parseAllInverseDiagnosticDoubleKey(run.output, "inverseIterations");
  const std::vector<double> converged = parseAllInverseDiagnosticDoubleKey(run.output, "inverseConverged");
  const std::vector<int> assembledStudyIterations {7, 13, 26, 42, 60, 87, 112};
  assertIterationsCloseToAssembledStudy(iterations, converged, assembledStudyIterations,
                                        "VLASOV_GRAPH_NORM_GMRES_CHEBYSHEV_ASSEMBLED_STUDY_MATCH",
                                        4, 0.25, out, success);
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, GMRESJacobiMatchesAssembledGMGSmootherOnlyP1ToP3 )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=1:3 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=gmres-jacobi "
      "--numInverseIterations=200 --inverseInitialGuess=zero --inverseRhs=constant-test-function "
      "--massWeight=1 --stiffnessWeight=1 --maxAssembledEntries=3000000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> iterations = parseAllInverseDiagnosticDoubleKey(run.output, "inverseIterations");
  const std::vector<double> converged = parseAllInverseDiagnosticDoubleKey(run.output, "inverseConverged");
  const std::vector<int> assembledStudyIterations {16, 52, 116};
  assertIterationsCloseToAssembledStudy(iterations, converged, assembledStudyIterations,
                                        "VLASOV_GRAPH_NORM_GMRES_JACOBI_ASSEMBLED_STUDY_MATCH",
                                        10, 0.30, out, success);
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, GMRESChebyshevConvergesForH1Gram )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=1:3 --cellsPerDim=1 "
      "--operator=h1-gram --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=gmres-chebyshev "
      "--numInverseIterations=80 --inversePreconditionerDegree=8 "
      "--inverseEigenRatio=30 --inverseSpectralEstimateIterations=10 "
      "--inverseInitialGuess=zero "
      "--massWeight=1 --stiffnessWeight=1 --maxAssembledEntries=2000000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> iterations = parseAllInverseDiagnosticDoubleKey(run.output, "inverseIterations");
  const std::vector<double> converged = parseAllInverseDiagnosticDoubleKey(run.output, "inverseConverged");
  TEST_EQUALITY(iterations.size(), static_cast<std::size_t>(3));
  TEST_EQUALITY(converged.size(), static_cast<std::size_t>(3));
  for (std::size_t ordinal=0; ordinal<iterations.size(); ordinal++)
  {
    const int p = static_cast<int>(ordinal) + 1;
    std::cout << "H1_GRAM_GMRES_CHEBYSHEV p=" << p
              << " iterations=" << iterations[ordinal]
              << " converged=" << converged[ordinal]
              << std::endl;
    TEST_ASSERT(std::isfinite(iterations[ordinal]));
    TEST_COMPARE(converged[ordinal], ==, 1.0);
    TEST_COMPARE(iterations[ordinal], <, 80.0);
  }
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
      "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=100000");

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

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, AssembledTrueEigenDiagnosticsForP2ToP5 )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=2:5 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=chebyshev "
      "--inverseSpectralEstimate=power --inverseSpectralEstimateIterations=20 "
      "--inverseTrueEigenDiagnostic=1 "
      "--inverseLowerSpectralEstimate=ratio --inverseEigenRatio=30 "
      "--numInverseIterations=6 --inverseInitialGuess=zero "
      "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=3000000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> trueMin = parseAllInverseDiagnosticDoubleKey(run.output, "trueLambdaMin");
  const std::vector<double> trueMax = parseAllInverseDiagnosticDoubleKey(run.output, "trueLambdaMax");
  const std::vector<double> trueCondition = parseAllInverseDiagnosticDoubleKey(run.output, "trueCondition");
  const std::vector<double> trueMaxOverGershgorin = parseAllInverseDiagnosticDoubleKey(run.output, "trueLambdaMaxOverAssembledGershgorin");
  const std::vector<double> trueMaxOverPowerRaw = parseAllInverseDiagnosticDoubleKey(run.output, "trueLambdaMaxOverPowerRaw");
  const std::vector<double> trueMinOverLowerEstimate = parseAllInverseDiagnosticDoubleKey(run.output, "trueLambdaMinOverLowerEstimate");
  const std::vector<double> residualReduction = parseAllInverseDiagnosticResidualReductions(run.output);
  TEST_EQUALITY(trueMin.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(trueMax.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(trueCondition.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(trueMaxOverGershgorin.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(trueMaxOverPowerRaw.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(trueMinOverLowerEstimate.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(residualReduction.size(), static_cast<std::size_t>(4));
  for (std::size_t ordinal=0; ordinal<trueMin.size(); ordinal++)
  {
    const int p = static_cast<int>(ordinal) + 2;
    std::cout << "VLASOV_GRAPH_NORM_TRUE_EIGEN p=" << p
              << " trueLambdaMin=" << trueMin[ordinal]
              << " trueLambdaMax=" << trueMax[ordinal]
              << " trueCondition=" << trueCondition[ordinal]
              << " trueLambdaMaxOverAssembledGershgorin=" << trueMaxOverGershgorin[ordinal]
              << " trueLambdaMaxOverPowerRaw=" << trueMaxOverPowerRaw[ordinal]
              << " trueLambdaMinOverLowerEstimate=" << trueMinOverLowerEstimate[ordinal]
              << " residualReduction=" << residualReduction[ordinal]
              << std::endl;
    TEST_ASSERT(std::isfinite(trueMin[ordinal]));
    TEST_ASSERT(std::isfinite(trueMax[ordinal]));
    TEST_ASSERT(std::isfinite(trueCondition[ordinal]));
    TEST_ASSERT(std::isfinite(trueMaxOverGershgorin[ordinal]));
    TEST_ASSERT(std::isfinite(trueMaxOverPowerRaw[ordinal]));
    TEST_ASSERT(std::isfinite(trueMinOverLowerEstimate[ordinal]));
    TEST_COMPARE(trueMin[ordinal], >, 0.0);
    TEST_COMPARE(trueMax[ordinal], >, trueMin[ordinal]);
    TEST_COMPARE(trueCondition[ordinal], >, 1.0);
    TEST_COMPARE(trueMaxOverGershgorin[ordinal], >, 0.0);
    TEST_COMPARE(trueMaxOverPowerRaw[ordinal], >, 0.0);
    TEST_COMPARE(trueMinOverLowerEstimate[ordinal], >, 0.0);
  }
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, OracleTrueLambdaMinChebyshevForP2ToP5 )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=2:5 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=chebyshev "
      "--inverseSpectralEstimate=power --inverseSpectralEstimateIterations=20 "
      "--inverseTrueEigenDiagnostic=1 "
      "--inverseLowerSpectralEstimate=true-eigen "
      "--numInverseIterations=30 --inverseInitialGuess=zero "
      "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=3000000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> reductions = parseAllInverseDiagnosticResidualReductions(run.output);
  const std::vector<double> lowerRatios = parseAllInverseDiagnosticDoubleKey(run.output, "trueLambdaMinOverLowerEstimate");
  TEST_EQUALITY(reductions.size(), static_cast<std::size_t>(4));
  TEST_EQUALITY(lowerRatios.size(), static_cast<std::size_t>(4));
  for (std::size_t ordinal=0; ordinal<reductions.size(); ordinal++)
  {
    const int p = static_cast<int>(ordinal) + 2;
    std::cout << "VLASOV_GRAPH_NORM_ORACLE_CHEBYSHEV p=" << p
              << " residualReduction=" << reductions[ordinal]
              << " trueLambdaMinOverLowerEstimate=" << lowerRatios[ordinal]
              << std::endl;
    TEST_ASSERT(std::isfinite(reductions[ordinal]));
    TEST_ASSERT(std::isfinite(lowerRatios[ordinal]));
    TEST_FLOATING_EQUALITY(lowerRatios[ordinal], 1.0, 1e-10);
  }
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, ConservativeTrueEigenBoundsAndOperatorSanity )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const DriverRun run = runDriver(
      "--dim=4 --polyOrder=2:3 --cellsPerDim=1 "
      "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
      "--inverseDiagnosticAlgorithm=chebyshev "
      "--inverseSpectralEstimate=true-eigen --inversePowerSafetyFactor=1.1 "
      "--inverseLowerSpectralEstimate=true-eigen-safe --inverseLowerPowerSafetyFactor=0.9 "
      "--inverseTrueEigenDiagnostic=1 --inverseOperatorCheckSamples=3 "
      "--numInverseIterations=6 --inverseInitialGuess=zero "
      "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=3000000");

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> reductions = parseAllInverseDiagnosticResidualReductions(run.output);
  const std::vector<double> actionRel = parseAllInverseDiagnosticDoubleKey(run.output, "operatorActionMaxRelDiff");
  const std::vector<double> symmetry = parseAllInverseDiagnosticDoubleKey(run.output, "operatorBilinearSymmetryAbsError");
  const std::vector<double> minRayleigh = parseAllInverseDiagnosticDoubleKey(run.output, "operatorMinRayleighQuotient");
  const std::vector<double> scalarAmplification = parseAllInverseDiagnosticDoubleKey(run.output, "chebyshevScalarMaxAbsAmplification");
  const std::vector<double> scalarAmplifiedModes = parseAllInverseDiagnosticDoubleKey(run.output, "chebyshevScalarNumModesAmplified");
  const std::vector<double> belowLower = parseAllInverseDiagnosticDoubleKey(run.output, "chebyshevScalarNumBelowLower");
  const std::vector<double> aboveUpper = parseAllInverseDiagnosticDoubleKey(run.output, "chebyshevScalarNumAboveUpper");
  TEST_EQUALITY(reductions.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(actionRel.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(symmetry.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(minRayleigh.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(scalarAmplification.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(scalarAmplifiedModes.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(belowLower.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(aboveUpper.size(), static_cast<std::size_t>(2));
  for (std::size_t ordinal=0; ordinal<reductions.size(); ordinal++)
  {
    const int p = static_cast<int>(ordinal) + 2;
    std::cout << "VLASOV_GRAPH_NORM_CHEBYSHEV_SANITY p=" << p
              << " residualReduction=" << reductions[ordinal]
              << " operatorActionMaxRelDiff=" << actionRel[ordinal]
              << " operatorBilinearSymmetryAbsError=" << symmetry[ordinal]
              << " operatorMinRayleighQuotient=" << minRayleigh[ordinal]
              << " chebyshevScalarMaxAbsAmplification=" << scalarAmplification[ordinal]
              << " chebyshevScalarNumModesAmplified=" << scalarAmplifiedModes[ordinal]
              << " chebyshevScalarNumBelowLower=" << belowLower[ordinal]
              << " chebyshevScalarNumAboveUpper=" << aboveUpper[ordinal]
              << std::endl;
    TEST_ASSERT(std::isfinite(reductions[ordinal]));
    // This is an interrogation/triage diagnostic: residual growth is allowed here
    // because it is the symptom we are trying to classify.  The stricter checks
    // below verify whether the assembled operator, matrix-free action, and scalar
    // Chebyshev polynomial are individually sane.
    TEST_COMPARE(actionRel[ordinal], <, 1e-10);
    TEST_COMPARE(symmetry[ordinal], <, 1e-10);
    TEST_COMPARE(minRayleigh[ordinal], >, 0.0);
    TEST_COMPARE(scalarAmplification[ordinal], <=, 1.0 + 1e-10);
    TEST_FLOATING_EQUALITY(scalarAmplifiedModes[ordinal], 0.0, 1e-12);
    TEST_FLOATING_EQUALITY(belowLower[ordinal], 0.0, 1e-12);
    TEST_FLOATING_EQUALITY(aboveUpper[ordinal], 0.0, 1e-12);
  }
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, PMultigridChebyshevCachedHgradSweepBeyondP2 )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  const std::vector<int> cycleCounts {1, 2};
  const std::vector<int> smootherDegrees {1, 2, 4};
  double bestReduction = std::numeric_limits<double>::infinity();
  int bestP = -1;
  int bestCycles = -1;
  int bestSmootherDegree = -1;

  for (const int cycles : cycleCounts)
  {
    for (const int smootherDegree : smootherDegrees)
    {
      std::ostringstream args;
      args << "--dim=4 --polyOrder=2:3 --cellsPerDim=1 "
           << "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
           << "--inverseDiagnosticAlgorithm=pmultigrid-chebyshev "
           << "--inverseSpectralEstimate=power --inverseSpectralEstimateIterations=20 "
           << "--inverseLowerSpectralEstimate=ratio --inverseEigenRatio=30 "
           << "--numInverseIterations=" << cycles << " "
           << "--pMultigridChebyshevSmootherIterations=" << smootherDegree << " "
           << "--pMultigridChebyshevSmootherLowerFraction=0.30 "
           << "--inverseInitialGuess=zero "
           << "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=3000000";
      const DriverRun run = runDriver(args.str());

      TEST_EQUALITY(run.status, 0);
      const std::vector<double> reductions = parseAllInverseDiagnosticResidualReductions(run.output);
      const std::vector<double> levels = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridLevels");
      const std::vector<double> coarseDofs = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridCoarseDofs");
      const std::vector<double> smootherFractions = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridChebyshevSmootherLowerFraction");
      TEST_EQUALITY(reductions.size(), static_cast<std::size_t>(2));
      TEST_EQUALITY(levels.size(), static_cast<std::size_t>(2));
      TEST_EQUALITY(coarseDofs.size(), static_cast<std::size_t>(2));
      TEST_EQUALITY(smootherFractions.size(), static_cast<std::size_t>(2));
      for (std::size_t ordinal=0; ordinal<reductions.size(); ordinal++)
      {
        const int p = static_cast<int>(ordinal) + 2;
        int expectedCoarseDofs = 1;
        for (int d=0; d<4; d++) expectedCoarseDofs *= p;
        std::cout << "VLASOV_GRAPH_NORM_PMULTIGRID_CHEBYSHEV_SWEEP p=" << p
                  << " cycles=" << cycles
                  << " smootherDegree=" << smootherDegree
                  << " smootherLowerFraction=" << (smootherFractions.empty() ? -1.0 : smootherFractions[ordinal])
                  << " residualReduction=" << reductions[ordinal]
                  << " levels=" << (levels.empty() ? -1.0 : levels[ordinal])
                  << " coarseDofs=" << (coarseDofs.empty() ? -1.0 : coarseDofs[ordinal])
                  << std::endl;
        TEST_ASSERT(std::isfinite(reductions[ordinal]));
        TEST_COMPARE(levels[ordinal], ==, 2.0);
        TEST_COMPARE(coarseDofs[ordinal], ==, static_cast<double>(expectedCoarseDofs));
        TEST_FLOATING_EQUALITY(smootherFractions[ordinal], 0.30, 1e-12);
        if (reductions[ordinal] < bestReduction)
        {
          bestReduction = reductions[ordinal];
          bestP = p;
          bestCycles = cycles;
          bestSmootherDegree = smootherDegree;
        }
      }
    }
  }

  std::cout << "VLASOV_GRAPH_NORM_PMULTIGRID_CHEBYSHEV_BEST p=" << bestP
            << " cycles=" << bestCycles
            << " smootherDegree=" << bestSmootherDegree
            << " residualReduction=" << bestReduction << std::endl;
  TEST_ASSERT(std::isfinite(bestReduction));
}

TEUCHOS_UNIT_TEST( TensorVlasovGraphNormChebyshev, PMultigridJacobiSmootherChebyshevCoarseDiagnostic )
{
  if (!shouldRunExternalDiagnosticsOnThisRank()) return;

  std::ostringstream args;
  args << "--dim=4 --polyOrder=2:3 --cellsPerDim=1 "
       << "--operator=vlasov-graph-norm-1d1t2v --basisMode=hgrad --mode=inverse-diagnostic "
       << "--inverseDiagnosticAlgorithm=pmultigrid-jacobi-chebyshev "
       << "--inverseSpectralEstimate=power --inverseSpectralEstimateIterations=20 "
       << "--inverseLowerSpectralEstimate=ratio --inverseEigenRatio=30 "
       << "--numInverseIterations=2 "
       << "--pMultigridJacobiSmootherIterations=2 "
       << "--pMultigridCoarseChebyshevIterations=6 "
       << "--pMultigridCoarseChebyshevLowerFraction=0.05 "
       << "--inverseInitialGuess=zero "
       << "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=3000000";
  const DriverRun run = runDriver(args.str());

  TEST_EQUALITY(run.status, 0);
  const std::vector<double> reductions = parseAllInverseDiagnosticResidualReductions(run.output);
  const std::vector<double> levels = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridLevels");
  const std::vector<double> coarseDofs = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridCoarseDofs");
  const std::vector<double> jacobiSweeps = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridJacobiSmootherIterations");
  const std::vector<double> coarseChebyshevIterations = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridCoarseChebyshevIterations");
  const std::vector<double> coarseChebyshevFractions = parseAllInverseDiagnosticDoubleKey(run.output, "pMultigridCoarseChebyshevLowerFraction");
  TEST_EQUALITY(reductions.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(levels.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(coarseDofs.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(jacobiSweeps.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(coarseChebyshevIterations.size(), static_cast<std::size_t>(2));
  TEST_EQUALITY(coarseChebyshevFractions.size(), static_cast<std::size_t>(2));
  for (std::size_t ordinal=0; ordinal<reductions.size(); ordinal++)
  {
    const int p = static_cast<int>(ordinal) + 2;
    std::cout << "VLASOV_GRAPH_NORM_PMULTIGRID_JACOBI_CHEBYSHEV_COARSE p=" << p
              << " residualReduction=" << reductions[ordinal]
              << " levels=" << levels[ordinal]
              << " coarseDofs=" << coarseDofs[ordinal]
              << " jacobiSweeps=" << jacobiSweeps[ordinal]
              << " coarseChebyshevIterations=" << coarseChebyshevIterations[ordinal]
              << " coarseChebyshevLowerFraction=" << coarseChebyshevFractions[ordinal]
              << std::endl;
    TEST_ASSERT(std::isfinite(reductions[ordinal]));
    TEST_COMPARE(levels[ordinal], ==, static_cast<double>(p));
    TEST_COMPARE(coarseDofs[ordinal], ==, 16.0);
    TEST_COMPARE(jacobiSweeps[ordinal], ==, 2.0);
    TEST_COMPARE(coarseChebyshevIterations[ordinal], ==, 6.0);
    TEST_FLOATING_EQUALITY(coarseChebyshevFractions[ordinal], 0.05, 1e-12);
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
           << "--massWeight=1e6 --stiffnessWeight=0.5 --maxAssembledEntries=1";
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
