#include "GMGOperator.h"
#include "MatrixFreeGMGPreconditioner.h"
#include "MatrixFreeGMGPreconditionerComparisonHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"

#include <string>
#include <vector>

using namespace Camellia;

namespace
{
  MatrixFreeGMGPreconditioner::StageTraceEntry stageTraceEntry(const size_t levelIndex,
                                                               const std::string &stageName,
                                                               const double rhsInfNorm,
                                                               const double solutionInfNorm,
                                                               const double residualInfNorm)
  {
    MatrixFreeGMGPreconditioner::StageTraceEntry entry;
    entry.levelIndex = levelIndex;
    entry.stageName = stageName;
    entry.rhsInfNorm = rhsInfNorm;
    entry.solutionInfNorm = solutionInfNorm;
    entry.residualInfNorm = residualInfNorm;
    return entry;
  }
}

TEUCHOS_UNIT_TEST( MatrixFreeGMGPreconditionerStageTraceComparison, AcceptsMatchingSequences )
{
  using StageTraceEntry = MatrixFreeGMGPreconditioner::StageTraceEntry;

  std::vector<StageTraceEntry> expected;
  expected.push_back(stageTraceEntry(0, "initialResidualNorm", 8.0, 0.0, 8.0));
  expected.push_back(stageTraceEntry(1, "postCoarseCorrection", 6.0, 12.0, 1.0));

  const std::vector<StageTraceEntry> actual = expected;
  Test::assertStageTracesMatch(expected, actual);

  const Test::StageTraceComparisonResult result =
    Test::compareStageTraces(expected, actual);
  TEST_EQUALITY(result.sizeMatches, true);
  TEST_EQUALITY(result.firstMismatchField, std::string());
  TEST_FLOATING_EQUALITY(result.maxDifference, 0.0, 1e-14);
}

TEUCHOS_UNIT_TEST( MatrixFreeGMGPreconditionerStageTraceComparison, AcceptsGMGOperatorTraceEntries )
{
  using GMGStageTraceEntry = GMGOperator::VcycleStageTraceEntry;
  using MatrixFreeStageTraceEntry = MatrixFreeGMGPreconditioner::StageTraceEntry;

  GMGStageTraceEntry gmgEntry;
  gmgEntry.levelIndex = 0;
  gmgEntry.stageName = "Initial";
  gmgEntry.rhsInfNorm = 8.0;
  gmgEntry.solutionInfNorm = 0.0;
  gmgEntry.residualInfNorm = 8.0;

  std::vector<GMGStageTraceEntry> expected;
  expected.push_back(gmgEntry);

  std::vector<MatrixFreeStageTraceEntry> actual;
  actual.push_back(stageTraceEntry(0, "Initial", 8.0, 0.0, 8.0));

  Test::assertStageTracesMatch(expected, actual);

  const Test::StageTraceComparisonResult result =
    Test::compareStageTraces(expected, actual);
  TEST_EQUALITY(result.sizeMatches, true);
  TEST_EQUALITY(result.firstMismatchField, std::string());
  TEST_FLOATING_EQUALITY(result.maxDifference, 0.0, 1e-14);
}

TEUCHOS_UNIT_TEST( MatrixFreeGMGPreconditionerStageTraceComparison, ReportsFirstNumericMismatch )
{
  using StageTraceEntry = MatrixFreeGMGPreconditioner::StageTraceEntry;

  std::vector<StageTraceEntry> expected;
  expected.push_back(stageTraceEntry(0, "initialResidualNorm", 8.0, 0.0, 8.0));
  expected.push_back(stageTraceEntry(1, "postCoarseCorrection", 6.0, 12.0, 1.0));

  std::vector<StageTraceEntry> actual = expected;
  actual[1].residualInfNorm = 1.25;

  const Test::StageTraceComparisonResult result =
    Test::compareStageTraces(expected, actual, 1e-12);
  TEST_EQUALITY(result.sizeMatches, true);
  TEST_EQUALITY(result.firstMismatchIndex, 1);
  TEST_EQUALITY(result.firstMismatchField, "residualInfNorm");
  TEST_COMPARE(result.maxDifference, >, 1e-12);
}

TEUCHOS_UNIT_TEST( MatrixFreeGMGPreconditionerStageTraceComparison, ReportsFirstMetadataMismatch )
{
  using StageTraceEntry = MatrixFreeGMGPreconditioner::StageTraceEntry;

  std::vector<StageTraceEntry> expected;
  expected.push_back(stageTraceEntry(0, "initialResidualNorm", 8.0, 0.0, 8.0));

  std::vector<StageTraceEntry> actual = expected;
  actual[0].stageName = "postSmoothing";

  const Test::StageTraceComparisonResult result =
    Test::compareStageTraces(expected, actual, 1e-12);
  TEST_EQUALITY(result.sizeMatches, true);
  TEST_EQUALITY(result.firstMismatchIndex, 0);
  TEST_EQUALITY(result.firstMismatchField, "stageName");
  TEST_FLOATING_EQUALITY(result.maxDifference, 0.0, 1e-14);
}

TEUCHOS_UNIT_TEST( MatrixFreeGMGPreconditionerStageTraceComparison, ReportsSizeMismatch )
{
  using StageTraceEntry = MatrixFreeGMGPreconditioner::StageTraceEntry;

  std::vector<StageTraceEntry> expected;
  expected.push_back(stageTraceEntry(0, "initialResidualNorm", 8.0, 0.0, 8.0));

  std::vector<StageTraceEntry> actual = expected;
  actual.push_back(stageTraceEntry(1, "postCoarseCorrection", 6.0, 12.0, 1.0));

  const Test::StageTraceComparisonResult result =
    Test::compareStageTraces(expected, actual);
  TEST_EQUALITY(result.sizeMatches, false);
  TEST_EQUALITY(result.firstMismatchIndex, 1);
  TEST_EQUALITY(result.firstMismatchField, "size");
}
