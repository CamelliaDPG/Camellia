#include "Teuchos_UnitTestHarness.hpp"

#include "TensorLocalPMultigrid.hpp"

#include "Kokkos_Core.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace Camellia;

namespace
{
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using Scalar = double;

  std::vector<Scalar> deterministicRhs(const int rows)
  {
    std::vector<Scalar> rhs(static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
      rhs[static_cast<size_t>(i)] = Scalar(1.0) + Scalar(0.125) * static_cast<Scalar>((i % 7) - 3);
    return rhs;
  }

  std::vector<Scalar> deterministicSPDMatrix(const int rows)
  {
    std::vector<Scalar> matrix(static_cast<size_t>(rows) * static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
    {
      for (int j=0; j<rows; j++)
      {
        Scalar value = Scalar(0.0);
        if (i == j) value = Scalar(4.0) + Scalar(0.05) * static_cast<Scalar>(i + 1);
        if (std::abs(i-j) == 1) value += Scalar(0.03);
        if (std::abs(i-j) == 2) value += Scalar(0.01);
        if (((i + j) % 17) == 0) value += Scalar(0.001);
        matrix[static_cast<size_t>(i) * static_cast<size_t>(rows) + static_cast<size_t>(j)] = value;
      }
    }
    return matrix;
  }

  Scalar dot(const std::vector<Scalar> &a, const std::vector<Scalar> &b)
  {
    Scalar value = Scalar(0.0);
    for (size_t i=0; i<a.size(); i++) value += a[i] * b[i];
    return value;
  }

  std::vector<Scalar> applyCoarseMatrixByExplicitRAP(const std::vector<Scalar> &fineMatrix,
                                                     const TensorHGradPTransfer<Scalar> &transfer,
                                                     const std::vector<Scalar> &coarseInput)
  {
    std::vector<Scalar> prolonged;
    applyTensorHGradPTransferProlongation(transfer, coarseInput, prolonged);
    const auto fineApplied = tensorLocalApplyDenseMatrix(fineMatrix, prolonged, transfer.fineRows);
    std::vector<Scalar> restricted;
    applyTensorHGradPTransferRestriction(transfer, fineApplied, restricted);
    return restricted;
  }
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, GalerkinCoarseMatrixHasRAPAction )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(2, 3);
  const auto fineMatrix = deterministicSPDMatrix(transfer.fineRows);
  const auto coarseMatrix = tensorLocalBuildGalerkinCoarseMatrix(fineMatrix, transfer);
  const auto coarseInput = deterministicRhs(transfer.coarseRows);

  const auto coarseApplied = tensorLocalApplyDenseMatrix(coarseMatrix, coarseInput, transfer.coarseRows);
  const auto explicitRAPApplied = applyCoarseMatrixByExplicitRAP(fineMatrix, transfer, coarseInput);

  TEST_EQUALITY(coarseApplied.size(), explicitRAPApplied.size());
  for (size_t i=0; i<coarseApplied.size(); i++)
    TEST_FLOATING_EQUALITY(coarseApplied[i], explicitRAPApplied[i], 1e-12);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, GalerkinCoarseMatrixPreservesCoarseEnergy )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(3, 2);
  const auto fineMatrix = deterministicSPDMatrix(transfer.fineRows);
  const auto coarseMatrix = tensorLocalBuildGalerkinCoarseMatrix(fineMatrix, transfer);
  const auto coarseInput = deterministicRhs(transfer.coarseRows);

  std::vector<Scalar> prolonged;
  applyTensorHGradPTransferProlongation(transfer, coarseInput, prolonged);

  const auto fineApplied = tensorLocalApplyDenseMatrix(fineMatrix, prolonged, transfer.fineRows);
  const auto coarseApplied = tensorLocalApplyDenseMatrix(coarseMatrix, coarseInput, transfer.coarseRows);

  const Scalar fineEnergy = dot(prolonged, fineApplied);
  const Scalar coarseEnergy = dot(coarseInput, coarseApplied);

  TEST_FLOATING_EQUALITY(coarseEnergy, fineEnergy, 1e-12);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, ZeroSmoothingVCycleIsGalerkinCoarseCorrection )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(2, 3);
  const auto fineMatrix = deterministicSPDMatrix(transfer.fineRows);
  const auto rhs = deterministicRhs(transfer.fineRows);
  const auto result = tensorLocalApplyTwoLevelPMultigridVcycle(fineMatrix,
                                                               transfer,
                                                               rhs,
                                                               0,
                                                               0);

  std::vector<Scalar> expectedRestrictedResidual;
  applyTensorHGradPTransferRestriction(transfer, rhs, expectedRestrictedResidual);

  TEST_EQUALITY(result.restrictedResidual.size(), expectedRestrictedResidual.size());
  for (size_t i=0; i<expectedRestrictedResidual.size(); i++)
    TEST_FLOATING_EQUALITY(result.restrictedResidual[i], expectedRestrictedResidual[i], 1e-12);

  const auto coarseMatrix = tensorLocalBuildGalerkinCoarseMatrix(fineMatrix, transfer);
  const auto coarseApplied = tensorLocalApplyDenseMatrix(coarseMatrix, result.coarseCorrection, transfer.coarseRows);
  for (size_t i=0; i<coarseApplied.size(); i++)
    TEST_FLOATING_EQUALITY(coarseApplied[i], result.restrictedResidual[i], 1e-11);

  std::vector<Scalar> expectedFineCorrection;
  applyTensorHGradPTransferProlongation(transfer, result.coarseCorrection, expectedFineCorrection);
  TEST_EQUALITY(result.finalSolution.size(), expectedFineCorrection.size());
  for (size_t i=0; i<expectedFineCorrection.size(); i++)
    TEST_FLOATING_EQUALITY(result.finalSolution[i], expectedFineCorrection[i], 1e-12);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, StageResidualsAreConsistentWithFineOperator )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(2, 3);
  const auto fineMatrix = deterministicSPDMatrix(transfer.fineRows);
  const auto rhs = deterministicRhs(transfer.fineRows);
  const auto result = tensorLocalApplyTwoLevelPMultigridVcycle(fineMatrix,
                                                               transfer,
                                                               rhs,
                                                               1,
                                                               1,
                                                               0.75);

  const auto expectedPreResidual = tensorLocalResidual(fineMatrix,
                                                       rhs,
                                                       result.preSmoothedSolution,
                                                       transfer.fineRows);
  const auto expectedCoarseResidual = tensorLocalResidual(fineMatrix,
                                                          rhs,
                                                          result.solutionAfterCoarseCorrection,
                                                          transfer.fineRows);
  const auto expectedPostResidual = tensorLocalResidual(fineMatrix,
                                                        rhs,
                                                        result.finalSolution,
                                                        transfer.fineRows);

  for (size_t i=0; i<expectedPreResidual.size(); i++)
    TEST_FLOATING_EQUALITY(result.residualAfterPreSmoothing[i], expectedPreResidual[i], 1e-12);
  for (size_t i=0; i<expectedCoarseResidual.size(); i++)
    TEST_FLOATING_EQUALITY(result.residualAfterCoarseCorrection[i], expectedCoarseResidual[i], 1e-12);
  for (size_t i=0; i<expectedPostResidual.size(); i++)
    TEST_FLOATING_EQUALITY(result.residualAfterPostSmoothing[i], expectedPostResidual[i], 1e-12);
}

namespace
{
  std::vector<Scalar> buildCoarseMatrixByExplicitRAP(const std::vector<Scalar> &fineMatrix,
                                                     const TensorHGradPTransfer<Scalar> &transfer)
  {
    std::vector<Scalar> coarseMatrix(static_cast<size_t>(transfer.coarseRows) * static_cast<size_t>(transfer.coarseRows),
                                     Scalar(0.0));
    for (int j=0; j<transfer.coarseRows; j++)
    {
      std::vector<Scalar> coarseBasis(static_cast<size_t>(transfer.coarseRows), Scalar(0.0));
      coarseBasis[static_cast<size_t>(j)] = Scalar(1.0);
      const auto coarseApplied = applyCoarseMatrixByExplicitRAP(fineMatrix, transfer, coarseBasis);
      for (int i=0; i<transfer.coarseRows; i++)
        coarseMatrix[static_cast<size_t>(i) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(j)] =
            coarseApplied[static_cast<size_t>(i)];
    }
    return coarseMatrix;
  }

  TensorLocalTwoLevelPMultigridResult<Scalar>
  expectedTensorPoissonLocalPMGStages(const std::vector<Scalar> &fineMatrix,
                                      const TensorHGradPTransfer<Scalar> &transfer,
                                      const std::vector<Scalar> &rhs,
                                      const int preSmoothIterations,
                                      const int postSmoothIterations,
                                      const Scalar jacobiDamping)
  {
    TensorLocalTwoLevelPMultigridResult<Scalar> expected;
    const auto inverseDiagonal = tensorLocalExtractInverseDiagonal(fineMatrix, transfer.fineRows);

    expected.preSmoothedSolution.assign(static_cast<size_t>(transfer.fineRows), Scalar(0.0));
    tensorLocalJacobiSmooth(fineMatrix, inverseDiagonal, rhs,
                            expected.preSmoothedSolution, transfer.fineRows,
                            preSmoothIterations, jacobiDamping);

    expected.residualAfterPreSmoothing = tensorLocalResidual(fineMatrix, rhs,
                                                             expected.preSmoothedSolution,
                                                             transfer.fineRows);
    applyTensorHGradPTransferRestriction(transfer,
                                         expected.residualAfterPreSmoothing,
                                         expected.restrictedResidual);

    const auto coarseMatrix = buildCoarseMatrixByExplicitRAP(fineMatrix, transfer);
    expected.coarseCorrection = tensorLocalSolveDenseLinearSystem(coarseMatrix,
                                                                  expected.restrictedResidual,
                                                                  transfer.coarseRows);
    applyTensorHGradPTransferProlongation(transfer,
                                          expected.coarseCorrection,
                                          expected.prolongatedCorrection);

    expected.solutionAfterCoarseCorrection = expected.preSmoothedSolution;
    for (int i=0; i<transfer.fineRows; i++)
      expected.solutionAfterCoarseCorrection[static_cast<size_t>(i)] +=
          expected.prolongatedCorrection[static_cast<size_t>(i)];

    expected.residualAfterCoarseCorrection = tensorLocalResidual(fineMatrix, rhs,
                                                                 expected.solutionAfterCoarseCorrection,
                                                                 transfer.fineRows);
    expected.finalSolution = expected.solutionAfterCoarseCorrection;
    tensorLocalJacobiSmooth(fineMatrix, inverseDiagonal, rhs,
                            expected.finalSolution, transfer.fineRows,
                            postSmoothIterations, jacobiDamping);
    expected.residualAfterPostSmoothing = tensorLocalResidual(fineMatrix, rhs,
                                                              expected.finalSolution,
                                                              transfer.fineRows);
    return expected;
  }

  struct LocalPMGStagePair
  {
    TensorLocalTwoLevelPMultigridResult<Scalar> actual;
    TensorLocalTwoLevelPMultigridResult<Scalar> expected;
  };

  LocalPMGStagePair makeLocalPMGStagePair()
  {
    const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(2, 3);
    const auto fineMatrix = deterministicSPDMatrix(transfer.fineRows);
    const auto rhs = deterministicRhs(transfer.fineRows);
    constexpr int preSmoothIterations = 2;
    constexpr int postSmoothIterations = 2;
    constexpr Scalar jacobiDamping = 0.75;

    return {tensorLocalApplyTwoLevelPMultigridVcycle(fineMatrix, transfer, rhs,
                                                    preSmoothIterations,
                                                    postSmoothIterations,
                                                    jacobiDamping),
            expectedTensorPoissonLocalPMGStages(fineMatrix, transfer, rhs,
                                                preSmoothIterations,
                                                postSmoothIterations,
                                                jacobiDamping)};
  }

  void assertVectorsNear(const std::vector<Scalar> &actual,
                         const std::vector<Scalar> &expected,
                         const Scalar tol,
                         Teuchos::FancyOStream &out,
                         bool &success)
  {
    TEST_EQUALITY(actual.size(), expected.size());
    const size_t count = actual.size() < expected.size() ? actual.size() : expected.size();
    for (size_t i=0; i<count; i++)
      TEST_FLOATING_EQUALITY(actual[i], expected[i], tol);
  }
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, PreSmoothingMatchesTensorPoissonLocalPMG )
{
  const auto stages = makeLocalPMGStagePair();
  assertVectorsNear(stages.actual.preSmoothedSolution,
                    stages.expected.preSmoothedSolution,
                    1e-11, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, RestrictedResidualMatchesTensorPoissonLocalPMG )
{
  const auto stages = makeLocalPMGStagePair();
  assertVectorsNear(stages.actual.restrictedResidual,
                    stages.expected.restrictedResidual,
                    1e-11, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, CoarseCorrectionMatchesTensorPoissonLocalPMG )
{
  const auto stages = makeLocalPMGStagePair();
  assertVectorsNear(stages.actual.coarseCorrection,
                    stages.expected.coarseCorrection,
                    1e-11, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, ProlongatedCorrectionMatchesTensorPoissonLocalPMG )
{
  const auto stages = makeLocalPMGStagePair();
  assertVectorsNear(stages.actual.prolongatedCorrection,
                    stages.expected.prolongatedCorrection,
                    1e-11, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, PostSmoothingMatchesTensorPoissonLocalPMG )
{
  const auto stages = makeLocalPMGStagePair();
  assertVectorsNear(stages.actual.finalSolution,
                    stages.expected.finalSolution,
                    1e-11, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, BlockDenseHelpersExtractPerCellQuantities )
{
  constexpr int rows = 3;
  const auto firstCell = deterministicSPDMatrix(rows);
  auto secondCell = deterministicSPDMatrix(rows);
  for (size_t i=0; i<secondCell.size(); i++) secondCell[i] += Scalar(0.01) * static_cast<Scalar>(i + 1);

  std::vector<Scalar> matrices;
  matrices.insert(matrices.end(), firstCell.begin(), firstCell.end());
  matrices.insert(matrices.end(), secondCell.begin(), secondCell.end());

  const auto extractedSecondCell = tensorLocalBlockDenseCellMatrix(matrices, 1, rows);
  assertVectorsNear(extractedSecondCell, secondCell, 1e-14, out, success);

  const auto inverseDiagonal = tensorLocalExtractInverseDiagonalFromBlockDenseMatrices(matrices, 2, rows);
  const auto expectedSecondInverseDiagonal = tensorLocalExtractInverseDiagonal(secondCell, rows);
  for (int i=0; i<rows; i++)
    TEST_FLOATING_EQUALITY(inverseDiagonal[static_cast<size_t>(rows + i)], expectedSecondInverseDiagonal[static_cast<size_t>(i)], 1e-14);

  const auto lambdaMaxByCell = tensorLocalExtractLambdaMaxFromBlockDenseMatrices(matrices, inverseDiagonal, 2, rows);
  Scalar expectedSecondLambdaMax = Scalar(0.0);
  for (int i=0; i<rows; i++)
  {
    Scalar scaledRowSum = Scalar(0.0);
    for (int j=0; j<rows; j++)
      scaledRowSum += std::abs(secondCell[static_cast<size_t>(i * rows + j)]) * expectedSecondInverseDiagonal[static_cast<size_t>(i)];
    expectedSecondLambdaMax = std::max(expectedSecondLambdaMax, scaledRowSum);
  }
  TEST_FLOATING_EQUALITY(lambdaMaxByCell[1], expectedSecondLambdaMax, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, DampedJacobiBlockSmootherMatchesCellReference )
{
  constexpr int rows = 3;
  const auto firstCell = deterministicSPDMatrix(rows);
  auto secondCell = deterministicSPDMatrix(rows);
  for (int i=0; i<rows; i++) secondCell[static_cast<size_t>(i * rows + i)] += Scalar(0.2);

  std::vector<Scalar> matrices;
  matrices.insert(matrices.end(), firstCell.begin(), firstCell.end());
  matrices.insert(matrices.end(), secondCell.begin(), secondCell.end());

  const auto inverseDiagonal = tensorLocalExtractInverseDiagonalFromBlockDenseMatrices(matrices, 2, rows);
  const auto lambdaMaxByCell = tensorLocalExtractLambdaMaxFromBlockDenseMatrices(matrices, inverseDiagonal, 2, rows);
  const auto secondInverseDiagonal = tensorLocalExtractInverseDiagonal(secondCell, rows);
  const std::vector<Scalar> rhs = {Scalar(1.0), Scalar(0.25), Scalar(-0.5)};
  std::vector<Scalar> actual = {Scalar(0.1), Scalar(-0.2), Scalar(0.05)};
  std::vector<Scalar> expected = actual;

  tensorLocalRunDampedJacobiBlockSmoother(matrices, inverseDiagonal, lambdaMaxByCell,
                                          1, rows, rhs, actual, 3);
  const Scalar damping = Scalar(1.0) / std::max(lambdaMaxByCell[1], Scalar(1.0));
  tensorLocalJacobiSmooth(secondCell, secondInverseDiagonal, rhs, expected, rows, 3, damping);
  assertVectorsNear(actual, expected, 1e-14, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, ChebyshevBlockSmootherFirstStepMatchesReferenceFormula )
{
  constexpr int rows = 3;
  const auto firstCell = deterministicSPDMatrix(rows);
  auto secondCell = deterministicSPDMatrix(rows);
  for (int i=0; i<rows; i++) secondCell[static_cast<size_t>(i * rows + i)] += Scalar(0.3);

  std::vector<Scalar> matrices;
  matrices.insert(matrices.end(), firstCell.begin(), firstCell.end());
  matrices.insert(matrices.end(), secondCell.begin(), secondCell.end());

  const auto inverseDiagonal = tensorLocalExtractInverseDiagonalFromBlockDenseMatrices(matrices, 2, rows);
  const auto lambdaMaxByCell = tensorLocalExtractLambdaMaxFromBlockDenseMatrices(matrices, inverseDiagonal, 2, rows);
  const auto secondInverseDiagonal = tensorLocalExtractInverseDiagonal(secondCell, rows);
  const std::vector<Scalar> rhs = {Scalar(0.5), Scalar(-0.125), Scalar(0.75)};
  const std::vector<Scalar> initial = {Scalar(0.2), Scalar(0.05), Scalar(-0.1)};
  std::vector<Scalar> actual = initial;
  std::vector<Scalar> expected = initial;

  constexpr Scalar lowerFraction = Scalar(0.25);
  tensorLocalRunChebyshevBlockSmoother(matrices, inverseDiagonal, lambdaMaxByCell,
                                       1, rows, rhs, actual, 1, lowerFraction);

  const auto applied = tensorLocalApplyDenseMatrix(secondCell, initial, rows);
  const Scalar lambdaMax = std::max(lambdaMaxByCell[1], Scalar(1.0));
  const Scalar lambdaMin = std::max(std::numeric_limits<Scalar>::epsilon(), lowerFraction * lambdaMax);
  const Scalar theta = Scalar(0.5) * (lambdaMax + lambdaMin);
  for (int i=0; i<rows; i++)
    expected[static_cast<size_t>(i)] += (Scalar(1.0) / theta) * secondInverseDiagonal[static_cast<size_t>(i)] * (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)]);

  assertVectorsNear(actual, expected, 1e-14, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, CoarseSolverPolicyParsesValidNamesAndRejectsInvalidNames )
{
  TEST_EQUALITY(std::string(tensorLocalPMultigridCoarseSolverPolicyName(tensorLocalPMultigridCoarseSolverPolicyFromName("direct"))), std::string("direct"));
  TEST_EQUALITY(std::string(tensorLocalPMultigridCoarseSolverPolicyName(tensorLocalPMultigridCoarseSolverPolicyFromName("chebyshev"))), std::string("chebyshev"));
  TEST_EQUALITY(std::string(tensorLocalPMultigridCoarseSolverPolicyName(tensorLocalPMultigridCoarseSolverPolicyFromName("jacobi"))), std::string("jacobi"));
  TEST_THROW(tensorLocalPMultigridCoarseSolverPolicyFromName("invalid", "LocalPMultigridTpetraPreconditioner"), std::invalid_argument);
}
