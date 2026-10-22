#include "Teuchos_UnitTestHarness.hpp"

#include "TensorLocalPMultigrid.hpp"

#include "Kokkos_Core.hpp"

#include "Ifpack2_Chebyshev.hpp"
#include "Teuchos_DefaultSerialComm.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

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

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, DeterministicPowerBlockLambdaMatchesCellHelper )
{
  constexpr int rows = 3;
  const auto firstCell = deterministicSPDMatrix(rows);
  auto secondCell = deterministicSPDMatrix(rows);
  for (int i=0; i<rows; i++) secondCell[static_cast<size_t>(i * rows + i)] += Scalar(0.35);

  std::vector<Scalar> matrices;
  matrices.insert(matrices.end(), firstCell.begin(), firstCell.end());
  matrices.insert(matrices.end(), secondCell.begin(), secondCell.end());

  const auto inverseDiagonal = tensorLocalExtractInverseDiagonalFromBlockDenseMatrices(matrices, 2, rows);
  const auto deterministicLambdaMaxByCell =
      tensorLocalExtractDeterministicPowerLambdaMaxFromBlockDenseMatrices(matrices, inverseDiagonal,
                                                                          2, rows, 7, 1);

  const auto firstInverseDiagonal = tensorLocalBlockDenseCellVector(inverseDiagonal, 0, rows);
  const auto secondInverseDiagonal = tensorLocalBlockDenseCellVector(inverseDiagonal, 1, rows);
  const Scalar expectedFirst = tensorLocalDenseDInvADeterministicPowerLambdaMax(firstCell, firstInverseDiagonal,
                                                                               rows, 7, 1);
  const Scalar expectedSecond = tensorLocalDenseDInvADeterministicPowerLambdaMax(secondCell, secondInverseDiagonal,
                                                                                rows, 7, 1);

  TEST_FLOATING_EQUALITY(deterministicLambdaMaxByCell[0], expectedFirst, 1e-14);
  TEST_FLOATING_EQUALITY(deterministicLambdaMaxByCell[1], expectedSecond, 1e-14);
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
  const Scalar boostFactor = Scalar(1.1); // Ifpack2 default
  const Scalar theta = Scalar(0.5) * (boostFactor * lambdaMax + lambdaMin);
  for (int i=0; i<rows; i++)
    expected[static_cast<size_t>(i)] += (Scalar(1.0) / theta) * secondInverseDiagonal[static_cast<size_t>(i)] * (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)]);

  assertVectorsNear(actual, expected, 1e-14, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, ChebyshevBlockSmootherMatchesIfpack2Chebyshev )
{
  // The p-multigrid coarse levels use tensorLocalRunChebyshevBlockSmoother, while the fine level uses Ifpack2::Chebyshev;
  // with the same lambdaMax, ratio, and initial guess, the two should produce the same iterate to round-off.
  constexpr int rows = 12;
  std::vector<Scalar> matrix(static_cast<size_t>(rows) * rows, Scalar(0.0));
  for (int i=0; i<rows; i++)
  {
    // 1D Laplacian plus a varying reaction term: a wide D^{-1} A spectrum, unlike deterministicSPDMatrix()
    matrix[static_cast<size_t>(i) * rows + i] = Scalar(2.0) + Scalar(0.1) * Scalar(i % 3);
    if (i > 0)        matrix[static_cast<size_t>(i) * rows + (i-1)] = Scalar(-1.0);
    if (i < rows - 1) matrix[static_cast<size_t>(i) * rows + (i+1)] = Scalar(-1.0);
  }
  const auto inverseDiagonal = tensorLocalExtractInverseDiagonalFromBlockDenseMatrices(matrix, 1, rows);
  const Scalar lambdaMax = Scalar(2.5); // a user-supplied bound, as both p-multigrid paths provide
  const std::vector<Scalar> lambdaMaxByCell = {lambdaMax};
  const Scalar ratio = Scalar(30.0);
  const int degree = 8;

  std::vector<Scalar> rhs(rows), initial(rows);
  for (int i=0; i<rows; i++)
  {
    rhs[static_cast<size_t>(i)]     = Scalar(1.0) + Scalar(0.25) * std::sin(Scalar(i));
    initial[static_cast<size_t>(i)] = Scalar(0.1) * std::cos(Scalar(2 * i));
  }
  std::vector<Scalar> actual = initial;
  tensorLocalRunChebyshevBlockSmoother(matrix, inverseDiagonal, lambdaMaxByCell, 0, rows, rhs, actual, degree, Scalar(1.0) / ratio);

  using Map = Tpetra::Map<>;
  using CrsMatrix = Tpetra::CrsMatrix<Scalar>;
  using MV = Tpetra::MultiVector<Scalar>;
  using RowMatrix = Tpetra::RowMatrix<Scalar>;
  Teuchos::RCP<const Teuchos::Comm<int>> serialComm = Teuchos::rcp(new Teuchos::SerialComm<int>());
  auto map = Teuchos::rcp(new Map(rows, 0, serialComm));
  auto A = Teuchos::rcp(new CrsMatrix(map, 3));
  for (int i=0; i<rows; i++)
  {
    std::vector<Map::global_ordinal_type> cols;
    std::vector<Scalar> values;
    for (int j=0; j<rows; j++)
    {
      const Scalar value = matrix[static_cast<size_t>(i) * rows + j];
      if (value != Scalar(0.0)) { cols.push_back(j); values.push_back(value); }
    }
    A->insertGlobalValues(i, Teuchos::ArrayView<const Map::global_ordinal_type>(cols), Teuchos::ArrayView<const Scalar>(values));
  }
  A->fillComplete();

  Ifpack2::Chebyshev<RowMatrix> chebyshev(A);
  Teuchos::ParameterList params;
  params.set("chebyshev: degree", degree);
  params.set("chebyshev: ratio eigenvalue", ratio);
  params.set("chebyshev: max eigenvalue", lambdaMax);
  params.set("chebyshev: min eigenvalue", lambdaMax / ratio);
  params.set("chebyshev: eigenvalue max iterations", 0);
  params.set("chebyshev: zero starting solution", false);
  chebyshev.setParameters(params);
  chebyshev.initialize();
  chebyshev.compute();

  MV B(map, 1), X(map, 1);
  {
    auto bView = B.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto xView = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (int i=0; i<rows; i++) { bView(i,0) = rhs[static_cast<size_t>(i)]; xView(i,0) = initial[static_cast<size_t>(i)]; }
  }
  chebyshev.apply(B, X);
  std::vector<Scalar> expected(rows);
  {
    auto xView = X.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (int i=0; i<rows; i++) expected[static_cast<size_t>(i)] = xView(i,0);
  }
  assertVectorsNear(actual, expected, 1e-13, out, success);
}

TEUCHOS_UNIT_TEST( TensorLocalPMultigrid, CoarseSolverPolicyParsesValidNamesAndRejectsInvalidNames )
{
  TEST_EQUALITY(std::string(tensorLocalPMultigridCoarseSolverPolicyName(tensorLocalPMultigridCoarseSolverPolicyFromName("direct"))), std::string("direct"));
  TEST_EQUALITY(std::string(tensorLocalPMultigridCoarseSolverPolicyName(tensorLocalPMultigridCoarseSolverPolicyFromName("chebyshev"))), std::string("chebyshev"));
  TEST_EQUALITY(std::string(tensorLocalPMultigridCoarseSolverPolicyName(tensorLocalPMultigridCoarseSolverPolicyFromName("jacobi"))), std::string("jacobi"));
  TEST_THROW(tensorLocalPMultigridCoarseSolverPolicyFromName("invalid", "LocalPMultigridTpetraPreconditioner"), std::invalid_argument);
}
