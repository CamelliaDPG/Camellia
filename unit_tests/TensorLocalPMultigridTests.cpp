#include "Teuchos_UnitTestHarness.hpp"

#include "TensorLocalPMultigrid.hpp"

#include "Kokkos_Core.hpp"

#include <cmath>
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
