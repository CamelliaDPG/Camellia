#include "Teuchos_UnitTestHarness.hpp"

#include "TensorHGradPTransfer.hpp"

#include "Kokkos_Core.hpp"

#include <cmath>
#include <vector>

using namespace Camellia;

namespace
{
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using Scalar = double;

  std::vector<int> extentsForP(const int dim, const int p)
  {
    return std::vector<int>(static_cast<size_t>(dim), p + 1);
  }

  std::vector<Scalar> deterministicFineVector(const int size)
  {
    std::vector<Scalar> values(static_cast<size_t>(size), Scalar(0.0));
    for (int i=0; i<size; i++)
      values[static_cast<size_t>(i)] = Scalar(0.125) + Scalar(0.5) * static_cast<Scalar>(i % 7) - Scalar(0.03125) * static_cast<Scalar>(i);
    return values;
  }

  std::vector<Scalar> deterministicCoarseVector(const int size)
  {
    std::vector<Scalar> values(static_cast<size_t>(size), Scalar(0.0));
    for (int i=0; i<size; i++)
      values[static_cast<size_t>(i)] = Scalar(1.0) + Scalar(0.25) * static_cast<Scalar>(i % 5) + Scalar(0.01) * static_cast<Scalar>(i);
    return values;
  }

  std::vector<Scalar> applyDenseFineMatrix(const std::vector<Scalar> &matrix,
                                           const std::vector<Scalar> &x,
                                           const int rows)
  {
    std::vector<Scalar> y(static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
      for (int j=0; j<rows; j++)
        y[static_cast<size_t>(i)] += matrix[static_cast<size_t>(i) * static_cast<size_t>(rows) + static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
    return y;
  }

  std::vector<Scalar> deterministicSPDLikeFineMatrix(const int rows)
  {
    std::vector<Scalar> matrix(static_cast<size_t>(rows) * static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
    {
      for (int j=0; j<rows; j++)
      {
        Scalar value = Scalar(0.0);
        if (i == j) value = Scalar(3.0) + Scalar(0.1) * static_cast<Scalar>(i + 1);
        if (std::abs(i-j) == 1) value += Scalar(0.05);
        if (((i + j) % 11) == 0) value += Scalar(0.002);
        matrix[static_cast<size_t>(i) * static_cast<size_t>(rows) + static_cast<size_t>(j)] = value;
      }
    }
    return matrix;
  }
}

TEUCHOS_UNIT_TEST( TensorHGradPTransfer, P2ToP1Dim4DimensionsAndTransposeRestriction )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(4, 2);

  TEST_EQUALITY(transfer.dim, 4);
  TEST_EQUALITY(transfer.fineP, 2);
  TEST_EQUALITY(transfer.coarseP, 1);
  TEST_EQUALITY(transfer.fineRows, 81);
  TEST_EQUALITY(transfer.coarseRows, 16);
  TEST_EQUALITY(transfer.prolongation.size(), static_cast<size_t>(transfer.fineRows * transfer.coarseRows));
  TEST_EQUALITY(transfer.restriction.size(), static_cast<size_t>(transfer.coarseRows * transfer.fineRows));

  for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
  {
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
    {
      const Scalar pValue = transfer.prolongation[static_cast<size_t>(fineOrdinal) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseOrdinal)];
      const Scalar rValue = transfer.restriction[static_cast<size_t>(coarseOrdinal) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineOrdinal)];
      TEST_FLOATING_EQUALITY(rValue, pValue, 1e-14);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorHGradPTransfer, HierarchicalCoarseModesEmbedInFineSpace )
{
  const int dim = 4;
  const int fineP = 2;
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(dim, fineP);

  const auto coarseExtents = extentsForP(dim, fineP - 1);
  const auto fineExtents = extentsForP(dim, fineP);

  for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
  {
    const auto coarseMultiIndex = tensorHGradPTransferMultiIndexFromOrdinal(coarseOrdinal, coarseExtents);
    const int embeddedFineOrdinal = tensorHGradPTransferOrdinalFromMultiIndex(coarseMultiIndex, fineExtents);

    for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
    {
      const Scalar expected = (fineOrdinal == embeddedFineOrdinal) ? Scalar(1.0) : Scalar(0.0);
      const Scalar actual = transfer.prolongation[static_cast<size_t>(fineOrdinal) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseOrdinal)];
      TEST_FLOATING_EQUALITY(actual, expected, 1e-13);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorHGradPTransfer, ApplyProlongationAndRestrictionMatchDenseMatrices )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(3, 4);
  const auto coarse = deterministicCoarseVector(transfer.coarseRows);
  const auto fineInput = deterministicFineVector(transfer.fineRows);

  std::vector<Scalar> fine;
  applyTensorHGradPTransferProlongation(transfer, coarse, fine);

  for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
  {
    Scalar expected = Scalar(0.0);
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
      expected += transfer.prolongation[static_cast<size_t>(fineOrdinal) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseOrdinal)] * coarse[static_cast<size_t>(coarseOrdinal)];
    TEST_FLOATING_EQUALITY(fine[static_cast<size_t>(fineOrdinal)], expected, 1e-14);
  }

  std::vector<Scalar> restricted;
  applyTensorHGradPTransferRestriction(transfer, fineInput, restricted);

  for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
  {
    Scalar expected = Scalar(0.0);
    for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
      expected += transfer.prolongation[static_cast<size_t>(fineOrdinal) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseOrdinal)] * fineInput[static_cast<size_t>(fineOrdinal)];
    TEST_FLOATING_EQUALITY(restricted[static_cast<size_t>(coarseOrdinal)], expected, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorHGradPTransfer, GalerkinCoarseActionMatchesRestrictionFineOperatorProlongation )
{
  const auto transfer = buildTensorHGradPTransfer<DeviceType,Scalar>(2, 3);
  const auto coarseInput = deterministicCoarseVector(transfer.coarseRows);
  const auto fineMatrix = deterministicSPDLikeFineMatrix(transfer.fineRows);

  std::vector<Scalar> prolonged;
  applyTensorHGradPTransferProlongation(transfer, coarseInput, prolonged);
  const auto fineApplied = applyDenseFineMatrix(fineMatrix, prolonged, transfer.fineRows);
  std::vector<Scalar> restrictedFineApplied;
  applyTensorHGradPTransferRestriction(transfer, fineApplied, restrictedFineApplied);

  std::vector<Scalar> coarseMatrix(static_cast<size_t>(transfer.coarseRows) * static_cast<size_t>(transfer.coarseRows), Scalar(0.0));
  for (int coarseI=0; coarseI<transfer.coarseRows; coarseI++)
  {
    for (int coarseJ=0; coarseJ<transfer.coarseRows; coarseJ++)
    {
      Scalar value = Scalar(0.0);
      for (int fineI=0; fineI<transfer.fineRows; fineI++)
      {
        const Scalar rEntry = transfer.restriction[static_cast<size_t>(coarseI) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineI)];
        if (rEntry == Scalar(0.0)) continue;
        for (int fineJ=0; fineJ<transfer.fineRows; fineJ++)
        {
          const Scalar pEntry = transfer.prolongation[static_cast<size_t>(fineJ) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseJ)];
          if (pEntry == Scalar(0.0)) continue;
          value += rEntry * fineMatrix[static_cast<size_t>(fineI) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineJ)] * pEntry;
        }
      }
      coarseMatrix[static_cast<size_t>(coarseI) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseJ)] = value;
    }
  }

  const auto coarseApplied = applyDenseFineMatrix(coarseMatrix, coarseInput, transfer.coarseRows);

  TEST_EQUALITY(coarseApplied.size(), restrictedFineApplied.size());
  for (size_t i=0; i<coarseApplied.size(); i++)
    TEST_FLOATING_EQUALITY(coarseApplied[i], restrictedFineApplied[i], 1e-12);
}
