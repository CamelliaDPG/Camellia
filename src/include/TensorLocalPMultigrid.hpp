#pragma once

#include "TensorHGradPTransfer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Camellia
{
  template<class Scalar>
  struct TensorLocalTwoLevelPMultigridResult
  {
    std::vector<Scalar> preSmoothedSolution;
    std::vector<Scalar> residualAfterPreSmoothing;
    std::vector<Scalar> restrictedResidual;
    std::vector<Scalar> coarseCorrection;
    std::vector<Scalar> prolongatedCorrection;
    std::vector<Scalar> solutionAfterCoarseCorrection;
    std::vector<Scalar> residualAfterCoarseCorrection;
    std::vector<Scalar> finalSolution;
    std::vector<Scalar> residualAfterPostSmoothing;
  };

  template<class Scalar>
  void tensorLocalCheckDenseMatrixSize(const std::vector<Scalar> &matrix,
                                       const int rows,
                                       const char *label)
  {
    if (rows <= 0)
      throw std::invalid_argument(std::string(label) + " requires rows > 0");
    const size_t expectedSize = static_cast<size_t>(rows) * static_cast<size_t>(rows);
    if (matrix.size() != expectedSize)
    {
      std::ostringstream message;
      message << label << " expected dense matrix size " << expectedSize
              << " but received " << matrix.size();
      throw std::invalid_argument(message.str());
    }
  }

  template<class Scalar>
  void tensorLocalCheckVectorSize(const std::vector<Scalar> &vector,
                                  const int rows,
                                  const char *label)
  {
    if (static_cast<int>(vector.size()) != rows)
    {
      std::ostringstream message;
      message << label << " expected vector size " << rows
              << " but received " << vector.size();
      throw std::invalid_argument(message.str());
    }
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalApplyDenseMatrix(const std::vector<Scalar> &matrix,
                                                  const std::vector<Scalar> &x,
                                                  const int rows)
  {
    tensorLocalCheckDenseMatrixSize(matrix, rows, "tensorLocalApplyDenseMatrix");
    tensorLocalCheckVectorSize(x, rows, "tensorLocalApplyDenseMatrix");

    std::vector<Scalar> y(static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
      for (int j=0; j<rows; j++)
        y[static_cast<size_t>(i)] +=
            matrix[static_cast<size_t>(i) * static_cast<size_t>(rows) + static_cast<size_t>(j)] *
            x[static_cast<size_t>(j)];
    return y;
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalSubtract(const std::vector<Scalar> &a,
                                          const std::vector<Scalar> &b)
  {
    if (a.size() != b.size())
      throw std::invalid_argument("tensorLocalSubtract requires equal vector sizes");

    std::vector<Scalar> result(a.size(), Scalar(0.0));
    for (size_t i=0; i<a.size(); i++)
      result[i] = a[i] - b[i];
    return result;
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalResidual(const std::vector<Scalar> &matrix,
                                          const std::vector<Scalar> &rhs,
                                          const std::vector<Scalar> &solution,
                                          const int rows)
  {
    const auto applied = tensorLocalApplyDenseMatrix(matrix, solution, rows);
    return tensorLocalSubtract(rhs, applied);
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalExtractInverseDiagonal(const std::vector<Scalar> &matrix,
                                                        const int rows)
  {
    tensorLocalCheckDenseMatrixSize(matrix, rows, "tensorLocalExtractInverseDiagonal");

    std::vector<Scalar> inverseDiagonal(static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
    {
      const Scalar diagonalValue = matrix[static_cast<size_t>(i) * static_cast<size_t>(rows) + static_cast<size_t>(i)];
      if (!std::isfinite(diagonalValue) || (diagonalValue <= std::numeric_limits<Scalar>::epsilon()))
      {
        std::ostringstream message;
        message << "tensorLocalExtractInverseDiagonal found nonpositive/nonfinite diagonal at row "
                << i << ": " << diagonalValue;
        throw std::runtime_error(message.str());
      }
      inverseDiagonal[static_cast<size_t>(i)] = Scalar(1.0) / diagonalValue;
    }
    return inverseDiagonal;
  }

  template<class Scalar>
  void tensorLocalJacobiSmooth(const std::vector<Scalar> &matrix,
                               const std::vector<Scalar> &inverseDiagonal,
                               const std::vector<Scalar> &rhs,
                               std::vector<Scalar> &solution,
                               const int rows,
                               const int iterations,
                               const Scalar damping = Scalar(1.0))
  {
    tensorLocalCheckDenseMatrixSize(matrix, rows, "tensorLocalJacobiSmooth");
    tensorLocalCheckVectorSize(inverseDiagonal, rows, "tensorLocalJacobiSmooth inverse diagonal");
    tensorLocalCheckVectorSize(rhs, rows, "tensorLocalJacobiSmooth rhs");
    tensorLocalCheckVectorSize(solution, rows, "tensorLocalJacobiSmooth solution");

    for (int iteration=0; iteration<iterations; iteration++)
    {
      const auto applied = tensorLocalApplyDenseMatrix(matrix, solution, rows);
      for (int i=0; i<rows; i++)
      {
        solution[static_cast<size_t>(i)] +=
            damping * inverseDiagonal[static_cast<size_t>(i)] *
            (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)]);
      }
    }
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalBuildGalerkinCoarseMatrix(const std::vector<Scalar> &fineMatrix,
                                                           const TensorHGradPTransfer<Scalar> &transfer)
  {
    tensorLocalCheckDenseMatrixSize(fineMatrix, transfer.fineRows, "tensorLocalBuildGalerkinCoarseMatrix");

    std::vector<Scalar> matrixTimesP(static_cast<size_t>(transfer.fineRows) * static_cast<size_t>(transfer.coarseRows),
                                     Scalar(0.0));
    for (int fineI=0; fineI<transfer.fineRows; fineI++)
    {
      for (int fineJ=0; fineJ<transfer.fineRows; fineJ++)
      {
        const Scalar matrixEntry =
            fineMatrix[static_cast<size_t>(fineI) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineJ)];
        if (matrixEntry == Scalar(0.0)) continue;
        for (int coarseJ=0; coarseJ<transfer.coarseRows; coarseJ++)
        {
          matrixTimesP[static_cast<size_t>(fineI) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseJ)] +=
              matrixEntry *
              transfer.prolongation[static_cast<size_t>(fineJ) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseJ)];
        }
      }
    }

    std::vector<Scalar> coarseMatrix(static_cast<size_t>(transfer.coarseRows) * static_cast<size_t>(transfer.coarseRows),
                                     Scalar(0.0));
    for (int coarseI=0; coarseI<transfer.coarseRows; coarseI++)
    {
      for (int fineI=0; fineI<transfer.fineRows; fineI++)
      {
        const Scalar restrictionEntry =
            transfer.restriction[static_cast<size_t>(coarseI) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineI)];
        if (restrictionEntry == Scalar(0.0)) continue;
        for (int coarseJ=0; coarseJ<transfer.coarseRows; coarseJ++)
        {
          coarseMatrix[static_cast<size_t>(coarseI) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseJ)] +=
              restrictionEntry *
              matrixTimesP[static_cast<size_t>(fineI) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseJ)];
        }
      }
    }

    return coarseMatrix;
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalSolveDenseLinearSystem(std::vector<Scalar> matrix,
                                                        std::vector<Scalar> rhs,
                                                        const int rows)
  {
    tensorLocalCheckDenseMatrixSize(matrix, rows, "tensorLocalSolveDenseLinearSystem");
    tensorLocalCheckVectorSize(rhs, rows, "tensorLocalSolveDenseLinearSystem");

    for (int pivot=0; pivot<rows; pivot++)
    {
      int pivotRow = pivot;
      Scalar pivotMagnitude = std::abs(matrix[static_cast<size_t>(pivot) * static_cast<size_t>(rows) + static_cast<size_t>(pivot)]);
      for (int row=pivot+1; row<rows; row++)
      {
        const Scalar candidateMagnitude = std::abs(matrix[static_cast<size_t>(row) * static_cast<size_t>(rows) + static_cast<size_t>(pivot)]);
        if (candidateMagnitude > pivotMagnitude)
        {
          pivotMagnitude = candidateMagnitude;
          pivotRow = row;
        }
      }

      if (pivotMagnitude <= std::numeric_limits<Scalar>::epsilon())
        throw std::runtime_error("tensorLocalSolveDenseLinearSystem encountered singular pivot");

      if (pivotRow != pivot)
      {
        for (int col=pivot; col<rows; col++)
          std::swap(matrix[static_cast<size_t>(pivot) * static_cast<size_t>(rows) + static_cast<size_t>(col)],
                    matrix[static_cast<size_t>(pivotRow) * static_cast<size_t>(rows) + static_cast<size_t>(col)]);
        std::swap(rhs[static_cast<size_t>(pivot)], rhs[static_cast<size_t>(pivotRow)]);
      }

      const Scalar pivotValue = matrix[static_cast<size_t>(pivot) * static_cast<size_t>(rows) + static_cast<size_t>(pivot)];
      for (int row=pivot+1; row<rows; row++)
      {
        const Scalar factor = matrix[static_cast<size_t>(row) * static_cast<size_t>(rows) + static_cast<size_t>(pivot)] / pivotValue;
        matrix[static_cast<size_t>(row) * static_cast<size_t>(rows) + static_cast<size_t>(pivot)] = Scalar(0.0);
        for (int col=pivot+1; col<rows; col++)
        {
          matrix[static_cast<size_t>(row) * static_cast<size_t>(rows) + static_cast<size_t>(col)] -=
              factor * matrix[static_cast<size_t>(pivot) * static_cast<size_t>(rows) + static_cast<size_t>(col)];
        }
        rhs[static_cast<size_t>(row)] -= factor * rhs[static_cast<size_t>(pivot)];
      }
    }

    std::vector<Scalar> solution(static_cast<size_t>(rows), Scalar(0.0));
    for (int row=rows-1; row>=0; row--)
    {
      Scalar sum = rhs[static_cast<size_t>(row)];
      for (int col=row+1; col<rows; col++)
        sum -= matrix[static_cast<size_t>(row) * static_cast<size_t>(rows) + static_cast<size_t>(col)] * solution[static_cast<size_t>(col)];
      solution[static_cast<size_t>(row)] =
          sum / matrix[static_cast<size_t>(row) * static_cast<size_t>(rows) + static_cast<size_t>(row)];
    }
    return solution;
  }

  template<class Scalar>
  TensorLocalTwoLevelPMultigridResult<Scalar>
  tensorLocalApplyTwoLevelPMultigridVcycle(const std::vector<Scalar> &fineMatrix,
                                           const TensorHGradPTransfer<Scalar> &transfer,
                                           const std::vector<Scalar> &rhs,
                                           const int preSmoothIterations,
                                           const int postSmoothIterations,
                                           const Scalar jacobiDamping = Scalar(1.0))
  {
    tensorLocalCheckDenseMatrixSize(fineMatrix, transfer.fineRows, "tensorLocalApplyTwoLevelPMultigridVcycle");
    tensorLocalCheckVectorSize(rhs, transfer.fineRows, "tensorLocalApplyTwoLevelPMultigridVcycle rhs");

    TensorLocalTwoLevelPMultigridResult<Scalar> result;
    const auto inverseDiagonal = tensorLocalExtractInverseDiagonal(fineMatrix, transfer.fineRows);

    result.preSmoothedSolution.assign(static_cast<size_t>(transfer.fineRows), Scalar(0.0));
    tensorLocalJacobiSmooth(fineMatrix,
                            inverseDiagonal,
                            rhs,
                            result.preSmoothedSolution,
                            transfer.fineRows,
                            preSmoothIterations,
                            jacobiDamping);

    result.residualAfterPreSmoothing = tensorLocalResidual(fineMatrix,
                                                           rhs,
                                                           result.preSmoothedSolution,
                                                           transfer.fineRows);

    applyTensorHGradPTransferRestriction(transfer,
                                         result.residualAfterPreSmoothing,
                                         result.restrictedResidual);

    const auto coarseMatrix = tensorLocalBuildGalerkinCoarseMatrix(fineMatrix, transfer);
    result.coarseCorrection = tensorLocalSolveDenseLinearSystem(coarseMatrix,
                                                                result.restrictedResidual,
                                                                transfer.coarseRows);

    applyTensorHGradPTransferProlongation(transfer,
                                          result.coarseCorrection,
                                          result.prolongatedCorrection);

    result.solutionAfterCoarseCorrection = result.preSmoothedSolution;
    for (int i=0; i<transfer.fineRows; i++)
      result.solutionAfterCoarseCorrection[static_cast<size_t>(i)] +=
          result.prolongatedCorrection[static_cast<size_t>(i)];

    result.residualAfterCoarseCorrection = tensorLocalResidual(fineMatrix,
                                                               rhs,
                                                               result.solutionAfterCoarseCorrection,
                                                               transfer.fineRows);

    result.finalSolution = result.solutionAfterCoarseCorrection;
    tensorLocalJacobiSmooth(fineMatrix,
                            inverseDiagonal,
                            rhs,
                            result.finalSolution,
                            transfer.fineRows,
                            postSmoothIterations,
                            jacobiDamping);

    result.residualAfterPostSmoothing = tensorLocalResidual(fineMatrix,
                                                            rhs,
                                                            result.finalSolution,
                                                            transfer.fineRows);
    return result;
  }

  enum class TensorLocalPMultigridCoarseSolverPolicy
  {
    Direct,
    Chebyshev,
    Jacobi
  };

  inline TensorLocalPMultigridCoarseSolverPolicy
  tensorLocalPMultigridCoarseSolverPolicyFromName(const std::string &name,
                                                  const char *label = "tensorLocalPMultigrid")
  {
    if (name == "direct") return TensorLocalPMultigridCoarseSolverPolicy::Direct;
    if (name == "chebyshev") return TensorLocalPMultigridCoarseSolverPolicy::Chebyshev;
    if (name == "jacobi") return TensorLocalPMultigridCoarseSolverPolicy::Jacobi;
    throw std::invalid_argument(std::string(label) + " coarse solver must be direct, chebyshev, or jacobi");
  }

  inline const char *tensorLocalPMultigridCoarseSolverPolicyName(const TensorLocalPMultigridCoarseSolverPolicy policy)
  {
    switch (policy)
    {
      case TensorLocalPMultigridCoarseSolverPolicy::Direct: return "direct";
      case TensorLocalPMultigridCoarseSolverPolicy::Chebyshev: return "chebyshev";
      case TensorLocalPMultigridCoarseSolverPolicy::Jacobi: return "jacobi";
    }
    throw std::invalid_argument("unknown TensorLocalPMultigridCoarseSolverPolicy");
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalBlockDenseCellMatrix(const std::vector<Scalar> &matrices,
                                                      const int cellOrdinal,
                                                      const int rows)
  {
    if (rows <= 0) throw std::invalid_argument("tensorLocalBlockDenseCellMatrix requires rows > 0");
    if (cellOrdinal < 0) throw std::invalid_argument("tensorLocalBlockDenseCellMatrix requires nonnegative cell ordinal");
    const size_t entries = static_cast<size_t>(rows) * static_cast<size_t>(rows);
    const size_t offset = static_cast<size_t>(cellOrdinal) * entries;
    if (offset + entries > matrices.size())
      throw std::invalid_argument("tensorLocalBlockDenseCellMatrix cell range exceeds matrix storage");
    return std::vector<Scalar>(matrices.begin() + offset,
                               matrices.begin() + offset + entries);
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalBlockDenseCellVector(const std::vector<Scalar> &values,
                                                      const int cellOrdinal,
                                                      const int rows)
  {
    if (rows <= 0) throw std::invalid_argument("tensorLocalBlockDenseCellVector requires rows > 0");
    if (cellOrdinal < 0) throw std::invalid_argument("tensorLocalBlockDenseCellVector requires nonnegative cell ordinal");
    const size_t entries = static_cast<size_t>(rows);
    const size_t offset = static_cast<size_t>(cellOrdinal) * entries;
    if (offset + entries > values.size())
      throw std::invalid_argument("tensorLocalBlockDenseCellVector cell range exceeds vector storage");
    return std::vector<Scalar>(values.begin() + offset,
                               values.begin() + offset + entries);
  }

  template<class Scalar>
  void tensorLocalApplyBlockDenseCellMatrix(const std::vector<Scalar> &matrices,
                                            const int cellOrdinal,
                                            const int rows,
                                            const std::vector<Scalar> &input,
                                            std::vector<Scalar> &output)
  {
    const auto cellMatrix = tensorLocalBlockDenseCellMatrix(matrices, cellOrdinal, rows);
    output = tensorLocalApplyDenseMatrix(cellMatrix, input, rows);
  }

  template<class Scalar>
  void tensorLocalRunDampedJacobiBlockSmoother(const std::vector<Scalar> &matrices,
                                               const std::vector<Scalar> &inverseDiagonal,
                                               const std::vector<Scalar> &lambdaMaxByCell,
                                               const int cellOrdinal,
                                               const int rows,
                                               const std::vector<Scalar> &rhs,
                                               std::vector<Scalar> &solution,
                                               const int iterations)
  {
    const Scalar lambdaMax = std::max(lambdaMaxByCell[static_cast<size_t>(cellOrdinal)], Scalar(1.0));
    const Scalar omega = Scalar(1.0) / lambdaMax;
    const auto cellMatrix = tensorLocalBlockDenseCellMatrix(matrices, cellOrdinal, rows);
    const auto cellInverseDiagonal = tensorLocalBlockDenseCellVector(inverseDiagonal, cellOrdinal, rows);
    tensorLocalJacobiSmooth(cellMatrix, cellInverseDiagonal, rhs, solution, rows, iterations, omega);
  }

  template<class Scalar>
  void tensorLocalRunChebyshevBlockSmoother(const std::vector<Scalar> &matrices,
                                            const std::vector<Scalar> &inverseDiagonal,
                                            const std::vector<Scalar> &lambdaMaxByCell,
                                            const int cellOrdinal,
                                            const int rows,
                                            const std::vector<Scalar> &rhs,
                                            std::vector<Scalar> &solution,
                                            const int iterations,
                                            const Scalar lowerFraction,
                                            Scalar *iter1LambdaUsed = nullptr,
                                            std::vector<Scalar> *iter1InverseDiagonal = nullptr,
                                            std::vector<Scalar> *iter1ResidualBefore = nullptr,
                                            std::vector<Scalar> *iter1Correction = nullptr,
                                            std::vector<Scalar> *iter1ResidualAfter = nullptr)
  {
    if (iterations < 1) return;
    std::vector<Scalar> correction(static_cast<size_t>(rows), Scalar(0.0));
    std::vector<Scalar> applied(static_cast<size_t>(rows), Scalar(0.0));
    const Scalar chebyshevLambdaMax = std::max(lambdaMaxByCell[static_cast<size_t>(cellOrdinal)], Scalar(1.0));
    const Scalar chebyshevLambdaMin = std::max(std::numeric_limits<Scalar>::epsilon(), lowerFraction * chebyshevLambdaMax);
    const Scalar ifpackDelta = Scalar(2.0) / (chebyshevLambdaMax - chebyshevLambdaMin);
    const Scalar ifpackTheta = Scalar(0.5) * (chebyshevLambdaMax + chebyshevLambdaMin);
    const Scalar s1 = ifpackTheta * ifpackDelta;
    Scalar rhok = Scalar(1.0) / s1;
    const size_t cellOffset = static_cast<size_t>(cellOrdinal) * static_cast<size_t>(rows);
    tensorLocalApplyBlockDenseCellMatrix(matrices, cellOrdinal, rows, solution, applied);
    std::vector<Scalar> iter1ResidualBeforeCapture;
    if (iter1ResidualBefore != nullptr) iter1ResidualBeforeCapture.assign(static_cast<size_t>(rows), Scalar(0.0));
    for (int i=0; i<rows; i++)
    {
      const size_t index = cellOffset + static_cast<size_t>(i);
      const Scalar residualBefore = rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)];
      if (iter1ResidualBefore != nullptr) iter1ResidualBeforeCapture[static_cast<size_t>(i)] = residualBefore;
      correction[static_cast<size_t>(i)] = (Scalar(1.0) / ifpackTheta) * inverseDiagonal[index] * residualBefore;
      solution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
    }
    if (iter1LambdaUsed != nullptr) *iter1LambdaUsed = chebyshevLambdaMax;
    if (iter1InverseDiagonal != nullptr)
    {
      iter1InverseDiagonal->assign(static_cast<size_t>(rows), Scalar(0.0));
      for (int i=0; i<rows; i++) (*iter1InverseDiagonal)[static_cast<size_t>(i)] = inverseDiagonal[cellOffset + static_cast<size_t>(i)];
    }
    if (iter1ResidualBefore != nullptr) *iter1ResidualBefore = iter1ResidualBeforeCapture;
    if (iter1Correction != nullptr) *iter1Correction = correction;
    if (iter1ResidualAfter != nullptr)
    {
      tensorLocalApplyBlockDenseCellMatrix(matrices, cellOrdinal, rows, solution, applied);
      iter1ResidualAfter->assign(static_cast<size_t>(rows), Scalar(0.0));
      for (int i=0; i<rows; i++) (*iter1ResidualAfter)[static_cast<size_t>(i)] = rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)];
    }
    for (int iteration=1; iteration<iterations; iteration++)
    {
      tensorLocalApplyBlockDenseCellMatrix(matrices, cellOrdinal, rows, solution, applied);
      const Scalar rhokp1 = Scalar(1.0) / (Scalar(2.0) * s1 - rhok);
      const Scalar dtemp1 = rhokp1 * rhok;
      const Scalar dtemp2 = Scalar(2.0) * rhokp1 * ifpackDelta;
      rhok = rhokp1;
      for (int i=0; i<rows; i++)
      {
        const size_t index = cellOffset + static_cast<size_t>(i);
        correction[static_cast<size_t>(i)] = dtemp2 * inverseDiagonal[index] * (rhs[static_cast<size_t>(i)] - applied[static_cast<size_t>(i)])
                                           + dtemp1 * correction[static_cast<size_t>(i)];
        solution[static_cast<size_t>(i)] += correction[static_cast<size_t>(i)];
      }
    }
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalExtractInverseDiagonalFromBlockDenseMatrices(
      const std::vector<Scalar> &matrices,
      const int cells,
      const int rows,
      const char *label = "tensorLocalExtractInverseDiagonalFromBlockDenseMatrices")
  {
    const size_t localLength =
        static_cast<size_t>(cells) * static_cast<size_t>(rows);

    const size_t expectedEntries =
        localLength * static_cast<size_t>(rows);

    if (matrices.size() != expectedEntries)
    {
      std::ostringstream message;
      message << label << " matrix size mismatch: "
              << "matrices.size()=" << matrices.size()
              << " expected=" << expectedEntries
              << " cells=" << cells
              << " rows=" << rows;
      throw std::invalid_argument(message.str());
    }

    std::vector<Scalar> inverseDiagonal(localLength, Scalar(0.0));

    for (int c=0; c<cells; c++)
    {
      const auto cellMatrix = tensorLocalBlockDenseCellMatrix(matrices, c, rows);
      const auto cellInverseDiagonal = tensorLocalExtractInverseDiagonal(cellMatrix, rows);
      const size_t cellOffset = static_cast<size_t>(c) * static_cast<size_t>(rows);
      std::copy(cellInverseDiagonal.begin(), cellInverseDiagonal.end(), inverseDiagonal.begin() + cellOffset);
    }

    return inverseDiagonal;
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalExtractLambdaMaxFromBlockDenseMatrices(
      const std::vector<Scalar> &matrices,
      const std::vector<Scalar> &inverseDiagonal,
      const int cells,
      const int rows,
      const char *label = "tensorLocalExtractLambdaMaxFromBlockDenseMatrices")
  {
    const size_t localLength = static_cast<size_t>(cells) * static_cast<size_t>(rows);
    if (inverseDiagonal.size() != localLength)
      throw std::invalid_argument(std::string(label) + " inverse diagonal size mismatch");
    if (matrices.size() != localLength * static_cast<size_t>(rows))
      throw std::invalid_argument(std::string(label) + " matrix size mismatch");

    std::vector<Scalar> lambdaMaxByCell(static_cast<size_t>(cells), Scalar(0.0));
    for (int c=0; c<cells; c++)
    {
      for (int i=0; i<rows; i++)
      {
        const size_t rowOffset = static_cast<size_t>((c * rows + i) * rows);
        Scalar scaledRowSum = Scalar(0.0);
        for (int j=0; j<rows; j++)
          scaledRowSum += std::abs(matrices[rowOffset + static_cast<size_t>(j)]) * inverseDiagonal[static_cast<size_t>(c * rows + i)];
        lambdaMaxByCell[static_cast<size_t>(c)] = std::max(lambdaMaxByCell[static_cast<size_t>(c)], scaledRowSum);
      }
    }
    return lambdaMaxByCell;
  }

  template<class Scalar>
  Scalar tensorLocalDenseDInvADeterministicPowerLambdaMax(
      const std::vector<Scalar> &matrix,
      const std::vector<Scalar> &inverseDiagonal,
      const int rows,
      const int numIters,
      const int eigNormalizationFreq = 1,
      const char *label = "tensorLocalDenseDInvADeterministicPowerLambdaMax",
      const Scalar unavailableValue = Scalar(-1.0),
      const Scalar tolerance = Scalar(1.0e-7))
  {
    tensorLocalCheckDenseMatrixSize(matrix, rows, label);
    tensorLocalCheckVectorSize(inverseDiagonal, rows, label);
    if (numIters <= 0)
      return unavailableValue;

    std::vector<Scalar> x(static_cast<size_t>(rows), Scalar(0.0));
    std::vector<Scalar> y(static_cast<size_t>(rows), Scalar(0.0));
    const int modulus = 97;
    for (int i=0; i<rows; i++)
      x[static_cast<size_t>(i)] = Scalar(1.0 + (i % modulus));

    auto l2Norm = [](const std::vector<Scalar> &values) -> Scalar
    {
      Scalar normSquared = Scalar(0.0);
      for (const auto value : values)
        normSquared += value * value;
      return std::sqrt(std::max(Scalar(0.0), normSquared));
    };

    auto scale = [](std::vector<Scalar> &values, const Scalar alpha)
    {
      for (auto &value : values)
        value *= alpha;
    };

    auto applyDInvA = [&](const std::vector<Scalar> &input, std::vector<Scalar> &output)
    {
      for (int i=0; i<rows; i++)
      {
        Scalar rowValue = Scalar(0.0);
        const size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(rows);
        for (int j=0; j<rows; j++)
          rowValue += matrix[rowOffset + static_cast<size_t>(j)] * input[static_cast<size_t>(j)];
        output[static_cast<size_t>(i)] = inverseDiagonal[static_cast<size_t>(i)] * rowValue;
      }
    };

    Scalar norm = l2Norm(x);
    if (norm == Scalar(0.0))
      return unavailableValue;
    scale(x, Scalar(1.0) / norm);

    const int normalizationFreq = std::max(1, eigNormalizationFreq);
    Scalar lambdaMax = Scalar(0.0);
    Scalar lambdaMaxOld = Scalar(1.0);

    // Match the deterministic Tpetra RowMatrix D^-1 A power helper used for controlled
    // Chebyshev lambda estimates: repeatedly apply D^-1 A, periodically normalize, and
    // return ||D^-1 A x||_2 after the final normalized iterate.
    for (int iter=0; iter<numIters-1; iter++)
    {
      applyDInvA(x, y);
      x.swap(y);
      if ((((iter + 1) % normalizationFreq) == 0) && (iter < numIters - 2))
      {
        norm = l2Norm(x);
        if (norm == Scalar(0.0))
          return Scalar(0.0);
        lambdaMaxOld = lambdaMax;
        lambdaMax = std::pow(norm, Scalar(1.0) / static_cast<Scalar>(normalizationFreq));
        if (std::abs(lambdaMax - lambdaMaxOld) < tolerance * std::abs(lambdaMax))
          return lambdaMax;
        scale(x, Scalar(1.0) / norm);
      }
    }

    norm = l2Norm(x);
    if (norm == Scalar(0.0))
      return Scalar(0.0);
    scale(x, Scalar(1.0) / norm);
    applyDInvA(x, y);
    return l2Norm(y);
  }

  template<class Scalar>
  std::vector<Scalar> tensorLocalExtractDeterministicPowerLambdaMaxFromBlockDenseMatrices(
      const std::vector<Scalar> &matrices,
      const std::vector<Scalar> &inverseDiagonal,
      const int cells,
      const int rows,
      const int numIters,
      const int eigNormalizationFreq = 1,
      const char *label = "tensorLocalExtractDeterministicPowerLambdaMaxFromBlockDenseMatrices")
  {
    const size_t localLength = static_cast<size_t>(cells) * static_cast<size_t>(rows);
    if (inverseDiagonal.size() != localLength)
      throw std::invalid_argument(std::string(label) + " inverse diagonal size mismatch");
    if (matrices.size() != localLength * static_cast<size_t>(rows))
      throw std::invalid_argument(std::string(label) + " matrix size mismatch");

    std::vector<Scalar> lambdaMaxByCell(static_cast<size_t>(cells), Scalar(0.0));
    for (int c=0; c<cells; c++)
    {
      const auto cellMatrix = tensorLocalBlockDenseCellMatrix(matrices, c, rows);
      const auto cellInverseDiagonal = tensorLocalBlockDenseCellVector(inverseDiagonal, c, rows);
      lambdaMaxByCell[static_cast<size_t>(c)] =
          tensorLocalDenseDInvADeterministicPowerLambdaMax(cellMatrix, cellInverseDiagonal, rows,
                                                          numIters, eigNormalizationFreq, label);
    }
    return lambdaMaxByCell;
  }
}

