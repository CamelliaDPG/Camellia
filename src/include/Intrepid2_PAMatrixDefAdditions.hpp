//
// © 2026 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#ifndef __INTREPID2_PAMATRIX_DEF_ADDITIONS_HPP__
#define __INTREPID2_PAMATRIX_DEF_ADDITIONS_HPP__

#include "Intrepid2_PAMatrix.hpp"
#include "Intrepid2_DataDimensionInfo.hpp"

#include "Kokkos_Core.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace Intrepid2 {

template<typename DeviceType,class Scalar>
Data<Scalar,DeviceType> PAMatrix<DeviceType,Scalar>::allocateDiagonalStorage() const
{
  const int CELL_DIM = 0;
  const auto cellMeasureData = _cellMeasures.getTensorComponent(0);

  DimensionInfo combinedCellDimInfo = cellMeasureData.getDimensionInfo(CELL_DIM);
  if (_basisValuesLeft.transform().isValid())
  {
    combinedCellDimInfo = combinedDimensionInfo(combinedCellDimInfo,
                                                _basisValuesLeft.transform().getDimensionInfo(CELL_DIM));
  }
  if (_basisValuesRight.transform().isValid())
  {
    combinedCellDimInfo = combinedDimensionInfo(combinedCellDimInfo,
                                                _basisValuesRight.transform().getDimensionInfo(CELL_DIM));
  }

  DataVariationType cellVariationType = combinedCellDimInfo.variationType;
  int cellDataExtent                  = combinedCellDimInfo.dataExtent;

  const int numCells       = _basisValuesLeft.numCells();
  const int numFieldsLeft  = _basisValuesLeft.numFields();
  const int numFieldsRight = _basisValuesRight.numFields();
  const int numDiagonalEntries = std::min(numFieldsLeft, numFieldsRight);

  if (_orientations.size() > 0)
  {
    // Orientations can make otherwise constant cell data cell-dependent.
    // Preserve per-cell diagonal storage so fallback assembly/copy and future
    // oriented diagonal paths do not collapse distinct oriented cells.
    cellVariationType = GENERAL;
    cellDataExtent = numCells;
  }

  Kokkos::Array<int,2> extents {numCells, numDiagonalEntries};
  Kokkos::Array<DataVariationType,2> variationTypes {cellVariationType, GENERAL};

  if (cellVariationType != CONSTANT)
  {
    Kokkos::View<Scalar**,DeviceType> data("Intrepid2::PAMatrix diagonal storage", cellDataExtent, numDiagonalEntries);
    return Data<Scalar,DeviceType>(data, extents, variationTypes);
  }
  else
  {
    Kokkos::View<Scalar*,DeviceType> data("Intrepid2::PAMatrix diagonal storage", numDiagonalEntries);
    return Data<Scalar,DeviceType>(data, extents, variationTypes);
  }
}

template<typename DeviceType,class Scalar>
bool pamatrixAssemble2UsesUnitColumnPath(const PAMatrix<DeviceType,Scalar> &matrix)
{
  bool useUnitColumnPath = (matrix._orientations.size() == 0);
  for (const auto &integralTuple : matrix.componentIntegralsToSum_)
  {
    const auto &leftOps       = std::get<0>(integralTuple);
    const typename PAMatrix<DeviceType,Scalar>::PointDataSpec &pointDataSpec = std::get<1>(integralTuple);
    const auto &rightOps      = std::get<2>(integralTuple);
    const int leftSpan        = std::get<4>(integralTuple);
    const int rightSpan       = std::get<6>(integralTuple);

    if ((leftOps.size() != rightOps.size()) || (rightOps.size() == 0) ||
        (rightOps.size() > 8) || (pointDataSpec.aSpan != 1) || (pointDataSpec.bSpan != 1))
    {
      useUnitColumnPath = false;
      break;
    }

    int rightFieldProduct = 1;
    int pointProduct = 1;
    for (size_t r=0; r<rightOps.size(); r++)
    {
      rightFieldProduct *= rightOps[r].N;
      pointProduct      *= rightOps[r].M;
    }
    int leftFieldProduct = 1;
    int leftPointProduct = 1;
    for (size_t r=0; r<leftOps.size(); r++)
    {
      leftFieldProduct *= leftOps[r].M;
      leftPointProduct *= leftOps[r].N;
    }
    if ((rightFieldProduct != rightSpan) || (leftFieldProduct != leftSpan) ||
        (pointProduct != pointDataSpec.P) || (leftPointProduct != pointDataSpec.P))
    {
      useUnitColumnPath = false;
      break;
    }
  }
  return useUnitColumnPath;
}

template<typename DeviceType,class Scalar>
void PAMatrix<DeviceType,Scalar>::assemble2(Data<Scalar,DeviceType> &integrals) const
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MultiVector = ScalarView<Scalar,DeviceType>;
  using ScratchView = Kokkos::View<Scalar*,DeviceType,Kokkos::MemoryUnmanaged>;

  const ordinal_type numCells       = _basisValuesLeft.numCells();
  const ordinal_type numFieldsLeft  = _basisValuesLeft.numFields();
  const ordinal_type numFieldsRight = _basisValuesRight.numFields();

  INTREPID2_TEST_FOR_EXCEPTION(integrals.rank() != 3, std::invalid_argument,
      "PAMatrix::assemble2() requires a rank-3 logical output container");
  INTREPID2_TEST_FOR_EXCEPTION(integrals.extent_int(0) != numCells ||
                               integrals.extent_int(1) != numFieldsLeft ||
                               integrals.extent_int(2) != numFieldsRight, std::invalid_argument,
      "PAMatrix::assemble2() output container has incompatible extents");

  const bool useUnitColumnPath = pamatrixAssemble2UsesUnitColumnPath(*this);

  const ordinal_type maxColumnBlockSize = 64;
  const ordinal_type columnBlockSize = std::min(numFieldsRight, maxColumnBlockSize);

  for (ordinal_type columnStart=0; columnStart<numFieldsRight; columnStart+=columnBlockSize)
  {
    const ordinal_type blockSize = std::min(columnBlockSize, numFieldsRight - columnStart);
    MultiVector outputColumns("PAMatrix assemble2 output columns", numCells, numFieldsLeft, blockSize);

    auto workspace = this->allocateWorkspace(numCells, blockSize);

    if (useUnitColumnPath)
    {
      Kokkos::deep_copy(outputColumns, Scalar(0));

      const int numIntegrationPasses = int(componentIntegralsToSum_.size());
      for (int integrationPass=0; integrationPass<numIntegrationPasses; integrationPass++)
      {
        const auto &integralTuple = componentIntegralsToSum_[integrationPass];
        const auto &leftOps       = std::get<0>(integralTuple);
        const PointDataSpec &pointDataSpec = std::get<1>(integralTuple);
        const auto &rightOps      = std::get<2>(integralTuple);
        const int leftOrdinalOffset  = std::get<3>(integralTuple);
        const int leftOutputSpan     = std::get<4>(integralTuple);
        const int rightOrdinalOffset = std::get<5>(integralTuple);
        const int rightInputSpan     = std::get<6>(integralTuple);
        const int numTensorComponents = int(rightOps.size());

        Kokkos::Array<typename PAMatrix<DeviceType,Scalar>::OpSpec,8> rightOpsArray;
        Kokkos::Array<typename PAMatrix<DeviceType,Scalar>::OpSpec,8> leftOpsArray;
        Kokkos::Array<int,8> rightFieldExtents;
        Kokkos::Array<int,8> pointExtents;
        for (int r=0; r<numTensorComponents; r++)
        {
          rightOpsArray[r] = rightOps[r];
          leftOpsArray[r]  = leftOps[r];
          rightFieldExtents[r] = rightOps[r].N;
          pointExtents[r]      = rightOps[r].M;
        }

        const int workspace1Size = maxIntermediateSize_ * numCells * blockSize;
        const int workspace2Size = maxIntermediateSize_ * numCells * blockSize;
        ScratchView workspace1(workspace.data(), workspace1Size);
        ScratchView workspace2(workspace.data() + workspace1Size, workspace2Size);
        ScratchView pointDataIn  = (numTensorComponents % 2 == 0) ? workspace1 : workspace2;
        ScratchView pointDataOut = (numTensorComponents % 2 == 0) ? workspace2 : workspace1;

        auto unitColumnPolicy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,pointDataSpec.P,blockSize});
        Kokkos::parallel_for("PAMatrix::assemble2(): unit trial columns to points", unitColumnPolicy,
        KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &pointOrdinal, const ordinal_type &blockColumn)
        {
          const int globalColumn = columnStart + blockColumn;
          Scalar value = 0;
          if ((globalColumn >= rightOrdinalOffset) && (globalColumn < rightOrdinalOffset + rightInputSpan))
          {
            int columnRemainder = globalColumn - rightOrdinalOffset;
            int pointRemainder = pointOrdinal;
            value = Scalar(1);
            for (int r=0; r<numTensorComponents; r++)
            {
              const int field = columnRemainder % rightFieldExtents[r];
              const int point = pointRemainder % pointExtents[r];
              columnRemainder /= rightFieldExtents[r];
              pointRemainder  /= pointExtents[r];
              const auto op = rightOpsArray[r];
              value *= op.opView(point + field * op.M);
            }
          }
          pointDataIn(pointOrdinal + (blockColumn + cell * blockSize) * pointDataSpec.P) = value;
        });
        ExecutionSpace().fence();

        auto pointData = _pointDataCache.find(pointDataSpec)->second;
        Impl::pointDataMultiply<DeviceType,Scalar>(pointDataSpec.C, pointDataSpec.P, pointDataSpec.aSpan, pointDataSpec.bSpan,
                                                   pointData.data(), pointDataIn.data(), pointDataOut.data(), blockSize);

        int Nr = pointDataSpec.P * numCells * blockSize * pointDataSpec.aSpan;
        const int numLeftIntegrals = int(leftOps.size());
        for (int i=0; i<numLeftIntegrals; i++)
        {
          auto in  = ((i+numTensorComponents+1)%2 == 0) ? workspace1 : workspace2;
          auto out = ((i+numTensorComponents+1)%2 == 0) ? workspace2 : workspace1;

          const int r = numLeftIntegrals-i-1;
          auto op = leftOpsArray[r];
          const ordinal_type & Fr = op.M;
          const ordinal_type & Pr = op.N;

          const auto A = op.opView.data();
          const ordinal_type LDA = Fr;
          const auto B = in.data();
          auto C = out.data();
          INTREPID2_TEST_FOR_EXCEPTION(Nr % Pr != 0, std::invalid_argument, "Error: Nr must be a multiple of Pr");
          Impl::gemm<Impl::GemmDeviceType>('N', 'T', Fr, Nr/Pr, Pr, Scalar(1.0), A, LDA, B, Scalar(0.0), C);
          Nr = (Nr * Fr) / Pr;
        }
        auto finalOut = ((numLeftIntegrals+numTensorComponents+1)%2 == 0) ? workspace1 : workspace2;
        auto sumPolicy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,leftOutputSpan,blockSize});
        Kokkos::parallel_for("PAMatrix::assemble2(): sum unit-column final output", sumPolicy,
        KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &field, const ordinal_type &blockColumn)
        {
          const ordinal_type idx = field + (blockColumn + cell * blockSize) * leftOutputSpan;
          outputColumns(cell,field+leftOrdinalOffset,blockColumn) += finalOut(idx);
        });
        ExecutionSpace().fence();
      }
    }
    else
    {
      MultiVector inputColumns ("PAMatrix assemble2 input unit columns", numCells, numFieldsRight, blockSize);
      auto identityPolicy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,numFieldsRight,blockSize});
      Kokkos::parallel_for("PAMatrix::assemble2(): set unit multivector", identityPolicy,
      KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &field, const ordinal_type &blockColumn)
      {
        inputColumns(cell,field,blockColumn) = (field == columnStart + blockColumn) ? Scalar(1) : Scalar(0);
      });
      this->apply(outputColumns, inputColumns, workspace, false, numCells);
    }

    if (integrals.getUnderlyingViewRank() == 3)
    {
      auto matrix = integrals.template getUnderlyingView<3>();
      auto copyPolicy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,numFieldsLeft,blockSize});
      Kokkos::parallel_for("PAMatrix::assemble2(): copy general columns", copyPolicy,
      KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &row, const ordinal_type &blockColumn)
      {
        matrix(cell,row,columnStart + blockColumn) = outputColumns(cell,row,blockColumn);
      });
    }
    else if (integrals.getUnderlyingViewRank() == 2)
    {
      auto matrix = integrals.template getUnderlyingView<2>();
      auto copyPolicy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{numFieldsLeft,blockSize});
      Kokkos::parallel_for("PAMatrix::assemble2(): copy constant-cell columns", copyPolicy,
      KOKKOS_LAMBDA(const ordinal_type &row, const ordinal_type &blockColumn)
      {
        matrix(row,columnStart + blockColumn) = outputColumns(0,row,blockColumn);
      });
    }
    else
    {
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument,
          "PAMatrix::assemble2() supports only rank-2 or rank-3 underlying matrix storage");
    }
  }
  ExecutionSpace().fence();
}

template<typename DeviceType,class Scalar>
void PAMatrix<DeviceType,Scalar>::extractDiagonal(const Data<Scalar,DeviceType> &diagonal) const
{
  using ExecutionSpace = typename DeviceType::execution_space;

  const ordinal_type numCells       = _basisValuesLeft.numCells();
  const ordinal_type numFieldsLeft  = _basisValuesLeft.numFields();
  const ordinal_type numFieldsRight = _basisValuesRight.numFields();
  const ordinal_type numDiagonalEntries = std::min(numFieldsLeft, numFieldsRight);

  INTREPID2_TEST_FOR_EXCEPTION(diagonal.rank() != 2, std::invalid_argument,
      "PAMatrix::extractDiagonal() requires a rank-2 logical output container");
  INTREPID2_TEST_FOR_EXCEPTION(diagonal.extent_int(0) != numCells ||
                               diagonal.extent_int(1) != numDiagonalEntries, std::invalid_argument,
      "PAMatrix::extractDiagonal() output container has incompatible extents");

  const auto assembleAndCopyDiagonal = [this, numCells, numFieldsLeft, numFieldsRight, numDiagonalEntries, diagonal]()
  {
    Kokkos::View<Scalar***,DeviceType> matrixView("PAMatrix extractDiagonal assembled temporary",
                                                  numCells, numFieldsLeft, numFieldsRight);
    Kokkos::Array<int,3> matrixExtents {numCells, numFieldsLeft, numFieldsRight};
    Kokkos::Array<DataVariationType,3> matrixVariationTypes {GENERAL, GENERAL, GENERAL};
    Data<Scalar,DeviceType> matrixData(matrixView, matrixExtents, matrixVariationTypes);
    this->assemble2(matrixData);

    if (diagonal.getUnderlyingViewRank() == 2)
    {
      auto diagonalView = diagonal.template getUnderlyingView<2>();
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{numCells,numDiagonalEntries});
      Kokkos::parallel_for("PAMatrix::extractDiagonal(): copy assembled diagonal", policy,
      KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &field)
      {
        diagonalView(cell,field) = matrixView(cell,field,field);
      });
    }
    else if (diagonal.getUnderlyingViewRank() == 1)
    {
      auto diagonalView = diagonal.template getUnderlyingView<1>();
      auto policy = Kokkos::RangePolicy<ExecutionSpace>(0,numDiagonalEntries);
      Kokkos::parallel_for("PAMatrix::extractDiagonal(): copy assembled constant-cell diagonal", policy,
      KOKKOS_LAMBDA(const ordinal_type &field)
      {
        diagonalView(field) = matrixView(0,field,field);
      });
    }
    else
    {
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument,
          "PAMatrix::extractDiagonal() supports only rank-1 or rank-2 underlying diagonal storage");
    }
    ExecutionSpace().fence();
  };

  bool useSumFactorizedPath = (_orientations.size() == 0);
  for (const auto &integralTuple : componentIntegralsToSum_)
  {
    const auto &leftOps       = std::get<0>(integralTuple);
    const PointDataSpec &pointDataSpec = std::get<1>(integralTuple);
    const auto &rightOps      = std::get<2>(integralTuple);
    const int leftOffset      = std::get<3>(integralTuple);
    const int leftSpan        = std::get<4>(integralTuple);
    const int rightOffset     = std::get<5>(integralTuple);
    const int rightSpan       = std::get<6>(integralTuple);

    if ((leftOps.size() != rightOps.size()) || (leftOps.size() == 0) ||
        (pointDataSpec.aSpan != 1) || (pointDataSpec.bSpan != 1) ||
        (leftOffset != rightOffset) || (leftSpan != rightSpan) ||
        (leftOffset + leftSpan > numDiagonalEntries))
    {
      useSumFactorizedPath = false;
      break;
    }

    int fieldProduct = 1;
    int pointProduct = 1;
    for (size_t r=0; r<leftOps.size(); r++)
    {
      if ((leftOps[r].M != rightOps[r].N) || (leftOps[r].N != rightOps[r].M))
      {
        useSumFactorizedPath = false;
        break;
      }
      fieldProduct *= leftOps[r].M;
      pointProduct *= leftOps[r].N;
    }
    if (!useSumFactorizedPath) break;
    if ((fieldProduct != leftSpan) || (pointProduct != pointDataSpec.P))
    {
      useSumFactorizedPath = false;
      break;
    }
  }

  if (!useSumFactorizedPath)
  {
    assembleAndCopyDiagonal();
    return;
  }

  diagonal.clear();
  Data<Scalar,DeviceType> diagonalData = diagonal;
  const int diagonalCellDataExtent = diagonal.getDataExtent(0);

  for (const auto &integralTuple : componentIntegralsToSum_)
  {
    const auto &leftOps       = std::get<0>(integralTuple);
    const PointDataSpec &pointDataSpec = std::get<1>(integralTuple);
    const auto &rightOps      = std::get<2>(integralTuple);
    const int diagonalOffset  = std::get<3>(integralTuple);
    const int numTensorComponents = static_cast<int>(leftOps.size());

    std::vector<int> fieldExtents(numTensorComponents);
    std::vector<int> pointExtents(numTensorComponents);
    int fieldProduct = 1;
    int pointProduct = 1;
    int maxScratchEntriesPerCell = 1;
    for (int r=0; r<numTensorComponents; r++)
    {
      fieldExtents[r] = leftOps[r].M;
      pointExtents[r] = leftOps[r].N;
      fieldProduct *= fieldExtents[r];
      pointProduct *= pointExtents[r];
    }
    for (int r=1; r<numTensorComponents; r++)
    {
      int prefixPointProduct = 1;
      for (int s=0; s<r; s++) prefixPointProduct *= pointExtents[s];
      int suffixFieldProduct = 1;
      for (int s=r; s<numTensorComponents; s++) suffixFieldProduct *= fieldExtents[s];
      maxScratchEntriesPerCell = std::max(maxScratchEntriesPerCell, prefixPointProduct * suffixFieldProduct);
    }

    auto pointData = _pointDataCache.find(pointDataSpec)->second;

    if (numTensorComponents == 1)
    {
      auto leftOp  = leftOps[0];
      auto rightOp = rightOps[0];
      const int numFields = fieldExtents[0];
      const int numPoints = pointExtents[0];
      const int pointDataCellExtent = pointDataSpec.C;
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{diagonalCellDataExtent,numFields});
      Kokkos::parallel_for("PAMatrix::extractDiagonal(): 1D sum-factorized diagonal", policy,
      KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &field)
      {
        const int pointDataCell = (pointDataCellExtent == 1) ? 0 : cell;
        Scalar value = 0;
        for (int point=0; point<numPoints; point++)
        {
          const Scalar L = leftOp.opView(field + point * leftOp.M);
          const Scalar R = rightOp.opView(point + field * rightOp.M);
          const Scalar W = pointData(pointDataCell + point * pointDataCellExtent);
          value += L * R * W;
        }
        diagonalData.getWritableEntry(cell, diagonalOffset + field) += value;
      });
      ExecutionSpace().fence();
      continue;
    }

    const int scratchEntries = diagonalCellDataExtent * maxScratchEntriesPerCell;
    Kokkos::View<Scalar*,DeviceType> scratchA("PAMatrix diagonal scratch A", scratchEntries);
    Kokkos::View<Scalar*,DeviceType> scratchB("PAMatrix diagonal scratch B", scratchEntries);

    const int lastDim = numTensorComponents - 1;
    {
      auto leftOp  = leftOps[lastDim];
      auto rightOp = rightOps[lastDim];
      int prefixPointProduct = 1;
      for (int s=0; s<lastDim; s++) prefixPointProduct *= pointExtents[s];
      const int numFieldsLast = fieldExtents[lastDim];
      const int numPointsLast = pointExtents[lastDim];
      const int pointDataCellExtent = pointDataSpec.C;
      const int totalEntries = diagonalCellDataExtent * prefixPointProduct * numFieldsLast;
      Kokkos::parallel_for("PAMatrix::extractDiagonal(): initialize last diagonal contraction", Kokkos::RangePolicy<ExecutionSpace>(0,totalEntries),
      KOKKOS_LAMBDA(const ordinal_type &entryOrdinal)
      {
        const int cell = entryOrdinal % diagonalCellDataExtent;
        int remainder = entryOrdinal / diagonalCellDataExtent;
        const int prefixPointOrdinal = remainder % prefixPointProduct;
        const int field = remainder / prefixPointProduct;
        const int pointDataCell = (pointDataCellExtent == 1) ? 0 : cell;
        Scalar value = 0;
        for (int point=0; point<numPointsLast; point++)
        {
          const int pointOrdinal = prefixPointOrdinal + point * prefixPointProduct;
          const Scalar L = leftOp.opView(field + point * leftOp.M);
          const Scalar R = rightOp.opView(point + field * rightOp.M);
          const Scalar W = pointData(pointDataCell + pointOrdinal * pointDataCellExtent);
          value += L * R * W;
        }
        scratchA(entryOrdinal) = value;
      });
      ExecutionSpace().fence();
    }

    bool scratchAIsCurrent = true;
    for (int r=numTensorComponents-2; r>=1; r--)
    {
      auto inScratch  = scratchAIsCurrent ? scratchA : scratchB;
      auto outScratch = scratchAIsCurrent ? scratchB : scratchA;
      auto leftOp  = leftOps[r];
      auto rightOp = rightOps[r];

      int prefixPointProduct = 1;
      for (int s=0; s<r; s++) prefixPointProduct *= pointExtents[s];
      const int pointExtent = pointExtents[r];
      const int previousPrefixPointProduct = prefixPointProduct * pointExtent;
      int suffixFieldProduct = 1;
      for (int s=r; s<numTensorComponents; s++) suffixFieldProduct *= fieldExtents[s];
      int suffixTailProduct = 1;
      for (int s=r+1; s<numTensorComponents; s++) suffixTailProduct *= fieldExtents[s];
      const int totalEntries = diagonalCellDataExtent * prefixPointProduct * suffixFieldProduct;

      Kokkos::parallel_for("PAMatrix::extractDiagonal(): diagonal tensor contraction stage", Kokkos::RangePolicy<ExecutionSpace>(0,totalEntries),
      KOKKOS_LAMBDA(const ordinal_type &entryOrdinal)
      {
        const int cell = entryOrdinal % diagonalCellDataExtent;
        int remainder = entryOrdinal / diagonalCellDataExtent;
        const int prefixPointOrdinal = remainder % prefixPointProduct;
        const int suffixFieldOrdinal = remainder / prefixPointProduct;
        const int field = suffixFieldOrdinal % leftOp.M;
        const int tailFieldOrdinal = suffixFieldOrdinal / leftOp.M;
        Scalar value = 0;
        for (int point=0; point<pointExtent; point++)
        {
          const int previousPrefixPointOrdinal = prefixPointOrdinal + point * prefixPointProduct;
          const int inputIndex = cell + (previousPrefixPointOrdinal + tailFieldOrdinal * previousPrefixPointProduct) * diagonalCellDataExtent;
          const Scalar L = leftOp.opView(field + point * leftOp.M);
          const Scalar R = rightOp.opView(point + field * rightOp.M);
          value += L * R * inScratch(inputIndex);
        }
        outScratch(entryOrdinal) = value;
      });
      ExecutionSpace().fence();
      scratchAIsCurrent = !scratchAIsCurrent;
    }

    auto finalScratch = scratchAIsCurrent ? scratchA : scratchB;
    {
      auto leftOp  = leftOps[0];
      auto rightOp = rightOps[0];
      const int pointExtent = pointExtents[0];
      int suffixTailProduct = 1;
      for (int s=1; s<numTensorComponents; s++) suffixTailProduct *= fieldExtents[s];
      const int totalEntries = diagonalCellDataExtent * fieldProduct;
      Kokkos::parallel_for("PAMatrix::extractDiagonal(): final diagonal accumulation", Kokkos::RangePolicy<ExecutionSpace>(0,totalEntries),
      KOKKOS_LAMBDA(const ordinal_type &entryOrdinal)
      {
        const int cell = entryOrdinal % diagonalCellDataExtent;
        const int fieldOrdinal = entryOrdinal / diagonalCellDataExtent;
        const int field0 = fieldOrdinal % leftOp.M;
        const int tailFieldOrdinal = fieldOrdinal / leftOp.M;
        Scalar value = 0;
        for (int point=0; point<pointExtent; point++)
        {
          const int inputIndex = cell + (point + tailFieldOrdinal * pointExtent) * diagonalCellDataExtent;
          const Scalar L = leftOp.opView(field0 + point * leftOp.M);
          const Scalar R = rightOp.opView(point + field0 * rightOp.M);
          value += L * R * finalScratch(inputIndex);
        }
        diagonalData.getWritableEntry(cell, diagonalOffset + fieldOrdinal) += value;
      });
      ExecutionSpace().fence();
    }
  }
}

} // namespace Intrepid2

#endif // __INTREPID2_PAMATRIX_DEF_ADDITIONS_HPP__
