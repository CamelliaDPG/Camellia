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

  const DataVariationType cellVariationType = combinedCellDimInfo.variationType;
  const int cellDataExtent                  = combinedCellDimInfo.dataExtent;

  const int numCells       = _basisValuesLeft.numCells();
  const int numFieldsLeft  = _basisValuesLeft.numFields();
  const int numFieldsRight = _basisValuesRight.numFields();
  const int numDiagonalEntries = std::min(numFieldsLeft, numFieldsRight);

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
void PAMatrix<DeviceType,Scalar>::assemble2(Data<Scalar,DeviceType> &integrals) const
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MultiVector = ScalarView<Scalar,DeviceType>;

  const ordinal_type numCells       = _basisValuesLeft.numCells();
  const ordinal_type numFieldsLeft  = _basisValuesLeft.numFields();
  const ordinal_type numFieldsRight = _basisValuesRight.numFields();

  INTREPID2_TEST_FOR_EXCEPTION(integrals.rank() != 3, std::invalid_argument,
      "PAMatrix::assemble2() requires a rank-3 logical output container");
  INTREPID2_TEST_FOR_EXCEPTION(integrals.extent_int(0) != numCells ||
                               integrals.extent_int(1) != numFieldsLeft ||
                               integrals.extent_int(2) != numFieldsRight, std::invalid_argument,
      "PAMatrix::assemble2() output container has incompatible extents");

  const ordinal_type maxColumnBlockSize = 64;
  const ordinal_type columnBlockSize = std::min(numFieldsRight, maxColumnBlockSize);

  for (ordinal_type columnStart=0; columnStart<numFieldsRight; columnStart+=columnBlockSize)
  {
    const ordinal_type blockSize = std::min(columnBlockSize, numFieldsRight - columnStart);
    MultiVector inputColumns ("PAMatrix assemble2 input unit columns", numCells, numFieldsRight, blockSize);
    MultiVector outputColumns("PAMatrix assemble2 output columns",     numCells, numFieldsLeft,  blockSize);

    auto identityPolicy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,numFieldsRight,blockSize});
    Kokkos::parallel_for("PAMatrix::assemble2(): set unit multivector", identityPolicy,
    KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &field, const ordinal_type &blockColumn)
    {
      inputColumns(cell,field,blockColumn) = (field == columnStart + blockColumn) ? Scalar(1) : Scalar(0);
    });

    auto workspace = this->allocateWorkspace(numCells, blockSize);
    this->apply(outputColumns, inputColumns, workspace, false, numCells);

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
    Kokkos::parallel_for("PAMatrix::extractDiagonal(): copy general diagonal", policy,
    KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &field)
    {
      diagonalView(cell,field) = matrixView(cell,field,field);
    });
  }
  else if (diagonal.getUnderlyingViewRank() == 1)
  {
    auto diagonalView = diagonal.template getUnderlyingView<1>();
    auto policy = Kokkos::RangePolicy<ExecutionSpace>(0,numDiagonalEntries);
    Kokkos::parallel_for("PAMatrix::extractDiagonal(): copy constant-cell diagonal", policy,
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
}

} // namespace Intrepid2

#endif // __INTREPID2_PAMATRIX_DEF_ADDITIONS_HPP__
