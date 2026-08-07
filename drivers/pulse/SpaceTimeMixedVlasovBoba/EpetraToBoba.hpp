#ifndef EPETRA_TO_BOBA_HPP
#define EPETRA_TO_BOBA_HPP

#include <map>
#include <set>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Exceptions.hpp>
#include <Epetra_Comm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FEVector.h>

#include "BOBA/boba.hpp"

template <boba::execution_space space, typename data_type, typename index_type, typename id_type>
boba::Vector<space, data_type>
extractBobaSubVector(const Teuchos::RCP<Epetra_FEVector>& originalVector,
                     const std::set<id_type>& allIdsToKeep) {
  // TODO construct on CPU, offload to general space?
  static_assert(space == boba::execution_space::CPU, "execution space must be CPU");

  TEUCHOS_TEST_FOR_EXCEPTION(originalVector->Comm().NumProc() != 1,
                             std::runtime_error,
                             "Boba vector extraction requires single-rank MPI execution.");

  // construct contiguous indices
  std::map<id_type, index_type> idToIndex;
  index_type numElem = 0;
  for (const auto &id : allIdsToKeep) {
    idToIndex[id] = numElem;
    ++numElem;
  }

  // allocate boba object for storing subvector
  boba::Vector<space, data_type> subVector({numElem});
  subVector.fill_with_zeros();

  // fill in subvector
  auto subVectorView = subVector.view();
  for (const auto &id : allIdsToKeep) {
    // first [0] index on RHS selects the first (only?) vector out of the MultiVector
    subVectorView(idToIndex[id]) += (*originalVector)[0][id];
  }

  return subVector;
}

template <boba::execution_space space, typename data_type, typename index_type, typename id_type>
boba::Matrix<space, data_type>
extractBobaSubMatrix(const Teuchos::RCP<Epetra_CrsMatrix>& originalMatrix,
                     const std::set<id_type>& allRowsToKeep,
                     const std::set<id_type>& allColsToKeep) {
  // TODO construct on CPU, offload to general space?
  static_assert(space == boba::execution_space::CPU, "execution space must be CPU");

  TEUCHOS_TEST_FOR_EXCEPTION(originalMatrix->Comm().NumProc() != 1,
                             std::runtime_error,
                             "Boba vector extraction requires single-rank MPI execution.");

  // construct contiguous indices for rows and columns
  std::map<id_type, index_type> rowIdToIndex;
  index_type numRow = 0;
  for (const auto &rowId : allRowsToKeep) {
    rowIdToIndex[rowId] = numRow;
    ++numRow;
  }

  std::map<id_type, index_type> colIdToIndex;
  index_type numCol = 0;
  for (const auto &colId : allColsToKeep) {
    colIdToIndex[colId] = numCol;
    ++numCol;
  }

  // allocate boba objects for storing submatrix
  boba::Matrix<space, data_type> subMatrix({numRow, numCol});
  subMatrix.fill_with_zeros();

  // fill in submatrix
  auto subMatrixView = subMatrix.view();
  for (const auto &rowId : allRowsToKeep) {
    const index_type rowIndex = rowIdToIndex[rowId];
    // following types are hard coded in Epetra
    int numEntries = 0;
    double *values = nullptr;
    int *indices = nullptr;
    originalMatrix->ExtractMyRowView(static_cast<int>(rowId), numEntries, values, indices);
    for (int i = 0; i < numEntries; ++i) {
      const id_type colId = static_cast<id_type>(indices[i]);
      const index_type colIndex = colIdToIndex[colId];
      if (allColsToKeep.count(colId) != 0) {
        subMatrixView({rowIndex, colIndex}) += values[i];
      }
    }
  }

  return subMatrix;
}

template <boba::execution_space space, typename data_type, typename index_type, typename id_type, size_t rows, size_t cols>
std::tuple<boba::BlockOperator<boba::Matrix<space, data_type>>,
           boba::BlockVector<boba::Vector<space, data_type>>,
           boba::BlockVector<boba::Vector<space, data_type>>>
setupBlockSystem(const Teuchos::RCP<Epetra_CrsMatrix>& originalMatrix,
                 const Teuchos::RCP<Epetra_FEVector>& originalRhsVector,
                 const Teuchos::RCP<Epetra_FEVector>& originalLhsVector,
                 const boba::Array<std::set<id_type>, rows>& rowPartition,
                 const boba::Array<std::set<id_type>, cols>& colPartition) {
  boba::BlockOperator<boba::Matrix<space, data_type>> blockMatrix(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      blockMatrix({i, j}) = extractBobaSubMatrix<space, double, size_t, id_type>(originalMatrix, rowPartition[i], colPartition[j]);
    }
  }

  boba::BlockVector<boba::Vector<space, data_type>> blockRhsVector(rows);
  for (size_t i = 0; i < rows; ++i) {
    blockRhsVector(i) = extractBobaSubVector<space, double, size_t, id_type>(originalRhsVector, rowPartition[i]);
  }

  boba::BlockVector<boba::Vector<space, data_type>> blockLhsVector(cols);
  for (size_t j = 0; j < cols; ++j) {
    blockLhsVector(j) = extractBobaSubVector<space, double, size_t, id_type>(originalLhsVector, colPartition[j]);
  }

  return {blockMatrix, blockRhsVector, blockLhsVector};
}

template <boba::execution_space space, typename data_type, typename index_type, typename id_type, size_t rows, size_t cols>
std::tuple<boba::BlockOperator<boba::Matrix<space, data_type>>,
           boba::BlockVector<boba::Vector<space, data_type>>>
setupBlockSystem(const Teuchos::RCP<Epetra_CrsMatrix>& originalMatrix,
                 const Teuchos::RCP<Epetra_FEVector>& originalRhsVector,
                 const boba::Array<std::set<id_type>, rows>& rowPartition,
                 const boba::Array<std::set<id_type>, cols>& colPartition) {
  boba::BlockOperator<boba::Matrix<space, data_type>> blockMatrix(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      blockMatrix({i, j}) = extractBobaSubMatrix<space, double, size_t, id_type>(originalMatrix, rowPartition[i], colPartition[j]);
    }
  }

  boba::BlockVector<boba::Vector<space, data_type>> blockRhsVector(rows);
  for (size_t i = 0; i < rows; ++i) {
    blockRhsVector(i) = extractBobaSubVector<space, double, size_t, id_type>(originalRhsVector, rowPartition[i]);
  }

  return {blockMatrix, blockRhsVector};
}

template <boba::execution_space space, typename data_type, typename index_type, typename id_type, size_t rows, size_t cols>
boba::BlockOperator<boba::Matrix<space, data_type>>
setupBlockSystem(const Teuchos::RCP<Epetra_CrsMatrix>& originalMatrix,
                 const boba::Array<std::set<id_type>, rows>& rowPartition,
                 const boba::Array<std::set<id_type>, cols>& colPartition) {
  boba::BlockOperator<boba::Matrix<space, data_type>> blockMatrix(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      blockMatrix({i, j}) = extractBobaSubMatrix<space, double, size_t, id_type>(originalMatrix, rowPartition[i], colPartition[j]);
    }
  }

  return blockMatrix;
}

#endif
