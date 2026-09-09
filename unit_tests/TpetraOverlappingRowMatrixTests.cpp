//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "TpetraOverlappingRowMatrix.h"

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

namespace
{
  using Scalar = double;
  using Map = Tpetra::Map<>;
  using LocalOrdinal = Map::local_ordinal_type;
  using GlobalOrdinal = Map::global_ordinal_type;
  using Node = Map::node_type;
  using RowMatrix = Tpetra::RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using CrsMatrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  Teuchos::RCP<CrsMatrix> diagonalMatrixWithTwoRowsPerRank()
  {
    auto comm = Tpetra::getDefaultComm();
    const Tpetra::global_size_t numGlobalRows = 2 * comm->getSize();
    const size_t numLocalRows = 2;
    auto rowMap = Teuchos::rcp(new Map(numGlobalRows, numLocalRows, 0, comm));
    auto matrix = Teuchos::rcp(new CrsMatrix(rowMap, 1));

    auto myGIDs = rowMap->getMyGlobalIndices();
    for (size_t i = 0; i < myGIDs.size(); i++)
    {
      const GlobalOrdinal gid = myGIDs[i];
      Teuchos::Array<GlobalOrdinal> cols(1, gid);
      Teuchos::Array<Scalar> vals(1, static_cast<Scalar>(gid + 1));
      matrix->insertGlobalValues(gid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
  }
}

TEUCHOS_UNIT_TEST(TpetraOverlappingRowMatrix, FactoryBypassesSerialNoExternalRows)
{
  auto matrix = diagonalMatrixWithTwoRowsPerRank();
  Teuchos::RCP<const RowMatrix> rowMatrix = Teuchos::rcp_implicit_cast<const RowMatrix>(matrix);
  Teuchos::Array<GlobalOrdinal> externalGIDs;

  auto overlapMatrix = Camellia::makeOverlappingRowMatrix(rowMatrix, externalGIDs);
  if (matrix->getComm()->getSize() == 1)
  {
    TEST_EQUALITY(overlapMatrix.getRawPtr(), rowMatrix.getRawPtr());
  }
  else
  {
    TEST_EQUALITY(overlapMatrix->getLocalNumRows(), rowMatrix->getLocalNumRows());
  }
}

TEUCHOS_UNIT_TEST(TpetraOverlappingRowMatrix, ExplicitExternalRowsAreAppended)
{
  auto matrix = diagonalMatrixWithTwoRowsPerRank();
  auto comm = matrix->getComm();
  Teuchos::RCP<const RowMatrix> rowMatrix = Teuchos::rcp_implicit_cast<const RowMatrix>(matrix);

  Teuchos::Array<GlobalOrdinal> externalGIDs;
  if (comm->getSize() > 1)
  {
    const int nextRank = (comm->getRank() + 1) % comm->getSize();
    externalGIDs.push_back(static_cast<GlobalOrdinal>(2 * nextRank));
  }

  auto overlapMatrix = Camellia::makeOverlappingRowMatrix(rowMatrix, externalGIDs);
  auto overlapMap = overlapMatrix->getRowMap();
  auto ownedMap = rowMatrix->getRowMap();
  const size_t ownedRowCount = ownedMap->getLocalNumElements();

  TEST_EQUALITY(overlapMap->getLocalNumElements(), ownedRowCount + externalGIDs.size());
  for (size_t i = 0; i < ownedRowCount; i++)
  {
    TEST_EQUALITY(overlapMap->getGlobalElement(static_cast<LocalOrdinal>(i)), ownedMap->getGlobalElement(static_cast<LocalOrdinal>(i)));
  }
  for (size_t i = 0; i < static_cast<size_t>(externalGIDs.size()); i++)
  {
    TEST_EQUALITY(overlapMap->getGlobalElement(static_cast<LocalOrdinal>(ownedRowCount + i)), externalGIDs[i]);
  }
}

TEUCHOS_UNIT_TEST(TpetraOverlappingRowMatrix, LocalRowViewsUseAppropriateColumnMaps)
{
  auto matrix = diagonalMatrixWithTwoRowsPerRank();
  auto comm = matrix->getComm();
  Teuchos::RCP<const RowMatrix> rowMatrix = Teuchos::rcp_implicit_cast<const RowMatrix>(matrix);

  Teuchos::Array<GlobalOrdinal> externalGIDs;
  if (comm->getSize() > 1)
  {
    const int nextRank = (comm->getRank() + 1) % comm->getSize();
    externalGIDs.push_back(static_cast<GlobalOrdinal>(2 * nextRank));
  }

  auto overlapMatrix = Camellia::makeOverlappingRowMatrix(rowMatrix, externalGIDs);
  const size_t ownedRowCount = rowMatrix->getLocalNumRows();

  RowMatrix::local_inds_host_view_type localCols;
  RowMatrix::values_host_view_type values;
  overlapMatrix->getLocalRowView(0, localCols, values);
  TEST_EQUALITY(localCols.extent(0), 1);
  TEST_EQUALITY(values.extent(0), 1);
  const GlobalOrdinal ownedRowGID = rowMatrix->getRowMap()->getGlobalElement(0);
  TEST_EQUALITY(rowMatrix->getColMap()->getGlobalElement(localCols(0)), ownedRowGID);
  TEST_EQUALITY(values(0), static_cast<Scalar>(ownedRowGID + 1));

  if (externalGIDs.size() > 0)
  {
    overlapMatrix->getLocalRowView(static_cast<LocalOrdinal>(ownedRowCount), localCols, values);
    TEST_EQUALITY(localCols.extent(0), 1);
    TEST_EQUALITY(values.extent(0), 1);
    TEST_EQUALITY(overlapMatrix->getColMap()->getGlobalElement(localCols(0)), externalGIDs[0]);
    TEST_EQUALITY(values(0), static_cast<Scalar>(externalGIDs[0] + 1));
  }
}
