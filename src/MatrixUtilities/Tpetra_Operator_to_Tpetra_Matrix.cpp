//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  Tpetra_Operator_to_Tpetra_Matrix.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/29/25.
//
//

#include "Tpetra_Operator_to_Tpetra_Matrix.h"

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_TestForException.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace Camellia
{
TMatrixPtr<double> Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(const Teuchos::RCP<Tpetra::Operator<double> > op, TMapPtr map)
{
  using Matrix = TMatrix<double>;
  using Vector = TVector<double>;
  using LO     = Matrix::local_ordinal_type;
  using GO     = Matrix::global_ordinal_type;

  TEUCHOS_TEST_FOR_EXCEPTION(op == Teuchos::null, std::invalid_argument,
                             "Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix() requires a non-null operator");
  TEUCHOS_TEST_FOR_EXCEPTION(map == Teuchos::null, std::invalid_argument,
                             "Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix() requires a non-null map");

  auto domainMap = op->getDomainMap();
  auto rangeMap  = op->getRangeMap();

  TEUCHOS_TEST_FOR_EXCEPTION(domainMap == Teuchos::null, std::invalid_argument,
                             "Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(): operator has null domain map");
  TEUCHOS_TEST_FOR_EXCEPTION(rangeMap == Teuchos::null, std::invalid_argument,
                             "Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(): operator has null range map");
  TEUCHOS_TEST_FOR_EXCEPTION(!rangeMap->isSameAs(*map), std::invalid_argument,
                             "Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(): supplied map does not match operator range map");

  const size_t maxEntriesPerRow = static_cast<size_t>(domainMap->getGlobalNumElements());
  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(rangeMap, maxEntriesPerRow));

  if ((domainMap->getGlobalNumElements() == 0) || (rangeMap->getGlobalNumElements() == 0))
  {
    matrix->fillComplete(domainMap, rangeMap);
    return matrix;
  }

  Vector X(domainMap, 1);
  Vector Y(rangeMap, 1);

  const double tol = 1e-15; // values below this will be considered zero
  const LO invalidLID = Teuchos::OrdinalTraits<LO>::invalid();
  const GO invalidGID = Teuchos::OrdinalTraits<GO>::invalid();

  // Tpetra::Operator::apply() is collective.  Therefore every rank must call
  // apply() the same number of times and for the same basis-vector columns.
  // Do not scan minGID..maxGID here: Tpetra maps may be non-contiguous, and
  // refined / p-multigrid Camellia maps often have gaps.  Scanning the whole
  // range can make these matrix-representation tests look like they have hung
  // under MPI.  Instead, gather the actual domain GIDs to every rank and
  // iterate that common list in a deterministic order.
  auto comm = domainMap->getComm();
  const int numRanks = comm->getSize();
  const auto localDomainGIDs = domainMap->getLocalElementList();
  const int localNumDomainGIDs = static_cast<int>(localDomainGIDs.size());

  int maxLocalNumDomainGIDs = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &localNumDomainGIDs, &maxLocalNumDomainGIDs);

  std::vector<GO> paddedLocalDomainGIDs(static_cast<size_t>(maxLocalNumDomainGIDs), invalidGID);
  for (int i = 0; i < localNumDomainGIDs; i++)
  {
    paddedLocalDomainGIDs[static_cast<size_t>(i)] = localDomainGIDs[i];
  }

  std::vector<GO> gatheredDomainGIDs(static_cast<size_t>(maxLocalNumDomainGIDs) * static_cast<size_t>(numRanks), invalidGID);
  if (maxLocalNumDomainGIDs > 0)
  {
    Teuchos::gatherAll(*comm, maxLocalNumDomainGIDs, paddedLocalDomainGIDs.data(),
                       static_cast<int>(gatheredDomainGIDs.size()), gatheredDomainGIDs.data());
  }

  std::vector<GO> domainGIDs;
  domainGIDs.reserve(gatheredDomainGIDs.size());
  for (GO gid : gatheredDomainGIDs)
  {
    if (gid != invalidGID)
    {
      domainGIDs.push_back(gid);
    }
  }
  std::sort(domainGIDs.begin(), domainGIDs.end());
  domainGIDs.erase(std::unique(domainGIDs.begin(), domainGIDs.end()), domainGIDs.end());

  for (GO colGID : domainGIDs)
  {
    X.putScalar(0.0);
    Y.putScalar(0.0);

    const LO colLID = domainMap->getLocalElement(colGID);
    if (colLID != invalidLID)
    {
      auto xView = X.getLocalViewHost(Tpetra::Access::ReadWrite);
      xView(colLID,0) = 1.0;
    }

    op->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);

    auto yView = Y.getLocalViewHost(Tpetra::Access::ReadOnly);
    const size_t numLocalRows = rangeMap->getLocalNumElements();
    for (size_t rowLIDOrdinal = 0; rowLIDOrdinal < numLocalRows; rowLIDOrdinal++)
    {
      const double value = yView(rowLIDOrdinal,0);
      if (std::abs(value) > tol)
      {
        const GO rowGID = rangeMap->getGlobalElement(static_cast<LO>(rowLIDOrdinal));
        Teuchos::Array<GO> cols(1, colGID);
        Teuchos::Array<double> values(1, value);
        matrix->insertGlobalValues(rowGID, cols(), values());
      }
    }
  }

  matrix->fillComplete(domainMap, rangeMap);
  return matrix;
}
}
