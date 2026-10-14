#pragma once

#include <Tpetra_Vector.hpp>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace Camellia
{
  template<class RowMatrixType>
  typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType
  tensorTpetraRowMatrixDInvAGershgorinLambdaMax(
      const Teuchos::RCP<const RowMatrixType> &rowMatrix,
      const std::string &label = "tensorTpetraRowMatrixDInvAGershgorinLambdaMax",
      const bool throwOnZeroDiagonal = true,
      const typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType unavailableValue =
          typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType(-1.0))
  {
    using Scalar = typename RowMatrixType::scalar_type;
    using LocalOrdinal = typename RowMatrixType::local_ordinal_type;
    using GlobalOrdinal = typename RowMatrixType::global_ordinal_type;
    using Node = typename RowMatrixType::node_type;
    using Magnitude = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
    using Vector = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    if (rowMatrix.is_null())
      throw std::invalid_argument(label + " requires a non-null RowMatrix");

    Vector diagonal(rowMatrix->getRowMap());
    rowMatrix->getLocalDiagCopy(diagonal);
    auto diagonalHost = diagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
    typename RowMatrixType::local_inds_host_view_type rowIndices;
    typename RowMatrixType::values_host_view_type rowValues;

    Magnitude localLambdaMax = throwOnZeroDiagonal ? Magnitude(0.0) : unavailableValue;
    const size_t localRows = rowMatrix->getLocalNumRows();
    for (size_t row=0; row<localRows; row++)
    {
      const Scalar diagonalValue = diagonalHost(row,0);
      const Magnitude absDiagonal = Teuchos::ScalarTraits<Scalar>::magnitude(diagonalValue);
      if (absDiagonal <= std::numeric_limits<Magnitude>::epsilon())
      {
        if (throwOnZeroDiagonal)
          throw std::runtime_error(label + " encountered a zero diagonal entry");
        continue;
      }

      rowMatrix->getLocalRowView(static_cast<LocalOrdinal>(row), rowIndices, rowValues);
      Magnitude rowBound = Magnitude(0.0);
      for (size_t entry=0; entry<rowValues.extent(0); entry++)
        rowBound += Teuchos::ScalarTraits<Scalar>::magnitude(rowValues(entry) / diagonalValue);
      localLambdaMax = std::max(localLambdaMax, rowBound);
    }

    Magnitude lambdaMax = localLambdaMax;
    Teuchos::reduceAll<int, Magnitude>(*rowMatrix->getComm(), Teuchos::REDUCE_MAX,
                                       localLambdaMax, Teuchos::outArg(lambdaMax));
    return lambdaMax;
  }
}
