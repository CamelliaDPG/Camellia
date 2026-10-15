#pragma once

#include <Tpetra_Vector.hpp>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <algorithm>
#include <cmath>
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

  template<class RowMatrixType>
  typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType
  tensorTpetraRowMatrixDInvADeterministicPowerLambdaMax(
      const Teuchos::RCP<const RowMatrixType> &rowMatrix,
      const int numIters,
      const int eigNormalizationFreq = 1,
      const std::string &label = "tensorTpetraRowMatrixDInvADeterministicPowerLambdaMax",
      const bool throwOnZeroDiagonal = false,
      const typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType unavailableValue =
          typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType(-1.0),
      const typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType tolerance =
          typename Teuchos::ScalarTraits<typename RowMatrixType::scalar_type>::magnitudeType(1.0e-7))
  {
    using Scalar = typename RowMatrixType::scalar_type;
    using LocalOrdinal = typename RowMatrixType::local_ordinal_type;
    using GlobalOrdinal = typename RowMatrixType::global_ordinal_type;
    using Node = typename RowMatrixType::node_type;
    using Magnitude = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
    using Vector = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    if (rowMatrix.is_null())
      throw std::invalid_argument(label + " requires a non-null RowMatrix");
    if (numIters <= 0)
      return unavailableValue;

    Vector diagonal(rowMatrix->getRowMap());
    Vector inverseDiagonal(rowMatrix->getRowMap());
    rowMatrix->getLocalDiagCopy(diagonal);

    int localDiagonalUsable = 1;
    {
      auto diagonalHost = diagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto inverseDiagonalHost = inverseDiagonal.getLocalViewHost(Tpetra::Access::OverwriteAll);
      const size_t localRows = rowMatrix->getLocalNumRows();
      for (size_t row=0; row<localRows; row++)
      {
        const Scalar diagonalValue = diagonalHost(row,0);
        const Magnitude absDiagonal = Teuchos::ScalarTraits<Scalar>::magnitude(diagonalValue);
        if (absDiagonal <= std::numeric_limits<Magnitude>::epsilon())
        {
          if (throwOnZeroDiagonal)
            throw std::runtime_error(label + " encountered a zero diagonal entry");
          inverseDiagonalHost(row,0) = Scalar(0.0);
          localDiagonalUsable = 0;
        }
        else
        {
          inverseDiagonalHost(row,0) = Scalar(1.0) / diagonalValue;
        }
      }
    }

    int globalDiagonalUsable = localDiagonalUsable;
    Teuchos::reduceAll<int, int>(*rowMatrix->getComm(), Teuchos::REDUCE_MIN,
                                 localDiagonalUsable, Teuchos::outArg(globalDiagonalUsable));
    if (!globalDiagonalUsable)
      return unavailableValue;

    Vector x(rowMatrix->getDomainMap());
    Vector y(rowMatrix->getRangeMap());

    {
      auto xHost = x.getLocalViewHost(Tpetra::Access::OverwriteAll);
      auto domainMap = rowMatrix->getDomainMap();
      const size_t localEntries = domainMap->getLocalNumElements();
      const GlobalOrdinal modulus = static_cast<GlobalOrdinal>(97);
      for (size_t localIndex=0; localIndex<localEntries; localIndex++)
      {
        const GlobalOrdinal gid = domainMap->getGlobalElement(static_cast<LocalOrdinal>(localIndex));
        GlobalOrdinal patternOrdinal = gid % modulus;
        if (patternOrdinal < static_cast<GlobalOrdinal>(0)) patternOrdinal = -patternOrdinal;
        const Magnitude pattern = Magnitude(1.0) + static_cast<Magnitude>(patternOrdinal);
        xHost(localIndex,0) = static_cast<Scalar>(pattern);
      }
    }

    const Scalar zero = Scalar(0.0);
    const Scalar one = Scalar(1.0);
    const int normalizationFreq = std::max(1, eigNormalizationFreq);

    Magnitude norm = x.norm2();
    if (norm == Magnitude(0.0))
      return unavailableValue;
    x.scale(one / static_cast<Scalar>(norm));

    Magnitude lambdaMax = Magnitude(0.0);
    Magnitude lambdaMaxOld = Magnitude(1.0);

    // Match Ifpack2::PowerMethod's spectral-radius route deterministically:
    // y=A*x, x=D^{-1}*y, periodically normalize, then return ||D^{-1} A x||_2.
    for (int iter=0; iter<numIters-1; iter++)
    {
      rowMatrix->apply(x, y);
      x.elementWiseMultiply(one, inverseDiagonal, y, zero);

      if ((((iter + 1) % normalizationFreq) == 0) && (iter < numIters - 2))
      {
        norm = x.norm2();
        if (norm == Magnitude(0.0))
          return Magnitude(0.0);
        lambdaMaxOld = lambdaMax;
        lambdaMax = std::pow(norm, Magnitude(1.0) / static_cast<Magnitude>(normalizationFreq));
        if (Teuchos::ScalarTraits<Magnitude>::magnitude(lambdaMax - lambdaMaxOld) <
            tolerance * Teuchos::ScalarTraits<Magnitude>::magnitude(lambdaMax))
          return lambdaMax;
        x.scale(one / static_cast<Scalar>(norm));
      }
    }

    norm = x.norm2();
    if (norm == Magnitude(0.0))
      return Magnitude(0.0);
    x.scale(one / static_cast<Scalar>(norm));
    rowMatrix->apply(x, y);
    y.elementWiseMultiply(one, inverseDiagonal, y, zero);
    lambdaMax = y.norm2();
    return lambdaMax;
  }
}
