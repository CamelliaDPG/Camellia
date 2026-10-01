//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TensorPAMatrixTpetraAdapters.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/30/26.
//
//

#include "TensorPAMatrixTpetraAdapters.hpp"

#include "Intrepid2_PAMatrix.hpp"

#include <sstream>
#include <stdexcept>

using Scalar = double;
using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
using TpetraMap = Tpetra::Map<>;
using TpetraCrsMatrix = Tpetra::CrsMatrix<Scalar, typename TpetraMap::local_ordinal_type, typename TpetraMap::global_ordinal_type, typename TpetraMap::node_type>;
using TpetraOperator = Tpetra::Operator<Scalar>;

namespace Camellia
{
  class LocalGMGSmootherOnlyJacobiOperator :
      public Tpetra::Operator<double>
  {
  public:
    using Scalar = double;
    using TpetraMap = Tpetra::Map<>;
    using TpetraMultiVector = Tpetra::MultiVector<Scalar>;
    using TpetraOperator = Tpetra::Operator<Scalar>;

    LocalGMGSmootherOnlyJacobiOperator(
        const Teuchos::RCP<const TpetraOperator> &A,
        const Teuchos::RCP<const TpetraOperator> &jacobi,
        const int smootherApplications)
      :
      _A(A),
      _jacobi(jacobi),
      _smootherApplications(smootherApplications)
    {
      if (_A.is_null())
        throw std::invalid_argument("LocalGMGSmootherOnlyJacobiOperator requires non-null A");
      if (_jacobi.is_null())
        throw std::invalid_argument("LocalGMGSmootherOnlyJacobiOperator requires non-null Jacobi operator");
      if (_smootherApplications < 1)
        throw std::invalid_argument("LocalGMGSmootherOnlyJacobiOperator requires smootherApplications >= 1");
    }

    Teuchos::RCP<const TpetraMap> getDomainMap() const override
    {
      return _A->getDomainMap();
    }

    Teuchos::RCP<const TpetraMap> getRangeMap() const override
    {
      return _A->getRangeMap();
    }

    void apply(const TpetraMultiVector &X,
               TpetraMultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      if (mode != Teuchos::NO_TRANS)
        throw std::invalid_argument("LocalGMGSmootherOnlyJacobiOperator::apply supports NO_TRANS only");

      if (X.getNumVectors() != Y.getNumVectors())
        throw std::invalid_argument("LocalGMGSmootherOnlyJacobiOperator::apply requires X and Y to have the same number of vectors");

      const auto domainMap = _A->getDomainMap();
      const auto rangeMap  = _A->getRangeMap();

      TpetraMultiVector result(domainMap, X.getNumVectors());
      TpetraMultiVector residual(rangeMap, X.getNumVectors());
      TpetraMultiVector correction(domainMap, X.getNumVectors());
      TpetraMultiVector Aresult(rangeMap, X.getNumVectors());

      // result = 0, residual = X.
      result.putScalar(Scalar(0.0));
      residual.update(Scalar(1.0), X, Scalar(0.0));

      // Pre-smoothing phase:
      //
      //   correction = J residual
      //   result += correction
      //   residual = X - A result
      //
      // Here X is the right-preconditioner input, i.e. the residual-like vector
      // to which the smoother-only preconditioner is being applied.
      for (int i=0; i<_smootherApplications; i++)
      {
        _jacobi->apply(residual, correction);
        result.update(Scalar(1.0), correction, Scalar(1.0));

        _A->apply(result, Aresult);
        residual.update(Scalar(1.0), X, Scalar(-1.0), Aresult, Scalar(0.0));
      }

      // Post-smoothing phase for GMGOperator::SMOOTHER_ONLY-like behavior.
      // For smootherApplications=1 this gives:
      //
      //   result = J X + J (X - A J X)
      //
      // i.e. two Jacobi/Richardson correction stages.
      for (int i=0; i<_smootherApplications; i++)
      {
        _jacobi->apply(residual, correction);
        result.update(Scalar(1.0), correction, Scalar(1.0));

        if (i + 1 < _smootherApplications)
        {
          _A->apply(result, Aresult);
          residual.update(Scalar(1.0), X, Scalar(-1.0), Aresult, Scalar(0.0));
        }
      }

      Y.update(alpha, result, beta);
    }

  private:
    Teuchos::RCP<const TpetraOperator> _A;
    Teuchos::RCP<const TpetraOperator> _jacobi;
    int _smootherApplications;
  };

  Teuchos::RCP<TpetraCrsMatrix> TensorPAMatrixTpetraAdapters::buildLocalCrsMatrixFromDense(
      const Teuchos::RCP<const TpetraMap> &map,
      const std::vector<Scalar> &localMatrices,
      const int cells,
      const int rowsPerCell)
  {
    using local_ordinal_type = typename TpetraMap::local_ordinal_type;
    using global_ordinal_type = typename TpetraMap::global_ordinal_type;

    if (map.is_null())
      throw std::invalid_argument("buildLocalCrsMatrixFromDense requires a non-null map");
    if (cells <= 0)
      throw std::invalid_argument("buildLocalCrsMatrixFromDense requires cells > 0");
    if (rowsPerCell <= 0)
      throw std::invalid_argument("buildLocalCrsMatrixFromDense requires rowsPerCell > 0");

    const size_t expectedLocalLength =
        static_cast<size_t>(cells) * static_cast<size_t>(rowsPerCell);
    const size_t mapLocalLength = map->getLocalNumElements();

    if (mapLocalLength != expectedLocalLength)
    {
      std::ostringstream message;
      message << "buildLocalCrsMatrixFromDense map/local matrix size mismatch: "
              << "map local length=" << mapLocalLength
              << ", cells=" << cells
              << ", rowsPerCell=" << rowsPerCell
              << ", expected local length=" << expectedLocalLength;
      throw std::invalid_argument(message.str());
    }

    const size_t expectedDenseEntries =
        static_cast<size_t>(cells) *
        static_cast<size_t>(rowsPerCell) *
        static_cast<size_t>(rowsPerCell);

    if (localMatrices.size() != expectedDenseEntries)
    {
      std::ostringstream message;
      message << "buildLocalCrsMatrixFromDense localMatrices size mismatch: "
              << "localMatrices.size()=" << localMatrices.size()
              << ", cells=" << cells
              << ", rowsPerCell=" << rowsPerCell
              << ", expected entries=" << expectedDenseEntries;
      throw std::invalid_argument(message.str());
    }

    auto crsMatrix = Teuchos::rcp(new TpetraCrsMatrix(
        map,
        static_cast<size_t>(rowsPerCell)));

    Teuchos::Array<global_ordinal_type> columns(
        static_cast<Teuchos::Array<global_ordinal_type>::size_type>(rowsPerCell));
    Teuchos::Array<Scalar> values(
        static_cast<Teuchos::Array<Scalar>::size_type>(rowsPerCell));

    const global_ordinal_type invalidGlobalOrdinal =
        Teuchos::OrdinalTraits<global_ordinal_type>::invalid();

    for (int c=0; c<cells; c++)
    {
      for (int j=0; j<rowsPerCell; j++)
      {
        const size_t localColumn =
            static_cast<size_t>(c) * static_cast<size_t>(rowsPerCell) +
            static_cast<size_t>(j);

        const auto localColumnOrdinal =
            static_cast<local_ordinal_type>(localColumn);

        const global_ordinal_type globalColumn =
            map->getGlobalElement(localColumnOrdinal);

        if (globalColumn == invalidGlobalOrdinal)
        {
          std::ostringstream message;
          message << "buildLocalCrsMatrixFromDense found invalid global column ordinal "
                  << "for local column " << localColumn;
          throw std::runtime_error(message.str());
        }

        columns[static_cast<Teuchos::Array<global_ordinal_type>::size_type>(j)] =
            globalColumn;
      }

      for (int i=0; i<rowsPerCell; i++)
      {
        const size_t localRow =
            static_cast<size_t>(c) * static_cast<size_t>(rowsPerCell) +
            static_cast<size_t>(i);

        const auto localRowOrdinal =
            static_cast<local_ordinal_type>(localRow);

        const global_ordinal_type globalRow =
            map->getGlobalElement(localRowOrdinal);

        if (globalRow == invalidGlobalOrdinal)
        {
          std::ostringstream message;
          message << "buildLocalCrsMatrixFromDense found invalid global row ordinal "
                  << "for local row " << localRow;
          throw std::runtime_error(message.str());
        }

        for (int j=0; j<rowsPerCell; j++)
        {
          const size_t denseIndex =
              (static_cast<size_t>(c) * static_cast<size_t>(rowsPerCell) +
               static_cast<size_t>(i)) *
                  static_cast<size_t>(rowsPerCell) +
              static_cast<size_t>(j);

          values[static_cast<Teuchos::Array<Scalar>::size_type>(j)] =
              localMatrices[denseIndex];
        }

        crsMatrix->insertGlobalValues(globalRow, columns(), values());
      }
    }

    crsMatrix->fillComplete(map, map);
    return crsMatrix;
  }

  Teuchos::RCP<TpetraOperator> TensorPAMatrixTpetraAdapters::localGMGSmootherOnlyJacobiOperator(const Teuchos::RCP<const TpetraOperator> &A, const Teuchos::RCP<const TpetraOperator> &jacobi, const int smootherApplications)
  {
    return Teuchos::rcp(new LocalGMGSmootherOnlyJacobiOperator(A,jacobi,smootherApplications));
  }
}
