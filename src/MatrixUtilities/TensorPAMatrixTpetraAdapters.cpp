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

#include <sstream>
#include <stdexcept>

using Scalar = double;
using TpetraMap = Tpetra::Map<>;
using TpetraCrsMatrix = Tpetra::CrsMatrix<Scalar, typename TpetraMap::local_ordinal_type, typename TpetraMap::global_ordinal_type, typename TpetraMap::node_type>;

namespace Camellia
{
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
}
