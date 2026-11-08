#ifndef Camellia_TensorCoefficientTpetraImport_h
#define Camellia_TensorCoefficientTpetraImport_h

#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <Teuchos_Array.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TestForException.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

#include "TensorCoefficientMap.h"
#include "TensorCoefficientValueEvaluator.h"

namespace Camellia
{
  /**
   * \brief Narrow Tpetra bridge that performs ghost import for tensor coefficient sampling.
   *
   * This helper deliberately does not evaluate basis values.  It verifies that the supplied
   * Tpetra vector uses the same simple rank partition as TensorCoefficientMap::ownedRangeForRank(),
   * imports the nonowned DOFs needed by volume-cell worksets or skeleton entities into overlap
   * vectors, and returns the explicit TensorCoefficientValueEvaluator::OwnedCoefficientValues
   * contract consumed by the host-side coefficient evaluator and Maxwell moment handoff.
   *
   * The volume-workset helper builds one overlap map/import for the union of nonowned DOFs
   * required by all cells in the workset, so production callers can amortize import setup
   * across multiple tensor cells instead of rebuilding a per-cell overlap plan.
   */
  class TensorCoefficientTpetraImport
  {
  public:
    using map_type = Tpetra::Map<>;
    using multivector_type = Tpetra::MultiVector<double>;
    using import_type = Tpetra::Import<typename map_type::local_ordinal_type,
                                       typename map_type::global_ordinal_type,
                                       typename map_type::node_type>;

    class VolumeCellWorksetImportPlan
    {
      std::pair<IndexType,IndexType> _ownedRange;
      std::vector<IndexType> _expectedOwnedIndices;
      std::vector<std::vector<IndexType>> _neededGhostIndicesByCell;
      Teuchos::RCP<const map_type> _ownedMap;
      Teuchos::RCP<const map_type> _overlapMap;
      Teuchos::RCP<import_type> _importer;

      void validateOwnedVectorMap(const multivector_type &ownedCoefficientVector) const
      {
        TEUCHOS_TEST_FOR_EXCEPTION(ownedCoefficientVector.getNumVectors() != 1, std::invalid_argument,
                                   "TensorCoefficientTpetraImport expects a single-column coefficient multivector.");
        auto ownedMap = ownedCoefficientVector.getMap();
        TEUCHOS_TEST_FOR_EXCEPTION(ownedMap == Teuchos::null, std::invalid_argument,
                                   "TensorCoefficientTpetraImport requires a coefficient vector with a non-null map.");
        TEUCHOS_TEST_FOR_EXCEPTION(ownedMap->getLocalNumElements() != _expectedOwnedIndices.size(), std::invalid_argument,
                                   "TensorCoefficientTpetraImport coefficient vector map does not match TensorCoefficientMap rank ownership size.");
        for (size_t localOrdinal=0; localOrdinal<_expectedOwnedIndices.size(); localOrdinal++)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<typename map_type::local_ordinal_type>(localOrdinal))) != _expectedOwnedIndices[localOrdinal],
                                     std::invalid_argument,
                                     "TensorCoefficientTpetraImport coefficient vector map does not match TensorCoefficientMap rank ownership order.");
        }
      }

    public:
      VolumeCellWorksetImportPlan(const TensorCoefficientMap &coefficientMap,
                                  const Teuchos::RCP<const map_type> &ownedMap,
                                  const std::vector<TensorMeshTopology::TensorCellTuple> &cellTuples)
      : _ownedMap(ownedMap)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(_ownedMap == Teuchos::null, std::invalid_argument,
                                   "TensorCoefficientTpetraImport requires a non-null owned map for a reusable import plan.");
        auto comm = _ownedMap->getComm();
        TEUCHOS_TEST_FOR_EXCEPTION(comm == Teuchos::null, std::invalid_argument,
                                   "TensorCoefficientTpetraImport requires an owned map with a non-null communicator.");

        const int rankOrdinal = comm->getRank();
        const int rankCount = comm->getSize();
        _ownedRange = coefficientMap.ownedRangeForRank(rankOrdinal, rankCount);
        _expectedOwnedIndices = coefficientMap.ownedIndicesForRank(rankOrdinal, rankCount);
        validateOwnedVectorMap(multivector_type(_ownedMap, 1));

        std::set<IndexType> neededGhostIndexUnion;
        _neededGhostIndicesByCell.reserve(cellTuples.size());
        for (const auto &cellTuple : cellTuples)
        {
          const auto neededGhostIndices = coefficientMap.neededIndicesForVolumeCell(cellTuple, rankOrdinal, rankCount);
          _neededGhostIndicesByCell.push_back(neededGhostIndices);
          neededGhostIndexUnion.insert(neededGhostIndices.begin(), neededGhostIndices.end());
        }

        Teuchos::Array<typename map_type::global_ordinal_type> overlapGIDs;
        for (IndexType gid : _expectedOwnedIndices)
        {
          overlapGIDs.push_back(static_cast<typename map_type::global_ordinal_type>(gid));
        }
        for (IndexType gid : neededGhostIndexUnion)
        {
          overlapGIDs.push_back(static_cast<typename map_type::global_ordinal_type>(gid));
        }

        _overlapMap = Teuchos::rcp(new map_type(static_cast<Tpetra::global_size_t>(coefficientMap.coefficientCount()),
                                                overlapGIDs(),
                                                static_cast<typename map_type::global_ordinal_type>(0),
                                                comm));
        _importer = Teuchos::rcp(new import_type(_ownedMap, _overlapMap));
      }

      size_t worksetSize() const { return _neededGhostIndicesByCell.size(); }
      Teuchos::RCP<const map_type> overlapMap() const { return _overlapMap; }
      const std::vector<std::vector<IndexType>> &neededGhostIndicesByCell() const { return _neededGhostIndicesByCell; }

      std::vector<TensorCoefficientValueEvaluator::OwnedCoefficientValues> ownedValues(const multivector_type &ownedCoefficientVector) const
      {
        validateOwnedVectorMap(ownedCoefficientVector);

        std::vector<double> ownedCoefficients(_expectedOwnedIndices.size(), 0.0);
        auto ownedHost = ownedCoefficientVector.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t localOrdinal=0; localOrdinal<_expectedOwnedIndices.size(); localOrdinal++)
        {
          ownedCoefficients[localOrdinal] = ownedHost(localOrdinal,0);
        }

        multivector_type overlapVector(_overlapMap, 1);
        overlapVector.doImport(ownedCoefficientVector, *_importer, Tpetra::INSERT);
        auto overlapHost = overlapVector.getLocalViewHost(Tpetra::Access::ReadOnly);

        std::vector<TensorCoefficientValueEvaluator::OwnedCoefficientValues> ownedValuesByCell;
        ownedValuesByCell.reserve(_neededGhostIndicesByCell.size());
        for (const auto &neededGhostIndices : _neededGhostIndicesByCell)
        {
          std::map<IndexType,double> ghostCoefficientsByGlobalIndex;
          for (IndexType gid : neededGhostIndices)
          {
            const auto localOrdinal = _overlapMap->getLocalElement(static_cast<typename map_type::global_ordinal_type>(gid));
            TEUCHOS_TEST_FOR_EXCEPTION(localOrdinal == Teuchos::OrdinalTraits<typename map_type::local_ordinal_type>::invalid(),
                                       std::invalid_argument,
                                       "TensorCoefficientTpetraImport failed to import a required ghost coefficient.");
            ghostCoefficientsByGlobalIndex[gid] = overlapHost(localOrdinal,0);
          }
          ownedValuesByCell.push_back(TensorCoefficientValueEvaluator::OwnedCoefficientValues(_ownedRange,
                                                                                              ownedCoefficients,
                                                                                              ghostCoefficientsByGlobalIndex));
        }
        return ownedValuesByCell;
      }
    };

    static Teuchos::RCP<const map_type> ownedMapForDefaultPartition(const TensorCoefficientMap &coefficientMap,
                                                                    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(comm == Teuchos::null, std::invalid_argument,
                                 "TensorCoefficientTpetraImport requires a non-null communicator.");

      Teuchos::Array<typename map_type::global_ordinal_type> ownedGIDs;
      for (IndexType gid : coefficientMap.ownedIndicesForRank(comm->getRank(), comm->getSize()))
      {
        ownedGIDs.push_back(static_cast<typename map_type::global_ordinal_type>(gid));
      }

      return Teuchos::rcp(new map_type(static_cast<Tpetra::global_size_t>(coefficientMap.coefficientCount()),
                                       ownedGIDs(),
                                       static_cast<typename map_type::global_ordinal_type>(0),
                                       comm));
    }

    static TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValuesForVolumeCell(const TensorCoefficientMap &coefficientMap,
                                                                                           const multivector_type &ownedCoefficientVector,
                                                                                           const TensorMeshTopology::TensorCellTuple &cellTuple)
    {
      auto worksetValues = ownedValuesForVolumeCells(coefficientMap,
                                                     ownedCoefficientVector,
                                                     std::vector<TensorMeshTopology::TensorCellTuple>{cellTuple});
      return worksetValues.front();
    }

    static std::vector<TensorCoefficientValueEvaluator::OwnedCoefficientValues> ownedValuesForVolumeCells(const TensorCoefficientMap &coefficientMap,
                                                                                                          const multivector_type &ownedCoefficientVector,
                                                                                                          const std::vector<TensorMeshTopology::TensorCellTuple> &cellTuples)
    {
      auto ownedMap = ownedCoefficientVector.getMap();
      VolumeCellWorksetImportPlan importPlan(coefficientMap, ownedMap, cellTuples);
      return importPlan.ownedValues(ownedCoefficientVector);
    }

    static TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValuesForSkeletonEntity(const TensorCoefficientMap &coefficientMap,
                                                                                                const multivector_type &ownedCoefficientVector,
                                                                                                const TensorMeshTopology::TensorSkeletonEntity &entity)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(ownedCoefficientVector.getNumVectors() != 1, std::invalid_argument,
                                 "TensorCoefficientTpetraImport expects a single-column coefficient multivector.");

      auto ownedMap = ownedCoefficientVector.getMap();
      TEUCHOS_TEST_FOR_EXCEPTION(ownedMap == Teuchos::null, std::invalid_argument,
                                 "TensorCoefficientTpetraImport requires a coefficient vector with a non-null map.");
      auto comm = ownedMap->getComm();
      TEUCHOS_TEST_FOR_EXCEPTION(comm == Teuchos::null, std::invalid_argument,
                                 "TensorCoefficientTpetraImport requires a coefficient vector with a non-null communicator.");

      const int rankOrdinal = comm->getRank();
      const int rankCount = comm->getSize();
      const auto ownedRange = coefficientMap.ownedRangeForRank(rankOrdinal, rankCount);
      const auto expectedOwnedIndices = coefficientMap.ownedIndicesForRank(rankOrdinal, rankCount);
      TEUCHOS_TEST_FOR_EXCEPTION(ownedMap->getLocalNumElements() != expectedOwnedIndices.size(), std::invalid_argument,
                                 "TensorCoefficientTpetraImport coefficient vector map does not match TensorCoefficientMap rank ownership size.");
      for (size_t localOrdinal=0; localOrdinal<expectedOwnedIndices.size(); localOrdinal++)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<typename map_type::local_ordinal_type>(localOrdinal))) != expectedOwnedIndices[localOrdinal],
                                   std::invalid_argument,
                                   "TensorCoefficientTpetraImport coefficient vector map does not match TensorCoefficientMap rank ownership order.");
      }

      std::vector<double> ownedCoefficients(expectedOwnedIndices.size(), 0.0);
      auto ownedHost = ownedCoefficientVector.getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t localOrdinal=0; localOrdinal<expectedOwnedIndices.size(); localOrdinal++)
      {
        ownedCoefficients[localOrdinal] = ownedHost(localOrdinal,0);
      }

      const auto neededGhostIndices = coefficientMap.neededIndicesForSkeletonEntity(entity, rankOrdinal, rankCount);
      Teuchos::Array<typename map_type::global_ordinal_type> overlapGIDs;
      for (IndexType gid : expectedOwnedIndices)
      {
        overlapGIDs.push_back(static_cast<typename map_type::global_ordinal_type>(gid));
      }
      for (IndexType gid : neededGhostIndices)
      {
        overlapGIDs.push_back(static_cast<typename map_type::global_ordinal_type>(gid));
      }

      auto overlapMap = Teuchos::rcp(new map_type(static_cast<Tpetra::global_size_t>(coefficientMap.coefficientCount()),
                                                  overlapGIDs(),
                                                  static_cast<typename map_type::global_ordinal_type>(0),
                                                  comm));
      import_type importer(ownedMap, overlapMap);
      multivector_type overlapVector(overlapMap, 1);
      overlapVector.doImport(ownedCoefficientVector, importer, Tpetra::INSERT);
      auto overlapHost = overlapVector.getLocalViewHost(Tpetra::Access::ReadOnly);

      std::map<IndexType,double> ghostCoefficientsByGlobalIndex;
      for (IndexType gid : neededGhostIndices)
      {
        const auto localOrdinal = overlapMap->getLocalElement(static_cast<typename map_type::global_ordinal_type>(gid));
        TEUCHOS_TEST_FOR_EXCEPTION(localOrdinal == Teuchos::OrdinalTraits<typename map_type::local_ordinal_type>::invalid(),
                                   std::invalid_argument,
                                   "TensorCoefficientTpetraImport failed to import a required ghost coefficient.");
        ghostCoefficientsByGlobalIndex[gid] = overlapHost(localOrdinal,0);
      }

      return TensorCoefficientValueEvaluator::OwnedCoefficientValues(ownedRange,
                                                                     ownedCoefficients,
                                                                     ghostCoefficientsByGlobalIndex);
    }
  };
}

#endif
