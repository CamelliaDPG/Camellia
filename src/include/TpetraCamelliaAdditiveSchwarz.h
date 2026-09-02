// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef CAMELLIA_TPETRA_CAMELLIA_ADDITIVE_SCHWARZ_H
#define CAMELLIA_TPETRA_CAMELLIA_ADDITIVE_SCHWARZ_H

#include "DofInterpreter.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "TypeDefs.h"

#include <Ifpack2_BlockRelaxation.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_OverlappingRowMatrix.hpp>
#include <Ifpack2_Preconditioner.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrix.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace Camellia
{

template<class Scalar=double,
         class LocalOrdinal=typename Tpetra::Map<>::local_ordinal_type,
         class GlobalOrdinal=typename Tpetra::Map<>::global_ordinal_type,
         class Node=typename Tpetra::Map<>::node_type>
class TpetraCamelliaAdditiveSchwarz : public Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>
{
public:
  using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using map_ptr = Teuchos::RCP<const map_type>;
  using multivector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using row_matrix_type = Tpetra::RowMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using crs_matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using crs_matrix_ptr = Teuchos::RCP<const crs_matrix_type>;
  using import_type = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
  using export_type = Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>;
  using preconditioner_type = Ifpack2::Preconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

private:
  struct Subdomain
  {
    GlobalIndexType ownedCellID;
    std::vector<GlobalOrdinal> gids;
    std::vector<LocalOrdinal> overlapLIDs;
    map_ptr localMap;
    Teuchos::RCP<crs_matrix_type> localMatrix;
    Teuchos::RCP<preconditioner_type> inverse;
  };

  crs_matrix_ptr A_;
  int overlapLevel_;
  MeshPtr mesh_;
  Teuchos::RCP<DofInterpreter> dofInterpreter_;
  bool hierarchical_;
  int dimensionForNeighborRelationship_;
  map_ptr ownedMap_;
  map_ptr overlapMap_;
  Teuchos::RCP<const import_type> ownedToOverlapImporter_;
  Teuchos::Array<GlobalOrdinal> externalGIDsThisRank_;
  std::vector<Subdomain> subdomains_;
  std::vector<int> localIncidenceCounts_;
  bool initialized_ = false;
  bool computed_ = false;

public:
  static std::set<GlobalIndexType> overlappingCells(GlobalIndexType cellID, MeshPtr mesh, int overlapLevel, bool hierarchical, int dimensionForNeighborRelation)
  {
    std::set<GlobalIndexType> cells = {cellID};
    std::set<GlobalIndexType> lastNeighbors = cells;
    for (int overlap = 0; overlap < overlapLevel; ++overlap)
    {
      std::set<GlobalIndexType> cellNeighbors;
      for (GlobalIndexType neighborCellID : lastNeighbors)
      {
        CellPtr cell = mesh->getTopology()->getCell(neighborCellID);
        int sideDim = cell->topology()->getDimension() - 1;
        if (dimensionForNeighborRelation < 0) dimensionForNeighborRelation = sideDim;
        std::set<GlobalIndexType> thisCellNeighbors = cell->getActiveNeighborIndices(dimensionForNeighborRelation, mesh->getTopology());
        cellNeighbors.insert(thisCellNeighbors.begin(), thisCellNeighbors.end());
      }
      cells.insert(cellNeighbors.begin(), cellNeighbors.end());
      lastNeighbors = cellNeighbors;
    }
    if (hierarchical)
    {
      std::set<GlobalIndexType> ancestors = {cellID};
      for (int overlap = 0; overlap < overlapLevel; ++overlap)
      {
        std::set<GlobalIndexType> previousAncestors = ancestors;
        ancestors.clear();
        for (GlobalIndexType ancestorCellID : previousAncestors)
        {
          CellPtr cell = mesh->getTopology()->getCell(ancestorCellID);
          if (cell->getParent() == Teuchos::null) ancestors.insert(ancestorCellID);
          else ancestors.insert(cell->getParent()->cellIndex());
        }
      }
      std::set<GlobalIndexType> descendantCells;
      for (GlobalIndexType ancestorCellID : ancestors)
      {
        CellPtr cell = mesh->getTopology()->getCell(ancestorCellID);
        std::set<GlobalIndexType> descendants = cell->getDescendants(mesh->getTopology());
        descendantCells.insert(descendants.begin(), descendants.end());
      }
      std::set<GlobalIndexType> neighborCells = cells;
      cells.clear();
      for (GlobalIndexType descendantCell : descendantCells)
      {
        if (neighborCells.find(descendantCell) != neighborCells.end()) cells.insert(descendantCell);
      }
    }
    return cells;
  }

private:
  Teuchos::RCP<const row_matrix_type> overlapRowMatrix() const
  {
    Teuchos::RCP<const row_matrix_type> rowMatrix = Teuchos::rcp_implicit_cast<const row_matrix_type>(A_);

    // Ifpack2::OverlappingRowMatrix construction performs collective Tpetra
    // communication over A_'s communicator.  In parallel, every rank must
    // participate even if this rank has no external rows to import; otherwise
    // ranks that do need overlap enter collectives while zero-overlap ranks move
    // on to later collectives, which can surface as MPI_ERR_TRUNCATE in
    // MPI_Allreduce on larger rank counts.  Only the true serial case can safely
    // bypass the wrapper.
    if (A_->getComm()->getSize() == 1) return rowMatrix;

    return Teuchos::rcp(new Ifpack2::OverlappingRowMatrix<row_matrix_type>(rowMatrix, externalGIDsThisRank_));
  }

  void buildLocalMatrixAndSolver(Subdomain &subdomain, const Teuchos::RCP<const row_matrix_type> &overlapMatrix) const
  {
    const size_t numRows = subdomain.gids.size();
    auto localComm = Teuchos::rcp(new Teuchos::SerialComm<int>());
    subdomain.localMap = Teuchos::rcp(new map_type(numRows, Teuchos::OrdinalTraits<GlobalOrdinal>::zero(), localComm, Tpetra::GloballyDistributed));
    subdomain.localMatrix = Teuchos::rcp(new crs_matrix_type(subdomain.localMap, numRows));

    std::map<GlobalOrdinal, GlobalOrdinal> localGIDForOriginalGID;
    for (size_t i = 0; i < numRows; i++) localGIDForOriginalGID[subdomain.gids[i]] = static_cast<GlobalOrdinal>(i);

    for (size_t rowOrdinal = 0; rowOrdinal < numRows; rowOrdinal++)
    {
      const LocalOrdinal overlapRowLID = subdomain.overlapLIDs[rowOrdinal];
      typename row_matrix_type::local_inds_host_view_type localCols;
      typename row_matrix_type::values_host_view_type rowVals;
      overlapMatrix->getLocalRowView(overlapRowLID, localCols, rowVals);

      // Ifpack2::OverlappingRowMatrix delegates base rows directly to A_, so
      // their local column ordinals are in A_->getColMap().  Imported extension
      // rows use the overlap matrix column map.  For the no-external-GID case,
      // overlapMatrix is just A_.
      map_ptr rowColMap = A_->getColMap();
      if ((externalGIDsThisRank_.size() != 0) &&
          (static_cast<size_t>(overlapRowLID) >= A_->getLocalNumRows()))
      {
        rowColMap = overlapMatrix->getColMap();
      }

      Teuchos::Array<GlobalOrdinal> compactCols;
      Teuchos::Array<Scalar> compactVals;
      compactCols.reserve(localCols.size());
      compactVals.reserve(rowVals.size());
      for (size_t entryOrdinal = 0; entryOrdinal < localCols.size(); entryOrdinal++)
      {
        const GlobalOrdinal originalColGID = rowColMap->getGlobalElement(localCols(entryOrdinal));
        auto localCol = localGIDForOriginalGID.find(originalColGID);
        if (localCol != localGIDForOriginalGID.end())
        {
          compactCols.push_back(localCol->second);
          compactVals.push_back(rowVals(entryOrdinal));
        }
      }
      subdomain.localMatrix->insertGlobalValues(static_cast<GlobalOrdinal>(rowOrdinal), compactCols(), compactVals());
    }
    subdomain.localMatrix->fillComplete(subdomain.localMap, subdomain.localMap);

    Teuchos::RCP<const row_matrix_type> localRowMatrix = Teuchos::rcp_implicit_cast<const row_matrix_type>(subdomain.localMatrix);
    auto inverse = Ifpack2::Factory::create<row_matrix_type>("AMESOS2", localRowMatrix);
    Teuchos::ParameterList List("Amesos2");
    inverse->setParameters(List);
    inverse->initialize();
    inverse->compute();
    subdomain.inverse = inverse;
  }

public:
  TpetraCamelliaAdditiveSchwarz(const Teuchos::RCP<const crs_matrix_type> &A, int overlapLevel, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter, bool hierarchical, int dimensionForNeighborRelationship)
  : A_(A), overlapLevel_(overlapLevel), mesh_(mesh), dofInterpreter_(dofInterpreter), hierarchical_(hierarchical), dimensionForNeighborRelationship_(dimensionForNeighborRelationship)
  {
    initialize();
    compute();
  }

  static int MaxGlobalOverlapSideNeighborCount(MeshPtr mesh, int overlap, bool hierarchical, int dimForNeighborRelationship)
  {
    const std::set<GlobalIndexType> &myCellIndices = mesh->cellIDsInPartition();
    Teuchos_CommPtr comm = mesh->Comm();

    int localMaxNeighbors = 0;
    MeshTopologyViewPtr meshTopo = mesh->getTopology();

    const int overlapNeighborRelationshipDimension = meshTopo->getDimension() - 1;
    std::set<GlobalIndexType> remoteCells;

    for (GlobalIndexType cellIndex : myCellIndices)
    {
      std::set<GlobalIndexType> subdomainCellsAndNeighbors = overlappingCells(cellIndex, mesh, overlap, hierarchical,
                                                                               dimForNeighborRelationship);
      for (GlobalIndexType subdomainCellIndex : subdomainCellsAndNeighbors)
      {
        if (!mesh->myCellsInclude(subdomainCellIndex)) remoteCells.insert(subdomainCellIndex);
      }
    }

    std::map<GlobalIndexType, std::vector<GlobalIndexType>> remoteCellNeighbors;

    int myRank = comm->getRank();
    std::map<int, std::vector<std::pair<int, GlobalIndexType>>> requests;
    for (GlobalIndexType remoteCellID : remoteCells)
    {
      int rank = mesh->partitionForCellID(remoteCellID);
      requests[rank].push_back({myRank, remoteCellID});
    }

    std::vector<std::pair<int, GlobalIndexType>> requestsReceived;
    MPIWrapper::sendDataVectors(comm, requests, requestsReceived);

    std::map<int, std::vector<std::pair<GlobalIndexType, GlobalIndexType>>> responsesToSend;
    std::vector<std::pair<GlobalIndexType, GlobalIndexType>> responsesReceived;
    for (auto request : requestsReceived)
    {
      int remotePID = request.first;
      GlobalIndexType myCellID = request.second;
      TEUCHOS_TEST_FOR_EXCEPTION(!mesh->myCellsInclude(myCellID), std::invalid_argument, "request received for non-owned cellID");
      CellPtr myCell = meshTopo->getCell(myCellID);
      std::set<GlobalIndexType> neighborIDs = myCell->getActiveNeighborIndices(meshTopo);
      for (GlobalIndexType neighborID : neighborIDs)
      {
        responsesToSend[remotePID].push_back({myCellID, neighborID});
      }
    }
    MPIWrapper::sendDataVectors(comm, responsesToSend, responsesReceived);
    for (auto response : responsesReceived)
    {
      GlobalIndexType remoteCellID = response.first;
      GlobalIndexType neighborID = response.second;
      remoteCellNeighbors[remoteCellID].push_back(neighborID);
    }

    for (GlobalIndexType cellIndex : myCellIndices)
    {
      std::set<GlobalIndexType> subdomainCellsAndNeighbors = overlappingCells(cellIndex, mesh, overlap, hierarchical,
                                                                               dimForNeighborRelationship);
      std::set<GlobalIndexType> subdomainNeighbors;
      for (GlobalIndexType subdomainCellIndex : subdomainCellsAndNeighbors)
      {
        if (mesh->myCellsInclude(subdomainCellIndex))
        {
          CellPtr subdomainCell = meshTopo->getCell(subdomainCellIndex);
          std::set<GlobalIndexType> cellNeighbors = subdomainCell->getActiveNeighborIndices(overlapNeighborRelationshipDimension, meshTopo);
          subdomainNeighbors.insert(cellNeighbors.begin(), cellNeighbors.end());
        }
        else
        {
          auto remoteEntry = remoteCellNeighbors.find(subdomainCellIndex);
          TEUCHOS_TEST_FOR_EXCEPTION(remoteEntry == remoteCellNeighbors.end(), std::invalid_argument, "Remote cell not found during send/receive");
          subdomainNeighbors.insert(remoteEntry->second.begin(), remoteEntry->second.end());
        }
      }

      subdomainCellsAndNeighbors.insert(subdomainNeighbors.begin(), subdomainNeighbors.end());
      localMaxNeighbors = std::max(localMaxNeighbors, static_cast<int>(subdomainCellsAndNeighbors.size()));
    }

    int globalMaxNeighbors = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &localMaxNeighbors, &globalMaxNeighbors);
    return globalMaxNeighbors;
  }

  int getMaxGlobalOverlapSideNeighborCount() const
  {
    return MaxGlobalOverlapSideNeighborCount(mesh_, overlapLevel_, hierarchical_, dimensionForNeighborRelationship_);
  }

  void initialize()
  {
    TEUCHOS_TEST_FOR_EXCEPTION(A_.is_null(), std::invalid_argument, "TpetraCamelliaAdditiveSchwarz requires a non-null matrix");
    TEUCHOS_TEST_FOR_EXCEPTION(mesh_ == Teuchos::null, std::invalid_argument, "TpetraCamelliaAdditiveSchwarz requires a non-null mesh");
    TEUCHOS_TEST_FOR_EXCEPTION(dofInterpreter_ == Teuchos::null, std::invalid_argument, "TpetraCamelliaAdditiveSchwarz requires a non-null dof interpreter");
    ownedMap_ = A_->getRowMap();
    subdomains_.clear();
    externalGIDsThisRank_.clear();

    std::set<GlobalOrdinal> gidsThisRank;
    std::set<GlobalIndexType> myCellIDs = mesh_->cellIDsInPartition();
    std::map<GlobalIndexType, std::set<GlobalIndexType>> overlapCellsForCell;
    std::set<GlobalIndexType> allCellsOfInterestToMe;
    for (GlobalIndexType cellID : myCellIDs)
    {
      std::set<GlobalIndexType> cellsInOverlap = overlappingCells(cellID, mesh_, overlapLevel_, hierarchical_, dimensionForNeighborRelationship_);
      overlapCellsForCell[cellID] = cellsInOverlap;
      allCellsOfInterestToMe.insert(cellsInOverlap.begin(), cellsInOverlap.end());
    }

    std::map<GlobalIndexType, std::set<GlobalIndexType>> globalIndicesForCells = dofInterpreter_->importGlobalIndicesMap(allCellsOfInterestToMe);
    for (GlobalIndexType cellID : myCellIDs)
    {
      std::set<GlobalOrdinal> globalDofIndices;
      for (GlobalIndexType overlapCell : overlapCellsForCell[cellID])
      {
        auto &cellDofs = globalIndicesForCells[overlapCell];
        globalDofIndices.insert(cellDofs.begin(), cellDofs.end());
      }
      Subdomain subdomain;
      subdomain.ownedCellID = cellID;
      subdomain.gids.assign(globalDofIndices.begin(), globalDofIndices.end());
      subdomains_.push_back(subdomain);
      gidsThisRank.insert(globalDofIndices.begin(), globalDofIndices.end());
    }

    for (GlobalOrdinal gid : gidsThisRank)
    {
      if (ownedMap_->getLocalElement(gid) == Teuchos::OrdinalTraits<LocalOrdinal>::invalid())
      {
        externalGIDsThisRank_.push_back(gid);
      }
    }

    Teuchos::RCP<const row_matrix_type> overlapMatrix = overlapRowMatrix();
    overlapMap_ = overlapMatrix->getRowMap();
    ownedToOverlapImporter_ = Teuchos::rcp(new import_type(ownedMap_, overlapMap_));

    localIncidenceCounts_.assign(overlapMap_->getLocalNumElements(), 0);
    for (auto &subdomain : subdomains_)
    {
      const size_t numIDs = subdomain.gids.size();
      subdomain.overlapLIDs.resize(numIDs);
      for (size_t i = 0; i < numIDs; i++)
      {
        const GlobalOrdinal gid = subdomain.gids[i];
        const LocalOrdinal lid = overlapMap_->getLocalElement(gid);
        TEUCHOS_TEST_FOR_EXCEPTION(lid == Teuchos::OrdinalTraits<LocalOrdinal>::invalid(), std::invalid_argument, "TpetraCamelliaAdditiveSchwarz::initialize(): overlap-map LID not found for GID " << gid);
        subdomain.overlapLIDs[i] = lid;
        localIncidenceCounts_[lid]++;
      }
    }
    initialized_ = true;
    computed_ = false;
  }

  void compute()
  {
    if (!initialized_) initialize();
    Teuchos::RCP<const row_matrix_type> overlapMatrix = overlapRowMatrix();
    for (auto &subdomain : subdomains_) buildLocalMatrixAndSolver(subdomain, overlapMatrix);
    computed_ = true;
  }

  map_ptr getDomainMap() const override { return A_->getDomainMap(); }
  map_ptr getRangeMap() const override { return A_->getRangeMap(); }
  bool hasTransposeApply() const override { return false; }

  void apply(const multivector_type &X, multivector_type &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS, Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(), Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::invalid_argument, "TpetraCamelliaAdditiveSchwarz supports NO_TRANS only");
    TEUCHOS_TEST_FOR_EXCEPTION(!computed_, std::logic_error, "TpetraCamelliaAdditiveSchwarz::apply() called before compute()");
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument, "TpetraCamelliaAdditiveSchwarz::apply(): X and Y must have the same number of vectors");

    const size_t numVectors = X.getNumVectors();
    multivector_type overlapX(overlapMap_, numVectors);
    overlapX.putScalar(Scalar(0));
    overlapX.doImport(X, *ownedToOverlapImporter_, Tpetra::INSERT);

    multivector_type overlapContrib(overlapMap_, numVectors);
    overlapContrib.putScalar(Scalar(0));

    {
      auto overlapXView = overlapX.getLocalViewHost(Tpetra::Access::ReadOnly);
      auto overlapContribView = overlapContrib.getLocalViewHost(Tpetra::Access::ReadWrite);

      for (const auto &subdomain : subdomains_)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(subdomain.inverse.is_null(), std::logic_error, "TpetraCamelliaAdditiveSchwarz::apply(): subdomain inverse is null");
        const size_t subdomainSize = subdomain.overlapLIDs.size();
        multivector_type localX(subdomain.localMap, numVectors);
        multivector_type localY(subdomain.localMap, numVectors);
        {
          auto localXView = localX.getLocalViewHost(Tpetra::Access::ReadWrite);
          for (size_t vectorOrdinal = 0; vectorOrdinal < numVectors; vectorOrdinal++)
          {
            for (size_t subdomainLID = 0; subdomainLID < subdomainSize; subdomainLID++)
            {
              localXView(subdomainLID, vectorOrdinal) = overlapXView(subdomain.overlapLIDs[subdomainLID], vectorOrdinal);
            }
          }
        }
        subdomain.inverse->apply(localX, localY);
        {
          auto localYView = localY.getLocalViewHost(Tpetra::Access::ReadOnly);
          for (size_t vectorOrdinal = 0; vectorOrdinal < numVectors; vectorOrdinal++)
          {
            for (size_t subdomainLID = 0; subdomainLID < subdomainSize; subdomainLID++)
            {
              overlapContribView(subdomain.overlapLIDs[subdomainLID], vectorOrdinal) += localYView(subdomainLID, vectorOrdinal);
            }
          }
        }
      }
    }

    multivector_type ownedResult(getRangeMap(), numVectors);
    ownedResult.putScalar(Scalar(0));
    export_type overlapToOwned(overlapMap_, getRangeMap());
    ownedResult.doExport(overlapContrib, overlapToOwned, Tpetra::ADD);
    Y.update(alpha, ownedResult, beta);
  }

  map_ptr getOverlapMap() const { return overlapMap_; }
  const std::vector<int> & getLocalIncidenceCounts() const { return localIncidenceCounts_; }
  int getOverlapLevel() const { return overlapLevel_; }

  int getMaxGlobalIncidenceCount() const
  {
    // localIncidenceCounts_ is indexed by overlapMap_, so in parallel the same
    // GID may have contributions on several ranks.  Match the legacy Epetra
    // implementation by summing those overlap-map contributions onto the owned
    // row map before taking the global maximum.
    Tpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> overlapCounts(overlapMap_, 1);
    overlapCounts.putScalar(0.0);

    auto overlapCountsView = overlapCounts.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (size_t lid = 0; lid < localIncidenceCounts_.size(); lid++)
    {
      overlapCountsView(static_cast<LocalOrdinal>(lid), 0) = static_cast<double>(localIncidenceCounts_[lid]);
    }

    Tpetra::MultiVector<double, LocalOrdinal, GlobalOrdinal, Node> ownedCounts(ownedMap_, 1);
    ownedCounts.putScalar(0.0);
    export_type overlapToOwned(overlapMap_, ownedMap_);
    ownedCounts.doExport(overlapCounts, overlapToOwned, Tpetra::ADD);

    int localMax = 0;
    auto ownedCountsView = ownedCounts.getLocalViewHost(Tpetra::Access::ReadOnly);
    const size_t numOwnedRows = ownedMap_->getLocalNumElements();
    for (size_t lid = 0; lid < numOwnedRows; lid++)
    {
      localMax = std::max(localMax, static_cast<int>(ownedCountsView(static_cast<LocalOrdinal>(lid), 0)));
    }

    int globalMax = 0;
    Teuchos::reduceAll(*A_->getComm(), Teuchos::REDUCE_MAX, 1, &localMax, &globalMax);
    return globalMax;
  }
};

} // namespace Camellia

#endif // CAMELLIA_TPETRA_CAMELLIA_ADDITIVE_SCHWARZ_H
