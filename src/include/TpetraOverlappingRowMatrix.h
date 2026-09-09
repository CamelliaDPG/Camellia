// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef CAMELLIA_TPETRA_OVERLAPPING_ROW_MATRIX_H
#define CAMELLIA_TPETRA_OVERLAPPING_ROW_MATRIX_H

#include <Ifpack2_Details_OverlappingRowGraph.hpp>
#include <Ifpack2_Details_RowMatrix.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_RowMatrix.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace Camellia
{

/// \brief Tpetra RowMatrix wrapper containing owned rows plus caller-selected ghost rows.
///
/// This is Camellia's compatibility adapter for the explicit-external-GID
/// constructor that Camellia previously used from a forked Ifpack2
/// OverlappingRowMatrix.  Base Ifpack2 only supports graph-distance overlap;
/// Camellia's additive Schwarz code already computes the mesh-defined overlap
/// rows explicitly, so this class keeps that behavior in Camellia rather than
/// relying on a non-upstream Ifpack2 constructor.
template<class MatrixType>
class TpetraOverlappingRowMatrix : virtual public Ifpack2::Details::RowMatrix<MatrixType>
{
public:
  using scalar_type = typename MatrixType::scalar_type;
  using local_ordinal_type = typename MatrixType::local_ordinal_type;
  using global_ordinal_type = typename MatrixType::global_ordinal_type;
  using node_type = typename MatrixType::node_type;
  using magnitude_type = typename Teuchos::ScalarTraits<scalar_type>::magnitudeType;
  using mag_type = typename MatrixType::mag_type;

  using global_inds_host_view_type = typename MatrixType::global_inds_host_view_type;
  using local_inds_host_view_type = typename MatrixType::local_inds_host_view_type;
  using values_host_view_type = typename MatrixType::values_host_view_type;
  using nonconst_global_inds_host_view_type = typename MatrixType::nonconst_global_inds_host_view_type;
  using nonconst_local_inds_host_view_type = typename MatrixType::nonconst_local_inds_host_view_type;
  using nonconst_values_host_view_type = typename MatrixType::nonconst_values_host_view_type;

  using row_matrix_type = Tpetra::RowMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using crs_matrix_type = Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using map_type = Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type>;
  using import_type = Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type>;
  using row_graph_type = Tpetra::RowGraph<local_ordinal_type, global_ordinal_type, node_type>;
  using vector_type = Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;

  static_assert(std::is_same<MatrixType, row_matrix_type>::value,
                "Camellia::TpetraOverlappingRowMatrix: MatrixType must be a Tpetra::RowMatrix specialization.");

  TpetraOverlappingRowMatrix(const Teuchos::RCP<const row_matrix_type> &A,
                             const Teuchos::Array<global_ordinal_type> &externalGIDsThisRank)
  : A_(Teuchos::rcp_dynamic_cast<const crs_matrix_type>(A, true))
  {
    using Teuchos::Array;
    using Teuchos::RCP;
    using Teuchos::REDUCE_SUM;
    using Teuchos::rcp;
    using Teuchos::reduceAll;
    using GST = Tpetra::global_size_t;
    using crs_graph_type = Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type>;

    TEUCHOS_TEST_FOR_EXCEPTION(A_.is_null(), std::runtime_error,
                               "Camellia::TpetraOverlappingRowMatrix: input matrix must be a Tpetra::CrsMatrix with matching template parameters.");

    const size_t numMyRowsA = A_->getLocalNumRows();
    const local_ordinal_type localInvalid = Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    const global_ordinal_type globalInvalid = Teuchos::OrdinalTraits<global_ordinal_type>::invalid();

    Array<global_ordinal_type> extElements;
    std::unordered_set<global_ordinal_type> extElementSet;
    extElements.reserve(externalGIDsThisRank.size());
    for (local_ordinal_type i = 0; i < externalGIDsThisRank.size(); i++)
    {
      const global_ordinal_type gid = externalGIDsThisRank[i];
      if ((A_->getRowMap()->getLocalElement(gid) == localInvalid) && extElementSet.insert(gid).second)
      {
        extElements.push_back(gid);
      }
    }

    Array<global_ordinal_type> rowElements(numMyRowsA + extElements.size());
    for (local_ordinal_type i = 0; static_cast<size_t>(i) < numMyRowsA; i++)
    {
      rowElements[i] = A_->getRowMap()->getGlobalElement(i);
    }
    for (local_ordinal_type i = 0; i < extElements.size(); i++)
    {
      rowElements[i + numMyRowsA] = extElements[i];
    }

    RowMap_ = rcp(new map_type(globalInvalid, rowElements(),
                               Teuchos::OrdinalTraits<global_ordinal_type>::zero(),
                               A_->getComm()));
    Importer_ = rcp(new import_type(A_->getRowMap(), RowMap_));
    ColMap_ = RowMap_;

    ExtMap_ = rcp(new map_type(globalInvalid, extElements(),
                               Teuchos::OrdinalTraits<global_ordinal_type>::zero(),
                               A_->getComm()));
    ExtImporter_ = rcp(new import_type(A_->getRowMap(), ExtMap_));

    {
      auto extMatrixDynGraph = rcp(new crs_matrix_type(ExtMap_, ColMap_, 0));
      extMatrixDynGraph->doImport(*A_, *ExtImporter_, Tpetra::INSERT);
      extMatrixDynGraph->fillComplete(A_->getDomainMap(), RowMap_);
      auto extLocalMatrix = extMatrixDynGraph->getLocalMatrixDevice();
      auto extMatrixStaticGraph = rcp(new crs_graph_type(extLocalMatrix.graph,
                                                         ExtMap_,
                                                         ColMap_,
                                                         extMatrixDynGraph->getDomainMap(),
                                                         extMatrixDynGraph->getRangeMap()));
      ExtMatrix_ = rcp(new crs_matrix_type(extMatrixStaticGraph, extLocalMatrix.values));
      ExtMatrix_->fillComplete();
    }

    const size_t numMyRowsB = ExtMatrix_->getLocalNumRows();
    const GST numMyNonzeros = A_->getLocalNumEntries() + ExtMatrix_->getLocalNumEntries();
    const GST numMyRows = numMyRowsA + numMyRowsB;
    {
      GST inArray[2], outArray[2];
      inArray[0] = numMyNonzeros;
      inArray[1] = numMyRows;
      outArray[0] = 0;
      outArray[1] = 0;
      reduceAll<int, GST>(*(A_->getComm()), REDUCE_SUM, 2, inArray, outArray);
      NumGlobalNonzeros_ = outArray[0];
      NumGlobalRows_ = outArray[1];
    }

    MaxNumEntries_ = std::max(A_->getLocalMaxNumRowEntries(), ExtMatrix_->getLocalMaxNumRowEntries());

    using row_graph_impl_type = Ifpack2::Details::OverlappingRowGraph<row_graph_type>;
    Teuchos::RCP<row_graph_impl_type> graph = rcp(new row_graph_impl_type(A_->getGraph(),
                                                                          ExtMatrix_->getGraph(),
                                                                          RowMap_,
                                                                          ColMap_,
                                                                          NumGlobalRows_,
                                                                          NumGlobalRows_,
                                                                          NumGlobalNonzeros_,
                                                                          MaxNumEntries_,
                                                                          Importer_,
                                                                          ExtImporter_));
    graph_ = Teuchos::rcp_const_cast<const row_graph_type>(Teuchos::rcp_implicit_cast<row_graph_type>(graph));

    Kokkos::resize(Indices_, MaxNumEntries_);
    Kokkos::resize(Values_, MaxNumEntries_);
  }

  Teuchos::RCP<const Teuchos::Comm<int>> getComm() const override { return A_->getComm(); }
  Teuchos::RCP<const map_type> getRowMap() const override { return RowMap_; }
  Teuchos::RCP<const map_type> getColMap() const override { return ColMap_; }
  Teuchos::RCP<const map_type> getDomainMap() const override { return ColMap_; }
  Teuchos::RCP<const map_type> getRangeMap() const override { return RowMap_; }
  Teuchos::RCP<const row_graph_type> getGraph() const override { return graph_; }

  Tpetra::global_size_t getGlobalNumRows() const override { return NumGlobalRows_; }
  Tpetra::global_size_t getGlobalNumCols() const override { return NumGlobalRows_; }
  size_t getLocalNumRows() const override { return A_->getLocalNumRows() + ExtMatrix_->getLocalNumRows(); }
  size_t getLocalNumCols() const override { return getLocalNumRows(); }
  global_ordinal_type getIndexBase() const override { return A_->getIndexBase(); }
  Tpetra::global_size_t getGlobalNumEntries() const override { return NumGlobalNonzeros_; }
  size_t getLocalNumEntries() const override { return A_->getLocalNumEntries() + ExtMatrix_->getLocalNumEntries(); }

  size_t getNumEntriesInGlobalRow(global_ordinal_type globalRow) const override
  {
    const local_ordinal_type localRow = RowMap_->getLocalElement(globalRow);
    if (localRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid()) return Teuchos::OrdinalTraits<size_t>::invalid();
    return getNumEntriesInLocalRow(localRow);
  }

  size_t getNumEntriesInLocalRow(local_ordinal_type localRow) const override
  {
    using Teuchos::as;
    const size_t numMyRowsA = A_->getLocalNumRows();
    if (as<size_t>(localRow) < numMyRowsA) return A_->getNumEntriesInLocalRow(localRow);
    return ExtMatrix_->getNumEntriesInLocalRow(as<local_ordinal_type>(localRow - numMyRowsA));
  }

  size_t getGlobalMaxNumRowEntries() const override
  {
    return std::max<size_t>(A_->getGlobalMaxNumRowEntries(), ExtMatrix_->getGlobalMaxNumRowEntries());
  }

  size_t getLocalMaxNumRowEntries() const override { return MaxNumEntries_; }
  local_ordinal_type getBlockSize() const override { return A_->getBlockSize(); }
  bool hasColMap() const override { return true; }
  bool isLocallyIndexed() const override { return true; }
  bool isGloballyIndexed() const override { return false; }
  bool isFillComplete() const override { return true; }
  bool supportsRowViews() const override { return false; }

  void getGlobalRowCopy(global_ordinal_type,
                        nonconst_global_inds_host_view_type &,
                        nonconst_values_host_view_type &,
                        size_t &) const override
  {
    throw std::runtime_error("Camellia::TpetraOverlappingRowMatrix::getGlobalRowCopy() not supported.");
  }

  void getLocalRowCopy(local_ordinal_type localRow,
                       nonconst_local_inds_host_view_type &indices,
                       nonconst_values_host_view_type &values,
                       size_t &numEntries) const override
  {
    using Teuchos::as;
    const size_t numMyRowsA = A_->getLocalNumRows();
    if (as<size_t>(localRow) < numMyRowsA)
    {
      A_->getLocalRowCopy(localRow, indices, values, numEntries);
    }
    else
    {
      ExtMatrix_->getLocalRowCopy(localRow - as<local_ordinal_type>(numMyRowsA), indices, values, numEntries);
    }
  }

  void getGlobalRowView(global_ordinal_type globalRow,
                        global_inds_host_view_type &indices,
                        values_host_view_type &values) const override
  {
    const local_ordinal_type localRow = RowMap_->getLocalElement(globalRow);
    if (localRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    {
      indices = global_inds_host_view_type();
      values = values_host_view_type();
    }
    else if (Teuchos::as<size_t>(localRow) < A_->getLocalNumRows())
    {
      A_->getGlobalRowView(globalRow, indices, values);
    }
    else
    {
      ExtMatrix_->getGlobalRowView(globalRow, indices, values);
    }
  }

  void getLocalRowView(local_ordinal_type localRow,
                       local_inds_host_view_type &indices,
                       values_host_view_type &values) const override
  {
    using Teuchos::as;
    const size_t numMyRowsA = A_->getLocalNumRows();
    if (as<size_t>(localRow) < numMyRowsA)
    {
      A_->getLocalRowView(localRow, indices, values);
    }
    else
    {
      ExtMatrix_->getLocalRowView(localRow - as<local_ordinal_type>(numMyRowsA), indices, values);
    }
  }

  void getLocalDiagCopy(vector_type &diag) const override
  {
    Teuchos::Array<scalar_type> baseDiagVals(A_->getLocalNumRows());
    Teuchos::Array<scalar_type> extDiagVals(ExtMatrix_->getLocalNumRows());

    vector_type baseDiag(A_->getRowMap());
    A_->getLocalDiagCopy(baseDiag);
    baseDiag.get1dCopy(baseDiagVals());

    vector_type extDiag(ExtMatrix_->getRowMap());
    ExtMatrix_->getLocalDiagCopy(extDiag);
    extDiag.get1dCopy(extDiagVals());

    Teuchos::ArrayRCP<scalar_type> allDiagVals = diag.getDataNonConst();
    TEUCHOS_TEST_FOR_EXCEPTION(allDiagVals.size() != baseDiagVals.size() + extDiagVals.size(), std::runtime_error,
                               "Camellia::TpetraOverlappingRowMatrix::getLocalDiagCopy(): diagonal length mismatch.");

    for (Teuchos::Ordinal i = 0; i < baseDiagVals.size(); i++) allDiagVals[i] = baseDiagVals[i];
    const Teuchos_Ordinal offset = baseDiagVals.size();
    for (Teuchos::Ordinal i = 0; i < extDiagVals.size(); i++) allDiagVals[i + offset] = extDiagVals[i];
  }

  void leftScale(const vector_type &) override
  {
    throw std::runtime_error("Camellia::TpetraOverlappingRowMatrix does not support leftScale.");
  }

  void rightScale(const vector_type &) override
  {
    throw std::runtime_error("Camellia::TpetraOverlappingRowMatrix does not support rightScale.");
  }

  mag_type getFrobeniusNorm() const override
  {
    throw std::runtime_error("Camellia::TpetraOverlappingRowMatrix does not support getFrobeniusNorm.");
  }

  void apply(const Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> &X,
             Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> &Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
             scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const override
  {
    using multivector_type = Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::runtime_error,
                               "Camellia::TpetraOverlappingRowMatrix::apply(): X and Y vector counts differ.");
    if (X.aliases(Y))
    {
      multivector_type XCopy(X, Teuchos::Copy);
      apply(XCopy, Y, mode, alpha, beta);
      return;
    }

    const auto &rowMap0 = *(A_->getRowMap());
    const auto &colMap0 = *(A_->getColMap());
    multivector_type X0(X, mode == Teuchos::NO_TRANS ? colMap0 : rowMap0, 0);
    multivector_type Y0(Y, mode == Teuchos::NO_TRANS ? rowMap0 : colMap0, 0);
    A_->localApply(X0, Y0, mode, alpha, beta);

    const auto &rowMap1 = *(ExtMatrix_->getRowMap());
    const auto &colMap1 = *(ExtMatrix_->getColMap());
    multivector_type X1(X, mode == Teuchos::NO_TRANS ? colMap1 : rowMap1, 0);
    multivector_type Y1(Y, mode == Teuchos::NO_TRANS ? rowMap1 : colMap1, A_->getLocalNumRows());
    ExtMatrix_->localApply(X1, Y1, mode, alpha, beta);
  }

  bool hasTransposeApply() const override { return true; }

  void importMultiVector(const Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> &X,
                         Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> &overlapX,
                         Tpetra::CombineMode combineMode = Tpetra::INSERT)
  {
    overlapX.doImport(X, *Importer_, combineMode);
  }

  void exportMultiVector(const Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> &overlapX,
                         Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> &X,
                         Tpetra::CombineMode combineMode = Tpetra::ADD)
  {
    X.doExport(overlapX, *Importer_, combineMode);
  }

  std::string description() const override
  {
    std::ostringstream oss;
    oss << "{ Camellia::TpetraOverlappingRowMatrix"
        << ", global rows: " << getGlobalNumRows()
        << ", global columns: " << getGlobalNumCols()
        << ", global entries: " << getGlobalNumEntries()
        << " }";
    return oss.str();
  }

  void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const override
  {
    if (verbLevel != Teuchos::VERB_NONE)
    {
      out << description() << std::endl;
      if (verbLevel >= Teuchos::VERB_MEDIUM)
      {
        out << "Row map:" << std::endl;
        RowMap_->describe(out, verbLevel);
      }
    }
  }

  Teuchos::RCP<const crs_matrix_type> getUnderlyingMatrix() const { return A_; }
  Teuchos::RCP<const crs_matrix_type> getExtMatrix() const { return ExtMatrix_; }

private:
  Teuchos::RCP<const crs_matrix_type> A_;
  Tpetra::global_size_t NumGlobalRows_ = 0;
  Tpetra::global_size_t NumGlobalNonzeros_ = 0;
  size_t MaxNumEntries_ = 0;

  Teuchos::RCP<const map_type> RowMap_;
  Teuchos::RCP<const map_type> ColMap_;
  Teuchos::RCP<const import_type> Importer_;
  Teuchos::RCP<crs_matrix_type> ExtMatrix_;
  Teuchos::RCP<const map_type> ExtMap_;
  Teuchos::RCP<const import_type> ExtImporter_;
  Teuchos::RCP<const row_graph_type> graph_;

  mutable nonconst_local_inds_host_view_type Indices_;
  mutable nonconst_values_host_view_type Values_;
};

template<class MatrixType>
Teuchos::RCP<const MatrixType>
makeOverlappingRowMatrix(const Teuchos::RCP<const MatrixType> &A,
                         const Teuchos::Array<typename MatrixType::global_ordinal_type> &externalGIDsThisRank)
{
  if ((A->getComm()->getSize() == 1) && (externalGIDsThisRank.size() == 0)) return A;
  return Teuchos::rcp_implicit_cast<const MatrixType>(Teuchos::rcp(new TpetraOverlappingRowMatrix<MatrixType>(A, externalGIDsThisRank)));
}

} // namespace Camellia

#endif // CAMELLIA_TPETRA_OVERLAPPING_ROW_MATRIX_H
