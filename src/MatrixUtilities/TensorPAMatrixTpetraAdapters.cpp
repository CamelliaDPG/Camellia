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

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <Tpetra_CrsGraph.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace Camellia
{
  using Adapters = TensorPAMatrixTpetraAdapters;
  using Scalar = Adapters::Scalar;
  using DeviceType = Adapters::DeviceType;
  using OperatorTerm = Adapters::OperatorTerm;
  using WorkspaceView = Adapters::WorkspaceView;
  using TpetraMap = Adapters::TpetraMap;
  using TpetraCrsGraph =
  Tpetra::CrsGraph<typename TpetraMap::local_ordinal_type,
  typename TpetraMap::global_ordinal_type,
  typename TpetraMap::node_type>;
  using TpetraCrsMatrix = Adapters::TpetraCrsMatrix;
  using TpetraMultiVector = Adapters::TpetraMultiVector;
  using TpetraOperator = Adapters::TpetraOperator;
  using TpetraRowMatrix = Adapters::TpetraRowMatrix;
  using TpetraVector = Adapters::TpetraVector;
}

using namespace Camellia;

namespace
{
  void addScaledOutput(Intrepid2::ScalarView<Scalar,DeviceType> &output,
                       const Intrepid2::ScalarView<Scalar,DeviceType> &termOutput,
                       const Scalar weight)
  {
    auto policy = Kokkos::MDRangePolicy<typename DeviceType::execution_space,Kokkos::Rank<2>>(
        {0,0}, {output.extent_int(0), output.extent_int(1)});

    Kokkos::parallel_for("TensorPAMatrixTpetraAdapters add weighted term", policy,
        KOKKOS_LAMBDA(const int c, const int f)
        {
          output(c,f) += weight * termOutput(c,f);
        });
  }

  bool allWeightsAreOne(const std::vector<OperatorTerm> &terms)
  {
    for (const auto &term : terms)
    {
      if (term.weight != Scalar(1.0)) return false;
    }
    return true;
  }

  class LocalPAMatrixTpetraOperator : public TpetraOperator
  {
    Teuchos::RCP<const TpetraMap> map_;
    mutable std::vector<OperatorTerm> *terms_;
    std::vector<WorkspaceView> workspaces_;
    int cells_ = 0;
    int localDofs_ = 0;
    int worksetSize_ = 0;

  public:
    LocalPAMatrixTpetraOperator(const Teuchos::RCP<const TpetraMap> &map, std::vector<OperatorTerm> &terms,
                                const std::vector<WorkspaceView> &workspaces, const int cells,
                                const int localDofs, const int worksetSize)
      : map_(map), terms_(&terms), workspaces_(workspaces), cells_(cells), localDofs_(localDofs), worksetSize_(worksetSize)
    {}

    Teuchos::RCP<const TpetraMap> getDomainMap() const override { return map_; }
    Teuchos::RCP<const TpetraMap> getRangeMap() const override { return map_; }
    bool hasTransposeApply() const override { return false; }

    void apply(const TpetraMultiVector &X, TpetraMultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      if (mode != Teuchos::NO_TRANS) throw std::invalid_argument("LocalPAMatrixTpetraOperator does not support transpose apply");
      const size_t localLength = static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
      if ((X.getLocalLength() != localLength) || (Y.getLocalLength() != localLength))
        throw std::invalid_argument("LocalPAMatrixTpetraOperator local vector length mismatch");
      if (X.getNumVectors() != Y.getNumVectors())
        throw std::invalid_argument("LocalPAMatrixTpetraOperator multivector count mismatch");

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      // With beta == 0, Y's prior contents must be ignored, not scaled by zero: Y may be uninitialized (e.g., a fresh
      // Belos Krylov vector), and 0 * NaN or 0 * Inf would produce NaN.
      const bool overwriteY = (beta == Teuchos::ScalarTraits<Scalar>::zero());
      auto yHost = overwriteY ? Y.getLocalViewHost(Tpetra::Access::OverwriteAll) : Y.getLocalViewHost(Tpetra::Access::ReadWrite);
      Intrepid2::ScalarView<Scalar,DeviceType> input("LocalPAMatrixTpetraOperator input", cells_, localDofs_);
      Intrepid2::ScalarView<Scalar,DeviceType> output("LocalPAMatrixTpetraOperator output", cells_, localDofs_);
      auto inputHost = Kokkos::create_mirror_view(input);

      for (size_t vectorOrdinal=0; vectorOrdinal<X.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
          for (int f=0; f<localDofs_; f++)
            inputHost(c,f) = xHost(static_cast<size_t>(c * localDofs_ + f), vectorOrdinal);
        Kokkos::deep_copy(input, inputHost);

        TensorPAMatrixTpetraAdapters::applyTerms(*terms_, workspaces_, input, output, worksetSize_);
        auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);

        for (int c=0; c<cells_; c++)
          for (int f=0; f<localDofs_; f++)
          {
            const size_t localOrdinal = static_cast<size_t>(c * localDofs_ + f);
            yHost(localOrdinal, vectorOrdinal) = overwriteY ? alpha * outputHost(c,f)
                                                            : alpha * outputHost(c,f) + beta * yHost(localOrdinal, vectorOrdinal);
          }
      }
    }
  };

  class LocalPAMatrixTpetraRowMatrix : public TpetraRowMatrix
  {
    using local_ordinal_type = typename TpetraRowMatrix::local_ordinal_type;
    using global_ordinal_type = typename TpetraRowMatrix::global_ordinal_type;
    using node_type = typename TpetraRowMatrix::node_type;
    using row_graph_type = Tpetra::RowGraph<local_ordinal_type, global_ordinal_type, node_type>;
    using global_inds_host_view_type = typename TpetraRowMatrix::global_inds_host_view_type;
    using local_inds_host_view_type = typename TpetraRowMatrix::local_inds_host_view_type;
    using values_host_view_type = typename TpetraRowMatrix::values_host_view_type;
    using nonconst_global_inds_host_view_type = typename TpetraRowMatrix::nonconst_global_inds_host_view_type;
    using nonconst_local_inds_host_view_type = typename TpetraRowMatrix::nonconst_local_inds_host_view_type;
    using nonconst_values_host_view_type = typename TpetraRowMatrix::nonconst_values_host_view_type;

    Teuchos::RCP<const TpetraMap> map_;
    mutable std::vector<OperatorTerm> *terms_;
    std::vector<WorkspaceView> workspaces_;
    int cells_ = 0;
    int localDofs_ = 0;
    int worksetSize_ = 0;
    long long maxDenseRowEntries_ = 0;
    mutable std::vector<Scalar> denseRows_;
    mutable bool denseRowsBuilt_ = false;
    mutable nonconst_local_inds_host_view_type localIndices_;
    mutable nonconst_global_inds_host_view_type globalIndices_;
    mutable nonconst_values_host_view_type values_;
    Teuchos::RCP<const row_graph_type> graph_;

    size_t localLength() const
    {
      return static_cast<size_t>(cells_) * static_cast<size_t>(localDofs_);
    }

    void checkLocalRow(const local_ordinal_type localRow) const
    {
      if ((localRow < 0) || (static_cast<size_t>(localRow) >= localLength()))
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix local row out of range");
    }

    void ensureDenseRowsBuilt() const
    {
      if (denseRowsBuilt_) return;
      const size_t length = localLength();
      const long long materializedEntries = (length > static_cast<size_t>(std::numeric_limits<long long>::max() / std::max<size_t>(size_t(1), length)))
                                       ? std::numeric_limits<long long>::max()
                                       : static_cast<long long>(length * length);
      if (materializedEntries > maxDenseRowEntries_)
        throw std::runtime_error("LocalPAMatrixTpetraRowMatrix dense row materialization is diagnostic-only and exceeded maxDenseRowEntries guard");
      denseRows_.assign(length * length, Scalar(0.0));
      TpetraMultiVector X(map_, 1);
      TpetraMultiVector Y(map_, 1);
      for (size_t column=0; column<length; column++)
      {
        X.putScalar(Scalar(0.0));
        {
          auto xHost = X.getLocalViewHost(Tpetra::Access::ReadWrite);
          xHost(column, 0) = Scalar(1.0);
        }
        apply(X, Y);
        auto yHost = Y.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t row=0; row<length; row++)
          denseRows_[row * length + column] = yHost(row, 0);
      }
      denseRowsBuilt_ = true;
    }

  public:
    LocalPAMatrixTpetraRowMatrix(const Teuchos::RCP<const TpetraMap> &map, std::vector<OperatorTerm> &terms,
                                 const std::vector<WorkspaceView> &workspaces, const int cells,
                                 const int localDofs, const int worksetSize, const long long maxDenseRowEntries)
      : map_(map), terms_(&terms), workspaces_(workspaces), cells_(cells), localDofs_(localDofs), worksetSize_(worksetSize), maxDenseRowEntries_(maxDenseRowEntries)
    {
      const size_t length = localLength();
      auto graph = Teuchos::rcp(new TpetraCrsGraph(map_, length));
      Teuchos::Array<global_ordinal_type> columns(length);
      for (size_t column=0; column<length; column++)
        columns[static_cast<Teuchos::Ordinal>(column)] = map_->getGlobalElement(static_cast<local_ordinal_type>(column));
      for (size_t row=0; row<length; row++)
        graph->insertGlobalIndices(map_->getGlobalElement(static_cast<local_ordinal_type>(row)), columns());
      graph->fillComplete(map_, map_);
      graph_ = graph;
    }

    Teuchos::RCP<const Teuchos::Comm<int>> getComm() const override { return map_->getComm(); }
    Teuchos::RCP<const TpetraMap> getRowMap() const override { return map_; }
    Teuchos::RCP<const TpetraMap> getColMap() const override { return map_; }
    Teuchos::RCP<const TpetraMap> getDomainMap() const override { return map_; }
    Teuchos::RCP<const TpetraMap> getRangeMap() const override { return map_; }
    Teuchos::RCP<const row_graph_type> getGraph() const override { return graph_; }

    Tpetra::global_size_t getGlobalNumRows() const override { return map_->getGlobalNumElements(); }
    Tpetra::global_size_t getGlobalNumCols() const override { return map_->getGlobalNumElements(); }
    size_t getLocalNumRows() const override { return localLength(); }
    size_t getLocalNumCols() const override { return localLength(); }
    global_ordinal_type getIndexBase() const override { return map_->getIndexBase(); }

    Tpetra::global_size_t getGlobalNumEntries() const override
    {
      const Tpetra::global_size_t localEntries = static_cast<Tpetra::global_size_t>(getLocalNumEntries());
      Tpetra::global_size_t globalEntries = 0;
      Teuchos::reduceAll<int, Tpetra::global_size_t>(*getComm(), Teuchos::REDUCE_SUM, 1, &localEntries, &globalEntries);
      return globalEntries;
    }

    size_t getLocalNumEntries() const override { return localLength() * localLength(); }

    size_t getNumEntriesInGlobalRow(global_ordinal_type globalRow) const override
    {
      const local_ordinal_type localRow = map_->getLocalElement(globalRow);
      if (localRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid()) return Teuchos::OrdinalTraits<size_t>::invalid();
      return getNumEntriesInLocalRow(localRow);
    }

    size_t getNumEntriesInLocalRow(local_ordinal_type localRow) const override
    {
      checkLocalRow(localRow);
      return localLength();
    }

    size_t getGlobalMaxNumRowEntries() const override { return localLength(); }
    size_t getLocalMaxNumRowEntries() const override { return localLength(); }
    local_ordinal_type getBlockSize() const override { return 1; }
    bool hasColMap() const override { return true; }
    bool isLocallyIndexed() const override { return true; }
    bool isGloballyIndexed() const override { return false; }
    bool isFillComplete() const override { return true; }
    bool supportsRowViews() const override { return true; }

    void getGlobalRowCopy(global_ordinal_type globalRow,
                          nonconst_global_inds_host_view_type &indices,
                          nonconst_values_host_view_type &values,
                          size_t &numEntries) const override
    {
      const local_ordinal_type localRow = map_->getLocalElement(globalRow);
      if (localRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix global row is not local");
      const size_t length = localLength();
      if ((indices.extent(0) < length) || (values.extent(0) < length))
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix global row copy buffers are too small");
      ensureDenseRowsBuilt();
      for (size_t column=0; column<length; column++)
      {
        indices(column) = map_->getGlobalElement(static_cast<local_ordinal_type>(column));
        values(column) = denseRows_[static_cast<size_t>(localRow) * length + column];
      }
      numEntries = length;
    }

    void getLocalRowCopy(local_ordinal_type localRow,
                         nonconst_local_inds_host_view_type &indices,
                         nonconst_values_host_view_type &values,
                         size_t &numEntries) const override
    {
      checkLocalRow(localRow);
      const size_t length = localLength();
      if ((indices.extent(0) < length) || (values.extent(0) < length))
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix local row copy buffers are too small");
      ensureDenseRowsBuilt();
      for (size_t column=0; column<length; column++)
      {
        indices(column) = static_cast<local_ordinal_type>(column);
        values(column) = denseRows_[static_cast<size_t>(localRow) * length + column];
      }
      numEntries = length;
    }

    void getGlobalRowView(global_ordinal_type globalRow,
                          global_inds_host_view_type &indices,
                          values_host_view_type &values) const override
    {
      const local_ordinal_type localRow = map_->getLocalElement(globalRow);
      if (localRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
      {
        indices = global_inds_host_view_type();
        values = values_host_view_type();
        return;
      }
      const size_t length = localLength();
      Kokkos::resize(globalIndices_, length);
      Kokkos::resize(values_, length);
      ensureDenseRowsBuilt();
      for (size_t column=0; column<length; column++)
      {
        globalIndices_(column) = map_->getGlobalElement(static_cast<local_ordinal_type>(column));
        values_(column) = denseRows_[static_cast<size_t>(localRow) * length + column];
      }
      indices = globalIndices_;
      values = values_;
    }

    void getLocalRowView(local_ordinal_type localRow,
                         local_inds_host_view_type &indices,
                         values_host_view_type &values) const override
    {
      checkLocalRow(localRow);
      const size_t length = localLength();
      Kokkos::resize(localIndices_, length);
      Kokkos::resize(values_, length);
      ensureDenseRowsBuilt();
      for (size_t column=0; column<length; column++)
      {
        localIndices_(column) = static_cast<local_ordinal_type>(column);
        values_(column) = denseRows_[static_cast<size_t>(localRow) * length + column];
      }
      indices = localIndices_;
      values = values_;
    }

    void getLocalDiagCopy(TpetraVector &diag) const override
    {
      const size_t length = localLength();
      if (diag.getLocalLength() != length)
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix diagonal length mismatch");
      auto diagHost = diag.getLocalViewHost(Tpetra::Access::OverwriteAll);
      const auto diagonal = TensorPAMatrixTpetraAdapters::extractCombinedDiagonal(*terms_, cells_, localDofs_);
      for (size_t localOrdinal=0; localOrdinal<length; localOrdinal++) diagHost(localOrdinal, 0) = diagonal[localOrdinal];
    }

    void leftScale(const TpetraVector &) override
    {
      throw std::runtime_error("LocalPAMatrixTpetraRowMatrix does not support leftScale in the diagnostic adapter.");
    }

    void rightScale(const TpetraVector &) override
    {
      throw std::runtime_error("LocalPAMatrixTpetraRowMatrix does not support rightScale in the diagnostic adapter.");
    }

    typename TpetraRowMatrix::mag_type getFrobeniusNorm() const override
    {
      ensureDenseRowsBuilt();
      Scalar normSquared = 0.0;
      for (const Scalar entry : denseRows_) normSquared += entry * entry;
      return std::sqrt(normSquared);
    }

    void apply(const TpetraMultiVector &X, TpetraMultiVector &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override
    {
      if (mode != Teuchos::NO_TRANS) throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix does not support transpose apply");
      const size_t length = localLength();
      if ((X.getLocalLength() != length) || (Y.getLocalLength() != length))
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix local vector length mismatch");
      if (X.getNumVectors() != Y.getNumVectors())
        throw std::invalid_argument("LocalPAMatrixTpetraRowMatrix multivector count mismatch");

      auto xHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      // With beta == 0, Y's prior contents must be ignored, not scaled by zero: Y may be uninitialized (e.g., a fresh
      // Belos Krylov vector), and 0 * NaN or 0 * Inf would produce NaN.
      const bool overwriteY = (beta == Teuchos::ScalarTraits<Scalar>::zero());
      auto yHost = overwriteY ? Y.getLocalViewHost(Tpetra::Access::OverwriteAll) : Y.getLocalViewHost(Tpetra::Access::ReadWrite);
      Intrepid2::ScalarView<Scalar,DeviceType> input("LocalPAMatrixTpetraRowMatrix input", cells_, localDofs_);
      Intrepid2::ScalarView<Scalar,DeviceType> output("LocalPAMatrixTpetraRowMatrix output", cells_, localDofs_);
      auto inputHost = Kokkos::create_mirror_view(input);

      for (size_t vectorOrdinal=0; vectorOrdinal<X.getNumVectors(); vectorOrdinal++)
      {
        for (int c=0; c<cells_; c++)
          for (int f=0; f<localDofs_; f++)
            inputHost(c,f) = xHost(static_cast<size_t>(c * localDofs_ + f), vectorOrdinal);
        Kokkos::deep_copy(input, inputHost);

        TensorPAMatrixTpetraAdapters::applyTerms(*terms_, workspaces_, input, output, worksetSize_);
        auto outputHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);

        for (int c=0; c<cells_; c++)
          for (int f=0; f<localDofs_; f++)
          {
            const size_t localOrdinal = static_cast<size_t>(c * localDofs_ + f);
            yHost(localOrdinal, vectorOrdinal) = overwriteY ? alpha * outputHost(c,f)
                                                            : alpha * outputHost(c,f) + beta * yHost(localOrdinal, vectorOrdinal);
          }
      }
    }

    bool hasTransposeApply() const override { return false; }

    std::string description() const override
    {
      return "LocalPAMatrixTpetraRowMatrix(local_flat_map_nonproduction_rowmatrix)";
    }

    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const override
    {
      if ((verbLevel == Teuchos::VERB_MEDIUM) || (verbLevel == Teuchos::VERB_HIGH) || (verbLevel == Teuchos::VERB_EXTREME)) out << description() << std::endl;
    }
  };
}

namespace Camellia
{
  std::vector<WorkspaceView>
  TensorPAMatrixTpetraAdapters::allocateWorkspaces(
      const std::vector<OperatorTerm> &terms,
      const int worksetSize)
  {
    std::vector<WorkspaceView> workspaces;
    workspaces.reserve(terms.size());
    for (const auto &term : terms)
    {
      workspaces.push_back(term.matrix.allocateWorkspace(worksetSize));
    }
    return workspaces;
  }


  double TensorPAMatrixTpetraAdapters::applyTerms(
      std::vector<OperatorTerm> &terms,
      const std::vector<WorkspaceView> &workspaces,
      const Intrepid2::ScalarView<Scalar,DeviceType> &input,
      Intrepid2::ScalarView<Scalar,DeviceType> &output,
      const int worksetSize)
  {
    if (terms.size() != workspaces.size())
      throw std::invalid_argument("TensorPAMatrixTpetraAdapters::applyTerms term/workspace size mismatch");

    Kokkos::Timer timer;

    if (allWeightsAreOne(terms))
    {
      bool sumInto = false;
      for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
      {
        terms[termOrdinal].matrix.apply(output,
                                        input,
                                        workspaces[termOrdinal],
                                        sumInto,
                                        worksetSize);
        sumInto = true;
      }
    }
    else
    {
      Kokkos::deep_copy(output, Scalar(0.0));

      Intrepid2::ScalarView<Scalar,DeviceType> termOutput(
          "TensorPAMatrixTpetraAdapters weighted term output",
          output.extent_int(0),
          output.extent_int(1));

      for (size_t termOrdinal=0; termOrdinal<terms.size(); termOrdinal++)
      {
        terms[termOrdinal].matrix.apply(termOutput,
                                        input,
                                        workspaces[termOrdinal],
                                        false,
                                        worksetSize);
        addScaledOutput(output, termOutput, terms[termOrdinal].weight);
      }
    }

    Kokkos::fence();
    return timer.seconds();
  }

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

  Teuchos::RCP<TpetraOperator>
  TensorPAMatrixTpetraAdapters::createLocalPAMatrixTpetraOperator(
      const Teuchos::RCP<const TpetraMap> &map,
      std::vector<OperatorTerm> &terms,
      const std::vector<WorkspaceView> &workspaces,
      const int cells,
      const int localDofs,
      const int worksetSize)
  {
    return Teuchos::rcp(new LocalPAMatrixTpetraOperator(
        map, terms, workspaces, cells, localDofs, worksetSize));
  }

  Teuchos::RCP<TpetraRowMatrix>
  TensorPAMatrixTpetraAdapters::createLocalPAMatrixTpetraRowMatrix(
      const Teuchos::RCP<const TpetraMap> &map,
      std::vector<OperatorTerm> &terms,
      const std::vector<WorkspaceView> &workspaces,
      const int cells,
      const int localDofs,
      const int worksetSize,
      const long long maxDenseRowEntries)
  {
    return Teuchos::rcp(new LocalPAMatrixTpetraRowMatrix(
        map, terms, workspaces, cells, localDofs, worksetSize, maxDenseRowEntries));
  }

  std::vector<Scalar> TensorPAMatrixTpetraAdapters::extractCombinedDiagonal(std::vector<OperatorTerm> &terms, const int cells, const int localDofs)
  {
    std::vector<Scalar> diagonal(static_cast<size_t>(cells * localDofs), Scalar(0.0));
    for (auto &term : terms)
    {
      auto diagonalData = term.matrix.allocateDiagonalStorage();
      term.matrix.extractDiagonal(diagonalData);
      Kokkos::fence();

      if (diagonalData.getUnderlyingViewRank() == 2)
      {
        auto diagonalHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.template getUnderlyingView<2>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<localDofs; i++)
            diagonal[static_cast<size_t>(c * localDofs + i)] += term.weight * diagonalHost(c,i);
      }
      else if (diagonalData.getUnderlyingViewRank() == 1)
      {
        auto diagonalHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagonalData.template getUnderlyingView<1>());
        for (int c=0; c<cells; c++)
          for (int i=0; i<localDofs; i++)
            diagonal[static_cast<size_t>(c * localDofs + i)] += term.weight * diagonalHost(i);
      }
      else
      {
        throw std::invalid_argument("Unexpected PAMatrix diagonal storage rank");
      }
    }
    return diagonal;
  }

  void TensorPAMatrixTpetraAdapters::fillInverseDiagonalFromTerms(
      std::vector<OperatorTerm> &terms,
      const int cells,
      const int localDofs,
      TpetraMultiVector &inverseDiagonal)
  {
    const size_t localLength =
        static_cast<size_t>(cells) * static_cast<size_t>(localDofs);

    if (inverseDiagonal.getLocalLength() != localLength)
      throw std::invalid_argument("fillInverseDiagonalFromTerms inverse diagonal length mismatch");

    auto inverseDiagonalHost =
        inverseDiagonal.getLocalViewHost(Tpetra::Access::OverwriteAll);

    const auto diagonal = TensorPAMatrixTpetraAdapters::extractCombinedDiagonal(terms, cells, localDofs);

    for (size_t localOrdinal=0; localOrdinal<localLength; localOrdinal++)
    {
      if (diagonal[localOrdinal] <= std::numeric_limits<Scalar>::epsilon())
        throw std::runtime_error("fillInverseDiagonalFromTerms found nonpositive diagonal entry");

      for (size_t vectorOrdinal=0; vectorOrdinal<inverseDiagonal.getNumVectors(); vectorOrdinal++)
      {
        inverseDiagonalHost(localOrdinal, vectorOrdinal) =
            Scalar(1.0) / diagonal[localOrdinal];
      }
    }
  }
}
