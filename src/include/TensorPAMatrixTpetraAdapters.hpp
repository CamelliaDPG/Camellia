#pragma once

#include "TensorPAMatrixBridge.h"

#include <Kokkos_Core.hpp>

#include <Intrepid2_PAMatrix.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrix.hpp>
#include <Tpetra_Vector.hpp>

#include <Teuchos_RCP.hpp>

#include <vector>

namespace Camellia
{
  class TensorPAMatrixTpetraAdapters
  {
  public:
    using Scalar = double;
    using DeviceType = Kokkos::DefaultExecutionSpace::device_type;

    using OperatorTerm = Camellia::TensorPAMatrixTerm<DeviceType,Scalar>;
    using WorkspaceView = Kokkos::View<Scalar*,DeviceType>;

    using TpetraMap = Tpetra::Map<>;
    using TpetraCrsMatrix =
        Tpetra::CrsMatrix<Scalar,
                          typename TpetraMap::local_ordinal_type,
                          typename TpetraMap::global_ordinal_type,
                          typename TpetraMap::node_type>;
    using TpetraMultiVector = Tpetra::MultiVector<Scalar>;
    using TpetraOperator = Tpetra::Operator<Scalar>;
    using TpetraRowMatrix = Tpetra::RowMatrix<Scalar>;
    using TpetraVector = Tpetra::Vector<Scalar>;

    static std::vector<WorkspaceView>
    allocateWorkspaces(const std::vector<OperatorTerm> &terms,
                       const int worksetSize);

    static double
    applyTerms(std::vector<OperatorTerm> &terms,
               const std::vector<WorkspaceView> &workspaces,
               const Intrepid2::ScalarView<Scalar,DeviceType> &input,
               Intrepid2::ScalarView<Scalar,DeviceType> &output,
               const int worksetSize);

    static std::vector<Scalar>
    extractCombinedDiagonal(std::vector<OperatorTerm> &terms,
                            const int cells,
                            const int localDofs);

    static void
    fillInverseDiagonalFromTerms(std::vector<OperatorTerm> &terms,
                                 const int cells,
                                 const int localDofs,
                                 TpetraMultiVector &inverseDiagonal);

    static Teuchos::RCP<TpetraOperator>
    createLocalPAMatrixTpetraOperator(const Teuchos::RCP<const TpetraMap> &map,
                                      std::vector<OperatorTerm> &terms,
                                      const std::vector<WorkspaceView> &workspaces,
                                      const int cells,
                                      const int localDofs,
                                      const int worksetSize);

    static Teuchos::RCP<TpetraRowMatrix>
    createLocalPAMatrixTpetraRowMatrix(const Teuchos::RCP<const TpetraMap> &map,
                                       std::vector<OperatorTerm> &terms,
                                       const std::vector<WorkspaceView> &workspaces,
                                       const int cells,
                                       const int localDofs,
                                       const int worksetSize,
                                       const long long maxDenseRowEntries);

    static Teuchos::RCP<TpetraCrsMatrix>
    buildLocalCrsMatrixFromDense(const Teuchos::RCP<const TpetraMap> &map,
                                 const std::vector<Scalar> &localMatrices,
                                 const int cells,
                                 const int rowsPerCell);
  };
}
