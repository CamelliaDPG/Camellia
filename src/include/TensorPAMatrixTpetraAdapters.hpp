//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TensorPAMatrixTpetraAdapters.hpp
//  Camellia
//
//  Created by Nate Roberts on 9/30/26.
//
//
#pragma once

#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrix.hpp>

#include <Teuchos_RCP.hpp>

#include <vector>

namespace Camellia
{
  class TensorPAMatrixTpetraAdapters
  {
  public:
    using Scalar = double;
    using TpetraMap = Tpetra::Map<>;
    using TpetraCrsMatrix = Tpetra::CrsMatrix<Scalar, typename TpetraMap::local_ordinal_type, typename TpetraMap::global_ordinal_type, typename TpetraMap::node_type>;
    using TpetraOperator = Tpetra::Operator<Scalar>;
    
    static Teuchos::RCP<TpetraCrsMatrix> buildLocalCrsMatrixFromDense(const Teuchos::RCP<const TpetraMap> &map,
                                                                      const std::vector<Scalar> &localMatrices,
                                                                      const int cells,
                                                                      const int rowsPerCell);
    
    static Teuchos::RCP<TpetraOperator> localGMGSmootherOnlyJacobiOperator(const Teuchos::RCP<const TpetraOperator> &A,
                                                                           const Teuchos::RCP<const TpetraOperator> &jacobi,
                                                                           const int smootherApplications);
  };
}
