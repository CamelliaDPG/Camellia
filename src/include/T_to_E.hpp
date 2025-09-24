// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  T_to_E.h
//  Camellia
//
// Convert Tpetra objects to Epetra objects.
//  Created by Nate Roberts on 9/23/25.
//

#ifndef Camellia_T_to_E_h
#define Camellia_T_to_E_h

#include "TypeDefs.h"

#include <Epetra_BlockMap.h>
#include <Epetra_CrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>
#include <Teuchos_RCP.hpp>

namespace Camellia
{

/// Utilities for converting from Tpetra to Epetra objects.
class T_to_E
{
public:
  /// Convert a Tpetra::CrsGraph into an Epetra_CrsGraph
  static Teuchos::RCP<Epetra_CrsGraph> graph(const TGraphConstPtr & tGraph);

  /// Convert a Tpetra::Map into an Epetra_BlockMap (with block size = 1)
  static Teuchos::RCP<Epetra_BlockMap> map(const TMapConstPtr & tMap);

  /// Convert a Tpetra::CrsMatrix<double> into an Epetra_CrsMatrix
  static Teuchos::RCP<Epetra_CrsMatrix> matrix(const TMatrixPtr<double> & tMatrix);

  /// Convert a Tpetra::MultiVector<double> into an Epetra_MultiVector
  static Teuchos::RCP<Epetra_MultiVector> vector(const TVectorPtr<double> & tVector);
};

} // namespace Camellia

#endif // Camellia_T_to_E_h
