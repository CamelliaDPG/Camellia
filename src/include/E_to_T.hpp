// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER
//
//  E_to_T.h
//  Camellia
//
//  Created by Nate Roberts on 7/29/25.
//
//

#ifndef Camellia_E_to_T_h
#define Camellia_E_to_T_h

#include "TypeDefs.h"

#include "Epetra_FECrsMatrix.h"

namespace Camellia
{
//! Utilities for converting from Epetra to Tpetra objects.  Intended as a temporary facility to allow piecemeal changes, upgrading one part of Camellia at a time to use Tpetra objects natively.
class E_to_T
{
public:
  static TGraphPtr graph(const Epetra_CrsGraph &eGraph);
  static TMapPtr map(const Epetra_BlockMap &eMap);
  static TMatrixPtr<double> matrix(Teuchos::RCP<const Epetra_CrsMatrix> eMatrix);
  static TVectorPtr<double> vector(Teuchos::RCP<const Epetra_MultiVector> eVector);
};
}

#endif
