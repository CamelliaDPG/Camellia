// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER
//
//  Tpetra_Operator_to_Tpetra_Matrix.h
//  Camellia
//
//  Created by Nate Roberts on 7/28/25.
//
//

#ifndef Camellia_Tpetra_Operator_to_Tpetra_Matrix_h
#define Camellia_Tpetra_Operator_to_Tpetra_Matrix_h

#include <Tpetra_Operator.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

#include "Teuchos_RCP.hpp"

#include "TypeDefs.h"

namespace Camellia
{
class Tpetra_Operator_to_Tpetra_Matrix
{
public:
  //! Belos at least considers the preconditioner operators provided to it to be approximations of the inverse.  What we implement here just naively constructs
  //! an explicitly stored matrix from a possibly implicitly specified operator.  This is based on the Epetra_Operator_to_Epetra_Matrix class.
  static Teuchos::RCP<TMatrix<double> > constructInverseMatrix(const Teuchos::RCP<Tpetra::Operator<double> > op, TMapPtr map);
};
}

#endif
