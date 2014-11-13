//
//  Epetra_Operator_to_Epetra_Matrix.h
//  Camellia
//
//  Created by Nate Roberts on 11/10/14.
//
//

#ifndef Camellia_Epetra_Operator_to_Epetra_Matrix_h
#define Camellia_Epetra_Operator_to_Epetra_Matrix_h

#include "Epetra_Operator.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Map.h"

#include "Teuchos_RCP.hpp"

class Epetra_Operator_to_Epetra_Matrix {
public:
//  static Teuchos::RCP<Epetra_CrsMatrix> constructMatrix(Epetra_Operator &op, Epetra_Map &map);
  static Teuchos::RCP<Epetra_CrsMatrix> constructInverseMatrix(const Epetra_Operator &op, const Epetra_Map &map);
};

#endif
