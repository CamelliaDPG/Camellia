//
//  Epetra_Operator_to_Epetra_Matrix.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/10/14.
//
//

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Vector.h"

#include <set>
#include <map>

namespace Camellia
{
Teuchos::RCP<Epetra_CrsMatrix> Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(const Epetra_Operator &op, const Epetra_Map &map)
{
  int numEntriesPerRow = 0;
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(new Epetra_CrsMatrix(::Copy, map, numEntriesPerRow));

  int numRows = map.NumGlobalElements();

  Epetra_Vector X(map);
  Epetra_Vector Y(map);

  double tol = 1e-15; // values below this will be considered 0

  for (int rowIndex=0; rowIndex<numRows; rowIndex++)
  {
    int lid = map.LID(rowIndex);
    if (lid != -1)
    {
      X[lid] = 1.0;
    }
    op.ApplyInverse(X, Y);
    if (lid != -1)
    {
      X[lid] = 0.0;
    }

    std::vector<double> values;
    std::vector<int> indices;
    for (int i=0; i<map.NumMyElements(); i++)
    {
      if (abs(Y[i]) > tol)
      {
        values.push_back(Y[i]);
        indices.push_back(map.GID(i));
      }
    }

    matrix->InsertGlobalValues(rowIndex, values.size(), &values[0], &indices[0]);
  }

  matrix->FillComplete();
  return matrix;
}
}
