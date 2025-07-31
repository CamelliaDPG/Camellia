//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  Tpetra_Operator_to_Tpetra_Matrix.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/29/25.
//
//

#include "Tpetra_Operator_to_Tpetra_Matrix.h"

#include <set>
#include <map>

namespace Camellia
{
Teuchos::RCP<TMatrix<double> > Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(const Tpetra::Operator<double> &op, const Tpetra::Map<> &map)
{
  Teuchos::RCP<TMatrix<double> > matrix;
  
//  int numEntriesPerRow = 0;
//  Teuchos::RCP<Tpetra_FECrsMatrix> matrix = Teuchos::rcp(new Tpetra_FECrsMatrix(::Copy, map, numEntriesPerRow));
//
//  int numRows = map.NumGlobalElements();
//
//  Tpetra_Vector X(map);
//  Tpetra_Vector Y(map);
//
//  double tol = 1e-15; // values below this will be considered 0
//
//  for (int rowIndex=0; rowIndex<numRows; rowIndex++)
//  {
//    int lid = map.LID(rowIndex);
//    if (lid != -1)
//    {
//      X[lid] = 1.0;
//    }
//    op.ApplyInverse(X, Y);
//    if (lid != -1)
//    {
//      X[lid] = 0.0;
//    }
//
//    std::vector<double> values;
//    std::vector<int> indices;
//    for (int i=0; i<map.NumMyElements(); i++)
//    {
//      if (abs(Y[i]) > tol)
//      {
//        values.push_back(Y[i]);
//        indices.push_back(map.GID(i));
//      }
//    }
//
//    matrix->InsertGlobalValues(rowIndex, values.size(), &values[0], &indices[0]);
//  }
//
//  matrix->GlobalAssemble();
  return matrix;
}
}
