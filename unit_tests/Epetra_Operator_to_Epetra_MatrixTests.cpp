//
//  Epetra_Operator_to_Epetra_MatrixTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/11/14.
//
//

#include "Epetra_Map.h"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Teuchos_UnitTestHarness.hpp"
/*namespace {
  TEUCHOS_UNIT_TEST( Epetra_Operator_to_Epetra_Matrix, MatrixRecovery )
  {
    // tests that an Epetra_CrsMatrix passed in is returned unchanged.
    int numRows = 10;
#ifdef HAVE_MPI
    Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
    Epetra_SerialComm comm;
#endif

    Epetra_Map map(numRows,0,comm);

    int entriesPerRow = 0;
    Epetra_CrsMatrix initialMatrix(::Copy, map, entriesPerRow);

    double rowStorage[entriesPerRow];
    for (int row=0; row<numRows; row++) {
      for (int col=0; col<numCols; col++) {
        rowStorage[col] = -row + row * row / (1 + col);
      }

    }
  }
} // namespace*/