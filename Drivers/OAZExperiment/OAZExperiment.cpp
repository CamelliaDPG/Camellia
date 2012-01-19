#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"

// Intrepid includes
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_Basis.hpp"

#include "Amesos_Klu.h"
#include "Amesos.h"
#include "Amesos_Utils.h"
//#include "Amesos_Mumps.h"

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "Epetra_LocalMap.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "ml_epetra_utils.h"
//#include "ml_common.h"

// Trilinos includes
#include "Intrepid_FieldContainer.hpp"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

int main(int argc, char *argv[]) {
  int numRows = 4;
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  TEST_FOR_EXCEPTION( (numProcs != 1) && (numProcs != 2) && (numProcs != 4),
                     std::invalid_argument, "numProcs = 1,2,4 supported" );
  
  int indexBase = 0;
  int numMyRows = numRows / numProcs;
  int myStartIndex = numMyRows * rank;
  int *myGlobalIndices = new int[ numMyRows ];
  set<int> myGlobalIndicesSet;
  
  for (int i=0; i<numMyRows; i++) {
    myGlobalIndices[i] = myStartIndex + i;
    myGlobalIndicesSet.insert(myGlobalIndices[i]);
  }

  Epetra_Map partMap = Epetra_Map(numRows, numMyRows, myGlobalIndices, indexBase, Comm);
  
  delete myGlobalIndices;
  
  Epetra_FECrsMatrix debugMatrix(Copy, partMap, numRows);
  
  for (set<int>::iterator indexIt1 = myGlobalIndicesSet.begin(); indexIt1 != myGlobalIndicesSet.end(); indexIt1++) {
    int globalIndex1 = *indexIt1;
    for (int globalIndex2=0; globalIndex2 < numRows; globalIndex2++) {
      double value = 2.0;
      debugMatrix.InsertGlobalValues(1, &globalIndex1, 1, &globalIndex2, &value);
    }
  }

  debugMatrix.GlobalAssemble();
  
  EpetraExt::RowMatrixToMatlabFile("debugMatrix_filled.dat",debugMatrix);
  
  int numBCs = numMyRows;
  FieldContainer<int> bcGlobalIndices(numBCs);
  int i=0;
  for (set<int>::iterator indexIt1 = myGlobalIndicesSet.begin(); indexIt1 != myGlobalIndicesSet.end(); indexIt1++) {
    int globalIndex1 = *indexIt1;
    bcGlobalIndices(i++) = debugMatrix.LRID(globalIndex1);
    cout << "rank " << rank << " applying BC to globalIndex " << globalIndex1 << endl;
  }
  
  ML_Epetra::Apply_OAZToMatrix(&bcGlobalIndices(0), numBCs, debugMatrix);
  
  EpetraExt::RowMatrixToMatlabFile("debugMatrix_afterOAZ.dat",debugMatrix);

}