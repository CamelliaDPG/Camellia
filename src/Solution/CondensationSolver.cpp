#include "Intrepid_FieldContainer.hpp"

#include "Amesos_Klu.h"
#include "Amesos.h"
#include "Amesos_Utils.h"

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_FEVector.h"
#include "Epetra_Time.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

#include "ml_epetra_utils.h"
#include <stdlib.h>

#include "CondensationSolver.h"

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;

// added by Jesse - static condensation solve. WARNING: will not take into account Lagrange multipliers or zero-mean constraints yet. Those must be condensed out separately
int CondensationSolver::solve(){

  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  Epetra_RowMatrix* stiffMat = problem().GetMatrix();
  Epetra_MultiVector* rhs = problem().GetRHS();
  Epetra_MultiVector* lhs = problem().GetLHS(); 

  init();

  // do stuff to get flux/field dofs/maps/etc
  // ------------------

  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;

  // all (including field) inds
  set<int> globalIndsForPartition = _mesh->globalDofIndicesForPartition(rank); 
  int numGlobalDofs = globalIndsForPartition.size();

  int numGlobalFluxDofs = _allFluxInds.size();

  //  Epetra_Map fluxPartMap = _solution->getPartitionMap(rank, _condensedFluxInds, numGlobalFluxDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints 
  Epetra_Map fluxPartMap = _solution->getPartitionMap(rank, _allFluxInds, numGlobalFluxDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints 
  cout << fluxPartMap;
  //  Epetra_Map partMap = _solution->getPartitionMap(rank, globalIndsForPartition, numGlobalDofs, 0, &Comm);
  //  cout << partMap;

  Epetra_FECrsMatrix K_cond(Copy, fluxPartMap, _mesh->rowSizeUpperBound()); // reduced matrix - soon to be schur complement
  Epetra_FEVector bflux(fluxPartMap);
  Epetra_FEVector lhsVector(fluxPartMap, true);

  // get dense solver for distributed portion
  Epetra_SerialDenseSolver denseSolver;

  getSubmatrices(stiffMat,K_cond);

  return 0;
}



// InsertGlobalValues to create the A flux submatrix (just go thru all partitions and check if the dofs are in the partition map)
// for rectangular matrix B, should just need field dofs (and "get rows" will cover the flux dofs due to storage of Epetra_RowMatrix)
// for block diagonal D, should just need field dofs

// =================================================================================================
// =================================================================================================

// gets flux/field dof data, builds maps, etc
void CondensationSolver::init(){
 
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  // _mesh->getFieldFluxDofInds(_localFluxInds, _localFieldInds);

  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;

  // compute set of all global flux dof indices, compute global-local map
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   

    // fluxes
    set<int> cellFluxInds = _localFluxInds[cellID];
    set<int>::iterator fluxIt;
    for (fluxIt = cellFluxInds.begin();fluxIt!=cellFluxInds.end();fluxIt++){
      int fluxInd = *fluxIt;
      int globalFluxInd = _mesh->globalDofIndex(cellID,fluxInd);
      _allFluxInds.insert(globalFluxInd);
    }

    // fields
    set<int> cellFieldInds = _localFieldInds[cellID];
    set<int>::iterator fieldIt;
    for (fieldIt=cellFieldInds.begin();fieldIt!=cellFieldInds.end();fieldIt++){
      int localFieldIndex = *fieldIt;
      int globalFieldIndex = _mesh->globalDofIndex(cellID,localFieldIndex);   
      _globalToLocalFieldInds[cellID][globalFieldIndex] = localFieldIndex; 
    }
  }

  // create maps between global flux inds and reduced matrix inds
  set<int>::iterator fluxIt;
  int count = 0;
  for (fluxIt = _allFluxInds.begin();fluxIt!= _allFluxInds.end();fluxIt++){
    int ind = *fluxIt;
    _globalToCondensedFluxInds[ind] = count;
    _condensedToGlobalFluxInds[count] = ind;
    _condensedFluxInds.insert(count);
    count++;
  }

  // create local maps between field inds and reduced matrices
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    int counter = 0;
    set<int> cellFluxInds = _localFluxInds[cellID];
    for (fluxIt = cellFluxInds.begin();fluxIt!=cellFluxInds.end();fluxIt++){      
      int localFieldInd = *fluxIt;
      _localToCondensedFieldInds[cellID][localFieldInd] = counter;
      _condensedToLocalFieldInds[cellID][counter] = localFieldInd;
    }
  }

  // allocate space for elem field matrices
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    int numFieldInds = _localFieldInds[cellID].size();
    Epetra_SerialDenseMatrix K_elem(numFieldInds,numFieldInds);
    _elemFieldMats[cellID] = K_elem;
  }
}

void CondensationSolver::getSubmatrices(const Epetra_RowMatrix* K,Epetra_FECrsMatrix &K_cond){
  
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  cout << "K->MaxNumEntries() = " << K->MaxNumEntries() << endl;


  int * globalColInds = K->RowMatrixColMap().MyGlobalElements();
  int numMyCols = K->RowMatrixColMap().NumMyElements();
  cout << "number of column elements on rank " << rank << " = " << numMyCols << endl;

  int numMyRows = K->RowMatrixRowMap().NumMyElements(); // number of rows stored on this proc
  for (int i = 0;i<numMyRows;++i){
    int numGlobalRowElements = K->RowMatrixRowMap().NumGlobalElements(); // number of rows on this proc

    // get inds and values of row
    int * globalRowInds = K->RowMatrixRowMap().MyGlobalElements();
    double * values = new double[numGlobalRowElements];
    int * indices = new int[numGlobalRowElements]; 
    int numEntries; // output from ExtractMyRowCopy
    K->ExtractMyRowCopy(i,numGlobalRowElements,numEntries,values,indices);    
   
    int rowInd = globalRowInds[i];
    bool rowIsFlux = _allFluxInds.find(rowInd)!=_allFluxInds.end();
    // loop through column values
    for (int j = 0;j<numEntries;++j){
      int colInd = globalColInds[indices[j]];
      bool colIsFlux = _allFluxInds.find(colInd)!=_allFluxInds.end();
      double value = values[j];

      //      cout << "K(" << rowInd << "," << colInd << ") = " << value << endl;

      // if the row index is a field index
      if (!rowIsFlux){
	int cellID = cellIDForGlobalFieldIndex(rowInd);
	if (cellID == -1){
	  cout << "Problem: returned negative cellID on rowInd = " << rowInd << endl;
	}
	int localRowInd = _globalToLocalFieldInds[cellID][rowInd];
	int condensedFieldRowInd = _localToCondensedFieldInds[cellID][localRowInd]; 

	if (!colIsFlux){ // if it's a field col index too, should have to a decoupled element matrix

	  int localColInd = _globalToLocalFieldInds[cellID][colInd];
	  int condensedFieldColInd = _localToCondensedFieldInds[cellID][localColInd]; 	  
	  _elemFieldMats[cellID](condensedFieldRowInd,condensedFieldColInd) = value;

	} else { // if it's a coupling term, store into 
	  
	  //	  int localColInd = _globalToCondensedInds[colInd];
	  //	  _couplingMatrices[cellID](condensedRowInd,localColInd) = values[j];

	}      

      }

      // if both are flux inds, use globalToCondensedInds map to sum into K_cond
      if (rowIsFlux && colIsFlux){

	int condensedFluxRowInd = _globalToCondensedFluxInds[rowInd];
	int condensedFluxColInd = _globalToCondensedFluxInds[colInd];
	//	K_cond.InsertGlobalValues(1,&condensedFluxRowInd,1,&condensedFluxColInd,&value);
	K_cond.InsertGlobalValues(1,&rowInd,1,&colInd,&value);

      }
      
    }
  }
  K_cond.GlobalAssemble();
  EpetraExt::RowMatrixToMatlabFile("test_mat.dat",*K);
}

// helper function - finds cellID for a field index
int CondensationSolver::cellIDForGlobalFieldIndex(int globalFieldIndex){
  vector< ElementPtr > elems = _mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    set<int> localInds = _localFieldInds[cellID];
    set<int>::iterator setIt;
    for (setIt=localInds.begin();setIt!=localInds.end();setIt++){
      int ind = *setIt;
      int globalInd = _mesh->globalDofIndex(cellID,ind);
      if (globalInd==globalFieldIndex){
	return cellID;
      }
    }
  }
  cout << "Did not find match." << endl;
  return -1;
}
