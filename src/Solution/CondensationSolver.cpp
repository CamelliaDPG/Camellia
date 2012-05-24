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
#include "Epetra_FECrsMatrix.h"
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

// added by Jesse - static condensation solve. WARNING: will not take into account Lagrange multipliers or zero-mean constraints yet. 
// UNFINISHED
void CondensationSolver::getDofIndices(){
  
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
  
  typedef Teuchos::RCP< DofOrdering > DofOrderingPtr;
  
  // get elemTypes
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes(rank);
  vector< ElementTypePtr >::iterator elemTypeIt;
  
  // determine any zero-mean constraints:
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  vector< int >::iterator trialIt;

  // get all element-local field/flux dof indices associated with elemTypes in partition
  vector<int> fieldDofInds;
  vector<int> fluxDofInds;
  _allFluxInds.clear();
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);

    vector<int> elemFieldInds;
    vector<int> elemFluxInds;
    for (trialIt = trialIDs.begin();trialIt!=trialIDs.end();trialIt++){
      int trialID = *(trialIt);
      if (!_mesh->bilinearForm()->isFluxOrTrace(trialID)){ 
	elemFieldInds = elemTypePtr->trialOrderPtr->getDofIndices(trialID, 0); // 0 = side index
      } else {
	int numSides = elemTypePtr->trialOrderPtr->getNumSidesForVarID(trialID);
	for (int sideIndex = 0;sideIndex<numSides;sideIndex++){
	  elemFluxInds = elemTypePtr->trialOrderPtr->getDofIndices(trialID, sideIndex); 
	}
      }
    } 

    // store cellID to 
    vector< ElementPtr > elemsOfType = _mesh->elementsOfType(rank, elemTypePtr); // all elems of type
    vector< ElementPtr >::iterator elemIt;       
    vector<int> globalFieldInds;
    vector<int> globalFluxInds;
    for (elemIt = elemsOfType.begin(); elemIt != elemsOfType.end(); elemIt++){   
      int cellID = (*(elemIt))->cellID();
      for (int i = 0;i<elemFieldInds.size();i++){
	int globalFieldIndex = _mesh->globalDofIndex(cellID,elemFieldInds[i]);
	globalFieldInds.push_back(globalFieldIndex);
      }
      for (int i = 0;i<elemFluxInds.size();i++){
	int globalFluxIndex = _mesh->globalDofIndex(cellID,elemFluxInds[i]);
	globalFluxInds.push_back(globalFluxIndex);
	_allFluxInds.insert(globalFluxIndex);
      }
      _GlobalFieldInds[cellID] = globalFieldInds;
      _GlobalFluxInds[cellID] = globalFluxInds;
      _ElemFieldInds[cellID] = elemFieldInds;
      _ElemFluxInds[cellID] = elemFluxInds;      
    }
  }   
}

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

  getDofIndices(); // get dof inds

  //  Teuchos::RCP<Epetra_RowMatrix> A = problem().GetMatrix();
  Epetra_RowMatrix* K = problem().GetMatrix();
  Epetra_MultiVector* f = problem().GetRHS();

  // modify load vector
  Epetra_SerialDenseSolver solver;
  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _GlobalFluxInds[cellID];
    vector<int> fieldInds = _GlobalFieldInds[cellID];

    // [D   B][fieldDofs] = [f]
    // [B^T A][flux Dofs] = [g]

    Epetra_SerialDenseMatrix D = getSubMatrix(K,fieldInds,fieldInds); // block field dof submatrix
    Epetra_SerialDenseMatrix B = getSubMatrix(K,fieldInds,fluxInds);  // coupling bw field/flux submat
    Epetra_SerialDenseMatrix b = getSubVector(f,fieldInds);           // field RHS
    Epetra_SerialDenseMatrix rhsMod;    

    solver.SetMatrix(D);
    solver.SetVectors(rhsMod,B);
    bool equilibrated = false;
    if ( solver.ShouldEquilibrate() ) {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }    
    solver.Solve();    
    if (equilibrated) {
      solver.UnequilibrateLHS();
    }
    
    // create reduced matrix 
    int numGlobalDofs = _allFluxInds.size();
    set<int> globalIndsForPartition = _mesh->globalDofIndicesForPartition(rank);
    Epetra_Map partMap = _solution->getPartitionMap(rank, globalIndsForPartition,numGlobalDofs,0,&Comm); // 0 = assumes no zero mean or Lagrange constraints imposed 
    Epetra_FECrsMatrix Aflux(Copy, partMap, numGlobalDofs); // reduced matrix - soon to be schur complement
    Epetra_FEVector bflux(partMap);        
    
  }
  
  return 0;
}


// TODO - finish implementing these
Epetra_SerialDenseMatrix CondensationSolver::getSubMatrix(Epetra_RowMatrix*  K,vector<int> rowInds, vector<int> colInds){
  Epetra_SerialDenseMatrix A;
  return A;
}

Epetra_SerialDenseMatrix CondensationSolver::getSubVector(Epetra_MultiVector*  f,vector<int> inds){
  Epetra_SerialDenseMatrix b;  
  return b;
}
