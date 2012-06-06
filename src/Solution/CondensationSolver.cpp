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
  Epetra_RowMatrix* stiffMat = problem().GetMatrix();
  Epetra_MultiVector* rhs = problem().GetRHS();

  // create reduced matrix 
  int numGlobalDofs = _allFluxInds.size();
  set<int> globalIndsForPartition = _mesh->globalDofIndicesForPartition(rank);
  Epetra_Map partMap = _solution->getPartitionMap(rank, globalIndsForPartition, numGlobalDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints 
  Epetra_FECrsMatrix Aflux(Copy, partMap, numGlobalDofs); // reduced matrix - soon to be schur complement
  Epetra_FEVector bflux(partMap);


  // get dense solver for distributed portion
  Epetra_SerialDenseSolver denseSolver;

  // modify stiffness/load vector
  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _GlobalFluxInds[cellID];
    vector<int> fieldInds = _GlobalFieldInds[cellID];

    // [D   B][fieldDofs] = [f]
    // [B^T A][flux Dofs] = [g]
    Epetra_SerialDenseMatrix D = getSubMatrix(stiffMat,fieldInds,fieldInds); // block field submatrix
    Epetra_SerialDenseMatrix B = getSubMatrix(stiffMat,fieldInds,fluxInds);  // field/flux submat
    Epetra_SerialDenseMatrix A = getSubMatrix(stiffMat,fluxInds,fluxInds);  // field/flux submat
    Epetra_SerialDenseMatrix f = getSubVector(rhs,fieldInds);           // field RHS
    Epetra_SerialDenseMatrix g = getSubVector(rhs,fluxInds);           // flux RHS -> will be reduced
    Epetra_SerialDenseMatrix Dinv_f;     // scratch storage
    Epetra_SerialDenseMatrix Dinv_B;     // scratch storage

    // modify load vector
    denseSolver.SetMatrix(D);
    denseSolver.SetVectors(Dinv_f,f); // solves Dx = b
    bool equilibrated = false;
    if ( denseSolver.ShouldEquilibrate() ) {
      denseSolver.EquilibrateMatrix();
      denseSolver.EquilibrateRHS();
      equilibrated = true;
    }    
    denseSolver.Solve();    
    if (equilibrated) {
      denseSolver.UnequilibrateLHS();
    }
    g.Multiply('T','N',-1.0,B,Dinv_f,1.0); // g := g - 1.0 * B^T*x , and x = inv(D) * f
    
    // modify stiffness matrix
    denseSolver.SetVectors(Dinv_B,B);
    denseSolver.Solve();    
    A.Multiply('T','N',1.0,B,Dinv_B,1.0);
    
    int numDofs = fluxInds.size();
    //    globalStiffMatrix.SumIntoGlobalValues(numDofs,&globalDofIndices(0),numDofs,&globalDofIndices(0),&finalStiffness(cellIndex,0,0));
    // TODO - figure out how to reference the array values like a FC
    //    rhsVector.SumIntoGlobalValues(numDofs,&globalDofIndices(0),&localRHSVector(cellIndex,0));

    //    Aflux.SumIntoGlobalValues()
    //    bflux.SumIntoGlobalValues()
    
  }

  Aflux.GlobalAssemble();
  bflux.GlobalAssemble();
  Epetra_FEVector lhsVector(partMap, true);
  Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&Aflux, &lhsVector, &bflux)); 

  _solver->setProblem(problem);
  int solveSuccess = _solver->solve(); // solve for the flux dofs (may need to modify this to make parallel)

   for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _GlobalFluxInds[cellID];
    vector<int> fieldInds = _GlobalFieldInds[cellID];

    // [D   B][fieldDofs] = [f]
    // [B^T A][flux Dofs] = [g]
    Epetra_SerialDenseMatrix D = getSubMatrix(stiffMat,fieldInds,fieldInds); // block field submatrix
    Epetra_SerialDenseMatrix B = getSubMatrix(stiffMat,fieldInds,fluxInds);  // field/flux submat
    Epetra_SerialDenseMatrix f_mod = getSubVector(rhs,fieldInds);           // field RHS
    Epetra_SerialDenseMatrix Dinv_rhs;     // scratch storage
    Epetra_SerialDenseMatrix y; // = getSubVector of flux solution y

    f_mod.Multiply('N','N',-1.0,B,y,1.0);
    denseSolver.SetMatrix(D);
    denseSolver.SetVectors(Dinv_rhs,f_mod);
    
    // TODO - set field degrees of freedom
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
