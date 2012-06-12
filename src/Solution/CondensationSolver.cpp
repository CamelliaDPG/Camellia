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
    vector< ElementPtr > elemsOfType = _mesh->elementsOfType(rank, elemTypePtr); // all elems of type on this processor
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
      _GlobalFieldInds[cellID] = globalFieldInds; // global over all dofs and partitions
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
  int numGlobalFluxDofs = _allFluxInds.size();
  set<int> globalIndsForPartition = _mesh->globalDofIndicesForPartition(rank); 
  // TODO: REMOVE field inds from above set

  Epetra_Map partMap = _solution->getPartitionMap(rank, globalIndsForPartition, numGlobalFluxDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints 
  Epetra_FECrsMatrix Aflux(Copy, partMap, numGlobalFluxDofs); // reduced matrix - soon to be schur complement
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
    int numFluxDofs = fluxInds.size();

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
    A.Multiply('T','N',-1.0,B,Dinv_B,1.0); // A := A - 1.0 * B^T*C, and C = inv(D)*B
    
    FieldContainer<int> fluxIndsFC(numFluxDofs);
    for (int i = 0;i<numFluxDofs;++i){
      fluxIndsFC(i) = fluxInds[i];
    }

    // NOTE: THIS IS NOT THE DISTRIBUTED VERSION
    //    Aflux.SumIntoGlobalValues(numFluxDofs,&fluxIndsFC(0),numDofs,&fluxIndsFC(0), A);
    //    bflux.SumIntoGlobalValues(numFluxDofs,&fluxIndsFC(0), g);
  }

  Aflux.GlobalAssemble();
  bflux.GlobalAssemble();
  Epetra_FEVector lhsVector(partMap, true);
  Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&Aflux, &lhsVector, &bflux)); 

  _solver->setProblem(problem);
  int solveSuccess = _solver->solve(); // solve for the flux dofs 

  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _GlobalFluxInds[cellID];
    vector<int> fieldInds = _GlobalFieldInds[cellID];
    int numFieldDofs = fieldInds.size();
    int numFluxDofs = fluxInds.size();

    // [D   B][fieldDofs] = [f]
    // [B^T A][flux Dofs] = [g]
    Epetra_SerialDenseMatrix D = getSubMatrix(stiffMat,fieldInds,fieldInds); // block field submatrix
    Epetra_SerialDenseMatrix B = getSubMatrix(stiffMat,fieldInds,fluxInds);  // field/flux submat
    Epetra_SerialDenseMatrix f_mod = getSubVector(rhs,fieldInds);           // field RHS
    Epetra_SerialDenseMatrix Dinv_rhs(numFieldDofs,1);     // scratch storage for field solutions
    Epetra_SerialDenseMatrix y(numFluxDofs,1); // = getSubVector of flux solution y

    f_mod.Multiply('N','N',-1.0,B,y,1.0); // x := inv(D)*(f - B*y)
    denseSolver.SetMatrix(D);
    denseSolver.SetVectors(Dinv_rhs,f_mod); 
    
    FieldContainer<double> fieldDofs(numFieldDofs);
    for (unsigned int i=0;i<numFieldDofs;++i){
      fieldDofs(i) = Dinv_rhs(i,1); 
    }
    
    vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
    FieldContainer<double> solnCoeffsToSet;
    for (int trialID = 0;trialID<trialIDs.size();++trialID){
      // TODO - set field degrees of freedom in lhsVector
      _solution->setSolnCoeffsForCellID(solnCoeffsToSet, cellID, trialID, 0);
    }
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


