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
  
  _mesh->getDofIndices(_allFluxInds,_globalFluxInds,_globalFieldInds,_localFluxInds,_localFieldInds); // get dof inds

  Epetra_RowMatrix* stiffMat = problem().GetMatrix();
  Epetra_MultiVector* rhs = problem().GetRHS();
  Epetra_MultiVector* lhs = problem().GetLHS();

  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;

  // create reduced matrix inds
  set<int> globalIndsForPartition = _mesh->globalDofIndicesForPartition(rank); 
  int numGlobalDofs = globalIndsForPartition.size();

  // remove field inds from above set
  set<int> globalFluxIndsForPartition;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _globalFluxInds[cellID];
    vector<int>::iterator indIt;
    for (indIt = fluxInds.begin(); indIt != fluxInds.end(); indIt++){
      int ind = (*indIt);
      const bool isInPartition = globalIndsForPartition.find(ind) != globalIndsForPartition.end();
      if (isInPartition){
	globalFluxIndsForPartition.insert(ind);
      }
    }
  } 
  int numGlobalFluxDofs = _allFluxInds.size();

  // global map to reproduce the solution map
  Epetra_Map globalPartMap = _solution->getPartitionMap(rank, globalIndsForPartition, numGlobalDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints 

  Epetra_Map fluxPartMap = _solution->getPartitionMap(rank, globalFluxIndsForPartition, numGlobalFluxDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints 

  //Epetra_Map fluxPartMap = getFluxMap(rank, &Comm); // NEED TO DEFINE

  Epetra_FECrsMatrix Aflux(Copy, fluxPartMap, numGlobalFluxDofs); // reduced matrix - soon to be schur complement
  Epetra_FEVector bflux(fluxPartMap);
  Epetra_FEVector lhsVector(fluxPartMap, true);

  // get dense solver for distributed portion
  Epetra_SerialDenseSolver denseSolver;

  // modify stiffness/load vector
  for (elemIt=elems.begin(); elemIt!=elems.end(); elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _globalFluxInds[cellID];
    vector<int> fieldInds = _globalFieldInds[cellID];

    // [D   B][fieldDofs] = [f]
    // [B^T A][flux Dofs] = [g]
    Epetra_SerialDenseMatrix D,B,A;
    getElemSubMatrices(cellID,stiffMat,A,B,D); 
 
    Epetra_SerialDenseMatrix f = getSubVector(rhs,fieldInds);           // field RHS
    Epetra_SerialDenseMatrix g = getSubVector(rhs,fluxInds);           // flux RHS -> will be reduced
    Epetra_SerialDenseMatrix Dinv_f;     // scratch storage
    Epetra_SerialDenseMatrix Dinv_B;     // scratch storage

    denseSolver.SetMatrix(D);

    // modify load vector
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
    g.Multiply('T','N',-1.0,B,Dinv_f,1.0); // g := g - 1.0 * B^T * inv(D) * f
    
    // modify stiffness matrix 
    denseSolver.SetVectors(Dinv_B,B);
    denseSolver.Solve();    
    A.Multiply('T','N',-1.0,B,Dinv_B,1.0); // A := A - 1.0 * B^T * inv(D) * B

    int numFluxDofs = fluxInds.size();
    FieldContainer<int> fluxIndsFC(numFluxDofs);
    for (int i = 0;i<numFluxDofs;++i){
      fluxIndsFC(i) = fluxInds[i];
    }

    // NOTE: THIS IS NOT THE DISTRIBUTED VERSION
    Aflux.SumIntoGlobalValues(numFluxDofs,&fluxIndsFC(0),numFluxDofs,&fluxIndsFC(0), &A(0,0));
    bflux.SumIntoGlobalValues(numFluxDofs,&fluxIndsFC(0), &g(0,0));
  }

  Aflux.GlobalAssemble();
  bflux.GlobalAssemble();  

  Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&Aflux, &lhsVector, &bflux)); 
  _solver->setProblem(problem);
  int solveSuccess = _solver->solve(); // solve for the flux dofs 
  if (solveSuccess!=0){
    cout << "Warning: solve status = " << solveSuccess << " in CondensationSolver.";
  }

  // element-wise local computations to recover field dofs and store flux dofs
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*(elemIt))->cellID();
    vector<int> fluxInds = _globalFluxInds[cellID];
    vector<int> fieldInds = _globalFieldInds[cellID];
    int numFieldDofs = fieldInds.size();
    int numFluxDofs = fluxInds.size();

    // [D   B][fieldDofs] = [f]
    // [B^T A][flux Dofs] = [g]
    Epetra_SerialDenseMatrix A,B,D;
    getElemSubMatrices(cellID,stiffMat,A,B,D);

    Epetra_SerialDenseMatrix f_mod = getSubVector(rhs,fieldInds);           // field RHS
    Epetra_SerialDenseMatrix Dinv_rhs(numFieldDofs,1);     // scratch storage for field solutions
    Epetra_SerialDenseMatrix y = getSubVector(&lhsVector,fluxInds); // sub vector of flux solution y in lhsVector

    f_mod.Multiply('N','N',-1.0,B,y,1.0); // x := inv(D)*(f - B*y)
    denseSolver.SetMatrix(D);
    denseSolver.SetVectors(Dinv_rhs,f_mod); 
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

    FieldContainer<double> fieldDofs(numFieldDofs);
    FieldContainer<int> FCFieldInds(numFieldDofs);
    for (unsigned int i=0;i<numFieldDofs;++i){
      FCFieldInds(i) = _globalFieldInds[cellID][i];
      fieldDofs(i) = Dinv_rhs(i,1); 
      if (rhs->NumVectors()>1){
	cout << "more than 1 rhs not supported at the moment. Solving only on the first RHS." << endl;
      }
      lhs->ReplaceGlobalValue(_globalFieldInds[cellID][i],0,Dinv_rhs(i,0));
    }    
    // store flux dofs
    for (unsigned int i=0;i<numFluxDofs;++i){
      lhs->ReplaceGlobalValue(_globalFluxInds[cellID][i],0,y(i,0));
    }
  }

  return 0;
}


// InsertGlobalValues to create the A flux submatrix (just go thru all partitions and check if the dofs are in the partition map)
// for rectangular matrix B, should just need field dofs (and "get rows" will cover the flux dofs due to storage of Epetra_RowMatrix)
// for block diagonal D, should just need field dofs
void CondensationSolver::getElemSubMatrices(int cellID, Epetra_RowMatrix* K, Epetra_SerialDenseMatrix A,Epetra_SerialDenseMatrix B,Epetra_SerialDenseMatrix D){
  vector<int> fieldInds = _localFieldInds[cellID];
  vector<int> fluxInds = _localFluxInds[cellID];
  int numFieldDof = fieldInds.size();
  int numFluxDof = fluxInds.size();
  D.Shape(numFieldDof,numFieldDof);
  A.Shape(numFluxDof,numFluxDof);
  B.Shape(numFieldDof,numFluxDof);
  
  
}

Epetra_SerialDenseMatrix CondensationSolver::getSubVector(Epetra_MultiVector*  f,vector<int> inds){
  Epetra_SerialDenseMatrix b;  
  int n = inds.size();
  b.Shape(n,1);
 
  
  return b;
}


