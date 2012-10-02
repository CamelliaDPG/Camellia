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

  //  setTimeResults(true);

  Epetra_Map timeMap(numProcs,0,Comm);
  Epetra_Time timer(Comm);
  Epetra_Time timerTotal(Comm);

  Epetra_RowMatrix* stiffMat = problem().GetMatrix();
  Epetra_MultiVector* rhs = problem().GetRHS();
  Epetra_MultiVector* lhs = problem().GetLHS(); 

  if (_timeResults){
    timer.ResetStartTime();
  }

  init();

  if (_timeResults){
    cout << "time on PID " << rank << " for initializing = " << timer.ElapsedTime() << endl;
    timer.ResetStartTime();
  }

  // do stuff to get flux/field dofs/maps/etc
  // ------------------

  // all (including field) inds
  set<int> globalIndsForPartition = _mesh->globalDofIndicesForPartition(rank); 
  int numGlobalDofs = globalIndsForPartition.size();

  int numGlobalFluxDofs = _allFluxInds.size();

  //  Epetra_Map fluxPartMap = _solution->getPartitionMap(rank, _condensedFluxInds, numGlobalFluxDofs, 0, &Comm); // 0 = assumes no zero mean/Lagrange constraints // WARNING: DOES NOT WORK. WHY?
  
  Epetra_Map fluxPartMap(numGlobalFluxDofs,0,Comm);

  int maxNnz = min(_mesh->condensedRowSizeUpperBound(),numGlobalFluxDofs);

  Epetra_FECrsMatrix K_cond(Copy, fluxPartMap, maxNnz); // condensed matrix - to be schur complement
  Epetra_FEVector rhs_cond(fluxPartMap);
  Epetra_FEVector lhs_cond(fluxPartMap, true);

  if (_timeResults){
    timer.ResetStartTime();
  }
  getCondensedData(stiffMat,rhs,K_cond,rhs_cond);
  if (_timeResults){
    cout << "time on PID " << rank << " for getting condensed data = " << timer.ElapsedTime() << endl;
    timer.ResetStartTime();
  }

  Teuchos::RCP<Epetra_LinearProblem> problem_cond = Teuchos::rcp( new Epetra_LinearProblem(&K_cond, &lhs_cond, &rhs_cond));
  
  _solver->setProblem(problem_cond);

  int solveSuccess = _solver->solve();

  if (_timeResults){
    cout << "time on PID " << rank << " for condensed solve = " << timer.ElapsedTime() << endl;
  }

  lhs_cond.GlobalAssemble();  

  //  EpetraExt::MultiVectorToMatrixMarketFile("lhs_cond.dat",lhs_cond,0,0,false);

  recoverAndStoreFieldDofs(K_cond,lhs,lhs_cond);

  if (_timeResults){
    cout << "total time on PID " << rank << " spent in solve() = " << timerTotal.ElapsedTime() << endl;
  }
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
#else
  Epetra_SerialComm Comm;
#endif

  Epetra_Map timeMap(numProcs,0,Comm);
  Epetra_Time timer(Comm);
  Epetra_Time timerTotal(Comm);

  _mesh->getFieldFluxDofInds(_localFluxInds, _localFieldInds);
  _mesh->getGlobalFieldFluxDofInds(_globalFluxInds, _globalFieldInds);

  vector< ElementPtr > elems = _mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;

  // compute set of all global flux dof indices, compute global-local map
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   

    // fluxes
    set<int> cellFluxInds = _globalFluxInds[cellID];
    set<int>::iterator fluxIt;
    for (fluxIt = cellFluxInds.begin();fluxIt!=cellFluxInds.end();fluxIt++){
      int globalFluxInd = *fluxIt;
      _allFluxInds.insert(globalFluxInd);
    }

    // fields
    set<int> cellFieldInds = _globalFieldInds[cellID];
    set<int>::iterator fieldIt;
    int fieldCount = 0;
    for (fieldIt=cellFieldInds.begin();fieldIt!=cellFieldInds.end();fieldIt++){
      int globalFieldIndex = *fieldIt;
      _globalToCondensedFieldInds[cellID][globalFieldIndex] = fieldCount;
      _globalFieldIndToCellID[globalFieldIndex] = cellID;
      fieldCount++;
    }
  }

  // create maps between global flux inds and condensed matrix inds
  set<int>::iterator fluxIt;
  int count = 0;
  for (fluxIt = _allFluxInds.begin();fluxIt!= _allFluxInds.end();fluxIt++){
    int ind = *fluxIt;
    _globalToCondensedFluxInds[ind] = count;
    _condensedToGlobalFluxInds[count] = ind;
    _condensedFluxInds.insert(count);
    count++;
  }

  // allocate space for elem field matrices
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    int numFieldInds = _globalFieldInds[cellID].size();
    Epetra_SerialDenseMatrix K_elem(numFieldInds,numFieldInds);
    _elemFieldMats[cellID] = K_elem;
  }

  // figure out nRHS for solves (nonzero coupling terms)
  Epetra_RowMatrix* K = problem().GetMatrix();
  int numMyCols = K->RowMatrixColMap().NumMyElements();
  int numGlobalRowElements = K->RowMatrixRowMap().NumGlobalElements(); // number of rows on this proc 
  // get inds and values of row
  int * globalColInds = K->RowMatrixColMap().MyGlobalElements();
  int * globalRowInds = K->RowMatrixRowMap().MyGlobalElements();

  int numMyRows = K->RowMatrixRowMap().NumMyElements(); // number of rows stored on this proc
  for (int i = 0;i<numMyRows;++i){
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
      // coupling terms - if row is field, col is flux (use symmetry o/w)
      if (!rowIsFlux && colIsFlux){
	//        int cellID = cellIDForGlobalFieldIndex(rowInd);
	int cellID = _globalFieldIndToCellID[rowInd];
        _elemCouplingInds[cellID].insert(colInd);
      }
    }
  }
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    int numRHS = _elemCouplingInds[cellID].size();
    int numFieldDofs = _localFieldInds[cellID].size();
    Epetra_SerialDenseMatrix elemCouplingMat(numFieldDofs,numRHS);
    _couplingMatrices[cellID] = elemCouplingMat;
    // TODO - add spot for creation of local RHS terms
    Epetra_SerialDenseVector elemFieldRHS(numFieldDofs);
    _fieldRHS[cellID] = elemFieldRHS;
  }
  
  // create local maps between flux inds and coupling matrices
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    set<int> couplingInds = _elemCouplingInds[cellID];
    int counter = 0;  
    for (fluxIt = couplingInds.begin();fluxIt!=couplingInds.end();fluxIt++){      
      int globalFluxInd = *fluxIt;
      _globalToReducedFluxInds[cellID][globalFluxInd] = counter;
      _reducedFluxToGlobalInds[cellID][counter] = globalFluxInd;
      counter++;
    }
  }  
}

void CondensationSolver::getCondensedData(const Epetra_RowMatrix* K, const Epetra_MultiVector* rhs, Epetra_FECrsMatrix &K_cond, Epetra_FEVector &rhs_cond){
  //getCondensedData(const Epetra_RowMatrix* K,Epetra_FECrsMatrix &K_cond){
  
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
  
  int * globalColInds = K->RowMatrixColMap().MyGlobalElements();
  int numMyCols = K->RowMatrixColMap().NumMyElements();
  int numGlobalRowElements = K->RowMatrixRowMap().NumGlobalElements(); // number of rows on this proc 
  int * globalRowInds = K->RowMatrixRowMap().MyGlobalElements();   // get inds and values of row
  
  int numMyRows = K->RowMatrixRowMap().NumMyElements(); // number of rows stored on this proc

  // import RHS onto every proc (INEFFICIENT: IS THERE ANOTHER WAY TO ACCESS NON-LOCAL DATA?)
  Epetra_Map partMap = K->RowMatrixRowMap();
  int numNodesGlobal = partMap.NumGlobalElements();
  Epetra_Map     rhsMap(numNodesGlobal, numNodesGlobal, 0, Comm);
  Epetra_Import  rhsImporter(rhsMap, partMap);
  Epetra_Vector  rhsImport(rhsMap);
  rhsImport.Import((*rhs), rhsImporter, Insert);
  
  for (int i = 0;i<numMyRows;++i){

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

      // if the row index is a field index
      if (!rowIsFlux){
	//	int cellID = cellIDForGlobalFieldIndex(rowInd);
	int cellID = _globalFieldIndToCellID[rowInd];
	if (cellID == -1){
	  cout << "Problem: returned negative cellID on rowInd = " << rowInd << endl;
	}

	int condensedFieldRowInd = _globalToCondensedFieldInds[cellID][rowInd]; 

	if (!colIsFlux){ // if it's a field col index too, should have to a decoupled element matrix
	  int condensedFieldColInd = _globalToCondensedFieldInds[cellID][colInd]; 
	  _elemFieldMats[cellID](condensedFieldRowInd,condensedFieldColInd) = value;
	} else { // if it's a coupling term, store into coupling matrices	  	  
	  int localReducedColInd = _globalToReducedFluxInds[cellID][colInd];
	  _couplingMatrices[cellID](condensedFieldRowInd,localReducedColInd) = value;
	}      
      }

      // if both are flux inds, use globalToCondensedInds map to sum into K_cond
      if (rowIsFlux && colIsFlux){
	int condensedFluxRowInd = _globalToCondensedFluxInds[rowInd];
	int condensedFluxColInd = _globalToCondensedFluxInds[colInd];
	K_cond.InsertGlobalValues(1,&condensedFluxRowInd,1,&condensedFluxColInd,&value);
      }      
    } // end of col loop


    // get RHS terms as well
    //    double value = (*rhs)[0][rowInd]; 
    double value = rhsImport[rowInd];   
    if (!rowIsFlux){	
      int cellID = _globalFieldIndToCellID[rowInd];
      int condensedFieldRowInd = _globalToCondensedFieldInds[cellID][rowInd]; 
      _fieldRHS[cellID][condensedFieldRowInd] = value;
    }else{ // if it's a flux row, just take and put into rhs_cond
      int condensedFluxRowInd = _globalToCondensedFluxInds[rowInd];
      if (condensedFluxRowInd==75){
	cout << "on PID " << rank << ", condRowInd = " << condensedFluxRowInd << " and value = " << value << endl;
      }
      rhs_cond.ReplaceGlobalValues(1,&condensedFluxRowInd,&value);
    }   

  } // end of row loop

  K_cond.GlobalAssemble(); // needed to redistribute data before modifying matrix
  rhs_cond.GlobalAssemble();

  EpetraExt::RowMatrixToMatlabFile("K_flux.dat",K_cond);     
  EpetraExt::MultiVectorToMatrixMarketFile("rhs_flux.dat",rhs_cond,0,0,false);

  // condense out matrices
  Epetra_SerialDenseSolver solver;

  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    Epetra_SerialDenseMatrix D = _elemFieldMats[cellID];
    Epetra_SerialDenseMatrix B = _couplingMatrices[cellID];    
    Epetra_SerialDenseMatrix Bcopy = B;

    int numElemFieldDofs = _globalFieldInds[cellID].size();
    set<int> fluxInds = _globalFluxInds[cellID];
    int numElemFluxDofs = fluxInds.size();

    Epetra_SerialDenseMatrix DinvB(numElemFieldDofs,numElemFluxDofs);
    Epetra_SerialDenseMatrix BtDinvB(numElemFluxDofs,numElemFluxDofs);

    // Local solve
    solver.SetMatrix(D);
    solver.SetVectors(DinvB,Bcopy);
    bool equilibrated = false;
    if ( solver.ShouldEquilibrate() ) {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }    
    solver.Solve();    
    if (equilibrated) 
      solver.UnequilibrateLHS();   

    BtDinvB.Multiply('T','N',-1.0,B,DinvB,0.0);

    Epetra_IntSerialDenseVector inds; 
    inds.Size(numElemFluxDofs);
    int count=0;
    set<int>::iterator fluxIt;   
    for (fluxIt = fluxInds.begin();fluxIt!=fluxInds.end();fluxIt++){
      int globalInd = *fluxIt;
      int condensedInd = _globalToCondensedFluxInds[globalInd];      
      inds[count] = condensedInd;
      count++;
    }

    K_cond.SumIntoGlobalValues(inds,BtDinvB,Epetra_FECrsMatrix::COLUMN_MAJOR);  
    
    // modify RHS vector    
    Epetra_SerialDenseVector Dinvf(numElemFieldDofs);
    Epetra_SerialDenseVector f = _fieldRHS[cellID];    
    Epetra_SerialDenseVector BtDinvf(numElemFluxDofs);

    solver.SetVectors(Dinvf,f);
    equilibrated = false;
    if ( solver.ShouldEquilibrate() ) {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }    
    solver.Solve();    
    if (equilibrated)
      solver.UnequilibrateLHS();

    BtDinvf.Multiply('T','N',-1.0,B,Dinvf,0.0);
    
    rhs_cond.SumIntoGlobalValues(inds,BtDinvf);
    
  }    

  // finish by assembling condensed matrix
  K_cond.GlobalAssemble();
  rhs_cond.GlobalAssemble();

  EpetraExt::RowMatrixToMatlabFile("K_cond.dat",K_cond);  
  EpetraExt::MultiVectorToMatrixMarketFile("rhs_cond.dat",rhs_cond,0,0,false);

} 

// assumes init() has already been called.
void CondensationSolver::recoverAndStoreFieldDofs(Epetra_FECrsMatrix &K_cond, Epetra_MultiVector* lhs, Epetra_FEVector &lhs_cond){
  
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
  
  Epetra_Map partMap = K_cond.RowMatrixRowMap();
  int numNodesGlobal = partMap.NumGlobalElements();
  Epetra_Map     lhsMap(numNodesGlobal, numNodesGlobal, 0, Comm);
  Epetra_Import  lhsImporter(lhsMap, partMap);
  Epetra_Vector  lhsImport(lhsMap);
  lhsImport.Import(lhs_cond, lhsImporter, Insert); 

  // redundant across processors - store all flux dofs.
  set<int>::iterator fluxIt;
  for (fluxIt = _allFluxInds.begin();fluxIt!=_allFluxInds.end();fluxIt++){
    int globalInd = *fluxIt;
    int condensedInd = _globalToCondensedFluxInds[globalInd];
    double value = lhsImport[condensedInd];      
    lhs->ReplaceGlobalValue(globalInd,0,value);
  }

  // recover field dofs
  Epetra_SerialDenseSolver solver;
  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  //  vector< ElementPtr > elems = _mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;
  for (elemIt = elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();   
    Epetra_SerialDenseMatrix D = _elemFieldMats[cellID];
    Epetra_SerialDenseMatrix B = _couplingMatrices[cellID]; // (field dofs) x (flux dofs)

    set<int>::iterator cplIt;
    set<int> fluxInds = _globalFluxInds[cellID];
    set<int> fieldInds = _globalFieldInds[cellID];
    int numElemFluxDofs = fluxInds.size();
    int numElemFieldDofs = fieldInds.size();

    // create "reduced" solution vector (corresponding to nonzeros of coupling matrix)
    set<int> couplingInds = _elemCouplingInds[cellID]; // set of nonzero flux indices
    Epetra_SerialDenseVector fluxDofs(couplingInds.size());
    for (cplIt=couplingInds.begin();cplIt!=couplingInds.end();cplIt++){
      int globalInd = (*cplIt);
      int reducedInd = _globalToReducedFluxInds[cellID][globalInd];
      int condensedInd = _globalToCondensedFluxInds[globalInd];
      fluxDofs[reducedInd] = lhsImport[condensedInd];
    }
    
    Epetra_SerialDenseVector f = _fieldRHS[cellID]; 

    f.Multiply('N','N',-1.0,B,fluxDofs,1.0); // (f-B*y)
    
    // Local solve for field dofs
    Epetra_SerialDenseVector fieldDofs(numElemFieldDofs);
    solver.SetMatrix(D);
    solver.SetVectors(fieldDofs,f);
    bool equilibrated = false;
    if ( solver.ShouldEquilibrate() ) {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }    
    solver.Solve();    
    if (equilibrated) 
      solver.UnequilibrateLHS();   
    
    // store field dofs
    set<int>::iterator fieldIt;
    for (fieldIt = fieldInds.begin();fieldIt!=fieldInds.end();fieldIt++){
      int globalInd = *fieldIt;
      int condensedInd = _globalToCondensedFieldInds[cellID][globalInd];
      double value = fieldDofs[condensedInd];
      lhs->ReplaceGlobalValue(globalInd,0,value);
    }
  } 
}

void CondensationSolver::writeFieldFluxIndsToFile(){

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

  // determine trialIDs
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  vector< int > fieldIDs;
  vector< int > fluxIDs;
  vector< int >::iterator idIt;

  for (idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){
    int trialID = *(idIt);
    if (!_mesh->bilinearForm()->isFluxOrTrace(trialID)){ // if field
      fieldIDs.push_back(trialID);
    } else {
      fluxIDs.push_back(trialID);
    }
  } 
  int numFieldInds = 0;
  map<int,vector<int> > globalFluxInds;   // from cellID to localDofInd vector
  map<int,vector<int> > globalFieldInds;   // from cellID to localDofInd vector
  map<int,vector<int> > localFieldInds;   // from cellID to localDofInd vector
  map<int,vector<int> > localFluxInds;   // from cellID to localDofInd vector
  set<int>              allFluxInds;    // unique set of all flux inds

  _mesh->getDofIndices(allFluxInds,globalFluxInds,globalFieldInds,localFluxInds,localFieldInds);

  if (rank==0){

    vector< ElementPtr > activeElems = _mesh->activeElements();
    vector< ElementPtr >::iterator elemIt;

    cout << "num flux dofs = " << allFluxInds.size() << endl;
    cout << "num field dofs = " << _mesh->numFieldDofs() << endl;
    cout << "num flux dofs = " << _mesh->numFluxDofs() << endl;
    elemIt = activeElems.begin();
    int cellID = (*elemIt)->cellID();
    cout << "num LOCAL field dofs = " << localFieldInds[cellID].size() << endl;
  
    ofstream fieldInds; 
    fieldInds.open("fieldInds.dat");
    for (elemIt = activeElems.begin();elemIt!=activeElems.end();elemIt++){
      int cellID = (*elemIt)->cellID();
      vector<int> inds = globalFieldInds[cellID];
      for (int i = 0;i<inds.size();++i){
	fieldInds << inds[i]+1 << endl;
      }
    }
    fieldInds.close();

    ofstream fluxInds;
    fluxInds.open("fluxInds.dat");
    set<int>::iterator fluxIt;
    for (fluxIt = allFluxInds.begin();fluxIt!=allFluxInds.end();fluxIt++){
      fluxInds << (*fluxIt)+1 << endl; // offset by 1 for matlab
    }
    fluxInds.close();
  }
}
