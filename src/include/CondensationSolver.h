#ifndef Condensation_Solver_h
#define Condensation_Solver_h

#include "Solver.h"
#include "Mesh.h"
#include "ElementType.h"
#include "Solution.h"

#include <Epetra_Map.h>

using namespace std;

class CondensationSolver : public Solver{
 private:
  // _problem inherited from Solver
  Teuchos::RCP<Mesh> _mesh; 
  Teuchos::RCP<Solution> _solution; // to get the partition map to create a FECrsMatrix
  Teuchos::RCP<Solver> _solver; // to get the partition map to create a FECrsMatrix

  set<int>              _allFluxInds;    // unique set of all flux inds
  map<int,vector<int> > _globalFluxInds; // from cellID to globalDofInd vector
  map<int,vector<int> > _globalFieldInds;
  map<int,vector<int> > _localFluxInds;   // from cellID to localDofInd vector
  map<int,vector<int> > _localFieldInds;  

  void getElemSubMatrices(int cellID, Epetra_RowMatrix* K, Epetra_SerialDenseMatrix A,Epetra_SerialDenseMatrix B,Epetra_SerialDenseMatrix D);
  Epetra_SerialDenseMatrix getSubVector(Epetra_MultiVector*  f,vector<int> inds);

 public:  
  
  CondensationSolver(Teuchos::RCP<Mesh> mesh,Teuchos::RCP<Solution> solution){
    _mesh = mesh;
    _solution = solution;
    _solver = Teuchos::rcp(new KluSolver()); // default to KLU for the reduced system for now
  }
  CondensationSolver(Teuchos::RCP<Mesh> mesh,Teuchos::RCP<Solution> solution, Teuchos::RCP<Solver> solver){
    _mesh = mesh;
    _solution = solution;
    _solver = solver; // for the reduced system
  }
  
  int solve();

  //  Epetra_Map getFluxMap(int rank, Epetra_Comm* Comm);

};

#endif
