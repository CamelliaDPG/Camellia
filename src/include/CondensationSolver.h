#ifndef Condensation_Solver_h
#define Condensation_Solver_h

#include "Solver.h"
#include "Mesh.h"
#include "ElementType.h"
#include "Solution.h"

using namespace std;

class CondensationSolver : public Solver{
 private:
  // _problem inherited from Solver
  Teuchos::RCP<Mesh> _mesh; 
  Teuchos::RCP<Solution> _solution; // to get the partition map to create a FECrsMatrix
  Teuchos::RCP<Solver> _solver; // to get the partition map to create a FECrsMatrix

  set<int>              _allFluxInds;    // unique set of all flux inds
  map<int,vector<int> > _GlobalFluxInds; // from cellID to globalDofInd vector
  map<int,vector<int> > _GlobalFieldInds;
  map<int,vector<int> > _ElemFluxInds;   // from cellID to localDofInd vector
  map<int,vector<int> > _ElemFieldInds;  
  void getDofIndices();
  Epetra_SerialDenseMatrix getSubMatrix(Epetra_RowMatrix* K, vector<int> rowInds, vector<int> colInds);
  Epetra_SerialDenseMatrix getSubVector(Epetra_MultiVector*  f,vector<int> inds);

 public:  

  CondensationSolver(Teuchos::RCP<Mesh> mesh,Teuchos::RCP<Solution> solution){
    _mesh = mesh;
    _solution = solution;
    _solver = Teuchos::rcp(new KluSolver()); // default to KLU for the reduced system
  }
  CondensationSolver(Teuchos::RCP<Mesh> mesh,Teuchos::RCP<Solution> solution, Teuchos::RCP<Solver> solver){
    _mesh = mesh;
    _solution = solution;
    _solver = solver; // for the reduced system
  }

  int solve();

};

#endif
