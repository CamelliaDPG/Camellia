#ifndef Condensation_Solver_h
#define Condensation_Solver_h

#include "Solver.h"
#include "Mesh.h"
#include "ElementType.h"
#include "Solution.h"

#include <Epetra_Map.h>
#include "Epetra_FECrsMatrix.h"

using namespace std;

class CondensationSolver : public Solver{
 private:
  // _problem inherited from Solver
  Teuchos::RCP<Mesh> _mesh; 
  Teuchos::RCP<Solution> _solution; // to get the partition map to create a FECrsMatrix
  Teuchos::RCP<Solver> _solver; // to get the partition map to create a FECrsMatrix

  // to get from _mesh
  map<int,set<int> > _localFluxInds;   // from cellID to localDofInd vector
  map<int,set<int> > _localFieldInds;  

  // to compute 
  set<int>               _allFluxInds;    // unique set of all flux inds
  set<int>               _condensedFluxInds;    // unique set of all flux inds but using the condensed ordering
  map<int,map<int,int> > _globalToLocalFieldInds; // cellID + globalFieldIndex -> local field index
  map<int,int> _globalToCondensedFluxInds;
  map<int,int> _condensedToGlobalFluxInds;
  map<int,map<int,int> > _localToCondensedFieldInds; // cellID + localFieldIndex -> field index
  map<int,map<int,int> > _condensedToLocalFieldInds; // cellID + fieldIndex -> local field index

  // matrices for static condensation: need to get by iterating through rows of global matrix
  map<int, Epetra_SerialDenseMatrix> _elemFieldMats; // from cellID to dense matrix
  map<int, Epetra_SerialDenseMatrix > _couplingMatrices; // from cellID to dense compacted matrix
  map<int, vector<int> > _couplingIndices; // relating cellID to column indices for that cell


  void init();
  int cellIDForGlobalFieldIndex(int globalFieldIndex);
  void getSubmatrices(const Epetra_RowMatrix* K,Epetra_FECrsMatrix &K_cond);
  /*
  void getSubMatrixData(Epetra_RowMatrix* K);
  void getSubMatrices(Epetra_RowMatrix* K);
  */
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
