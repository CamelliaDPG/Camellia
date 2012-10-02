#ifndef Condensation_Solver_h
#define Condensation_Solver_h

#include "Solver.h"
#include "Mesh.h"
#include "ElementType.h"
#include "Solution.h"

#include <Epetra_Map.h>
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"

using namespace std;

class CondensationSolver : public Solver{
 private:

  bool _timeResults;

  // _problem inherited from Solver
  Teuchos::RCP<Mesh> _mesh; 
  Teuchos::RCP<Solution> _solution; // to get the partition map to create a FECrsMatrix
  Teuchos::RCP<Solver> _solver; // to get the partition map to create a FECrsMatrix

  // to get from _mesh
  map<int,set<int> > _globalFluxInds;   // from cellID to localDofInd vector
  map<int,set<int> > _globalFieldInds;  
  // CRUFT
  map<int,set<int> > _localFluxInds;   // from cellID to localDofInd vector
  map<int,set<int> > _localFieldInds;  

  // to compute 
  set<int>               _allFluxInds;    // unique set of all global flux inds
  set<int>               _condensedFluxInds;    // unique set of all flux inds but using the condensed ordering

  map<int,int> _globalFieldIndToCellID; // lookup table
  
  map<int,map<int,int> > _globalToCondensedFieldInds; // cellID + globalFieldIndex -> local field index
  map<int,int> _globalToCondensedFluxInds;
  map<int,int> _condensedToGlobalFluxInds;
  //  map<int,map<int,int> > _localToCondensedFieldInds; // cellID + localFieldIndex -> field index
  //  map<int,map<int,int> > _condensedToLocalFieldInds; // cellID + fieldIndex -> local field index
  

  // matrices for static condensation: need to get by iterating through rows of global matrix
  map<int, Epetra_SerialDenseMatrix> _elemFieldMats; // from cellID to dense matrix  

  map<int,map<int,int> > _globalToReducedFluxInds; // from cellID/global flux ind to local coupling matrix ind (i.e. index into # of nonzero columns of elem coupling matrix)
  map<int,map<int,int> > _reducedFluxToGlobalInds; // vice versa of above

  map<int, set<int> > _elemCouplingInds; // map from cellID to set of nonzero global flux indices
  map<int, Epetra_SerialDenseMatrix > _couplingMatrices; // from cellID to dense compacted matrix
  map<int, Epetra_SerialDenseVector > _fieldRHS; // from cellID to dense compacted matrix


  void init();
  void getCondensedData(const Epetra_RowMatrix* K, const Epetra_MultiVector* rhs, Epetra_FECrsMatrix &K_cond, Epetra_FEVector &rhs_cond);
  void recoverAndStoreFieldDofs(Epetra_FECrsMatrix &K_cond, Epetra_MultiVector* lhs, Epetra_FEVector &lhs_cond);
  /*
  void getSubMatrixData(Epetra_RowMatrix* K);
  void getSubMatrices(Epetra_RowMatrix* K);
  */
 public:  
  
  CondensationSolver(Teuchos::RCP<Mesh> mesh,Teuchos::RCP<Solution> solution){
    _mesh = mesh;
    _solution = solution;
    _solver = Teuchos::rcp(new KluSolver()); // default to KLU for the reduced system for now
    _timeResults = false;
  }
  CondensationSolver(Teuchos::RCP<Mesh> mesh,Teuchos::RCP<Solution> solution, Teuchos::RCP<Solver> solver){
    _mesh = mesh;
    _solution = solution;
    _solver = solver; // for the reduced system
    _timeResults = false;
  }
  
  void setTimeResults(bool value){
    _timeResults = value;
  }
  int solve();
  void writeFieldFluxIndsToFile();
};

#endif
