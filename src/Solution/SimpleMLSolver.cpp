//
//  SimpleMLSolver.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/24/14.
//
//

#include "Solver.h"

#include "SimpleMLSolver.h"

using namespace Camellia;

SimpleMLSolver::SimpleMLSolver(bool saveFactorization, double residualTolerance, int maxIterations) {
//  if (saveFactorization) {
//    int rank = Teuchos::GlobalMPISession::getRank();
//    
//    if (rank==0) {
//      cout << "WARNING: SimpleMLSolver may have issues with saveFactorization=true; overriding with false.\n";
//    }
//    saveFactorization = false;
//  }
  _saveFactorization = saveFactorization;
  _resTol = residualTolerance;
  _maxIters = maxIterations;
}
int SimpleMLSolver::solve() {
  Epetra_LinearProblem problem(_stiffnessMatrix.get(), _lhs.get(), _rhs.get());
  AztecOO *solver = new AztecOO( problem );
  // create a parameter list for ML options
  Teuchos::ParameterList MLList;
  
  // Sets default parameters for classic smoothed aggregation. After this
  // call, MLList contains the default values for the ML parameters,
  // as required by typical smoothed aggregation for symmetric systems.
  // Other sets of parameters are available for non-symmetric systems
  // ("DD" and "DD-ML"), and for the Maxwell equations ("maxwell").
  
  //    int maxLevels = 8;
  //    char parameter[80];
  
  ML_Epetra::MultiLevelPreconditioner* MLPrec;
  
  ML_Epetra::SetDefaults("SA",MLList);
  
  Epetra_RowMatrix *A = problem.GetMatrix();
  
//  MLList.set("ML output", 10);
  
  MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);
  
  solver->SetPrecOperator(MLPrec);
  solver->SetAztecOption(AZ_output, 0);
  int result = solver->Iterate(_maxIters, _resTol);
  
  if (! _saveFactorization) {
    // destroy the preconditioner
    delete MLPrec;
    // destroy the solver object
    delete solver;
  } else {
    _savedSolver = Teuchos::rcp( solver );
    _savedPreconditioner = Teuchos::rcp( MLPrec );
  }
  
  return result;
}
int SimpleMLSolver::resolve() {
  if (_savedSolver.get() != NULL) {
    return _savedSolver->Iterate(_maxIters, _resTol);;
  } else {
    return solve();
  }
}
// void SimpleMLSolver::setProblem(Teuchos::RCP< Epetra_LinearProblem > problem) {
//   _savedSolver = Teuchos::rcp((AztecOO*)NULL);
//   this->_problem = problem;
// }