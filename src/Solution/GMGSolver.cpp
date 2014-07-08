//
//  GMGSolver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/7/14.
//
//

#include "GMGSolver.h"

GMGSolver::GMGSolver( BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
                     Epetra_Map finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver) : _gmgOperator(zeroBCs,coarseMesh,coarseIP,fineMesh,finePartitionMap,coarseSolver),
                    _finePartitionMap(finePartitionMap) {
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;         
}

void GMGSolver::setPrintToConsole(bool printToConsole) {
  _printToConsole = printToConsole;
}

void GMGSolver::setTolerance(double tol) {
  _tol = tol;
}

int GMGSolver::solve() {
  AztecOO solver(problem());
  
  Epetra_RowMatrix *A = problem().GetMatrix();
  
  Epetra_Vector diagA(A->RowMatrixRowMap());
  A->ExtractDiagonalCopy(diagA);
  
  Teuchos::RCP<Epetra_MultiVector> diagA_ptr = Teuchos::rcp( &diagA, false );
  
  _gmgOperator.setStiffnessDiagonal(diagA_ptr);

//  solver.SetAztecOption(AZ_solver, AZ_cg);
  solver.SetAztecOption(AZ_solver, AZ_gmres);
  solver.SetPrecOperator(&_gmgOperator);
  solver.SetAztecOption(AZ_precond, AZ_user_precond);
  solver.SetAztecOption(AZ_scaling, AZ_none);
  
  int solveResult = solver.Iterate(_maxIters,_tol);
  
  const double* status = solver.GetAztecStatus();
  int whyTerminated = status[AZ_why];
  switch (whyTerminated) {
    case AZ_normal:
      cout << "whyTerminated: AZ_normal " << endl;
      break;
    case AZ_param:
      cout << "whyTerminated: AZ_param " << endl;
      break;
    case AZ_breakdown:
      cout << "whyTerminated: AZ_breakdown " << endl;
      break;
    case AZ_loss:
      cout << "whyTerminated: AZ_loss " << endl;
      break;
    case AZ_ill_cond:
      cout << "whyTerminated: AZ_ill_cond " << endl;
      break;
    case AZ_maxits:
      cout << "whyTerminated: AZ_maxits " << endl;
      break;
    default:
      break;
  }
  
  double norminf = A->NormInf();
  double normone = A->NormOne();
  
  int numIters = solver.NumIters();
  
  if (_printToConsole) {
    cout << "\n Inf-norm of stiffness matrix after scaling = " << norminf;
    cout << "\n One-norm of stiffness matrix after scaling = " << normone << endl << endl;
    cout << "Num iterations: " << numIters << endl;
  }
  
  _gmgOperator.setStiffnessDiagonal(Teuchos::rcp((Epetra_MultiVector*) NULL ));
  
  return solveResult;
}
