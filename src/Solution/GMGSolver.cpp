//
//  GMGSolver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/7/14.
//
//

#include "GMGSolver.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

GMGSolver::GMGSolver( BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
                     Epetra_Map finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver) : _gmgOperator(zeroBCs,coarseMesh,coarseIP,fineMesh,finePartitionMap,coarseSolver),
                    _finePartitionMap(finePartitionMap) {
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
  _diagonalSmoothing = true;
  _azOutput = AZ_warnings;
}

void GMGSolver::setApplySmoothingOperator(bool applySmoothingOp) {
  _diagonalSmoothing = applySmoothingOp;
}

void GMGSolver::setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap) {
  _gmgOperator.setFineMesh(fineMesh, finePartitionMap);
}

void GMGSolver::setPrintToConsole(bool printToConsole) {
  _printToConsole = printToConsole;
}

void GMGSolver::setTolerance(double tol) {
  _tol = tol;
}

int GMGSolver::solve() {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  AztecOO solver(problem());
  
  Epetra_RowMatrix *A = problem().GetMatrix();
//  Epetra_MultiVector *b = problem().GetRHS();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/b_gmg.dat",*b);

  Epetra_Vector diagA(A->RowMatrixRowMap());
  A->ExtractDiagonalCopy(diagA);
  
  Teuchos::RCP<Epetra_MultiVector> diagA_ptr = Teuchos::rcp( &diagA, false );
  
  if (_diagonalSmoothing)
    _gmgOperator.setStiffnessDiagonal(diagA_ptr);

  solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
//  solver.SetAztecOption(AZ_solver, AZ_gmres);
  solver.SetPrecOperator(&_gmgOperator);
//  solver.SetAztecOption(AZ_precond, AZ_none);
  solver.SetAztecOption(AZ_precond, AZ_user_precond);
  solver.SetAztecOption(AZ_scaling, AZ_none);
  solver.SetAztecOption(AZ_conv, AZ_rhs);
//  solver.SetAztecOption(AZ_output, AZ_last);
  solver.SetAztecOption(AZ_output, _azOutput);
  
  int solveResult = solver.Iterate(_maxIters,_tol);
  
  const double* status = solver.GetAztecStatus();
  int whyTerminated = status[AZ_why];
  if (rank==0) {
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
  }
  
//  Epetra_MultiVector *x = problem().GetLHS();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/x.dat",*x);
  
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

void GMGSolver::setAztecOutput(int value) {
  _azOutput = value;
}
