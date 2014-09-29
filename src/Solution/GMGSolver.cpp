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

const bool DIAGONAL_SCALING_DEFAULT = true;

GMGSolver::GMGSolver( BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                     MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
                     int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
                     _gmgOperator(zeroBCs,coarseMesh,coarseIP,fineMesh,fineDofInterpreter,
                                  finePartitionMap,coarseSolver, useStaticCondensation, DIAGONAL_SCALING_DEFAULT),
                    _finePartitionMap(finePartitionMap) {
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
  _diagonalSmoothing = true;
  _diagonalScaling = DIAGONAL_SCALING_DEFAULT;
                      
  _computeCondest = true;
  _azOutput = AZ_warnings;
}

void GMGSolver::setApplySmoothingOperator(bool applySmoothingOp) {
  _diagonalSmoothing = applySmoothingOp;
  _gmgOperator.setApplyDiagonalSmoothing(_diagonalSmoothing);
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
  
  bool useAztecToScaleDiagonally = true;
  
  AztecOO solver(problem());
  
  Epetra_RowMatrix *A = problem().GetMatrix();
  
//  EpetraExt::RowMatrixToMatlabFile("/tmp/A_pre_scaling.dat",*A);

//  Epetra_MultiVector *b = problem().GetRHS();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/b_pre_scaling.dat",*b);

//  Epetra_MultiVector *x = problem().GetLHS();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/x_initial_guess.dat",*x);
  
  const Epetra_Map* map = &A->RowMatrixRowMap();
  
  Epetra_Vector diagA(*map);
  A->ExtractDiagonalCopy(diagA);

  // in place of doing the scaling ourselves, for the moment I've switched
  // over to using Aztec's built-in scaling.  Not sure this is any different.
//  EpetraExt::MultiVectorToMatlabFile("/tmp/diagA.dat",diagA);
//  
  Epetra_Vector scale_vector(*map);
  Epetra_Vector diagA_sqrt_inv(*map);
  Epetra_Vector diagA_inv(*map);
  
  if (_diagonalScaling && !useAztecToScaleDiagonally) {
    int length = scale_vector.MyLength();
    for (int i=0; i<length; i++) scale_vector[i] = 1.0 / sqrt(fabs(diagA[i]));

    
//    EpetraExt::MultiVectorToMatlabFile("/tmp/diagA_inv.dat",diagA_inv);
    problem().LeftScale(scale_vector);
    problem().RightScale(scale_vector);
//    A->LeftScale(diagA_sqrt_inv);
//    A->RightScale(diagA_sqrt_inv);
//    b->Multiply(1.0, diagA_inv, *b, 0);
//    EpetraExt::MultiVectorToMatlabFile("/tmp/b_post_scaling.dat",*b);
    
//    EpetraExt::RowMatrixToMatlabFile("/tmp/A_post_scaling.dat",*A);
  }
  
  Teuchos::RCP<Epetra_MultiVector> diagA_ptr = Teuchos::rcp( &diagA, false );

  _gmgOperator.setStiffnessDiagonal(diagA_ptr);
  
  _gmgOperator.setApplyDiagonalSmoothing(_diagonalSmoothing);
  _gmgOperator.setFineSolverUsesDiagonalScaling(_diagonalScaling);

  if (_diagonalScaling && useAztecToScaleDiagonally) {
    solver.SetAztecOption(AZ_scaling, AZ_sym_diag);
  } else {
    solver.SetAztecOption(AZ_scaling, AZ_none);
  }
  if (_computeCondest) {
    solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
  } else {
    solver.SetAztecOption(AZ_solver, AZ_cg);
  }
  
  solver.SetPrecOperator(&_gmgOperator);
//  solver.SetAztecOption(AZ_precond, AZ_none);
  solver.SetAztecOption(AZ_precond, AZ_user_precond);
  solver.SetAztecOption(AZ_conv, AZ_rhs);
//  solver.SetAztecOption(AZ_output, AZ_last);
  solver.SetAztecOption(AZ_output, _azOutput);
  
  int solveResult = solver.Iterate(_maxIters,_tol);
  
  const double* status = solver.GetAztecStatus();
  int remainingIters = _maxIters;

  int whyTerminated = status[AZ_why];
  int maxRestarts = 1;
  int numRestarts = 0;
  while ((whyTerminated==AZ_loss) && (numRestarts < maxRestarts)) {
    remainingIters -= status[AZ_its];
    if (rank==0) cout << "Aztec warned that the recursive residual indicates convergence even though the true residual is too large.  Restarting with the new solution as initial guess, with maxIters = " << remainingIters << endl;
    solver.Iterate(remainingIters,_tol);
    whyTerminated = status[AZ_why];
    numRestarts++;
  }
  
  if (rank==0) {
    switch (whyTerminated) {
      case AZ_normal:
//        cout << "whyTerminated: AZ_normal " << endl;
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
  
  if (_diagonalScaling && !useAztecToScaleDiagonally) {
    // reverse the scaling here
    scale_vector.Reciprocal(scale_vector);
    problem().LeftScale(scale_vector);
    problem().RightScale(scale_vector);
//    A->LeftScale(diagA_sqrt);
//    A->RightScale(diagA_sqrt);
//    b->Multiply(1.0, diagA, *b, 0);
//    EpetraExt::MultiVectorToMatlabFile("/tmp/b_post_unscaling.dat",*b);

//    Epetra_MultiVector *x = problem().GetLHS();
//    EpetraExt::MultiVectorToMatlabFile("/tmp/x.dat",*x);
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

void GMGSolver::setAztecOutput(int value) {
  _azOutput = value;
}

void GMGSolver::setComputeConditionNumberEstimate(bool value) {
  _computeCondest = value;
}

void GMGSolver::setUseDiagonalScaling(bool value) {
  _diagonalScaling = value;
  _gmgOperator.setFineSolverUsesDiagonalScaling(value);
}
