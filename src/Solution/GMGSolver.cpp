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

const bool DIAGONAL_SCALING_DEFAULT = false;

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
                      
  _useCG = true;
  _azConvergenceOption = AZ_rhs;
}

double GMGSolver::condest() {
  return _condest;
}

int GMGSolver::iterationCount() {
  return _iterationCount;
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
  
  // in place of doing the scaling ourselves, for the moment I've switched
  // over to using Aztec's built-in scaling.  This appears to be functionally identical.
  bool useAztecToScaleDiagonally = true;
  
  AztecOO solver(problem());
  
  Epetra_CrsMatrix *A = dynamic_cast<Epetra_CrsMatrix *>( problem().GetMatrix() );
  
  if (A == NULL) {
    cout << "Error: GMGSolver requires an Epetra_CrsMatrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: GMGSolver requires an Epetra_CrsMatrix.\n");
  }
  
//  EpetraExt::RowMatrixToMatlabFile("/tmp/A_pre_scaling.dat",*A);

//  Epetra_MultiVector *b = problem().GetRHS();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/b_pre_scaling.dat",*b);

//  Epetra_MultiVector *x = problem().GetLHS();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/x_initial_guess.dat",*x);
  
  const Epetra_Map* map = &A->RowMatrixRowMap();
  
  Epetra_Vector diagA(*map);
  A->ExtractDiagonalCopy(diagA);

//  EpetraExt::MultiVectorToMatlabFile("/tmp/diagA.dat",diagA);
//  
  Epetra_Vector scale_vector(*map);
  Epetra_Vector diagA_sqrt_inv(*map);
  Epetra_Vector diagA_inv(*map);
  
  if (_diagonalScaling && !useAztecToScaleDiagonally) {
    int length = scale_vector.MyLength();
    for (int i=0; i<length; i++) scale_vector[i] = 1.0 / sqrt(fabs(diagA[i]));

    problem().LeftScale(scale_vector);
    problem().RightScale(scale_vector);
  }
  
  Teuchos::RCP<Epetra_MultiVector> diagA_ptr = Teuchos::rcp( &diagA, false );

  _gmgOperator.setStiffnessDiagonal(diagA_ptr);
  
  _gmgOperator.setApplyDiagonalSmoothing(_diagonalSmoothing);
  _gmgOperator.setFineSolverUsesDiagonalScaling(_diagonalScaling);
  
  _gmgOperator.computeCoarseStiffnessMatrix(A);

  if (_diagonalScaling && useAztecToScaleDiagonally) {
    solver.SetAztecOption(AZ_scaling, AZ_sym_diag);
  } else {
    solver.SetAztecOption(AZ_scaling, AZ_none);
  }
  if (_useCG) {
    if (_computeCondest) {
      solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
    } else {
      solver.SetAztecOption(AZ_solver, AZ_cg);
    }
  } else {
    solver.SetAztecOption(AZ_kspace, 200); // default is 30
    if (_computeCondest) {
      solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);
    } else {
      solver.SetAztecOption(AZ_solver, AZ_gmres);
    }
  }
  
  solver.SetPrecOperator(&_gmgOperator);
//  solver.SetAztecOption(AZ_precond, AZ_none);
  solver.SetAztecOption(AZ_precond, AZ_user_precond);
  solver.SetAztecOption(AZ_conv, _azConvergenceOption);
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
    solveResult = solver.Iterate(remainingIters,_tol);
    whyTerminated = status[AZ_why];
    numRestarts++;
  }
  remainingIters -= status[AZ_its];
  _iterationCount = _maxIters - remainingIters;
  _condest = solver.Condest(); // will be -1 if running without condest
  
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

void GMGSolver::setAztecConvergenceOption(int value) {
  _azConvergenceOption = value;
}

void GMGSolver::setAztecOutput(int value) {
  _azOutput = value;
}

void GMGSolver::setComputeConditionNumberEstimate(bool value) {
  _computeCondest = value;
}

void GMGSolver::setUseConjugateGradient(bool value) {
  _useCG = value;
}

void GMGSolver::setUseDiagonalScaling(bool value) {
  _diagonalScaling = value;
  _gmgOperator.setFineSolverUsesDiagonalScaling(value);
}
