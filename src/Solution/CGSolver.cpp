//
//  CGSolver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "CGSolver.h"

CGSolver::CGSolver(int maxIters, double tol) {
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
}

void CGSolver::setPrintToConsole(bool printToConsole) {
  _printToConsole = printToConsole;
}

void CGSolver::setTolerance(double tol) {
  _tol = tol;
}

int CGSolver::solve() {
  // compute some statistics for the original problem
//  double condest = -1;
//  AztecOO solverForConditionEstimate(problem());
//  solverForConditionEstimate.SetAztecOption(AZ_solver, AZ_cg_condnum);
//  solverForConditionEstimate.ConstructPreconditioner(condest);
//  Epetra_RowMatrix *A = problem().GetMatrix();
//  double norminf = A->NormInf();
//  double normone = A->NormOne(); 
//  if (_printToConsole) {
//    cout << "\n Inf-norm of stiffness matrix = " << norminf;
//    cout << "\n One-norm of stiffness matrix = " << normone << endl << endl;
//    cout << "Condition number estimate: " << condest << endl;
//  }
  
  AztecOO solver(problem());

  // COMBO KNOWN TO WORK FOR STOKES (at least): GMRES + Jacobi.  It can be slow to converge, though.
  // (I've used a tol of 1e-8.)
  
//  solver.SetAztecOption(AZ_solver, AZ_cg);        
  solver.SetAztecOption(AZ_solver, AZ_gmres);
//  solver.SetAztecOption(AZ_precond, AZ_none);     // no preconditioner
  solver.SetAztecOption(AZ_precond, AZ_Jacobi);   // Jacobi preconditioner
//  solver.SetAztecOption(AZ_precond, AZ_sym_GS);   // Gauss-Seidel preconditioner
    
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
  
  Epetra_RowMatrix *A = problem().GetMatrix();
  double norminf = A->NormInf();
  double normone = A->NormOne(); 
  
  int numIters = solver.NumIters();
  
  if (_printToConsole) {
    cout << "\n Inf-norm of stiffness matrix after scaling = " << norminf;
    cout << "\n One-norm of stiffness matrix after scaling = " << normone << endl << endl;
    cout << "Num iterations: " << numIters << endl;
  }
  
  return solveResult;
}
