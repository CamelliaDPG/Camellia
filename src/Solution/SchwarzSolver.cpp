//
//  SchwarzSolver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "AztecOO.h"
#include "SchwarzSolver.h"

SchwarzSolver::SchwarzSolver(int overlapLevel, int maxIters, double tol) {
  _overlapLevel = overlapLevel;
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
}

void SchwarzSolver::setPrintToConsole(bool printToConsole) {
  _printToConsole = printToConsole;
}

void SchwarzSolver::setTolerance(double tol) {
  _tol = tol;
}

int SchwarzSolver::solve() {
  // compute some statistics for the original problem
  double condest = -1;
  AztecOO solverForConditionEstimate(problem());
  solverForConditionEstimate.SetAztecOption(AZ_solver, AZ_cg_condnum);
  solverForConditionEstimate.ConstructPreconditioner(condest);
  Epetra_RowMatrix *A = problem().GetMatrix();
  double norminf = A->NormInf();
  double normone = A->NormOne(); 
  if (_printToConsole) {
    cout << "\n Inf-norm of stiffness matrix before scaling = " << norminf;
    cout << "\n One-norm of stiffness matrix before scaling = " << normone << endl << endl;
    cout << "Condition number estimate: " << condest << endl;
  }
  
  AztecOO solver(problem());
  
  int otherRows = A->NumGlobalRows() - A->NumMyRows();
  int overlapLevel = std::min(otherRows,_overlapLevel);
  
  solver.SetAztecOption(AZ_precond, AZ_dom_decomp);   // additive schwarz
  solver.SetAztecOption(AZ_overlap, overlapLevel);   // level of overlap for schwarz
  solver.SetAztecOption(AZ_type_overlap, AZ_symmetric);
  solver.SetAztecOption(AZ_solver, AZ_cg_condnum); // more expensive than AZ_cg, but allows estimate of condition #
  solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut); // TODO: look these up (copied from example)
  solver.SetAztecParam(AZ_ilut_fill, 1.0);            // TODO: look these up (copied from example)
  solver.SetAztecParam(AZ_drop, 0.0);                 // TODO: look these up (copied from example)
  
  
  int solveResult = solver.Iterate(_maxIters,_tol);
//  int solveResult = solver.AdaptiveIterate(_maxIters,1,_tol); // an experiment (was Iterate())
  
  norminf = A->NormInf();
  normone = A->NormOne();
  condest = solver.Condest();
  int numIters = solver.NumIters();
  
  if (_printToConsole) {
    cout << "\n Inf-norm of stiffness matrix after scaling = " << norminf;
    cout << "\n One-norm of stiffness matrix after scaling = " << normone << endl << endl;
    cout << "Condition number estimate: " << condest << endl;
    cout << "Num iterations: " << numIters << endl;
  }
  
  return solveResult;
}
