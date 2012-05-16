//
//  Solver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Solver_h
#define Camellia_Solver_h

#include "Epetra_LinearProblem.h"
#include "Amesos_Klu.h"
#include "AztecOO.h"

// abstract class for solving Epetra_LinearProblem problems
class Solver {
private:
  Teuchos::RCP< Epetra_LinearProblem > _problem;
public:
  virtual Epetra_LinearProblem & problem() { return *(_problem.get()); }
  virtual void setProblem(Teuchos::RCP< Epetra_LinearProblem > problem) {
    _problem = problem;
  }
  virtual int solve() = 0; // solve with an error code response
};

// some concrete implementations follow…
class KluSolver : public Solver {
public:
  int solve() {
    Amesos_Klu klu(problem());
    return klu.Solve();
  }
};

// only use MUMPS when we have MPI
#ifdef HAVE_MPI
#include "Amesos_Mumps.h"
class MumpsSolver : public Solver {
public:
  int solve() {
    Amesos_Mumps mumps(problem());
    mumps.SymbolicFactorization();
    mumps.NumericFactorization();
    int relaxationParam = 0; // the default
    int* info = mumps.GetINFO();
    int numErrors = 0;
    while (info[0] < 0) { // error occurred on another processor
      numErrors++;
      if (info[0] == -9) { // out of memory
        // mumps_par%ICNTL(14) = mumps_par%ICNTL(14) + 20
        relaxationParam += 20;
        mumps.SetICNTL(14, relaxationParam);
        cout << "MUMPS memory allocation too small.  Relaxation Param: " << relaxationParam << endl;
      }
      mumps.SymbolicFactorization();
      mumps.NumericFactorization();
      if (numErrors > 20) {
        cout << "Too many errors in MUMPS factorization.\n";
        break;
      }
    }
    return mumps.Solve();
  }
};
#else
class MumpsSolver : public Solver {
public:
  int solve() {
    cout << "ERROR: no MUMPS support for non-MPI runs yet (because Nate hasn't built MUMPS for his serial-debug Trilinos).\n";
    return -1;
  }
};
#endif

#endif
