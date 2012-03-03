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
